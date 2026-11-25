"""
Autotrading orchestration: mirrors the signal engine into real OKX EU positions
for every onboarded user.

Lifecycle glue (called from main.py):
  open_positions_for_signal(sig)  — after a signal is published
  update_trailing(sig, stop_px)   — every monitor cycle while TP1_PARTIAL
  mirror_transition(sig, status)  — when the engine closes/transitions a signal
  poll_exchange_closes()          — every monitor cycle, ALL open positions:
    the exchange-side OCO fires the instant SL/TP is touched, but the signal
    engine only learns about it on its own kline-based check (adds minutes
    of lag for the DM). This polls real position size directly so the user
    is notified within one monitor cycle of the real close, independent of
    when/whether the engine's own status transition catches up.

Per-user flow on open:
  decrypt keys → current balance → client percentage size → set 10x isolated →
  market entry → protection OCO (SL at engine stop, TP at tp2) → DM the user.

Fail-safe rules:
  - a user erroring out never blocks other users;
  - if the protection OCO can't be placed, the naked position is closed
    immediately — never leave a position without a stop;
  - closes treat "already flat" (exchange algo fired first) as success.
"""
import logging
import json
import hashlib
import math
from datetime import datetime, timezone
import threading
from collections import defaultdict
import time

import requests

from src.risk_limits import (
    account_margin_allows, account_position_count_allows, equity_guard)
from config import (
    AUTOTRADE_MAX_DAILY_LOSS, AUTOTRADE_MAX_DRAWDOWN, AUTOTRADE_EQUITY_PEAK_FLOOR,
    AUTOTRADE_MAX_ACCOUNT_MARGIN_FRACTION, AUTOTRADE_ACCOUNT_WRITER_COUNT,
    AUTOTRADE_MAX_ACCOUNT_POSITIONS,
    TELEGRAM_TOKEN,
    AUTOTRADE_ENABLED, AUTOTRADE_LEVERAGE,
    AUTOTRADE_CONTACT,
    STOCK_LIVE_FORWARD_WINDOW, STOCK_LIVE_FORWARD_MIN_TOTAL_R,
    STOP_CLOSE_CONFIRM, STOP_EXCHANGE_BACKSTOP_R,
)
from src.db import (
    get_bot_state, set_bot_state, compare_and_set_bot_state,
    at_get_active_traders, at_get, at_set_balance,
    at_create_entry_intent, at_mark_entry_submitting, at_release_unsent_entry,
    at_update_entry_intent_protection, at_activate_entry_intent,
    at_open_positions_for_signal, at_update_position_sl,
    at_close_position, at_all_open_positions, at_reduce_position_sz,
    at_has_open_position,
    get_signal_by_id,
    get_route_live_readiness,
    set_signal_size_mult,
    record_signal_execution_book,
)
from src.keystore import decrypt_secret, keystore_ready
from src.autotrade_risk_status import risk_status_text
from src.drawdown_recovery import (
    checked_equity_guard, drawdown_reset_version, failed_equity_check, manual_reset_equity_guard,
)
from src import okx_trader as okx
from src.shared_capacity_runtime import SharedCapacityRuntime
from src.binance_client import get_xperp_instruments

log = logging.getLogger(__name__)

_STATUS_RU = {
    "TP1_PARTIAL": "TP1 достигнут — остаток под защитой стопа",
    "TP2_HIT":     "TP2 достигнут — позиция закрыта",
    "TP1_TRAIL":   "трейлинг-стоп сработал — позиция закрыта",
    "SL_HIT":      "стоп-лосс — позиция закрыта",
    "BREAKEVEN":   "сработал стоп б/у — позиция закрыта",
    "EXPIRED":     "сигнал истёк — позиция закрыта",
    "TP1_EXPIRED": "истёк после TP1 — позиция закрыта",
    "TP1_STOP":    "остаток закрылся по исходному стопу после TP1",
}

# Statuses that require closing whatever is still open on the exchange.
_CLOSE_STATUSES = ("TP2_HIT", "TP1_TRAIL", "SL_HIT", "BREAKEVEN",
                   "EXPIRED", "TP1_EXPIRED", "TP1_STOP")

# A position younger than this is never declared closed from a zero size read —
# /account/positions can lag a fresh market fill. One monitor cycle is 60s, so
# this costs at most one extra cycle of notification latency on a real close.
_CLOSE_CONFIRM_MIN_AGE_SEC = 90
_ENTRY_PENDING_GRACE_SEC = 90
_ACCOUNT_SNAPSHOT_MAX_AGE_SEC = 15
_LOCAL_ACCOUNT_SLOTS = 8
_shared_capacity = SharedCapacityRuntime('stocks')


def maintain_shared_capacity() -> None:
    """Independent job: coordinator outages must not delay TP/SL monitoring."""
    try:
        _shared_capacity.maintenance(_creds_of)
    except Exception as exc:
        log.warning('shared capacity maintenance deferred (%s)',type(exc).__name__)


def _dm(user_id: int, text: str) -> None:
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": user_id, "text": text, "parse_mode": "Markdown"},
            timeout=10,
        )
        # Markdown parse failure (unbalanced _ / * in an error string or
        # username) silently drops the message — retry as plain text.
        if resp.status_code != 200:
            requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                json={"chat_id": user_id, "text": text},
                timeout=10,
            )
    except Exception as e:
        log.warning(f"autotrade DM to {user_id} failed: {e}")


def _creds_of(u: dict) -> dict | None:
    try:
        return {
            "api_key":    decrypt_secret(u["api_key_enc"]),
            "api_secret": decrypt_secret(u["api_secret_enc"]),
            "passphrase": decrypt_secret(u["passphrase_enc"]),
        }
    except Exception as e:
        log.warning(f"autotrade decrypt failed for {u['user_id']}: {e}")
        return None


def _base_of(symbol: str) -> str:
    return symbol[:-4] if symbol.endswith("USDT") else symbol


def _inst_id_of(symbol: str) -> str | None:
    return get_xperp_instruments().get(_base_of(symbol))


def _margin_for(u: dict, balance: float) -> float:
    """Return margin as the client's percentage of the latest balance."""
    if u.get("size_mode") != "percent":
        return 0.0
    pct = float(u.get("size_value") or 0)
    if not 1.0 <= pct <= 10.0 or balance <= 0:
        return 0.0
    return balance * pct / 100.0


_entry_locks = defaultdict(threading.Lock)
_account_entry_locks = defaultdict(threading.Lock)


def reset_user_drawdown(user_id: int, version: str) -> tuple[str, dict | None]:
    """Rebase the saved drawdown using a fresh balance from this user's account."""
    lock = _entry_locks[user_id]
    if not lock.acquire(blocking=False):
        return "busy", None
    try:
        user = at_get(user_id)
        if not user or not user.get("allowed"):
            return "not_connected", None
        key = f"risk_guard:{user_id}"
        raw = get_bot_state(key)
        state = json.loads(raw or "{}")
        if not version or drawdown_reset_version(state) != version:
            return "stale", None
        creds = _creds_of(user)
        if not creds:
            return "not_connected", None
        ok, balance = okx.get_balance(creds)
        if not ok:
            return "balance_unavailable", None
        latest = at_get(user_id)
        if not latest or not latest.get("allowed") or _creds_of(latest) != creds:
            return "stale", None
        state = manual_reset_equity_guard(state, equity=balance, now=time.time(),
            max_daily_loss=AUTOTRADE_MAX_DAILY_LOSS, max_drawdown=AUTOTRADE_MAX_DRAWDOWN)
        if not compare_and_set_bot_state(key, raw, json.dumps(state)):
            return "stale", None
        log.info("User reset a drawdown pause using fresh account equity")
        return "reset", state
    except Exception as exc:
        log.warning("Drawdown reset unavailable (%s)", type(exc).__name__)
        return "unavailable", None
    finally:
        lock.release()


def maintain_equity_guards() -> None:
    """Fresh checks for current connected clients; independent of TP/SL jobs."""
    if not AUTOTRADE_ENABLED:
        return
    try:
        users=at_get_active_traders()
    except Exception:
        log.warning('Equity recovery users unavailable')
        return
    for row in users:
        uid=row['user_id']
        lock=_entry_locks[uid]
        if not lock.acquire(blocking=False):
            continue
        try:
            user=at_get(uid)
            if not user or not user.get('allowed') or not user.get('active'):
                continue
            key=f'risk_guard:{uid}'
            raw=get_bot_state(key)
            state=json.loads(raw or '{}')
            if not isinstance(state,dict) or state.get('paused') is not True:
                continue
            creds=_creds_of(user)
            ok,balance=okx.get_balance(creds) if creds else (False,None)
            now=time.time()
            if not ok:
                result=failed_equity_check(state,now)
            else:
                result,blocked=checked_equity_guard(state,equity=balance,now=now,
                    max_daily_loss=AUTOTRADE_MAX_DAILY_LOSS,max_drawdown=AUTOTRADE_MAX_DRAWDOWN,
                    peak_floor=max(float(user.get('last_balance') or 0),AUTOTRADE_EQUITY_PEAK_FLOOR))
                result.update(last_block=blocked,equity=balance,checked_at=now,
                    max_daily_loss=AUTOTRADE_MAX_DAILY_LOSS,max_drawdown=AUTOTRADE_MAX_DRAWDOWN)
            if compare_and_set_bot_state(key,raw,json.dumps(result)) and not result['paused']:
                log.info('Equity drawdown pause recovered after sustained fresh checks inside existing limits')
        except Exception as exc:
            log.warning('Equity recovery check deferred (%s)',type(exc).__name__)
        finally:
            lock.release()


def _entry_skip(user_id: int, sig: dict, inst_id: str, reason: str,
                message: str | None = None) -> None:
    """Persist a user-visible reason whenever a published signal cannot open."""
    payload = {
        "ts": time.time(),
        "signal_id": sig.get("id"),
        "symbol": sig.get("symbol"),
        "direction": sig.get("direction"),
        "inst_id": inst_id,
        "reason": reason,
    }
    log.info("autotrade entry skipped: signal=%s instrument=%s direction=%s reason=%s",
             payload["signal_id"], inst_id, payload["direction"], reason)
    try:
        set_bot_state(f"autotrade_last_skip:{user_id}", json.dumps(payload))
    except Exception as exc:
        log.warning("autotrade skip audit failed %s %s: %s",
                    user_id, sig.get("id"), exc)
    if message:
        _dm(user_id, message)


def _entry_client_order_id(user_id: int, sig: dict, inst_id: str) -> str:
    """Stable 32-char OKX id for recovering a lost market-order response."""
    material = "|".join((
        "stock-v1", str(user_id), str(sig.get("id") or ""), inst_id,
        str(sig.get("opened_at") or ""), str(sig.get("source") or ""),
    ))
    return "s1" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:30]


def _open_for_user(u: dict, sig: dict, inst_id: str, disp: str) -> None:
    # Keep balance, deduplication, risk budgeting and entry atomic per local user.
    with _entry_locks[u['user_id']]:
        _open_for_user_locked(u, sig, inst_id, disp)


def _open_for_user_locked(u: dict, sig: dict, inst_id: str, disp: str) -> None:
    uid = u["user_id"]
    creds = _creds_of(u)
    if not creds:
        _entry_skip(
            uid, sig, inst_id, "API-ключи не удалось расшифровать",
            f"⚠️ Автотрейдинг: {disp} пропущен — не удалось прочитать "
            "подключённые API-ключи. Открой настройки и замени ключи.")
        return

    # One instrument, one position. A second entry on the same instId cannot be
    # protected: OKX permits a single closeFraction=1 TP/SL algo per position,
    # so place_protection_oco below would fail and the handler would close the
    # position — flattening the FIRST trade too. Cost a live trade on the sister
    # crypto bot 2026-08-13 (PUMP), caused by a duplicate signal.
    if at_has_open_position(uid, inst_id):
        log.warning(f"autotrade: {uid} already holds {inst_id} — second entry refused")
        _dm(uid, f"ℹ️ Автотрейдинг: по {disp} уже есть открытая позиция — "
                 f"повторный сигнал пропущен.")
        return

    account_ok, account_check = okx.validate_autotrade_account(creds)
    if not account_ok:
        reason = str((account_check or {}).get("reason") or "config_unavailable")
        log.warning("autotrade account incompatible for %s: %s", uid, account_check)
        messages = {
            "long_short_mode": (
                "режим позиций Long/Short",
                f"⚠️ Автотрейдинг: {disp} пропущен — аккаунт OKX работает в "
                "режиме Long/Short. Закрой открытые позиции и переключи режим "
                "позиций на Net (односторонний)."),
            "trade_permission_missing": (
                "у API-ключа нет права Trade",
                f"⚠️ Автотрейдинг: {disp} пропущен — у API-ключа OKX нет права "
                "Trade. Создай ключ с правами Read + Trade."),
        }
        audit_reason, message = messages.get(reason, (
            "конфигурацию API-ключа OKX не удалось проверить",
            f"⚠️ Автотрейдинг: {disp} пропущен — не удалось проверить "
            "конфигурацию API-ключа OKX."))
        from src.okx_connection_check import account_failure_label
        key_failure = account_failure_label(account_check or {})
        if key_failure:
            audit_reason = key_failure
            message = (f"⚠️ Автотрейдинг: {disp} пропущен. {key_failure}. "
                       "В «Автотрейдинге» нажми «Проверить подключение OKX».")
        _entry_skip(
            uid, sig, inst_id, audit_reason,
            message + " Бот отказался от входа для безопасности.")
        return

    account_key = okx.account_key(account_check)
    if not account_key:
        _entry_skip(uid, sig, inst_id, "OKX account identity unavailable",
                    f"⚠️ Автотрейдинг: {disp} пропущен — OKX не подтвердил "
                    "идентификатор счёта для проверки общего лимита позиций.")
        return
    with _account_entry_locks[account_key]:
        _open_for_account_locked(u, sig, inst_id, disp, creds, account_key)


def _open_for_account_locked(
    u: dict, sig: dict, inst_id: str, disp: str, creds: dict,
    account_key: str | None,
) -> None:
    uid = u["user_id"]
    if not _shared_capacity.prepare(_creds_of):
        _entry_skip(uid, sig, inst_id, "shared reservation is not ready",
                    f"⏸ Автотрейдинг: {disp} пропущен — общая бронь счёта "
                    "ещё не готова. Открытые позиции продолжают сопровождаться.")
        return
    _writers = 1 if _shared_capacity.enabled else AUTOTRADE_ACCOUNT_WRITER_COUNT
    deadline_ms = None
    native_signal = sig.get("source") == "stock_xperp_native_shadow"
    if native_signal:
        from src.stock_xperp_native_shadow import MAX_SIGNAL_AGE_SECONDS, backing_level_allows_entry

        entry_bar = sig.get("strategy_entry_bar_ts")
        if entry_bar:
            deadline_ms = int((float(entry_bar) + MAX_SIGNAL_AGE_SECONDS) * 1000)
    if deadline_ms is not None and time.time() * 1000 >= deadline_ms:
        _entry_skip(uid, sig, inst_id, "истёк срок входа",
                    f"ℹ️ Автотрейдинг: {disp} пропущен — истёк срок входа.")
        return
    if native_signal:
        try:
            backing_level = float(sig.get("_native_backing_level"))
            if not math.isfinite(backing_level) or backing_level <= 0:
                raise ValueError("Invalid native backing level")
        except (TypeError, ValueError, OverflowError):
            _entry_skip(uid, sig, inst_id, "опорный уровень сигнала недоступен",
                        f"ℹ️ Автотрейдинг: {disp} пропущен — опорный уровень сигнала недоступен.")
            return
        backing = {"direction": sig["direction"], "backing_level": backing_level}

    ok, balance = okx.get_balance(creds)
    if not ok:
        log.warning(f"autotrade balance failed for {uid}: {balance}")
        _dm(uid, f"⚠️ Автотрейдинг: не смог прочитать баланс OKX — сделка по {disp} пропущена.\n`{balance}`")
        return
    at_set_balance(uid, balance)
    try:
        state_key = f"risk_guard:{uid}"
        state = json.loads(get_bot_state(state_key) or '{}')
        previous_balance = float(u.get('last_balance') or 0)
        peak_floor = max(previous_balance, AUTOTRADE_EQUITY_PEAK_FLOOR)
        state, blocked = checked_equity_guard(state, equity=balance,
            now=time.time(),
            max_daily_loss=AUTOTRADE_MAX_DAILY_LOSS, max_drawdown=AUTOTRADE_MAX_DRAWDOWN,
            peak_floor=peak_floor)
        state["last_block"] = blocked
        state.update(equity=balance, checked_at=time.time(),
                     max_daily_loss=AUTOTRADE_MAX_DAILY_LOSS,
                     max_drawdown=AUTOTRADE_MAX_DRAWDOWN)
        set_bot_state(state_key, json.dumps(state))
        if blocked:
            _entry_skip(
                uid, sig, inst_id, f"лимит потерь: {blocked}",
                f"⏸ Автотрейдинг: {disp} пропущен по лимиту потерь. "
                "Открытые позиции продолжают сопровождаться.\n"
                + risk_status_text(state, now=time.time()))
            log.warning("autotrade risk guard: %s; drawdown_pct=%.3f; "
                        "drawdown_limit_pct=%.3f; latched=%s; snapshot_day=%s",
                        blocked, 100 * max(0., 1 - balance / state["peak"]),
                        100 * AUTOTRADE_MAX_DRAWDOWN, state["paused"], state["day"])
            return
    except Exception as exc:
        log.error("autotrade risk state unavailable; entry refused: %s", exc)
        _entry_skip(
            uid, sig, inst_id, "проверка лимита потерь недоступна",
            f"⚠️ Автотрейдинг: {disp} пропущен — временно не удалось "
            "проверить лимит потерь. Бот отказался от входа для безопасности.")
        return

    # The client owns position sizing. Strategy/regime fields may decide
    # whether a setup is traded, but they must not change the requested money.
    _size_mult = 1.0
    try:
        set_signal_size_mult(sig["id"], _size_mult)
    except Exception as _sm_err:
        log.warning("could not record client size for signal %s: %s",
                    sig.get("id"), _sm_err)
    margin = _margin_for(u, balance)
    if margin <= 0:
        _entry_skip(
            uid, sig, inst_id, "неверно задан процент сделки",
            f"⚠️ Автотрейдинг: {disp} пропущен — процент сделки не задан "
            "или находится вне диапазона 1–10%. Проверь настройки.")
        return
    if margin > balance:
        _dm(uid, f"⚠️ Автотрейдинг: на балансе ${balance:.2f} меньше, чем размер сделки ${margin:.2f} — {disp} пропущен.")
        return

    # The exchange is the only shared state visible to both Railway services.
    # Refuse on read/parse failure: proceeding would make the 90% account cap a
    # guess. Reserve one same-sized peer entry so simultaneous crypto + stock
    # decisions cannot both pass from an 87%-occupied snapshot and reach 93%.
    margin_ok, account_positions = okx.get_open_position_margin(creds)
    if not margin_ok:
        log.warning("autotrade account margin unavailable for %s: %s",
                    uid, account_positions)
        _entry_skip(uid, sig, inst_id, "OKX account snapshot unavailable",
                    "⚠️ Автотрейдинг: не смог проверить общую загрузку счёта OKX — "
                 f"сделка по {disp} пропущена.\n`{account_positions}`")
        return
    # A submitted local entry can be absent from a lagging OKX snapshot.
    # A visible partial fill may still have an unresolved entry intent.
    local_positions = [
        row for row in at_all_open_positions()
        if (row["user_id"] == uid
            or (account_key and row.get("account_key") == account_key))
    ]
    local_instruments = {row["inst_id"] for row in local_positions}
    pending_entry = any(row.get("status") == "ENTRY_PENDING" for row in local_positions)
    if pending_entry or not local_instruments.issubset(account_positions["inst_ids"]):
        reason = ("предыдущая заявка ещё сверяется с OKX" if pending_entry else
                  "локальный список позиций расходится со снимком OKX")
        _entry_skip(uid, sig, inst_id, reason,
                    f"Автотрейдинг: {disp} пропущен. {reason}.")
        return
    if inst_id in account_positions["inst_ids"]:
        log.warning("autotrade: %s already holds %s on exchange — entry refused",
                    uid, inst_id)
        _entry_skip(uid, sig, inst_id, "instrument already open on shared account",
                    f"ℹ️ Автотрейдинг: по {disp} уже есть позиция на общем счёте — "
                 "повторный сигнал пропущен.")
        return
    if not account_position_count_allows(
            occupied_positions=account_positions["count"],
            max_positions=AUTOTRADE_MAX_ACCOUNT_POSITIONS,
            simultaneous_writers=_writers):
        count = int(account_positions["count"])
        log.warning("autotrade shared position cap %s: open %s, limit %s",
                    uid, count, AUTOTRADE_MAX_ACCOUNT_POSITIONS)
        _entry_skip(uid, sig, inst_id, "shared account position cap",
                    f"ℹ️ Автотрейдинг: на общем счёте уже {count} "
                 f"открытых позиций. Лимит {AUTOTRADE_MAX_ACCOUNT_POSITIONS}; "
                 f"сделка по {disp} пропущена.")
        return
    if not account_margin_allows(
            equity=balance,
            occupied_margin=account_positions["margin_usd"],
            requested_margin=margin,
            max_fraction=AUTOTRADE_MAX_ACCOUNT_MARGIN_FRACTION,
            simultaneous_writers=_writers):
        used = 100*float(account_positions["margin_usd"])/balance
        limit = 100*AUTOTRADE_MAX_ACCOUNT_MARGIN_FRACTION
        log.warning("autotrade shared margin cap %s: used %.2f%%, request %.2f%%",
                    uid, used, 100*margin/balance)
        _entry_skip(uid, sig, inst_id, "shared account margin cap",
                    f"ℹ️ Автотрейдинг: общая загрузка счёта уже {used:.1f}%. "
                 f"Для безопасного лимита {limit:.0f}% сделка по {disp} пропущена.")
        return

    spec = okx.get_xperp_spec(inst_id)
    try:
        if not spec:
            raise ValueError("contract specification unavailable")
        pos_ok, exchange_size = okx.get_position_size(creds, inst_id)
        if not pos_ok or exchange_size != 0:
            raise ValueError("exchange instrument is not confirmed flat")
        planned, original_sl = float(sig['entry_price']), float(sig['sl'])
        direction = sig['direction']
        target1 = float(sig['tp1'])
        if not (original_sl < planned < target1 if direction == 'LONG'
                else target1 < planned < original_sl if direction == 'SHORT' else False):
            raise ValueError("invalid entry bracket")
    except (TypeError, ValueError, KeyError) as exc:
        log.warning("autotrade risk validation refused %s: %s", inst_id, exc)
        detail = str(exc)
        friendly = {
            "contract specification unavailable": "биржа не отдала параметры контракта",
            "exchange instrument is not confirmed flat":
                "по инструменту уже есть позиция либо биржа не подтвердила нулевой остаток",
        }.get(detail, "вход не прошёл проверку цены и риска")
        _entry_skip(
            uid, sig, inst_id, friendly,
            f"ℹ️ Автотрейдинг: {disp} пропущен — {friendly}.")
        return
    # Confirm the leverage used to convert the client's margin to contracts.
    _lev_ok, _lev_err = okx.ensure_leverage(creds, inst_id, AUTOTRADE_LEVERAGE)
    if not _lev_ok:
        log.warning(f"autotrade leverage set failed {uid} {inst_id}: {_lev_err}")
        _entry_skip(uid, sig, inst_id, "OKX leverage setup failed",
                    "\n".join([
            f"⚠️ Автотрейдинг: не смог выставить плечо {AUTOTRADE_LEVERAGE}x по {disp}.",
            "Сделка пропущена — иначе размер позиции не соответствовал бы риску.",
            f"`{_lev_err}`",
        ]))
        return

    # Refresh executable depth after leverage setup; last-trade prices may be stale.
    from src.execution_guard import (execution_quote_with_retry, order_book_snapshot,
                                     SpreadLimitError, validate_book_freshness)

    def size_for_price(price):
        return okx.calc_contracts(margin, AUTOTRADE_LEVERAGE, price, spec)

    backing_crossed = "executable price crossed original backing level"
    backing_rejection = "цена в стакане пробила опорный уровень сигнала"

    def validate_entry_quote(quote):
        quote.validate_bracket(original_sl, target1)
        if native_signal and not all(backing_level_allows_entry(backing, price)
                                     for price in (quote.best, quote.worst)):
            raise ValueError(backing_crossed)

    try:
        book, quote = execution_quote_with_retry(
            lambda: okx.get_order_book(inst_id), direction, None,
            clock_ms=lambda: time.time()*1000, deadline_ms=deadline_ms,
            size_for_price=size_for_price)
        book_observed_at = time.time()
        validate_entry_quote(quote)
        sz, px = quote.contracts, quote.average
    except (TypeError, ValueError, KeyError, IndexError) as exc:
        log.warning("autotrade execution filter refused %s: %s", inst_id, exc)
        detail = str(exc)
        friendly = {
            "order size below contract minimum":
                f"маржа ${margin:.2f} меньше минимального объёма по цене стакана",
            "order book unavailable": "стакан OKX недоступен",
            "stale order book": "стакан OKX устарел",
            "entry deadline elapsed": "истёк срок входа",
            "spread exceeds limit": "спред в стакане слишком широкий",
            "insufficient displayed depth": "в стакане недостаточно объёма",
            "depth slippage exceeds limit": "ожидаемое проскальзывание слишком большое",
            "executable price outside signal bracket":
                "исполняемая цена уже вышла за диапазон сигнала",
            backing_crossed: backing_rejection,
        }.get(detail, "стакан или исполняемая цена не прошли проверку")
        if isinstance(exc, SpreadLimitError):
            friendly += (f" ({exc.measured * 100:.4f}%, лимит "
                         f"{exc.limit * 100:.4f}%; 3 проверки)")
        _entry_skip(
            uid, sig, inst_id, friendly,
            f"ℹ️ Автотрейдинг: {disp} пропущен — {friendly}.")
        return

    if deadline_ms is not None and time.time() * 1000 >= deadline_ms:
        _entry_skip(uid, sig, inst_id, "истёк срок входа",
                    f"ℹ️ Автотрейдинг: {disp} пропущен — истёк срок входа.")
        return

    client_order_id = _entry_client_order_id(uid, sig, inst_id)
    try:
        intent_id = at_create_entry_intent(
            sig["id"], uid, inst_id, sig["direction"], sz, px, margin,
            float(sig["sl"]), client_order_id, account_key=account_key,
            slot_limit=_LOCAL_ACCOUNT_SLOTS, expected_api_key_enc=u.get("api_key_enc"))
    except Exception as intent_error:
        log.error("autotrade entry intent failed %s %s: %s",
                  uid, inst_id, intent_error)
        _dm(uid, f"❌ Автотрейдинг: не смог безопасно записать вход по {disp}; "
                 "ордер не отправлен.")
        return
    if not intent_id:
        _entry_skip(uid, sig, inst_id,
                    "локальная бронь входа отклонена")
        return
    account_snapshot_started = time.monotonic()
    account_snapshot_at = time.time()
    margin_ok, latest_positions = okx.get_open_position_margin(creds)
    latest_balance_ok, latest_balance = okx.get_balance(creds)
    if deadline_ms is not None and time.time() * 1000 >= deadline_ms:
        at_close_position(intent_id, "ENTRY_EXPIRED")
        _entry_skip(uid, sig, inst_id, "истёк срок входа",
                    f"ℹ️ Автотрейдинг: {disp} пропущен — истёк срок входа.")
        return
    if (not margin_ok or not latest_balance_ok or latest_balance <= 0
            or inst_id in latest_positions["inst_ids"]
            or not local_instruments.issubset(latest_positions["inst_ids"])
            or not account_position_count_allows(
                occupied_positions=latest_positions["count"],
                max_positions=AUTOTRADE_MAX_ACCOUNT_POSITIONS,
                simultaneous_writers=_writers)
            or not account_margin_allows(
                equity=min(balance, latest_balance),
                occupied_margin=latest_positions["margin_usd"],
                requested_margin=margin,
                max_fraction=AUTOTRADE_MAX_ACCOUNT_MARGIN_FRACTION,
                simultaneous_writers=_writers)
            or time.monotonic() - account_snapshot_started > _ACCOUNT_SNAPSHOT_MAX_AGE_SEC):
        at_close_position(intent_id, "ENTRY_SNAPSHOT_EXPIRED")
        _entry_skip(uid, sig, inst_id, "общий лимит OKX не подтверждён перед отправкой",
                    f"ℹ️ Автотрейдинг: {disp} пропущен — общий лимит OKX не подтверждён.")
        return
    try:
        shared_claim = _shared_capacity.reserve(
            intent_id, account_key, inst_id, client_order_id, margin,
            latest_positions, min(balance, latest_balance), account_snapshot_at)
        _shared_capacity.begin(shared_claim, latest_positions,
                               min(balance, latest_balance), account_snapshot_at)
    except Exception:
        at_close_position(intent_id, "ENTRY_SHARED_REFUSED")
        _entry_skip(uid, sig, inst_id, "shared account reservation refused",
                    f"⏸ Автотрейдинг: {disp} пропущен — общая бронь входа "
                    "не подтверждена. Ордер не отправлен.")
        return
    if _shared_capacity.enabled and (
            time.monotonic() - account_snapshot_started > _ACCOUNT_SNAPSHOT_MAX_AGE_SEC
            or (deadline_ms is not None and time.time() * 1000 >= deadline_ms)):
        at_close_position(intent_id, "ENTRY_SHARED_EXPIRED")
        _entry_skip(uid, sig, inst_id, "entry expired during shared reservation")
        return
    try:
        validate_book_freshness(book, time.time()*1000)
    except ValueError:
        try:
            # The claim already covers a fixed quantity. A fresh book must
            # support that quantity inside the signal and reserved budget.
            book, quote = execution_quote_with_retry(
                lambda: okx.get_order_book(inst_id, timeout=1.0), direction, sz,
                attempts=1, clock_ms=lambda: time.time()*1000, deadline_ms=deadline_ms)
            book_observed_at = time.time()
            validate_entry_quote(quote)
            if sz > size_for_price(max(quote.best, quote.worst)):
                raise ValueError("reserved order size exceeds the refreshed margin budget")
            if time.monotonic() - account_snapshot_started > _ACCOUNT_SNAPSHOT_MAX_AGE_SEC:
                raise ValueError("account snapshot expired during book refresh")
            px = quote.average
        except (TypeError, ValueError, KeyError, IndexError, ArithmeticError) as exc:
            at_close_position(intent_id, "ENTRY_BOOK_CHANGED")
            log.warning("autotrade refreshed book refused %s: %s", inst_id, exc)
            reason = (backing_rejection if str(exc) == backing_crossed
                      else "стакан изменился во время общей брони")
            _entry_skip(uid, sig, inst_id, reason, f"ℹ️ Автотрейдинг: {disp} пропущен — {reason}.")
            return
    if not at_mark_entry_submitting(
            intent_id, account_short_count=latest_positions.get("short_count"),
            account_unknown_side_count=latest_positions.get("unknown_side_count"),
            expected_signal_id=sig["id"]):
        at_release_unsent_entry(intent_id)
        _entry_skip(uid, sig, inst_id, "сигнал или бронь больше не доступны перед отправкой")
        return
    if sig["direction"] == "SHORT":
        log.info(
            "shared short-cap audit signal_id=%s inst=%s shorts=%d unknown_sides=%d total=%d",
            sig["id"], inst_id, latest_positions.get("short_count", 0),
            latest_positions.get("unknown_side_count", 0), latest_positions["count"])
    ok, ord_id = okx.place_market_entry(
        creds, inst_id, sig["direction"], sz, client_order_id,
        deadline_ms=deadline_ms)
    if not ok:
        ambiguous = okx.is_ambiguous_request_error(ord_id)
        if ambiguous:
            recovered, order = okx.get_order_by_client_id(
                creds, inst_id, client_order_id)
            if (recovered and str(order.get("state") or "").lower()
                    in {"live", "partially_filled", "filled"}):
                ok, ord_id = True, str(order.get("ordId") or client_order_id)
                log.warning("autotrade recovered lost entry response %s %s: %s",
                            uid, inst_id, ord_id)
            else:
                pos_ok, current_size = okx.get_position_size(creds, inst_id)
                if pos_ok and current_size > 0:
                    ok, ord_id = True, client_order_id
                    log.warning("autotrade recovered ambiguous entry from position %s %s",
                                uid, inst_id)
                else:
                    # A transport timeout is not an exchange rejection. Keep a
                    # durable unprotected row so the minute reconciler will
                    # close a late-visible fill instead of forgetting it.
                    _dm(uid, "\n".join([
                        f"🚨 *Неопределённый ответ OKX по {disp}*",
                        "",
                        "Биржа не ответила на запрос открытия. Бот будет проверять "
                        "счёт и автоматически закроет позицию, если она появилась.",
                        "До проверки не открывай этот тикер вручную.",
                        f"ID: `{client_order_id}`",
                    ]))
                    log.error("autotrade AMBIGUOUS ENTRY %s %s: %s; lookup=%s",
                              uid, inst_id, ord_id, order)
                    return
        if not ok:
            at_close_position(intent_id, "ENTRY_REJECTED", error=str(ord_id))
            log.warning(f"autotrade entry failed for {uid} {inst_id}: {ord_id}")
            _entry_skip(uid, sig, inst_id, "OKX rejected the market entry",
                        f"❌ Автотрейдинг: не смог открыть {disp} {sig['direction']}.\n`{ord_id}`\nЕсли не понимаешь причину — напиши `{AUTOTRADE_CONTACT}`.")
            return

    tick  = (spec or {}).get("tickSz", 0)
    # Close-confirmed stop: the bot closes the position itself once a 15m candle
    # CLOSES beyond the signal's stop, so the exchange stop must NOT sit on that
    # same level or it fires first on every wick and defeats the mechanism. It
    # moves out to STOP_EXCHANGE_BACKSTOP_R purely as a disaster backstop for
    # when the bot is down (deploy/restart/network) — without it a 10x position
    # would be naked. TP levels are untouched: still anchored to the original 1R.
    # Anchor the backstop to what was REALLY paid, not to the signal's price:
    # entry_price is measured on the global feed at scan time, the market order
    # fills on the X-Perp seconds later, off by the basis plus slippage —
    # always in the same direction.
    _avg_px  = okx.get_position_avg_px(creds, inst_id)
    # The TP1 partial further down is a FIXED-size reduce-only order, so it has
    # to be sized off what the entry ACTUALLY filled rather than what was asked
    # for. A market order that fills short leaves a position smaller than sz;
    # reduceOnly then caps the TP1 order at the position, so "close half at TP1"
    # silently becomes "close all of it" and the runner the strategy is built
    # around never exists. Nothing reverses or over-closes — it just quietly
    # stops being the trade that was measured.
    # Falls back to sz when the read fails, which is the old behaviour.
    _sz_ok, _filled_sz = okx.get_position_size(creds, inst_id)
    _pos_sz = _filled_sz if (_sz_ok and _filled_sz and _filled_sz > 0) else sz
    if _pos_sz != sz:
        log.warning(f"autotrade partial fill {uid} {inst_id}: asked {sz}, got {_pos_sz}")
    _fill_px = _avg_px or px
    _sl_level = float(sig["sl"])
    strategy_stop = sig.get("strategy_stop_on_close")
    close_confirm = STOP_CLOSE_CONFIRM if strategy_stop is None else bool(strategy_stop)
    if close_confirm:
        # Risk distance stays the signal's (that is the strategy's 1R); only the
        # point it is measured FROM moves to the real fill.
        _risk = abs(float(sig["entry_price"]) - _sl_level)
        if _risk > 0:
            _back = _risk * max(1.0, float(STOP_EXCHANGE_BACKSTOP_R))
            _sl_level = (_fill_px - _back) if sig["direction"] == "LONG" else (_fill_px + _back)
    sl_px = okx.round_to_tick(_sl_level, tick)
    tp_px = okx.round_to_tick(float(sig["tp2"]), tick)
    ok, algo_id = okx.place_protection_oco(creds, inst_id, sig["direction"], sl_px, tp_px)
    if not ok:
        # Never leave a naked position: close it right back — and CHECK that the
        # close actually happened. The result used to be discarded, so a failed
        # close still told the owner "closed for safety" while a leveraged
        # position sat open with no stop on it. Retry once, then say plainly that
        # it is still open, because that needs a human immediately.
        _closed, _cerr = okx.close_position_market(creds, inst_id)
        if not _closed:
            _closed, _cerr = okx.close_position_market(creds, inst_id)
        if _closed:
            # Keep the intent until reconciliation confirms that the entry
            # order is terminal and the position is flat.
            log.warning("autotrade OCO failed for %s %s: %s — close submitted",
                        uid, inst_id, algo_id)
            _dm(uid, f"❌ Автотрейдинг: не смог поставить стоп по {disp} — "
                     f"отправлено аварийное закрытие. Бот проверит исполнение.\n`{algo_id}`")
        else:
            log.error(f"autotrade NAKED POSITION {uid} {inst_id}: OCO failed ({algo_id}) "
                      f"AND close failed ({_cerr}) — position is open WITHOUT a stop")
            # Keep failed emergency closes discoverable by reconciliation.
            try:
                if not at_activate_entry_intent(
                        intent_id, _pos_sz, _fill_px, margin, "", sl_px):
                    log.critical("unprotected intent activation failed %s %s",
                                 uid, inst_id)
            except Exception as tracking_error:
                log.critical("unprotected intent could not be activated %s %s: %s",
                             uid, inst_id, tracking_error)
            _msg = "\n".join([
                f"🚨 *СРОЧНО: {disp} открыта БЕЗ СТОПА*",
                "",
                "Не удалось ни поставить защиту, ни закрыть позицию.",
                "Зайди на биржу и закрой её вручную.",
                "",
                f"стоп: `{algo_id}`",
                f"закрытие: `{_cerr}`",
            ])
            _dm(uid, _msg)
        return

    try:
        at_update_entry_intent_protection(intent_id, algo_id, sl_px)
    except Exception as tracking_error:
        log.error("protected entry intent update failed %s %s: %s",
                  uid, inst_id, tracking_error)

    # Optional: pre-place the user's chosen TP1 partial-close as its OWN real
    # exchange order (fixed size, reduceOnly) right now — so the exchange
    # itself fires it the instant TP1 triggers, no bot round-trip latency.
    # Independent of the OCO above (SL/TP2 keep covering whatever remains).
    tp1_algo_id = None
    tp1_sz      = None
    try:
        close_pct = float(u.get("tp1_close_pct") or 0)
    except (TypeError, ValueError):
        close_pct = 0.0
    tp1_line = "TP1 и трейлинг ведёт бот автоматически."
    if close_pct > 0:
        from src.tp1_plan import plan_tp1_close
        partial = plan_tp1_close(_pos_sz, close_pct, spec or {},
            ensure_minimum=bool(u.get('tp1_minimum_close')))
        raw_tp1_sz = partial["size"]
        if raw_tp1_sz > 0:
            tp1_px = okx.round_to_tick(float(sig["tp1"]), tick)
            ok, tp1_result = okx.place_tp1_partial(creds, inst_id, sig["direction"], tp1_px, raw_tp1_sz)
            if ok:
                tp1_algo_id, tp1_sz = tp1_result, raw_tp1_sz
                try:
                    at_update_entry_intent_protection(
                        intent_id, algo_id, sl_px, tp1_algo_id, tp1_sz)
                except Exception as tracking_error:
                    log.error("TP1 intent update failed %s %s: %s",
                              uid, inst_id, tracking_error)
                # Report the share that will REALLY close, not the setting.
                # raw_tp1_sz is floored to lotSz, so "50%" of an odd contract
                # count is never exactly 50% — 50% of 99 lots is 49, i.e. 49.5%.
                _act_pct = partial["actual_pct"]
                _pct_s = (f"{_act_pct:.0f}%" if abs(_act_pct - close_pct) < 0.5
                          else f"{_act_pct:.1f}% (просил {close_pct:.0f}%, "
                               f"ровно не делится на контракты)")
                tp1_line = (f"На TP1 ({tp1_px}) закроется {_pct_s} "
                            f"({okx._fmt_sz(raw_tp1_sz)} из {okx._fmt_sz(_pos_sz)} контр.), "
                            f"остаток — под трейлинг.")
                if partial['full_close']:
                    tp1_line = (f"На TP1 ({tp1_px}) закроется вся позиция "
                        f"({okx._fmt_sz(raw_tp1_sz)} контр.): выбран режим минимального объёма, "
                        "на допустимые части она не делится. Остатка к TP2 не будет.")
                elif u.get('tp1_minimum_close'):
                    tp1_line = (f"На TP1 ({tp1_px}) закроется {_act_pct:.1f}% "
                        f"({okx._fmt_sz(raw_tp1_sz)} из {okx._fmt_sz(_pos_sz)} контр.). "
                        "Выбран минимум допустимого объёма; остаток — к TP2 под защитой стопа.")
            else:
                log.warning(f"autotrade TP1 order failed for {uid} {inst_id}: {tp1_result}")
                tp1_line = "⚠️ Не смог поставить ордер на частичное закрытие TP1 — весь объём пойдёт под трейлинг."
        else:
            minimum_pct = partial["minimum_pct"]
            tp1_line = (
                f"На TP1 выбрано {close_pct:g}%, минимум для позиции "
                f"{minimum_pct:.2f}%. Весь объём остаётся под защитой."
                if minimum_pct else
                "Не удалось рассчитать объём TP1. Весь объём остаётся под защитой."
            )

    # Record the REAL fill, not the pre-entry quote used for sizing — this is
    # what "breakeven" is later measured against.
    # Store the size we ACTUALLY hold. The monitor subtracts this from the live
    # position size to report how much TP1 closed, so a requested size that never
    # filled would overstate that share to the owner.
    #
    # Promote the durable pre-order intent now that the real fill and protection
    # are known. If this final write fails, ENTRY_PENDING remains visible to the
    # reconciler after restart, while the exchange OCO still protects the fill.
    try:
        activated = at_activate_entry_intent(
            intent_id, _pos_sz, _fill_px, margin, algo_id, sl_px,
            tp1_algo_id=tp1_algo_id, tp1_sz=tp1_sz,
            entry_filled_sz=_filled_sz if _sz_ok and _filled_sz and _filled_sz > 0 else None)
        if not activated:
            raise RuntimeError("entry intent is missing or no longer pending")
    except Exception as _db_err:
        log.error(f"autotrade POSITION ACTIVATION FAILED {uid} {inst_id}: entry and OCO are LIVE "
                  f"but entry intent activation failed ({_db_err})")
        _dm(uid, "\n".join([
            f"🚨 *{disp}*: позиция открыта, но бот её НЕ ВЕДЁТ.",
            "",
            "Стоп и цель на бирже стоят, но трейлинг работать не будет.",
            "Закрой её вручную или следи сам.",
            "",
            f"инструмент: `{inst_id}`",
            f"защита: `{algo_id}`",
            f"ошибка: `{_db_err}`",
        ]))
        raise
    try:
        from src.factor_exposure_observer import record as record_exposure
        record_exposure(intent_id, sig, inst_id, latest_positions, account_snapshot_at)
    except Exception:
        log.warning("factor exposure shadow unavailable for position=%s", intent_id)
    if _avg_px and planned > 0:
        sign = 1 if sig["direction"] == "LONG" else -1
        log.info(
            "entry execution signal=%s inst=%s adverse_vs_signal_bps=%.2f "
            "adverse_vs_book_bps=%.2f",
            sig["id"], inst_id, sign * (_avg_px / planned - 1) * 10_000,
            sign * (_avg_px / px - 1) * 10_000,
        )
    # Diagnostics come strictly after the live fill has both exchange protection
    # and durable OPEN tracking. SQLite must never delay the protective OCO.
    try:
        book_snapshot = order_book_snapshot(book, spec["ctVal"])
        recorded = record_signal_execution_book(
            sig["id"], spread_bps=book_snapshot.spread_bps,
            depth_usd=book_snapshot.depth_usd,
            imbalance=book_snapshot.imbalance, observed_at=book_observed_at)
        if not recorded:
            log.info("autotrade book snapshot had no linked setup for signal %s", sig.get("id"))
    except Exception as exc:
        log.warning("autotrade book snapshot write failed signal %s: %s", sig.get("id"), exc)
    lev = AUTOTRADE_LEVERAGE
    # Report what was actually BOUGHT, not what was requested. Size is in
    # contracts and OKX floors to lotSz, so the real notional is
    # sz * ctVal * fill. The old line printed margin*lev (the request) and the
    # pre-entry quote, so both numbers were right by construction and could
    # never show a real fill or a rounding gap.
    _ctval = float((spec or {}).get("ctVal") or 0) or 0.0
    _real_notional = _pos_sz * _ctval * _fill_px if _ctval else margin * lev
    _real_margin   = _real_notional / lev if lev else margin
    _px_line = (f"Вход: {okx._fmt_px(_fill_px)}" if _avg_px
                else f"Вход: ~{px} _(биржа не отдала цену заливки)_")
    _dm(uid, (f"🤖 *Сделка открыта: {disp} {sig['direction']}*\n"
              f"Объём: {okx._fmt_sz(_pos_sz)} контр. (${_real_notional:.2f} позиция, "
              f"${_real_margin:.2f} маржа, {lev}x)\n"
              f"{_px_line}\nSL: {sl_px}\nTP2: {tp_px}\n"
              f"{tp1_line}"))


def open_positions_for_signal(sig: dict) -> None:
    """Fire-and-forget: open this signal for every active autotrader."""
    if not AUTOTRADE_ENABLED or not sig:
        return
    if int(sig.get("autotrade_eligible", 1) or 0) != 1:
        log.info("autotrade: shadow-only signal %s refused", sig.get("id"))
        return
    if sig.get("source") == "stock_xperp_native_shadow":
        health = get_route_live_readiness(
            "stock_xperp_native_shadow",
            STOCK_LIVE_FORWARD_WINDOW,
            STOCK_LIVE_FORWARD_MIN_TOTAL_R,
        )
        if not health["ready"]:
            log.warning(
                "autotrade: stock forward gate refused signal %s "
                "(%s/%s resolved, %+.2fR, %.1f%% wins; price_return=%s; "
                "missing_outcomes=%s; missing_geometry=%s)",
                sig.get("id"), health["resolved"], health["window"],
                health["total_r"], health["win_rate"],
                health.get("total_price_return"), health.get("missing_outcomes"),
                health.get("missing_geometry"),
            )
            return
    traders = at_get_active_traders()
    if not traders:
        return
    if not keystore_ready():
        log.warning("autotrade: keystore not ready — no positions opened")
        return
    inst_id = _inst_id_of(sig["symbol"])
    if not inst_id:
        log.warning(f"autotrade: no X-Perp for {sig['symbol']} — skipped")
        return
    disp = sig["symbol"].replace("USDT", "/USDC")

    def _run():
        for u in traders:
            try:
                _open_for_user(u, sig, inst_id, disp)
            except Exception as e:
                log.warning(f"autotrade open failed for {u['user_id']}: {e}")

    threading.Thread(target=_run, daemon=True, name=f"autotrade-{sig['id']}").start()


def update_trailing(sig: dict, stop_px: float) -> None:
    """Engine's current post-TP1 trail moved → amend every user's exchange SL.
    Called each monitor cycle while the signal stays TP1_PARTIAL."""
    if not AUTOTRADE_ENABLED or stop_px is None:
        return
    positions = at_open_positions_for_signal(sig["id"])
    if not positions:
        return
    for pos in positions:
        try:
            # Only push amendments that actually move the stop (avoid API spam)
            old = float(pos.get("sl_px") or 0)
            tick = (okx.get_xperp_spec(pos["inst_id"]) or {}).get("tickSz", 0)
            new = okx.round_to_tick(float(stop_px), tick)
            if old and abs(new - old) < (tick or 1e-9):
                continue
            # A trailing stop only ever moves TOWARDS price. The engine
            # recomputes the post-TP1 peak from scratch every cycle out of the
            # candles that came back, so a short window — or the fallback to
            # the deep feed when the X-Perp call fails, whose highs are lower
            # than the thin X-Perp's — understates the peak and hands us a
            # LOOSER stop. Amending to it would push the exchange stop back
            # away from price and give up profit already locked in, purely
            # because one data fetch came back thin. Ratchet against the last
            # stop actually set, which is what sl_px records.
            if old:
                _long = str(pos.get("direction", "")).upper() == "LONG"
                if (new < old) if _long else (new > old):
                    log.info(f"autotrade trail held pos#{pos['id']}: "
                             f"engine asked {new}, keeping {old}")
                    continue
            u = at_get(pos["user_id"])
            creds = _creds_of(u) if u else None
            if not creds:
                continue
            ok, live = okx.get_live_protection_sl(
                creds, pos["inst_id"], pos["sl_algo_id"])
            if not ok:
                log.warning("autotrade trail lookup failed pos#%s: %s", pos["id"], live)
                continue
            is_long = pos["direction"] == "LONG"
            if (live >= new if is_long else live <= new):
                if live != old:
                    at_update_position_sl(pos["id"], live)
                continue
            ok, err = okx.amend_protection_sl(creds, pos["inst_id"], pos["sl_algo_id"], new)
            if ok:
                at_update_position_sl(pos["id"], new)
            else:
                log.warning(f"autotrade SL amend failed pos#{pos['id']}: {err}")
        except Exception as e:
            log.warning(f"autotrade trailing failed pos#{pos['id']}: {e}")


def _clear_closed_protection(pos: dict, creds: dict, *, notify=None) -> bool:
    """Clear the old fixed-size TP1 before releasing a confirmed-flat position."""
    inst_id = pos["inst_id"]
    if pos.get("tp1_algo_id"):
        canceled, _ = okx.cancel_protection(creds, inst_id, pos["tp1_algo_id"])
        if not canceled:
            log.warning("TP1 cancellation unconfirmed pos#%s %s; retaining tracking", pos["id"], inst_id)
            if notify is not None:
                base = inst_id.split("-")[0]
                notify(pos["user_id"],
                       f"⚠️ *{base}/USDC*: OKX ещё не подтвердил снятие старого ордера TP1.\n"
                       "Бот повторит проверку. Новый вход по этому инструменту пока заблокирован.")
            return False
    # OKX also retires the closeFraction OCO when the position becomes flat.
    if pos.get("sl_algo_id"):
        okx.cancel_protection(creds, inst_id, pos["sl_algo_id"])
    return True


def mirror_transition(sig: dict, new_status: str, exit_px: float, *, notify=None) -> None:
    """Engine transitioned a signal → mirror it on the exchange for every user."""
    if not AUTOTRADE_ENABLED:
        return
    positions = at_open_positions_for_signal(sig["id"])
    if not positions:
        return
    disp = sig["symbol"].replace("USDT", "/USDC")
    label = _STATUS_RU.get(new_status, new_status)
    send = _dm if notify is None else notify

    for pos in positions:
        try:
            u = at_get(pos["user_id"])
            creds = _creds_of(u) if u else None
            if not creds:
                log.error("Cannot manage position %s: credentials unavailable; keeping tracked", pos["id"])
                continue

            if new_status in _CLOSE_STATUSES:
                # Confirm flatness before cancelling the associated exits.
                ok, err = okx.close_position_market(creds, pos["inst_id"])
                if not ok:
                    ok, err = okx.close_position_market(creds, pos["inst_id"])
                if not ok:
                    log.error(f"autotrade CLOSE PENDING {pos['user_id']} "
                              f"{pos['inst_id']}: close not confirmed; protection retained "
                              f"({err}) — position remains tracked for retry")
                    send(pos["user_id"], "\n".join([
                        f"🚨 *СРОЧНО: {disp} пока не закрыта*",
                        "",
                        "Закрытие пока не подтверждено. Существующая защита не отменялась.",
                        "Зайди на биржу и закрой её вручную.",
                        "",
                        f"причина: `{err}`",
                    ]))
                    continue
                if not _clear_closed_protection(pos, creds, notify=send):
                    continue
                at_close_position(pos["id"], new_status, error=None)
                send(pos["user_id"], f"🤖 *{disp}*: {label} (~{exit_px}).")
            elif new_status == "TP1_PARTIAL":
                # Move the exchange stop to breakeven RIGHT NOW — don't wait
                # for the next monitor cycle's trail computation, which would
                # leave the original (loss) SL live on the exchange for up to
                # ~1 min after TP1 actually triggered. update_trailing() only
                # ever raises this further (max(entry, ...)), so moving to
                # entry immediately is always safe, never premature.
                try:
                    tick = (okx.get_xperp_spec(pos["inst_id"]) or {}).get("tickSz", 0)
                    # Breakeven means THIS user's breakeven: their real fill,
                    # not the signal's global-feed entry.
                    _be_src = float(pos.get("entry_px") or sig["entry_price"])
                    be_px = okx.round_to_tick(_be_src, tick)
                    old = float(pos.get("sl_px") or 0)
                    already_protected = old > 0 and (
                        old >= be_px if pos["direction"] == "LONG" else old <= be_px)
                    if not already_protected:
                        ok, live = okx.get_live_protection_sl(
                            creds, pos["inst_id"], pos["sl_algo_id"])
                        if not ok:
                            log.warning("autotrade breakeven lookup failed pos#%s: %s",
                                        pos["id"], live)
                        elif (live >= be_px if pos["direction"] == "LONG" else live <= be_px):
                            at_update_position_sl(pos["id"], live)
                        else:
                            ok, err = okx.amend_protection_sl(
                                creds, pos["inst_id"], pos["sl_algo_id"], be_px)
                            if ok:
                                at_update_position_sl(pos["id"], be_px)
                            else:
                                log.warning("autotrade breakeven amend failed pos#%s: %s",
                                            pos["id"], err)
                except Exception as e:
                    log.warning(f"autotrade breakeven amend failed pos#{pos['id']}: {e}")

                detail = ("Биржевое исполнение TP1 подтверждается отдельно."
                          if pos.get("tp1_algo_id") else "Весь объём ведётся под защитой стопа.")
                send(pos["user_id"], f"🤖 *{disp}*: {label}.\n{detail}")
        except Exception as e:
            log.warning(f"autotrade mirror failed pos#{pos['id']}: {e}")


def _reconcile_partial_tp1(pos: dict, size: float, creds: dict) -> None:
    if not pos.get("tp1_algo_id") or size >= float(pos.get("sz") or 0):
        return
    try:
        from src.exit_summary import confirmed_partial_tp1
        fill = confirmed_partial_tp1(pos, size, creds, okx)
        if fill is None or not at_reduce_position_sz(pos["id"], size, expected_sz=pos["sz"]):
            return
        base = pos["inst_id"].split("-")[0]
        pct = 100 * fill["closed"] / fill["original"]
        _dm(pos["user_id"], f"🤖 *{base}/USDC*: OKX подтвердил исполнение TP1.\n"
            f"Всего закрыто {pct:.1f}% ({okx._fmt_sz(fill['closed'])} из "
            f"{okx._fmt_sz(fill['original'])} контр.). Остаток {okx._fmt_sz(size)} контр.")
    except Exception as exc:
        log.warning("TP1 reconciliation deferred pos#%s (%s)", pos["id"], type(exc).__name__)


def poll_exchange_closes() -> None:
    """Every monitor cycle: check the REAL position size on the exchange for
    every DB-open autotrade position. The exchange-side OCO closes the
    position the instant SL/TP triggers — this catches that immediately
    instead of waiting for the signal engine's own (slower, kline-based)
    status transition to reach mirror_transition()."""
    if not AUTOTRADE_ENABLED:
        return
    positions = at_all_open_positions()
    if not positions:
        return
    for pos in positions:
        try:
            pending_entry = pos.get("status") == "ENTRY_PENDING"
            submitted_at = pos.get("entry_submit_started_at")
            pending_age = time.time() - float(submitted_at or pos.get("opened_at") or 0)
            if pending_entry and pending_age < _ENTRY_PENDING_GRACE_SEC:
                continue
            if pending_entry and submitted_at == 0:
                at_release_unsent_entry(pos["id"])
                continue
            u = at_get(pos["user_id"])
            creds = _creds_of(u) if u else None
            if not creds:
                continue
            ok, size = okx.get_position_size(creds, pos["inst_id"])
            if not ok:
                continue   # transient API error — retry next cycle
            if pending_entry:
                if size > 0:
                    closed, err = okx.close_position_market(
                        creds, pos["inst_id"])
                    if not closed:
                        closed, err = okx.close_position_market(
                            creds, pos["inst_id"])
                    if not closed:
                        log.error("pending entry recovery close failed pos#%s: %s",
                                  pos["id"], err)
                        _dm(pos["user_id"], "\n".join([
                            f"🚨 *СРОЧНО: {pos['inst_id']} осталась открыта*",
                            "Незавершённый вход найден после сбоя, но закрытие не подтверждено.",
                            "Закрой позицию вручную на OKX.",
                            f"`{err}`",
                        ]))
                        continue
                    # A close acknowledgement does not confirm flatness.
                    # Keep the intent and protection until a later poll
                    # sees the entry order terminal and OKX flat.
                    if pos.get("client_order_id"):
                        continue
                    if pos.get("sl_algo_id"):
                        okx.cancel_protection(
                            creds, pos["inst_id"], pos["sl_algo_id"])
                    if pos.get("tp1_algo_id"):
                        okx.cancel_protection(
                            creds, pos["inst_id"], pos["tp1_algo_id"])
                    at_close_position(pos["id"], "ENTRY_RECOVERY_CLOSED")
                    _dm(pos["user_id"],
                        f"🤖 *{pos['inst_id']}*: найден незавершённый вход после сбоя; "
                        "позиция закрыта для безопасности.")
                    continue
                # A flat position does not prove that an ambiguous market
                # order cannot still fill. Keep the intent until OKX confirms
                # a terminal order state, then recheck the position.
                client_id = pos.get("client_order_id")
                if not client_id:
                    continue
                found, order = okx.get_order_by_client_id(
                    creds, pos["inst_id"], client_id)
                if not found or order.get("state") not in {"filled", "canceled"}:
                    continue
                ok2, size2 = okx.get_position_size(creds, pos["inst_id"])
                if not ok2 or size2 > 0:
                    continue
                if not _clear_closed_protection(pos, creds):
                    continue
                at_close_position(pos["id"], "ENTRY_TERMINAL_FLAT")
                continue
            if size > 0:
                # The engine marks a signal closed BEFORE mirroring that close
                # to the exchange, and mirror_transition wraps each position in
                # its own except. A failure there leaves the signal finished,
                # this record open, and the position still running — with
                # nobody to retry it, because mirror_transition only fires on
                # the transition itself and that has already passed. The OCO
                # still guards it, so this is not a naked position, but the
                # trail is dead and the trade has stopped being the one the
                # engine believes it closed. Catch it here, where we are
                # already asking the exchange what is really open.
                unprotected = not pos.get("sl_algo_id")
                _sig = {"status": "UNPROTECTED_CLOSED"} if unprotected else get_signal_by_id(pos["signal_id"])
                if not unprotected and (not _sig or _sig.get("status") not in _CLOSE_STATUSES):
                    _reconcile_partial_tp1(pos, size, creds)
                    continue   # still open on the exchange, nothing to do
                log.warning(f"autotrade reconcile pos#{pos['id']} "
                            f"{pos['inst_id']}: signal already {_sig['status']} "
                            f"but position still open on the exchange")
                _rok, _rerr = okx.close_position_market(creds, pos["inst_id"])
                if not _rok:
                    _rok, _rerr = okx.close_position_market(creds, pos["inst_id"])
                _rbase = pos["inst_id"].split("-")[0]
                if not _rok:
                    log.error(f"autotrade CLOSE PENDING {pos['user_id']} "
                              f"{pos['inst_id']}: reconcile close not confirmed; protection retained "
                              f"({_rerr}) — position remains tracked for retry")
                    _dm(pos["user_id"], "\n".join([
                        f"🚨 *СРОЧНО: {_rbase} пока не закрыта*",
                        "",
                        "Сделка по сигналу уже закрыта, а позицию на бирже "
                        "закрыть не удалось (2 попытки).",
                        "Зайди на биржу и закрой её вручную.",
                        "",
                        f"причина: `{_rerr}`",
                    ]))
                    continue
                if not _clear_closed_protection(pos, creds):
                    continue
                at_close_position(pos["id"], _sig["status"])
                _dm(pos["user_id"],
                    f"🤖 *{_rbase}*: позиция осталась висеть на бирже после "
                    f"закрытия сигнала — закрыл её сейчас.")
                continue

            # A zero read is acted on IRREVERSIBLY below (protection cancelled,
            # record closed), so it must not be believed on a single sample.
            # OKX can report 0 for a position that is really open: a brand-new
            # entry not yet visible on /account/positions, or a transient empty
            # payload. Either would strip the stop off a live position and stop
            # tracking it — the worst outcome the autotrader can produce.
            if time.time() - float(pos.get("opened_at") or 0) < _CLOSE_CONFIRM_MIN_AGE_SEC:
                continue
            ok2, size2 = okx.get_position_size(creds, pos["inst_id"])
            if not ok2 or size2 > 0:
                log.info(f"autotrade pos#{pos['id']}: flat read not confirmed "
                         f"(second read ok={ok2} size={size2}) — leaving open")
                continue

            # Keep the local and shared reservations while the old TP1 could still fire.
            if not _clear_closed_protection(pos, creds):
                continue
            at_close_position(pos["id"], "EXCHANGE_FLAT")
            base = pos["inst_id"].split("-")[0]
            # Bound all exit fills to this position's protective orders and
            # full original entry quantity. A final BE order alone omits TP1.
            from src.exit_summary import protective_exit_summary, format_exit_summary
            summary = None
            try:
                summary = protective_exit_summary(pos, creds, okx)
            except Exception as lookup_error:
                log.warning("protective exit summary unavailable pos#%s: %s",
                            pos['id'], type(lookup_error).__name__)
            _tick = (okx.get_xperp_spec(pos["inst_id"]) or {}).get("tickSz", 0)
            _px_line = format_exit_summary(summary, lambda price: okx.fmt_px_display(price, _tick))
            _dm(pos["user_id"],
                f"🤖 *{base}/USDC*: позиция закрыта на бирже.\n"
                f"{_px_line}")
        except Exception as e:
            log.warning(f"autotrade exchange-close poll failed pos#{pos['id']}: {e}")
