"""
Autotrading orchestration: mirrors the signal engine into real OKX EU positions
for every onboarded user.

Lifecycle glue (called from main.py):
  open_positions_for_signal(sig)  — after a signal is published
  update_trailing(sig, stop_px)   — every monitor cycle while TP1_PARTIAL
  mirror_transition(sig, status)  — when the engine closes/transitions a signal

Per-user flow on open:
  decrypt keys → balance (threshold-cross check) → size → set 10x isolated →
  market entry → protection OCO (SL at engine stop, TP at tp2) → DM the user.

Fail-safe rules:
  - a user erroring out never blocks other users;
  - if the protection OCO can't be placed, the naked position is closed
    immediately — never leave a position without a stop;
  - closes treat "already flat" (exchange algo fired first) as success.
"""
import logging
import threading

import requests

from config import (
    TELEGRAM_TOKEN,
    AUTOTRADE_ENABLED, AUTOTRADE_LEVERAGE, AUTOTRADE_BALANCE_THRESHOLD,
    AUTOTRADE_CONTACT,
)
from src.db import (
    at_get_active_traders, at_get, at_set_balance, at_set_mode_prompt,
    at_log_position, at_open_positions_for_signal, at_update_position_sl,
    at_close_position,
)
from src.keystore import decrypt_secret, keystore_ready
from src import okx_trader as okx
from src.binance_client import get_xperp_instruments

log = logging.getLogger(__name__)

_STATUS_RU = {
    "TP1_PARTIAL": "TP1 достигнут — включён трейлинг-стоп",
    "TP2_HIT":     "TP2 достигнут — позиция закрыта",
    "TP1_TRAIL":   "трейлинг-стоп сработал — позиция закрыта",
    "SL_HIT":      "стоп-лосс — позиция закрыта",
    "BREAKEVEN":   "безубыток — позиция закрыта",
    "EXPIRED":     "сигнал истёк — позиция закрыта",
    "TP1_EXPIRED": "истёк после TP1 — позиция закрыта",
}

# Statuses that require closing whatever is still open on the exchange.
_CLOSE_STATUSES = ("TP2_HIT", "TP1_TRAIL", "SL_HIT", "BREAKEVEN",
                   "EXPIRED", "TP1_EXPIRED")


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
    """Margin ($) this user puts into one trade under their chosen mode."""
    if u.get("size_mode") == "percent":
        return balance * float(u.get("size_value") or 0) / 100.0
    return float(u.get("size_value") or 0)


def _check_threshold_cross(u: dict, balance: float) -> None:
    """Balance crossed $100 against the chosen mode → ask once, keep trading
    under the current mode until the user answers."""
    mode = u.get("size_mode")
    if not mode or u.get("mode_prompt_pending"):
        return
    below = balance < AUTOTRADE_BALANCE_THRESHOLD
    if (below and mode == "percent") or (not below and mode == "fixed"):
        at_set_mode_prompt(u["user_id"], True)
        cur  = (f"{u['size_value']:.0f}% от депозита" if mode == "percent"
                else f"${u['size_value']:.2f} на сделку")
        alt  = "фиксированную сумму ($)" if mode == "percent" else "процент от депозита (1-10%)"
        verb = "упал ниже" if below else "вырос выше"
        kb = {"inline_keyboard": [[
            {"text": "Оставить как есть", "callback_data": "at_mode_keep"},
            {"text": "Сменить режим",     "callback_data": "at_mode_switch"},
        ]]}
        try:
            requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                json={
                    "chat_id": u["user_id"],
                    "text": (f"⚖️ Твой депозит {verb} ${AUTOTRADE_BALANCE_THRESHOLD:.0f} "
                             f"(сейчас ${balance:.2f}).\n"
                             f"Текущий режим: *{cur}*.\n"
                             f"Хочешь перейти на {alt}?"),
                    "parse_mode": "Markdown",
                    "reply_markup": kb,
                },
                timeout=10,
            )
        except Exception as e:
            log.warning(f"threshold prompt to {u['user_id']} failed: {e}")


def _open_for_user(u: dict, sig: dict, inst_id: str, disp: str) -> None:
    uid = u["user_id"]
    creds = _creds_of(u)
    if not creds:
        return

    ok, balance = okx.get_balance(creds)
    if not ok:
        log.warning(f"autotrade balance failed for {uid}: {balance}")
        _dm(uid, f"⚠️ Автотрейдинг: не смог прочитать баланс OKX — сделка по {disp} пропущена.\n`{balance}`")
        return
    at_set_balance(uid, balance)
    _check_threshold_cross(u, balance)

    margin = _margin_for(u, balance)
    if margin <= 0:
        return
    if margin > balance:
        _dm(uid, f"⚠️ Автотрейдинг: на балансе ${balance:.2f} меньше, чем размер сделки ${margin:.2f} — {disp} пропущен.")
        return

    spec = okx.get_xperp_spec(inst_id)
    px   = okx.get_last_price(inst_id) or float(sig["entry_price"])
    sz   = okx.calc_contracts(margin, AUTOTRADE_LEVERAGE, px, spec or {})
    if sz <= 0:
        _dm(uid, f"⚠️ Автотрейдинг: размер сделки ${margin:.2f} слишком мал для минимального контракта {disp} — пропущено.")
        return

    okx.set_leverage(creds, inst_id, AUTOTRADE_LEVERAGE)

    ok, ord_id = okx.place_market_entry(creds, inst_id, sig["direction"], sz)
    if not ok:
        log.warning(f"autotrade entry failed for {uid} {inst_id}: {ord_id}")
        _dm(uid, f"❌ Автотрейдинг: не смог открыть {disp} {sig['direction']}.\n`{ord_id}`\nЕсли не понимаешь причину — напиши `{AUTOTRADE_CONTACT}`.")
        return

    tick  = (spec or {}).get("tickSz", 0)
    sl_px = okx.round_to_tick(float(sig["sl"]), tick)
    tp_px = okx.round_to_tick(float(sig["tp2"]), tick)
    ok, algo_id = okx.place_protection_oco(creds, inst_id, sig["direction"], sl_px, tp_px)
    if not ok:
        # Never leave a naked position: close it right back.
        okx.close_position_market(creds, inst_id)
        log.warning(f"autotrade OCO failed for {uid} {inst_id}: {algo_id} — position closed")
        _dm(uid, f"❌ Автотрейдинг: не смог поставить стоп по {disp} — позиция закрыта для безопасности.\n`{algo_id}`")
        return

    at_log_position(sig["id"], uid, inst_id, sig["direction"], sz, px, margin,
                    algo_id, sl_px)
    lev = AUTOTRADE_LEVERAGE
    _dm(uid, (f"🤖 *Сделка открыта: {disp} {sig['direction']}*\n"
              f"Объём: {okx._fmt_sz(sz)} контр. (~${margin * lev:.2f} позиция, ${margin:.2f} маржа, {lev}x)\n"
              f"Вход: ~{px}\nSL: {sl_px}\nTP2: {tp_px}\n"
              f"TP1 и трейлинг ведёт бот автоматически."))


def open_positions_for_signal(sig: dict) -> None:
    """Fire-and-forget: open this signal for every active autotrader."""
    if not AUTOTRADE_ENABLED or not sig:
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
            u = at_get(pos["user_id"])
            creds = _creds_of(u) if u else None
            if not creds:
                continue
            ok, err = okx.amend_protection_sl(creds, pos["inst_id"], pos["sl_algo_id"], new)
            if ok:
                at_update_position_sl(pos["id"], new)
            else:
                log.warning(f"autotrade SL amend failed pos#{pos['id']}: {err}")
        except Exception as e:
            log.warning(f"autotrade trailing failed pos#{pos['id']}: {e}")


def mirror_transition(sig: dict, new_status: str, exit_px: float) -> None:
    """Engine transitioned a signal → mirror it on the exchange for every user."""
    if not AUTOTRADE_ENABLED:
        return
    positions = at_open_positions_for_signal(sig["id"])
    if not positions:
        return
    disp = sig["symbol"].replace("USDT", "/USDC")
    label = _STATUS_RU.get(new_status, new_status)

    for pos in positions:
        try:
            u = at_get(pos["user_id"])
            creds = _creds_of(u) if u else None
            if not creds:
                at_close_position(pos["id"], new_status, error="no creds")
                continue

            if new_status in _CLOSE_STATUSES:
                # Cancel protection first so its market close can't double-fire,
                # then flatten whatever the algo hasn't already closed.
                okx.cancel_protection(creds, pos["inst_id"], pos["sl_algo_id"])
                ok, err = okx.close_position_market(creds, pos["inst_id"])
                at_close_position(pos["id"], new_status,
                                  error=None if ok else str(err))
                _dm(pos["user_id"], f"🤖 *{disp}*: {label} (~{exit_px}).")
            elif new_status == "TP1_PARTIAL":
                # Full position stays on (TP1_CLOSE_FRAC=0) — trailing takes
                # over from the next monitor cycle. Just tell the user.
                _dm(pos["user_id"], f"🤖 *{disp}*: {label}.")
        except Exception as e:
            log.warning(f"autotrade mirror failed pos#{pos['id']}: {e}")
