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
import time

import requests

from config import (
    RISK_NORMALIZED_SIZING, RISK_REFERENCE_PCT, RISK_SIZE_MULT_MIN,
    EXTENSION_FRESH_THRESHOLD, EXTENSION_FRESH_SIZE_MULT,
    OPEN_SESSION_SIZE_MULT, OPEN_VOL_MIN,
    HTF_NEUTRAL_1H_SIZE_MULT,
    VOLUME_SPIKE_SIZE_MULT, VOLUME_SPIKE_BOOST_MIN, OFF_SESSION_SIZE_MULT,
    ORDERLY_EFF_MIN, ORDERLY_ATR_MAX, ORDERLY_EXT_MIN, ORDERLY_SIZE_MULT,
    SIZE_MULT_MAX,
    TELEGRAM_TOKEN,
    AUTOTRADE_ENABLED, AUTOTRADE_LEVERAGE, AUTOTRADE_BALANCE_THRESHOLD,
    AUTOTRADE_CONTACT,
    STOP_CLOSE_CONFIRM, STOP_EXCHANGE_BACKSTOP_R,
)
from src.db import (
    at_get_active_traders, at_get, at_set_balance, at_set_mode_prompt,
    at_log_position, at_open_positions_for_signal, at_update_position_sl,
    at_close_position, at_all_open_positions, at_reduce_position_sz,
    at_has_open_position,
    get_signal_by_id,
    set_signal_size_mult,
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

# A position younger than this is never declared closed from a zero size read —
# /account/positions can lag a fresh market fill. One monitor cycle is 60s, so
# this costs at most one extra cycle of notification latency on a real close.
_CLOSE_CONFIRM_MIN_AGE_SEC = 90


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

    ok, balance = okx.get_balance(creds)
    if not ok:
        log.warning(f"autotrade balance failed for {uid}: {balance}")
        _dm(uid, f"⚠️ Автотрейдинг: не смог прочитать баланс OKX — сделка по {disp} пропущена.\n`{balance}`")
        return
    at_set_balance(uid, balance)
    _check_threshold_cross(u, balance)

    # Risk-normalised sizing: a wide stop gets less size so that 1R costs the
    # same money regardless of where structure put the stop. See config.py —
    # stop distance here spans 0.40%-1.50%, so without this the widest trades
    # risk 3.8x the tightest ones under an identical "size".
    _size_mult = 1.0
    # Normalisation is held aside and applied AFTER the ceiling, matching the
    # crypto bot. Folding it in here and capping the product afterwards lets
    # the ceiling undo it: with boosts 3.0 and a 0.7 normalisation the old
    # order sent min(2.1, 2.0) = 2.0 where the intended size is 0.7 * 2.0 =
    # 1.4 -- 43% more money at risk than the wide stop was supposed to get.
    # Normalisation exists precisely so 1R costs the same money at any stop
    # width, which is the assumption every R figure in this project rests on,
    # so it has to have the last word.
    _norm = 1.0
    _stack = 1.0   # BOOSTS only: the ceiling never applied to the trim
    if RISK_NORMALIZED_SIZING:
        try:
            _e, _sl = float(sig["entry_price"]), float(sig["sl"])
            _risk_pct = abs(_e - _sl) / _e if _e > 0 else 0.0
            if _risk_pct > RISK_REFERENCE_PCT:
                _norm = max(RISK_SIZE_MULT_MIN, RISK_REFERENCE_PCT / _risk_pct)
        except (TypeError, ValueError, ZeroDivisionError, KeyError) as _e:
            # Fails OPEN: without this the multiplier stays 1.0 and a wide-stop
            # trade goes out at FULL size, which is the case this whole block
            # exists to prevent. Say so rather than sizing up in silence.
            log.warning(f"risk-normalised sizing failed for {sig.get('symbol')} "
                        f"({_e}) — full size used")
    # Fresh-break trim — see EXTENSION_FRESH_THRESHOLD in config.py. Reads None
    # on signals written before the 2026-08-25 migration, which falls through to
    # no trim rather than mis-sizing.
    try:
        # `or 99.0` also swallowed a legitimate ZERO — price sitting right ON
        # the break, the freshest entry there is, and exactly what this trim
        # exists to catch. None still means "no break level found" and still
        # falls through to no trim. Mirrors backtest._fld.
        _ext = sig.get("bos_extension_atr")
        _ext = 99.0 if _ext is None or _ext == "" else float(_ext)
        if _ext <= EXTENSION_FRESH_THRESHOLD:
            _size_mult *= float(EXTENSION_FRESH_SIZE_MULT)
    except (TypeError, ValueError):
        pass
    # Orderly trend rides bigger — see ORDERLY_SIZE_MULT in config.py. All
    # three fields are absent on rows written before the 2026-08-27 migration,
    # and an absent field means no boost rather than one granted on no
    # evidence — matching what the backtest does with a missing field.
    if ORDERLY_SIZE_MULT != 1.0:
        try:
            _er, _ap, _ex = (sig.get("eff_ratio"), sig.get("vol_atr_pct"),
                             sig.get("bos_extension_atr"))
            if (_er is not None and _ap is not None and _ex is not None
                    and float(_er) >= ORDERLY_EFF_MIN
                    and float(_ap) < ORDERLY_ATR_MAX
                    and float(_ex) >= ORDERLY_EXT_MIN):
                _size_mult *= float(ORDERLY_SIZE_MULT)
                _stack *= float(ORDERLY_SIZE_MULT)
        except (TypeError, ValueError):
            pass
    # Opening bell WITH volume rides bigger — see OPEN_SESSION_SIZE_MULT in
    # config.py. Ungated by volume the rule fails, so a row missing
    # volume_ratio (written before the 2026-08-27 migration) gets NO boost
    # rather than an unconditional one: granting extra size on absent evidence
    # is the wrong direction to fail in. Matches what the backtest does with an
    # absent field, so the two paths agree.
    if OPEN_SESSION_SIZE_MULT != 1.0 and str(sig.get("session") or "") == "OPEN":
        try:
            _vr = sig.get("volume_ratio")
            if _vr is not None and float(_vr) >= OPEN_VOL_MIN:
                _size_mult *= float(OPEN_SESSION_SIZE_MULT)
                _stack *= float(OPEN_SESSION_SIZE_MULT)
        except (TypeError, ValueError):
            pass
    # Volume spike and OFF session — added 2026-08-29 to close a real gap: the
    # backtest applied both and this function did not, so a size rule could be
    # "shipped", measured, and never reach a live order. That is exactly what
    # happened to the volume-spike boost on 2026-08-28. Both multipliers are 1.0
    # today, so nothing changes now; the point is that the two paths can no
    # longer drift apart silently. Missing fields fail toward NO boost, matching
    # the backtest's _fld() substitutes.
    if VOLUME_SPIKE_SIZE_MULT != 1.0:
        try:
            _vr = sig.get("volume_ratio")
            if _vr is not None and float(_vr) >= VOLUME_SPIKE_BOOST_MIN:
                _size_mult *= float(VOLUME_SPIKE_SIZE_MULT)
                _stack *= float(VOLUME_SPIKE_SIZE_MULT)
        except (TypeError, ValueError):
            pass
    if OFF_SESSION_SIZE_MULT != 1.0 and str(sig.get("session") or "") == "OFF":
        _size_mult *= float(OFF_SESSION_SIZE_MULT)
        _stack *= float(OFF_SESSION_SIZE_MULT)
    # Ceiling on the stacked product — see SIZE_MULT_MAX in config.py. Mirrors
    # backtest.py, which applies the same cap to the folded multipliers.
    # The ceiling applies to the BOOSTS, not to the whole product, matching
    # backtest._size_mult_for. This changes NOTHING reachable today and is not
    # presented as a fix: capping the total differs only where the trim and the
    # boosts co-occur, and they cannot -- the trim needs bos_extension_atr
    # <= 0.7 while ORDERLY needs >= 1.271, which leaves a maximum stack of 1.5
    # against a 2.0 ceiling. Aligned anyway so the two paths agree by
    # construction rather than by a coincidence of two thresholds that nothing
    # stops anyone from moving.
    # A neutral 1h trend rides bigger — see HTF_NEUTRAL_1H_SIZE_MULT in
    # config.py. Absent on rows written before this knob existed, which sizes
    # at 1.0 rather than guessing, matching what the backtest does.
    if (float(HTF_NEUTRAL_1H_SIZE_MULT) != 1.0
            and str(sig.get("trend_1h") or "").lower() == "neutral"):
        _hm = float(HTF_NEUTRAL_1H_SIZE_MULT)
        _size_mult *= _hm
        _stack *= _hm
    if _stack > float(SIZE_MULT_MAX):
        _size_mult *= float(SIZE_MULT_MAX) / _stack
    # Record what this signal actually traded at, HERE — above the _norm
    # step below, which resizes so that 1R costs the same money at any stop
    # width and therefore does not scale R. The rule multipliers above DO
    # scale it, and are what the backtest multiplies gross_r/net_r by.
    # Best-effort: the trade matters more than the bookkeeping, so a failure
    # here is logged and the open continues.
    try:
        set_signal_size_mult(sig["id"], _size_mult)
    except Exception as _sm_err:
        log.warning(f"could not record size_mult for signal {sig.get('id')}: "
                    f"{_sm_err}")
    _size_mult *= _norm
    margin = _margin_for(u, balance) * _size_mult
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

    # The order size above was computed FOR this leverage. The result used to
    # be discarded, so a failed call meant the exchange kept whatever leverage
    # the instrument already had and filled a size priced for a different one —
    # silently more or less risk than the strategy asked for. Refusing the trade
    # is the safe failure here: a skipped setup costs nothing, a mis-levered
    # position costs whatever the market does next.
    _lev_ok, _lev_err = okx.ensure_leverage(creds, inst_id, AUTOTRADE_LEVERAGE)
    if not _lev_ok:
        log.warning(f"autotrade leverage set failed {uid} {inst_id}: {_lev_err}")
        _dm(uid, "\n".join([
            f"⚠️ Автотрейдинг: не смог выставить плечо {AUTOTRADE_LEVERAGE}x по {disp}.",
            "Сделка пропущена — иначе размер позиции не соответствовал бы риску.",
            f"`{_lev_err}`",
        ]))
        return

    ok, ord_id = okx.place_market_entry(creds, inst_id, sig["direction"], sz)
    if not ok:
        log.warning(f"autotrade entry failed for {uid} {inst_id}: {ord_id}")
        _dm(uid, f"❌ Автотрейдинг: не смог открыть {disp} {sig['direction']}.\n`{ord_id}`\nЕсли не понимаешь причину — напиши `{AUTOTRADE_CONTACT}`.")
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
    _fill_px = _avg_px or float(sig["entry_price"])
    _sl_level = float(sig["sl"])
    if STOP_CLOSE_CONFIRM:
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
            log.warning(f"autotrade OCO failed for {uid} {inst_id}: {algo_id} — position closed")
            _dm(uid, f"❌ Автотрейдинг: не смог поставить стоп по {disp} — "
                     f"позиция закрыта для безопасности.\n`{algo_id}`")
        else:
            log.error(f"autotrade NAKED POSITION {uid} {inst_id}: OCO failed ({algo_id}) "
                      f"AND close failed ({_cerr}) — position is open WITHOUT a stop")
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
        lot = (spec or {}).get("lotSz") or 1.0
        raw_tp1_sz = (_pos_sz if close_pct >= 100
                      else int((_pos_sz * close_pct / 100.0) / lot) * lot)
        if raw_tp1_sz > 0:
            tp1_px = okx.round_to_tick(float(sig["tp1"]), tick)
            ok, tp1_result = okx.place_tp1_partial(creds, inst_id, sig["direction"], tp1_px, raw_tp1_sz)
            if ok:
                tp1_algo_id, tp1_sz = tp1_result, raw_tp1_sz
                # Report the share that will REALLY close, not the setting.
                # raw_tp1_sz is floored to lotSz, so "50%" of an odd contract
                # count is never exactly 50% — 50% of 99 lots is 49, i.e. 49.5%.
                _act_pct = raw_tp1_sz / _pos_sz * 100.0 if _pos_sz else close_pct
                _pct_s = (f"{_act_pct:.0f}%" if abs(_act_pct - close_pct) < 0.5
                          else f"{_act_pct:.1f}% (просил {close_pct:.0f}%, "
                               f"ровно не делится на контракты)")
                tp1_line = (f"На TP1 ({tp1_px}) закроется {_pct_s} "
                            f"({okx._fmt_sz(raw_tp1_sz)} из {okx._fmt_sz(_pos_sz)} контр.), "
                            f"остаток — под трейлинг.")
            else:
                log.warning(f"autotrade TP1 order failed for {uid} {inst_id}: {tp1_result}")
                tp1_line = "⚠️ Не смог поставить ордер на частичное закрытие TP1 — весь объём пойдёт под трейлинг."

    # Record the REAL fill, not the pre-entry quote used for sizing — this is
    # what "breakeven" is later measured against.
    # Store the size we ACTUALLY hold. The monitor subtracts this from the live
    # position size to report how much TP1 closed, so a requested size that never
    # filled would overstate that share to the owner.
    #
    # This write is the LAST step, after the entry and its protection are already
    # live on the exchange, and the caller only logs a warning on exception. So a
    # failure here leaves a real position that this bot does not know about: the
    # monitor never sees it, the trail never moves, and the owner is told nothing
    # while the log says "open failed". The OCO still protects it, but nobody is
    # managing it. Say so loudly instead, with what is needed to find it by hand.
    try:
        at_log_position(sig["id"], uid, inst_id, sig["direction"], _pos_sz, _fill_px, margin,
                        algo_id, sl_px, tp1_algo_id=tp1_algo_id, tp1_sz=tp1_sz)
    except Exception as _db_err:
        log.error(f"autotrade UNTRACKED POSITION {uid} {inst_id}: entry and OCO are LIVE "
                  f"but at_log_position failed ({_db_err}) — bot will not manage it")
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
                # then flatten whatever the algo hasn't already closed. Also
                # cancel any still-resting TP1 partial-close order — if TP1
                # was never reached (e.g. straight SL_HIT) it would otherwise
                # sit orphaned on the exchange after the position is flat.
                okx.cancel_protection(creds, pos["inst_id"], pos["sl_algo_id"])
                if pos.get("tp1_algo_id"):
                    # This one is checked while the OCO above is not, and the asymmetry is
                    # deliberate. The OCO is placed with closeFraction, so the exchange
                    # retires it when the position goes flat. The TP1 partial is a FIXED
                    # size reduce-only order: nothing retires it, and a leftover would sit
                    # armed until the next position in the same instrument, then close part
                    # of it at a price chosen for a trade that ended days ago.
                    _c_ok, _c_err = okx.cancel_protection(creds, pos["inst_id"], pos["tp1_algo_id"])
                    if not _c_ok:
                        log.error(f"autotrade STALE TP1 ORDER {pos['user_id']} {pos['inst_id']}: "
                                  f"cancel failed ({_c_err}) — it can fire on a later position")
                        _dm(pos["user_id"], "\n".join([
                            f"⚠️ *{disp}*: не смог снять ордер частичного закрытия TP1.",
                            "Он может сработать по следующей сделке этого же тикера.",
                            "Проверь открытые алго-ордера на бирже.",
                            f"`{_c_err}`",
                        ]))
                # The protection was cancelled a few lines up, so if this
                # close does not go through the position is sitting on the
                # exchange with NOTHING under it. Marking the record closed
                # here would also stop us tracking it and tell the owner it
                # is over. Retry once, exactly like the naked-position guard
                # on the open path, and if it still will not close, keep the
                # record OPEN and say so: poll_exchange_closes will retire it
                # properly once the position really is flat.
                ok, err = okx.close_position_market(creds, pos["inst_id"])
                if not ok:
                    ok, err = okx.close_position_market(creds, pos["inst_id"])
                if not ok:
                    log.error(f"autotrade NAKED POSITION {pos['user_id']} "
                              f"{pos['inst_id']}: protection cancelled but close "
                              f"failed twice ({err}) — position open WITHOUT a stop")
                    _dm(pos["user_id"], "\n".join([
                        f"🚨 *СРОЧНО: {disp} осталась БЕЗ СТОПА*",
                        "",
                        "Защита снята, а закрыть позицию не удалось (2 попытки).",
                        "Зайди на биржу и закрой её вручную.",
                        "",
                        f"причина: `{err}`",
                    ]))
                    continue
                at_close_position(pos["id"], new_status, error=None)
                _dm(pos["user_id"], f"🤖 *{disp}*: {label} (~{exit_px}).")
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
                    ok, err = okx.amend_protection_sl(creds, pos["inst_id"], pos["sl_algo_id"], be_px)
                    if ok:
                        at_update_position_sl(pos["id"], be_px)
                    else:
                        log.warning(f"autotrade breakeven amend failed pos#{pos['id']}: {err}")
                except Exception as e:
                    log.warning(f"autotrade breakeven amend failed pos#{pos['id']}: {e}")

                # If the user has a TP1 %, the exchange's own pre-placed order
                # (set at trade-open, see _open_for_user) should have fired
                # the instant TP1 triggered — no need to re-issue anything
                # here. Just sync our record to the REAL remaining size so
                # bookkeeping/notifications reflect what actually happened.
                if pos.get("tp1_algo_id"):
                    try:
                        ok, real_sz = okx.get_position_size(creds, pos["inst_id"])
                        if ok and real_sz < float(pos["sz"]):
                            if real_sz <= 0:
                                at_close_position(pos["id"], "TP1_FULL_CLOSE")
                                _dm(pos["user_id"],
                                    f"🤖 *{disp}*: {label}.\nЗакрыто по твоей настройке — вся позиция.")
                                continue
                            at_reduce_position_sz(pos["id"], real_sz)
                            # Say how much actually closed: the exchange fills in
                            # whole lots, so the realised share is rarely the
                            # configured one.
                            _orig = float(pos["sz"])
                            _closed = _orig - real_sz
                            _pct = _closed / _orig * 100.0 if _orig else 0.0
                            _dm(pos["user_id"],
                                f"🤖 *{disp}*: {label}.\n"
                                f"Закрыто {_pct:.1f}% "
                                f"({okx._fmt_sz(_closed)} из {okx._fmt_sz(_orig)} контр.), "
                                f"остаток {okx._fmt_sz(real_sz)} контр. идёт под трейлинг.")
                            continue
                        elif ok:
                            log.warning(f"autotrade TP1 order pos#{pos['id']} hasn't filled yet "
                                       f"(real_sz={real_sz} >= tracked {pos['sz']}) — will recheck next cycle")
                    except Exception as e:
                        log.warning(f"autotrade TP1 size sync failed pos#{pos['id']}: {e}")

                _dm(pos["user_id"], f"🤖 *{disp}*: {label}.")
        except Exception as e:
            log.warning(f"autotrade mirror failed pos#{pos['id']}: {e}")


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
            u = at_get(pos["user_id"])
            creds = _creds_of(u) if u else None
            if not creds:
                continue
            ok, size = okx.get_position_size(creds, pos["inst_id"])
            if not ok:
                continue   # transient API error — retry next cycle
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
                _sig = get_signal_by_id(pos["signal_id"])
                if not _sig or _sig.get("status") not in _CLOSE_STATUSES:
                    continue   # still open on the exchange, nothing to do
                log.warning(f"autotrade reconcile pos#{pos['id']} "
                            f"{pos['inst_id']}: signal already {_sig['status']} "
                            f"but position still open on the exchange")
                okx.cancel_protection(creds, pos["inst_id"], pos["sl_algo_id"])
                if pos.get("tp1_algo_id"):
                    okx.cancel_protection(creds, pos["inst_id"], pos["tp1_algo_id"])
                _rok, _rerr = okx.close_position_market(creds, pos["inst_id"])
                if not _rok:
                    _rok, _rerr = okx.close_position_market(creds, pos["inst_id"])
                _rbase = pos["inst_id"].split("-")[0]
                if not _rok:
                    log.error(f"autotrade NAKED POSITION {pos['user_id']} "
                              f"{pos['inst_id']}: reconcile close failed twice "
                              f"({_rerr}) — position open WITHOUT a stop")
                    _dm(pos["user_id"], "\n".join([
                        f"🚨 *СРОЧНО: {_rbase} осталась БЕЗ СТОПА*",
                        "",
                        "Сделка по сигналу уже закрыта, а позицию на бирже "
                        "закрыть не удалось (2 попытки).",
                        "Зайди на биржу и закрой её вручную.",
                        "",
                        f"причина: `{_rerr}`",
                    ]))
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

            # Exchange is flat but our DB still says OPEN → SL/TP/trail
            # already fired for real. Close out the record and tell the user
            # right away; the signal's own status message follows separately.
            okx.cancel_protection(creds, pos["inst_id"], pos["sl_algo_id"])
            if pos.get("tp1_algo_id"):
                # Same hazard as the close path above, which checks this and
                # this path did not. The OCO is closeFraction, so the exchange
                # retires it once the position is flat; the TP1 partial is a
                # FIXED size reduce-only order that nothing retires. A leftover
                # sits armed until the next position in this instrument, then
                # closes part of it at a price chosen for a trade already over.
                _c_ok, _c_err = okx.cancel_protection(
                    creds, pos["inst_id"], pos["tp1_algo_id"])
                if not _c_ok:
                    _d = pos["inst_id"].split("-")[0]
                    log.error(f"autotrade STALE TP1 ORDER {pos['user_id']} "
                              f"{pos['inst_id']}: cancel failed ({_c_err}) "
                              f"on the exchange-flat path - it can fire later")
                    _dm(pos["user_id"], "\n".join([
                        f"⚠️ *{_d}*: не смог снять ордер частичного закрытия TP1.",
                        "Он может сработать по следующей сделке этого же тикера.",
                        "Проверь открытые алго-ордера на бирже.",
                        f"`{_c_err}`",
                    ]))
            at_close_position(pos["id"], "EXCHANGE_FLAT")
            base = pos["inst_id"].split("-")[0]
            # Report the REAL fill. The signal engine sends its own message
            # later -- it waits for a 15m candle to close beyond the stop --
            # and quotes that candle close, a price this position never
            # traded at. Ported from the crypto bot, which hit this live.
            _tick = (okx.get_xperp_spec(pos["inst_id"]) or {}).get("tickSz", 0)
            _fill = okx.get_last_fill_px(creds, pos["inst_id"])
            _entry = float(pos.get("entry_px") or 0)
            if _fill and _entry > 0:
                _mv = ((_fill - _entry) / _entry * 100.0
                       if str(pos.get("direction", "")).upper() == "LONG"
                       else (_entry - _fill) / _entry * 100.0)
                _px_line = (f"Закрыто по `{okx.fmt_px_display(_fill, _tick)}` "
                            f"(вход `{okx.fmt_px_display(_entry, _tick)}`, "
                            f"{_mv:+.2f}%, x{AUTOTRADE_LEVERAGE}: "
                            f"{_mv * AUTOTRADE_LEVERAGE:+.0f}%)")
            else:
                _px_line = "Цену заливки биржа не отдала."
            _dm(pos["user_id"],
                f"🤖 *{base}/USDC*: позиция закрыта на бирже (стоп/тейк сработал).\n"
                f"{_px_line}\n"
                f"_Это реальная цена сделки. Сообщение по сигналу придёт позже "
                f"и покажет цену закрытия свечи — она может отличаться._")
        except Exception as e:
            log.warning(f"autotrade exchange-close poll failed pos#{pos['id']}: {e}")
