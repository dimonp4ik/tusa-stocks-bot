"""
TUSA Stocks Bot — entry point.

Flow every N minutes:
  1. Fetch top 45 USDT pairs from KuCoin (by 24h volume)
  2. Run SMC technical filter (BOS + FVG + OB + multi-timeframe)
  3. Send only strong setups to Claude Sonnet
  4. Claude returns LONG / SHORT / NO TRADE
  5. Telegram receives only actionable signals
"""

import json
import logging
import os
import time
import threading
from datetime import datetime, timezone

from flask import Flask, request as flask_request
from apscheduler.schedulers.background import BackgroundScheduler

import requests as _requests

from config import (
    SCAN_INTERVAL_MINUTES, SIGNAL_COOLDOWN_HOURS, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID,
    TRADING_HOURS_START, TRADING_HOURS_END, TRADE_WEEKENDS,
    MAX_SETUPS_TO_CLAUDE, ALLOWED_SYMBOLS, KLINES_INTERVAL_SEC, SIGNAL_EXPIRY_HOURS,
    SIGNAL_EXPIRY_MAX_DAYS,
    CLAUDE_HEAVY_MIN_SCORE, CLAUDE_HEAVY_MAX_PER_SCAN, CLAUDE_MEMORY_LIMIT,
    TRAIL_RUNNER_ENABLED, TRAIL_ATR_MULT,
    STOP_CLOSE_CONFIRM, MAX_SAME_DIRECTION_POSITIONS, STOP_EXCHANGE_BACKSTOP_R,
    MTF_MIN_SCORE, TP1_R_MULT,
    TP1_CLOSE_FRAC, EXIT_PROFILE,
    CLAUDE_GATE_ENABLED,
    LEVELS_FROM_STRUCTURE,
    POST_TP1_STRONG_TRAIL_ATR_MULT, POST_TP1_WEAK_TRAIL_ATR_MULT,
    POST_TP1_STRONG_CLOSE_PROGRESS, POST_TP1_STRONG_WICK_PROGRESS,
    POST_TP1_WEAK_CLOSE_PROGRESS,
    KNN_RISK_OVERLAY, KNN_DEEP_CANDLES, KNN_MAX_HISTORY, KNN_SHAPE_LEN,
    KNN_HORIZON, KNN_K, KNN_MIN_HISTORY, KNN_HIGH_SCORE, KNN_HIGH_MULT,
    KNN_LOW_SCORE, KNN_LOW_MULT, KNN_RISK_MAX_MULT, KNN_RISK_MIN_MULT,
)
from src.binance_client import (
    get_top_coins, get_klines, get_klines_1h, get_klines_4h, get_klines_1d,
    get_btc_change_1h, get_btc_change_1d, get_funding_rate, get_current_price,
    get_open_interest, get_xperp_instruments, get_xperp_price, get_klines_xperp,
)
from src.signal_filter import analyze_coin_smc
from src.knn_analog import knn_direction_score, knn_risk_mult
from src.claude_analyzer import analyze_batch_with_claude, analyze_heavy
from src.telegram_notifier import send_signal, send_status, send_news_alert, send_signal_update, calculate_tp_sl, send_morning_digest, send_weekly_digest, send_daily_prayer, send_commandments, send_evening_prayer, send_evening_ritual, _disp_sym
from src.news_filter import check_news_sentiment
from src.news_agent import (
    get_market_news, detect_major_events, fetch_recent_headlines,
    get_daily_digest, get_upcoming_high_impact_events, get_day_events,
    generate_weekly_commentary,
)
from config import EVENT_WARN_HOURS
from config import LIVE_PRICE_MAX_DRIFT_PCT
from config import EXTENDED_SESSION_WINDOWS
from src.market_hours import session_hours_between
from src.db import (
    init_db, get_open_signals, update_signal_status, get_stats,
    set_sl_xperp_only, get_sl_wick_stats,
    auto_block_bad_symbols, is_symbol_auto_blocked, get_active_symbol_blocks,
    get_recent_outcomes, unblock_symbol, set_symbol_block, get_symbols_performance,
    upsert_user, get_user_by_id, get_all_users, get_users_count,
    add_dynamic_admin, remove_dynamic_admin, get_dynamic_admins, is_dynamic_admin,
    get_dynamic_role,
    delete_signal, get_recent_signals,
    get_signals_count, get_signals_page, get_distinct_signal_symbols,
    get_claude_spend_stats,
    get_bot_state, set_bot_state,
    log_setup_candidate, mark_setup_sent, get_setups_by_date,
    mark_setup_blocked, get_cap_impact_stats, get_skew_response_stats,
    get_all_setups_since, get_all_signals_since,
    get_unresolved_setups, mark_setup_resolved, get_setup_accuracy,
    link_setup_to_signal, resolve_sent_setups_from_signals, backfill_setup_signal_links,
    get_similar_resolved_setups, seed_backtest_outcomes, backfill_backtest_net_r,
    delete_backtest_seed_rows,
    get_today_sl_streak,
    get_weekly_stats,
    at_add_allowed, at_remove, at_get, at_all_allowed, at_set_keys,
    at_set_mode, at_set_active, at_set_balance, at_set_mode_prompt,
    at_set_tp1_close_pct,
    get_latest_open_signal, get_signal_by_id,
)
from src import autotrader
from src.keystore import keystore_ready, encrypt_secret
from src import okx_trader as _okx_trade
from config import ADMIN_IDS, AUTOTRADE_BALANCE_THRESHOLD, AUTOTRADE_CONTACT
from config import REJECT_COOLDOWN_HOURS, KILL_SWITCH_SL_STREAK

# ── Admin helpers ─────────────────────────────────────────────────────────────

def _is_admin(user_id: int) -> bool:
    """True for config super-admins OR dynamically added DB admins."""
    return user_id in ADMIN_IDS or is_dynamic_admin(user_id)


def _is_super_admin(user_id: int) -> bool:
    """True only for hardcoded config admins (can manage other admins)."""
    return user_id in ADMIN_IDS


def _admin_role(user_id: int) -> str:
    """'super' | 'admin' | 'moderator' | 'none' — used to gate admin-panel actions."""
    if user_id in ADMIN_IDS:
        return "super"
    return get_dynamic_role(user_id) or "none"


# Moderators get monitoring + reversible actions + the autotrade allow-list —
# NOT strategy/filter params, Claude budget, signal deletion, or admin management.
_MODERATOR_ALLOWED_EXACT = {
    "adm_sec_trading", "adm_sec_settings", "adm_sec_analytics", "adm_sec_people",
    "adm_stats", "adm_open", "adm_setups", "adm_setups_today", "adm_setups_date",
    "adm_manblock", "adm_mb_input",
    "adm_top", "adm_worst", "adm_ai_acc",
    "adm_users", "adm_users_p0", "adm_users_search",
    "adm_autotrade", "adm_at_add",
    "adm_back", "adm_open_new",
}
_MODERATOR_ALLOWED_PREFIX = (
    "adm_setups_pg_", "adm_mb_block_", "adm_mb_unblock_",
    "adm_top_d", "adm_worst_d",
    "adm_users_p", "adm_users_q_",
    "adm_at_rm_",
)


def _moderator_may(data: str) -> bool:
    return data in _MODERATOR_ALLOWED_EXACT or data.startswith(_MODERATOR_ALLOWED_PREFIX)


# State: super-admin is typing a new admin's Telegram ID.
_pending_add_admin: dict = {}
# State: admin is typing a symbol name to manually block.
_pending_block_chat: dict = {}
# State: admin is typing a user search query.
_pending_users_search: dict = {}
# State: admin is typing a date to view setup history.
_pending_setups_date: dict = {}
_pending_report_date: dict = {}
# State: admin is typing a user ID to allow autotrading.
_pending_add_autotrade: dict = {}
# State: user is inside the autotrade onboarding dialog.
# chat_id → {"step": str, "data": {...}}  (steps: api_key → api_secret →
# passphrase → size_percent | size_fixed; 'switch_*' reuse the size steps)
_at_onboarding: dict = {}
# State: which admin sub-section each chat is currently in, so detail-view
# "« Назад" returns to that section menu instead of the top-level panel.
_admin_section: dict = {}
_photo_panel_messages: set = set()


def _sec_back_cb(chat_id) -> str:
    """Callback for a detail-view back button: the chat's current section,
    or the top-level panel if no section is tracked."""
    return _admin_section.get(chat_id, "adm_back")


def _parse_date_to_ts(date_str: str):
    """DD.MM / DD.MM.YYYY / YYYY-MM-DD -> epoch at Riga midnight. None if unparseable."""
    from datetime import datetime as _dt
    s = (date_str or "").strip()
    for fmt in ("%d.%m.%Y", "%d.%m", "%Y-%m-%d"):
        try:
            p = _dt.strptime(s, fmt)
            if fmt == "%d.%m":
                p = p.replace(year=_dt.now().year)
            return p.replace(tzinfo=_riga_tz()).timestamp()
        except ValueError:
            continue
    return None


def _build_and_send_report(chat_id: int, message_id, since_ts: float,
                           window_label: str) -> None:
    """Full report over an explicit window, as a text file + raw CSV dumps.

    The window is caller-chosen on purpose: filter settings change, and a fixed
    look-back silently mixes results produced by different configs, which makes
    every rate in here meaningless.
    """
    try:
        rz = _riga_tz()
        now = time.time()
        since7 = now - 7 * 86400
        L = []
        A = L.append
        A("# ПОЛНЫЙ ОТЧЁТ БОТА (АКЦИИ)")
        A(f"Сформирован: {datetime.now(rz).strftime('%Y-%m-%d %H:%M')} Riga")
        A(f"ОКНО ДАННЫХ: {window_label}")
        if since_ts > 0:
            A(f"  с {datetime.fromtimestamp(since_ts, rz).strftime('%Y-%m-%d %H:%M')} по сейчас")
            A(f"  длительность: {(now - since_ts) / 86400:.1f} дней")
        else:
            A("  всё, что есть в базе")
        A("")

        for label, cut in ([(window_label, since_ts)] +
                           ([("последние 7 дней", since7)] if since_ts < since7 else [])):
            s = get_stats(since_ts=cut) if cut > 0 else get_stats(days=36500)
            A(f"## ЖИВЫЕ РЕЗУЛЬТАТЫ — {label}")
            A(f"  сигналов: {s['total']}  закрыто: {s['closed']}  "
              f"открыто: {s['open']}+{s['tp1_partial_open']}")
            A(f"  винрейт: {s['win_rate']}%")
            A(f"  TP1: {s['tp1_hit']} ({s['tp1_rate']}%)  TP2: {s['tp2_hit']}  "
              f"SL: {s['sl_hit']}  истекло: {s['expired']}")
            A(f"  итог: {s['total_r']}R  ({s['r_per_trade']}R/сделка)")
            _l, _sh = s.get("long") or {}, s.get("short") or {}
            A(f"  LONG  n={_l.get('total', 0)} WR={_l.get('win_rate', 0)}% R={_l.get('total_r', 0)}")
            A(f"  SHORT n={_sh.get('total', 0)} WR={_sh.get('win_rate', 0)}% R={_sh.get('total_r', 0)}")
            A("")

        A(f"## КЛОД: ТОЧНОСТЬ И ДОЛЯ ОДОБРЕНИЙ ({window_label})")
        acc = get_setup_accuracy(since_ts)
        _s, _r = acc.get("sent") or {}, acc.get("rejected") or {}
        _tot = (_s.get("n") or 0) + (_r.get("n") or 0)
        if _tot:
            A(f"  одобрил и отправлено: {_s.get('n', 0)}  отклонил: {_r.get('n', 0)}  "
              f"доля одобрений: {(_s.get('n', 0) / _tot * 100):.0f}%")
            # get_setup_accuracy returns tp1_pct, not tp1_rate. The mismatched
            # key meant this block printed 0%/0%/+0пп for every window since it
            # was written — the one number that says whether Claude's filtering
            # helps at all was silently blank.
            A(f"  TP1 у отправленных: {_s.get('tp1_pct', 0):.1f}%")
            A(f"  TP1 у отклонённых:  {_r.get('tp1_pct', 0):.1f}%")
            A(f"  разрыв (больше = лучше отбирает): "
              f"{(_s.get('tp1_pct', 0) - _r.get('tp1_pct', 0)):+.1f}пп")
            A(f"  зеркало отклонённых: {_r.get('mirror_r', 0):+.1f}R (n={_r.get('n', 0)})")
        else:
            A("  данных нет")
        A("")

        A("## ПРИРОДА СТОПОВ (X-Perp фитиль vs реальный разворот)")
        for label, cut in (("последние 7 дней", since7), (window_label, since_ts)):
            w = get_sl_wick_stats(cut)
            if w["n"]:
                A(f"  {label}: стопов {w['n']}  "
                  f"X-Perp шум {w['xperp_only']} ({w['xperp_only_pct']:.0f}%)  "
                  f"реальный разворот {w['confirmed']}")
            else:
                A(f"  {label}: данных нет")
        A("")

        A(f"## ВЛИЯНИЕ ЛИМИТОВ ({window_label})")
        _caps = get_cap_impact_stats(since_ts) or {}
        for code, title in (("dir_cap", f"лимит одной стороны ({MAX_SAME_DIRECTION_POSITIONS})"),
                            ("scan_cap", "лимит 3 за скан"),
                            ("send_failed", "⚠️ СБОЙ ОТПРАВКИ (баг, не лимит — должно быть 0)")):
            st = _caps.get(code) or {}
            if st.get("n"):
                A(f"  {title}: срезано {st['n']}  "
                  f"дошли бы до TP1 {st['reached_tp1']} ({st['tp1_pct']:.0f}%)  "
                  f"в стоп {st['sl']} ({st['sl_pct']:.0f}%)  итог {st['saved_r']:+.1f}R")
            else:
                A(f"  {title}: не срабатывал")
        A("")
        A(f"## РЕАКЦИЯ КЛОДА НА ПЕРЕКОС КНИГИ ({window_label})")
        sk = get_skew_response_stats(since_ts)
        if sk:
            for b in sk:
                A(f"  открыто {b['bucket']}: одобрил {b['approve_pct']:.0f}% из {b['n']}  "
                  f"(TP1 {b['tp1_pct']:.0f}%, закрыто {b['resolved']})")
        else:
            A("  данных нет")
        A("")
        A("## КОНФИГ НА МОМЕНТ ОТЧЁТА")
        A(f"  MTF_MIN_SCORE={MTF_MIN_SCORE}  TP1_R_MULT={TP1_R_MULT}")
        A(f"  STOP_CLOSE_CONFIRM={STOP_CLOSE_CONFIRM}  BACKSTOP_R={STOP_EXCHANGE_BACKSTOP_R}")
        A(f"  КЛОД: {'ФИЛЬТР (его вердикт решает)' if _claude_gate_enabled() else 'ТЕНЬ (торгуют правила, вердикт только пишется)'}")
        A(f"  MAX_SAME_DIRECTION_POSITIONS={MAX_SAME_DIRECTION_POSITIONS}")

        report = "\n".join(L)
        if message_id:
            _edit_admin_text(chat_id, message_id,
                             f"📦 *Полный отчёт* — {window_label}\n"
                             "Отправляю файлы, перешли их Клоду.", _KB_ANALYTICS)

        import csv as _csv, io as _io
        stamp = datetime.now(rz).strftime("%Y%m%d")
        files = [("отчёт", f"stocks_report_{stamp}.txt", report.encode("utf-8"))]
        for name, rows in (("setups", get_all_setups_since(since_ts)),
                           ("signals", get_all_signals_since(since_ts))):
            if not rows:
                continue
            buf = _io.StringIO()
            w = _csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
            files.append((name, f"stocks_{name}_{stamp}.csv", buf.getvalue().encode("utf-8")))
        for lbl, fn, blob in files:
            try:
                _requests.post(
                    f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendDocument",
                    data={"chat_id": chat_id, "caption": f"📦 {lbl} — {window_label}"},
                    files={"document": (fn, blob)}, timeout=45,
                )
            except Exception as e:
                log.warning(f"full report send failed ({lbl}): {e}")
    except Exception as e:
        log.warning(f"full report failed: {e}")
        _send_admin_text(chat_id, f"❌ Ошибка отчёта: {e}", _KB_ANALYTICS)


def _send_setups_for_date(chat_id, date_input: str):
    """Look up setup history for date_input and send it (with a plain-text
    Markdown fallback inside _send_admin_text). Always replies."""
    back_kb = {"inline_keyboard": [[
        {"text": "📅 Другая дата", "callback_data": "adm_setups_date"},
        {"text": "« Назад",        "callback_data": _sec_back_cb(chat_id)},
    ]]}
    try:
        rows = get_setups_by_date(date_input)
        if rows is None or (not rows and date_input):
            _send_admin_text(
                chat_id,
                f"📅 Нет сетапов за *{date_input}* или неверный формат.\nФормат: `10.06` или `10.06.2026`",
                back_kb,
            )
        else:
            text, total_pages = _format_setups_page(rows, date_input, 0)
            kb = {"inline_keyboard": _setups_kb(chat_id, date_input, 0, total_pages)}
            _send_admin_text(chat_id, text, kb)
    except Exception as e:
        log.warning(f"setups date '{date_input}' failed: {e}")
        _send_admin_text(chat_id, f"Ошибка при загрузке сетапов за {date_input}.", back_kb)

# Last scan summary — populated by run_scan(), shown in 📊 Фильтры panel.
_last_scan_stats: dict = {
    "coins": 0, "setups": 0, "fresh": 0, "enriched": 0, "sent": 0, "ts": 0.0
}

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Flask (keeps Render dyno alive) ──────────────────────────────────────────
app = Flask(__name__)
ADMIN_PANEL_IMAGE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "assets",
    "tusa_crypto_banner.png",
)


@app.route("/")
def health():
    return "TUSA Stocks Bot is running.", 200


@app.route("/status")
def status():
    return f"Scanning every {SCAN_INTERVAL_MINUTES} min. Signal cache: {len(_signal_cache)} entries.", 200


# ── Admin panel helpers ───────────────────────────────────────────────────────

# Persistent bottom-bar keyboards — set once via /start, stay forever in DM.
_USER_KB = {
    "keyboard": [
        [{"text": "📋 Открытые сделки"}, {"text": "📈 Результаты"}],
        [{"text": "🤖 Автотрейдинг"}, {"text": "📰 Новости на сегодня"}],
        [{"text": "❓ Помощь"}],
    ],
    "resize_keyboard": True,
    "is_persistent":   True,
}
_ADMIN_KB = {
    "keyboard": [
        [{"text": "🛠 Админ панель"}],
        [{"text": "📋 Открытые сделки"}, {"text": "📈 Результаты"}],
        [{"text": "🤖 Автотрейдинг"}, {"text": "📰 Новости на сегодня"}],
        [{"text": "❓ Помощь"}],
    ],
    "resize_keyboard": True,
    "is_persistent":   True,
}
# Group chats get no autotrade button — it's a DM-only feature.
_GROUP_KB = {
    "keyboard": [
        [{"text": "📋 Открытые сделки"}, {"text": "📈 Результаты"}],
        [{"text": "📰 Новости на сегодня"}],
        [{"text": "❓ Помощь"}],
    ],
    "resize_keyboard": True,
    "is_persistent":   True,
}

# Inline keyboard shown inside the panel message — 4 sections + market status.
_ADMIN_KEYBOARD = {
    "inline_keyboard": [[
        {"text": "📈 Торговля",   "callback_data": "adm_sec_trading"},
        {"text": "🔧 Настройки", "callback_data": "adm_sec_settings"},
    ], [
        {"text": "📊 Аналитика", "callback_data": "adm_sec_analytics"},
        {"text": "👥 Люди",      "callback_data": "adm_sec_people"},
    ], [
        {"text": "🕐 Рынок США", "callback_data": "adm_market"},
    ]]
}

_BACK_ROW = [{"text": "« Назад", "callback_data": "adm_back"}]

_KB_TRADING = {"inline_keyboard": [[
    {"text": "📊 Статистика",      "callback_data": "adm_stats"},
    {"text": "📋 Открытые сделки", "callback_data": "adm_open"},
], [
    {"text": "🗑 Управление",      "callback_data": "adm_deals"},
    {"text": "🔍 История сетапов", "callback_data": "adm_setups"},
], [_BACK_ROW[0]]]}

def _in_extended_window(now=None) -> bool:
    """Inside an allowed extended-session window (see EXTENDED_SESSION_WINDOWS).

    Separate from the all-or-nothing off-session toggle because the two are not
    the same thing. Measured 2026-08-20: the London / US-pre-market window
    (04:00-09:00 ET) carries 51% of regular-session candle volume and quoted a
    3.63 bps median spread across all 26 tickers, while weekends carry 5% and
    are the one window whose backtest halves fall apart. Lumping them together
    was costing real trades to protect against a risk only some hours carry.

    Weekends are excluded here regardless of what the windows say.
    """
    if not EXTENDED_SESSION_WINDOWS:
        return False
    try:
        from zoneinfo import ZoneInfo
        et = (now or datetime.now(ZoneInfo("America/New_York")))
        if et.tzinfo is None:
            et = et.replace(tzinfo=ZoneInfo("America/New_York"))
        et = et.astimezone(ZoneInfo("America/New_York"))
        if et.weekday() >= 5:
            return False
        h = et.hour
        return any(lo <= h < hi for lo, hi in EXTENDED_SESSION_WINDOWS)
    except Exception:
        return False


def _off_session_enabled() -> bool:
    """Off-session scanning toggle: DB state wins, env OFF_SESSION_SIGNALS is
    the initial default. X-Perps trade 24/7 but off-session tape is thin."""
    state = get_bot_state("off_session_signals")
    if state is not None:
        return state == "1"
    from config import OFF_SESSION_SIGNALS
    return OFF_SESSION_SIGNALS


def _session_windows_text() -> str:
    """Human-readable summary of when the bot is allowed to look for setups.

    The single toggle was misleading: it read "вне сессии ВЫКЛ" while the bot
    in fact also scans the London window, and reading "ВКЛ" silently opened the
    post-market and weekends — the worst hours of the day per the 2026-08-20
    measurement. The panel now states the real windows.
    """
    from zoneinfo import ZoneInfo
    RG = ZoneInfo("Europe/Riga")
    ET = ZoneInfo("America/New_York")
    try:
        now = datetime.now(ET)
        shift = int((now.astimezone(RG).utcoffset() - now.utcoffset()).total_seconds() // 3600)
    except Exception:
        shift = 7
    lines_out = ["🇺🇸 Сессия США — всегда (16:30-23:00 Рига)"]
    # Merge windows that touch, so 20-24 and 0-4 read as one night block
    # instead of two adjacent lines the reader has to join up themselves.
    merged: list = []
    for lo, hi in sorted(EXTENDED_SESSION_WINDOWS):
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    if len(merged) > 1 and merged[0][0] == 0 and merged[-1][1] == 24:
        merged = [(merged[-1][0], merged[0][1] + 24)] + merged[1:-1]
    for lo, hi in merged:
        a, b = (lo + shift) % 24, (hi + shift) % 24
        lines_out.append(f"🌍 {a:02d}:00-{b:02d}:00 Рига  ({lo % 24:02d}-{hi % 24:02d} Нью-Йорк)")
    if not EXTENDED_SESSION_WINDOWS:
        lines_out.append("🌍 Доп. окна: выключены")
    lines_out.append("🚫 Выходные — закрыты всегда")
    if _off_session_enabled():
        lines_out.append("")
        lines_out.append("⚠️ Тумблер «вне сессии» ВКЛ — он перекрывает окна и "
                         "открывает пост-маркет и выходные, худшие часы суток.")
    return chr(10).join(lines_out)



def _claude_gate_enabled() -> bool:
    """Is Claude a GATE, or an observer? DB state wins, config is the default.

    Runtime-switchable on purpose: this is a live experiment, and its whole
    value is being able to stop it the moment it looks wrong, without a
    redeploy. backtest.py never calls Claude, so with the gate OFF the live bot
    trades the same population the model measures — the only way to score his
    verdict against real fills instead of simulated ones.
    """
    state = get_bot_state("claude_gate_enabled")
    if state is not None:
        return state == "1"
    return CLAUDE_GATE_ENABLED


def _kb_settings():
    """Settings keyboard — built per-render so the off-session toggle shows
    its live state."""
    night = "🌙 Вне сессии: ВКЛ ✅" if _off_session_enabled() else "🌙 Вне сессии: ВЫКЛ"
    gate = ("🤖 Клод: ФИЛЬТР ✅" if _claude_gate_enabled()
            else "🤖 Клод: ТЕНЬ 👁")
    return {"inline_keyboard": [[
        {"text": "📊 Фильтры",       "callback_data": "adm_filters"},
        {"text": "🔒 Блок тикеров",  "callback_data": "adm_manblock"},
    ], [
        {"text": "🚫 Авто-блок",     "callback_data": "adm_blocks"},
        {"text": "💰 Бюджет Claude", "callback_data": "adm_budget"},
    ], [
        {"text": night,              "callback_data": "adm_night_toggle"},
    ], [
        {"text": gate,               "callback_data": "adm_claude_toggle"},
    ], [_BACK_ROW[0]]]}

_KB_ANALYTICS = {"inline_keyboard": [[
    {"text": "🏆 Топ тикеров",     "callback_data": "adm_top"},
    {"text": "💀 Худшие тикеры", "callback_data": "adm_worst"},
], [
    {"text": "🎯 Точность ИИ",   "callback_data": "adm_ai_acc"},
], [
    {"text": "📦 Полный отчёт", "callback_data": "adm_fullreport"},
], [_BACK_ROW[0]]]}

_KB_PEOPLE = {"inline_keyboard": [[
    {"text": "👥 Пользователи", "callback_data": "adm_users"},
    {"text": "👮 Админы",       "callback_data": "adm_admins"},
], [
    {"text": "🤖 Автотрейдинг", "callback_data": "adm_autotrade"},
], [_BACK_ROW[0]]]}


def _send_persistent_menu(chat_id: int, is_admin: bool = False, is_dm: bool = True):
    """Send the persistent bottom-bar keyboard. Admins get extra admin button;
    groups get the keyboard without the DM-only autotrade button."""
    kb = (_ADMIN_KB if is_admin else _USER_KB) if is_dm else _GROUP_KB
    text = ("✅ Меню активировано.\n🛠 Админ панель доступна."
            if is_admin else
            "✅ Меню активировано. Кнопки внизу всегда доступны.")
    try:
        _requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": chat_id, "text": text, "reply_markup": kb},
            timeout=10,
        )
    except Exception as e:
        log.warning(f"_send_persistent_menu failed: {e}")


def _panel_message_key(chat_id: int, message_id: int):
    if chat_id is None or message_id is None:
        return None
    try:
        return (str(chat_id), int(message_id))
    except (TypeError, ValueError):
        return None


def _mark_photo_panel_message(chat_id: int, message_id: int):
    key = _panel_message_key(chat_id, message_id)
    if key:
        _photo_panel_messages.add(key)


def _send_admin_text(chat_id: int, text: str, reply_markup: dict):
    resp = _requests.post(
        f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
        json={
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "Markdown",
            "reply_markup": reply_markup,
        },
        timeout=10,
    )
    # Markdown parse failures return 400 and silently drop the message. Retry as
    # plain text so the admin always gets a reply (e.g. a setup reason with an
    # unbalanced `*`/`_`/backtick that slipped past escaping).
    if resp.status_code != 200 and "parse" in _telegram_error(resp).lower():
        resp = _requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": chat_id, "text": text, "reply_markup": reply_markup},
            timeout=10,
        )
    return resp


def _clear_inline_keyboard(chat_id: int, message_id: int):
    try:
        _requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/editMessageReplyMarkup",
            json={
                "chat_id": chat_id,
                "message_id": message_id,
                "reply_markup": {"inline_keyboard": []},
            },
            timeout=5,
        )
    except Exception as e:
        log.warning(f"_clear_inline_keyboard failed: {e}")


def _telegram_error(resp) -> str:
    try:
        return str(resp.json().get("description", ""))
    except Exception:
        return getattr(resp, "text", "") or ""


def _should_send_replacement(desc: str) -> bool:
    desc = (desc or "").lower()
    return (
        "no text in the message" in desc
        or "message is not a text message" in desc
        or "message can't be edited" in desc
        or "message to edit not found" in desc
    )


def _edit_admin_text(chat_id: int, message_id: int, text: str, reply_markup: dict):
    key = _panel_message_key(chat_id, message_id)
    if key in _photo_panel_messages:
        _clear_inline_keyboard(chat_id, message_id)
        return _send_admin_text(chat_id, text, reply_markup)

    resp = _requests.post(
        f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/editMessageText",
        json={
            "chat_id": chat_id,
            "message_id": message_id,
            "text": text,
            "parse_mode": "Markdown",
            "reply_markup": reply_markup,
        },
        timeout=10,
    )
    if resp.status_code == 200:
        return resp

    desc = _telegram_error(resp)
    if "message is not modified" in desc.lower():
        return resp

    # Markdown parse failure on edit (e.g. a setup reason with an unbalanced
    # `/[/_/* that slipped past escaping) — retry the SAME edit as plain text so
    # the button never appears dead. Without this the panel silently stops working.
    if "parse" in desc.lower():
        resp = _requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/editMessageText",
            json={
                "chat_id": chat_id,
                "message_id": message_id,
                "text": text,
                "reply_markup": reply_markup,
            },
            timeout=10,
        )
        if resp.status_code == 200:
            return resp
        desc = _telegram_error(resp)

    if _should_send_replacement(desc):
        _clear_inline_keyboard(chat_id, message_id)
        return _send_admin_text(chat_id, text, reply_markup)

    log.warning(f"_edit_admin_text failed: {resp.status_code} {desc}")
    return resp


def _send_keyboard(chat_id: int, text: str):
    """Send message with admin inline keyboard."""
    try:
        if os.path.exists(ADMIN_PANEL_IMAGE_PATH):
            with open(ADMIN_PANEL_IMAGE_PATH, "rb") as photo:
                resp = _requests.post(
                    f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto",
                    data={"chat_id": chat_id},
                    files={"photo": photo},
                    timeout=10,
                )
            if resp.status_code != 200:
                log.warning(f"admin banner send failed: {resp.status_code} {_telegram_error(resp)}")

        _send_admin_text(chat_id, text, _ADMIN_KEYBOARD)
    except Exception as e:
        log.warning(f"_send_keyboard failed: {e}")


def _answer_callback(callback_id: str, text: str = ""):
    """Acknowledge a button press (stops Telegram spinner)."""
    try:
        _requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/answerCallbackQuery",
            json={"callback_query_id": callback_id, "text": text},
            timeout=5,
        )
    except Exception:
        pass


def _format_open_signal(s: dict) -> str:
    """One-liner display for an open signal with live price, nearest TP and SL."""
    import time as _t
    age_h = round((_t.time() - s["opened_at"]) / 3600, 1)
    icon  = "🟡" if s["status"] == "TP1_PARTIAL" else "🟢"
    entry = float(s["entry_price"])
    tp1   = float(s["tp1"])
    tp2   = float(s["tp2"])
    sl    = float(s["sl"])
    direction = s["direction"]

    # Nearest TP depends on status
    nearest_tp    = tp2 if s["status"] == "TP1_PARTIAL" else tp1
    nearest_label = "TP2" if s["status"] == "TP1_PARTIAL" else "TP1"

    # Live price: use cached value from last 1-min monitor run (no extra API call).
    # Falls back to a fresh OKX request only if cache is empty (bot just started).
    cur = _last_prices.get(s["symbol"]) or get_current_price(s["symbol"])
    if cur and entry:
        pct   = (cur - entry) / entry * 100
        arrow = "📈" if direction == "LONG" else "📉"
        # positive pnl means price moved in our favour
        pnl   = pct if direction == "LONG" else -pct
        pnl_s = f"+{pnl:.2f}%" if pnl >= 0 else f"{pnl:.2f}%"
        price_line = f"   {arrow} `{cur}` ({pnl_s}) | 🎯 {nearest_label}: `{nearest_tp}` | 🛑 SL: `{sl}`"
    else:
        price_line = f"   🎯 {nearest_label}: `{nearest_tp}` | 🛑 SL: `{sl}`"

    header = f"{icon} *{_disp_sym(s['symbol'])}* {direction} @ `{entry}`  _{age_h}ч_"
    return f"{header}\n{price_line}"


# ── Deals journal: pagination + symbol search ─────────────────────────────────
_DEALS_PER_PAGE = 5
_DEAL_STATUS_ICON = {
    "OPEN": "🟢", "TP1_PARTIAL": "🟡", "TP2_HIT": "✅",
    "BREAKEVEN": "⚖️", "SL_HIT": "❌", "EXPIRED": "⏱",
    "TP1_EXPIRED": "⏱", "TP1_HIT": "✅", "TP1_TRAIL": "✅",
}


def _edit_with_keyboard(chat_id: int, message_id: int, text: str, kb_rows: list):
    """Edit a message with a custom inline keyboard (Markdown)."""
    try:
        _edit_admin_text(chat_id, message_id, text, {"inline_keyboard": kb_rows})
    except Exception as e:
        log.warning(f"_edit_with_keyboard failed: {e}")


def _render_deals_page(chat_id: int, message_id: int, page: int, symbol: str = None):
    """Render one page (5 deals) of the journal with delete + nav + search buttons."""
    per   = _DEALS_PER_PAGE
    total = get_signals_count(symbol)
    if total == 0:
        msg = (f"🗑 *Сделки по {_disp_sym(symbol)}*\n\nНет сделок." if symbol
               else "🗑 *Управление сделками*\n\nСделок в базе нет.")
        _edit_message(chat_id, message_id, msg)
        return

    pages  = (total + per - 1) // per
    page   = max(0, min(page, pages - 1))
    offset = page * per
    sigs   = get_signals_page(per, offset, symbol)
    _RIGA  = _riga_tz()

    title = f"🗑 *Сделки по {_disp_sym(symbol)}*" if symbol else "🗑 *Управление сделками*"
    lines = [f"{title}  (стр. {page + 1}/{pages}, всего {total})\n"]
    kb_rows = []
    for s in sigs:
        icon   = _DEAL_STATUS_ICON.get(s["status"], "•")
        opened = datetime.fromtimestamp(s["opened_at"], tz=_RIGA).strftime("%d.%m %H:%M")
        lines.append(
            f"{icon} `#{s['id']}` *{_disp_sym(s['symbol'])}* {s['direction']} "
            f"@ {s['entry_price']}  [{s['status']}]  {opened}"
        )
        kb_rows.append([{
            "text": f"🗑 Удалить #{s['id']} {_disp_sym(s['symbol'])} {s['direction']}",
            "callback_data": f"adm_del_sig_{s['id']}",
        }])

    # Nav row: ◀️ (prev) | 🔍 Поиск | ▶️ (next)
    def _pcb(p):
        return f"adm_dsym_{symbol}_p{p}" if symbol else f"adm_deals_p{p}"
    nav = []
    if page > 0:
        nav.append({"text": "◀️", "callback_data": _pcb(page - 1)})
    nav.append({"text": "🔍 Поиск", "callback_data": "adm_deals_search"})
    if page < pages - 1:
        nav.append({"text": "▶️", "callback_data": _pcb(page + 1)})
    kb_rows.append(nav)

    if symbol:
        kb_rows.append([{"text": "📋 Все сделки", "callback_data": "adm_deals_p0"}])
    kb_rows.append([{"text": "« Назад", "callback_data": _sec_back_cb(chat_id)}])

    _edit_with_keyboard(chat_id, message_id, "\n".join(lines), kb_rows)


def _render_deals_symbol_picker(chat_id: int, message_id: int):
    """Show all journal symbols as buttons; click → that symbol's deals."""
    symbols = get_distinct_signal_symbols()
    if not symbols:
        _edit_message(chat_id, message_id, "🔍 *Поиск*\n\nВ журнале нет тикеров.")
        return
    kb_rows, row = [], []
    for sym in symbols:
        row.append({"text": _disp_sym(sym), "callback_data": f"adm_dsym_{sym}_p0"})
        if len(row) == 3:
            kb_rows.append(row); row = []
    if row:
        kb_rows.append(row)
    kb_rows.append([{"text": "« Назад к сделкам", "callback_data": "adm_deals_p0"}])
    _edit_with_keyboard(
        chat_id, message_id,
        "🔍 *Поиск по тикеру*\nВыбери тикер — покажу все её сделки:",
        kb_rows,
    )


def _render_manual_block_panel(chat_id: int, message_id: int):
    """Show manual block/unblock panel: active blocks + recently traded symbols."""
    _RIGA = _riga_tz()
    import time as _t

    # Active blocks (auto + manual)
    blocks = get_active_symbol_blocks()

    # Recently traded symbols (last 30 signals) — candidates to block
    recent_syms = get_distinct_signal_symbols()[:20]
    blocked_syms = {b["symbol"] for b in blocks}

    lines = ["🔒 *Ручная блокировка тикеров*\n"]
    kb_rows = []

    if blocks:
        lines.append("*Заблокированы сейчас:*")
        for b in blocks:
            until = datetime.fromtimestamp(b["blocked_until"], tz=_RIGA).strftime("%d.%m %H:%M")
            lines.append(f"• *{_disp_sym(b['symbol'])}* до {until}")
            kb_rows.append([{
                "text": f"✅ Разблок {_disp_sym(b['symbol'])}",
                "callback_data": f"adm_mb_unblock_{b['symbol']}",
            }])
    else:
        lines.append("_Нет активных блоков._")

    # Symbols not currently blocked → show as block buttons (3 per row)
    free_syms = [s for s in recent_syms if s not in blocked_syms]
    if free_syms:
        lines.append("\n*Тикеры из журнала:*")
        row = []
        for sym in free_syms:
            row.append({"text": f"🚫 {_disp_sym(sym)}", "callback_data": f"adm_mb_block_{sym}"})
            if len(row) == 3:
                kb_rows.append(row); row = []
        if row:
            kb_rows.append(row)

    kb_rows.append([{"text": "✏️ Ввести символ вручную", "callback_data": "adm_mb_input"}])
    kb_rows.append([{"text": "« Назад", "callback_data": _sec_back_cb(chat_id)}])

    _edit_with_keyboard(chat_id, message_id, "\n".join(lines), kb_rows)


def _edit_message(chat_id: int, message_id: int, text: str):
    """Edit a detail view and attach a back button to the chat's current
    section (not the top-level panel), so the admin stays where they were."""
    try:
        back = {"inline_keyboard": [[{"text": "« Назад", "callback_data": _sec_back_cb(chat_id)}]]}
        _edit_admin_text(chat_id, message_id, text, back)
    except Exception as e:
        log.warning(f"_edit_message failed: {e}")


_USERS_PER_PAGE = 10


def _render_users_page(chat_id: int, message_id: int, page: int, query: str = ""):
    """Render paginated users list with search support."""
    total = get_users_count(query)
    pages = max(1, (total + _USERS_PER_PAGE - 1) // _USERS_PER_PAGE)
    page  = max(0, min(page, pages - 1))
    users = get_all_users(limit=_USERS_PER_PAGE, offset=page * _USERS_PER_PAGE, query=query)

    title = f"👥 *Пользователи*" + (f" — поиск: `{query}`" if query else "")
    lines = [f"{title}  ({total} чел., стр. {page + 1}/{pages})\n"]
    for u in users:
        fn    = u.get("first_name") or ""
        ln    = u.get("last_name") or ""
        name  = (fn + (" " + ln if ln else "")).strip() or "—"
        uname = f"@{u['username']}" if u.get("username") else f"`{u['user_id']}`"
        last  = datetime.fromtimestamp(u["last_seen"], tz=_riga_tz()).strftime("%d.%m %H:%M")
        lines.append(f"• {name} {uname} — {last} ({u.get('message_count', 1)} сообщ.)")

    kb_rows = []
    # Pagination nav
    def _ucb(p, q): return f"adm_users_q_{q}_p{p}" if q else f"adm_users_p{p}"
    nav = []
    if page > 0:
        nav.append({"text": "◀️", "callback_data": _ucb(page - 1, query)})
    nav.append({"text": "🔍 Поиск", "callback_data": "adm_users_search"})
    if page < pages - 1:
        nav.append({"text": "▶️", "callback_data": _ucb(page + 1, query)})
    if nav:
        kb_rows.append(nav)
    if query:
        kb_rows.append([{"text": "❌ Сбросить поиск", "callback_data": "adm_users_p0"}])
    kb_rows.append([{"text": "« Назад", "callback_data": _sec_back_cb(chat_id)}])
    _edit_with_keyboard(chat_id, message_id, "\n".join(lines), kb_rows)


def _handle_admin_callback(callback_id: str, chat_id: int,
                           message_id: int, data: str, user_id: int = 0):
    """Dispatch inline-button presses for the admin panel."""
    if _admin_role(user_id) == "moderator" and not _moderator_may(data):
        _answer_callback(callback_id, "⛔ Недостаточно прав.")
        return

    # Block/unblock use delayed answer with confirmation text — skip default here
    _silent_cb = data.startswith("adm_mb_block_") or data.startswith("adm_mb_unblock_")
    if not _silent_cb:
        _answer_callback(callback_id)

    if data in ("adm_sec_trading", "adm_sec_settings", "adm_sec_analytics", "adm_sec_people"):
        _admin_section[chat_id] = data

    if data == "adm_sec_trading":
        _edit_admin_text(chat_id, message_id, "📈 *Торговля*\nВыбери раздел:", _KB_TRADING)

    elif data == "adm_sec_settings":
        _hdr = "🔧 *Настройки*\n\n*Когда бот ищет сетапы:*\n"
        _edit_admin_text(chat_id, message_id,
                         _hdr + _session_windows_text() + "\n\nВыбери раздел:",
                         _kb_settings())

    elif data == "adm_claude_toggle":
        new_val = "0" if _claude_gate_enabled() else "1"
        set_bot_state("claude_gate_enabled", new_val)
        _on = [
            "🤖 Клод: *ФИЛЬТР*",
            "Его вердикт решает — отклонённые сетапы не торгуются.",
            "Так бот работал всегда.",
        ]
        _off = [
            "🤖 Клод: *ТЕНЬ*",
            "Торгуют только правила. Клод по-прежнему считает и пишет вердикт,",
            "но ничего не задерживает.",
            "",
            "Он одобряет ~80%, так что сделок станет примерно на четверть больше.",
            "Смысл: бэктест Клода не вызывает, поэтому в тени живой бот торгует",
            "ту же совокупность, что меряет модель — и вердикт можно оценить по",
            "РЕАЛЬНЫМ сделкам, а не по симулированным.",
            "",
            "Судить не раньше чем через 2 недели.",
        ]
        note = "\n".join(_on if new_val == "1" else _off)
        _edit_admin_text(chat_id, message_id, "🔧 *Настройки*\n\n" + note,
                         _kb_settings())

    elif data == "adm_night_toggle":
        new_val = "0" if _off_session_enabled() else "1"
        set_bot_state("off_session_signals", new_val)
        note = ("🌙 Сигналы вне сессии США: *ВКЛЮЧЕНЫ*\n"
                "Бот сканит 24/7. Ночью движение тонкое — сигналы могут быть хуже."
                if new_val == "1" else
                "🌙 Сигналы вне сессии США: *ВЫКЛЮЧЕНЫ*\n"
                "Сканим только пока рынок США открыт (рекомендуется).")
        _edit_admin_text(chat_id, message_id, f"🔧 *Настройки*\n\n{note}", _kb_settings())

    elif data == "adm_market":
        from src.market_hours import status_text as _mh_status
        _mkt = _mh_status().replace("<b>", "*").replace("</b>", "*")
        _scan_mode = ("🌙 Скан вне сессии: ВКЛ — бот работает 24/7"
                      if _off_session_enabled() else
                      "Скан: только в сессию (вне сессии бот следит за открытыми позициями)")
        _edit_admin_text(chat_id, message_id, f"{_mkt}\n\n{_scan_mode}",
                         {"inline_keyboard": [[{"text": "🔄 Обновить", "callback_data": "adm_market"}],
                                              [{"text": "« Назад", "callback_data": "adm_back"}]]})

    elif data == "adm_sec_analytics":
        _edit_admin_text(chat_id, message_id, "📊 *Аналитика*\nВыбери раздел:", _KB_ANALYTICS)

    elif data == "adm_sec_people":
        _edit_admin_text(chat_id, message_id, "👥 *Люди*\nВыбери раздел:", _KB_PEOPLE)

    elif data == "adm_stats":
        try:
            _now_riga      = datetime.now(_riga_tz())
            _riga_midnight = _now_riga.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
            sd  = get_stats(since_ts=_riga_midnight)
            s7  = get_stats(days=7)
            s30 = get_stats(days=30)
            txt = (
                f"📈 *СТАТИСТИКА*\n\n"
                f"*Сегодня:*\n"
                f"  Сигналов: {sd['total']}  Закрыто: {sd['closed']}\n"
                f"  TP1: {sd['tp1_hit']} ({sd['tp1_rate']}%)  TP2: {sd['tp2_hit']}\n"
                f"  BE: {sd['breakeven']}  SL: {sd['sl_hit']}  Expired: {sd['expired']}\n"
                f"  Win rate: *{sd['win_rate']}%*\n\n"
                f"*За 7 дней:*\n"
                f"  Сигналов: {s7['total']}  Закрыто: {s7['closed']}\n"
                f"  TP1: {s7['tp1_hit']} ({s7['tp1_rate']}%)  TP2: {s7['tp2_hit']}\n"
                f"  BE: {s7['breakeven']}  SL: {s7['sl_hit']}  Expired: {s7['expired']}\n"
                f"  Win rate: *{s7['win_rate']}%*\n\n"
                f"*За 30 дней:*\n"
                f"  Сигналов: {s30['total']}  Закрыто: {s30['closed']}\n"
                f"  TP1: {s30['tp1_hit']} ({s30['tp1_rate']}%)  TP2: {s30['tp2_hit']}\n"
                f"  BE: {s30['breakeven']}  SL: {s30['sl_hit']}  Expired: {s30['expired']}\n"
                f"  Win rate: *{s30['win_rate']}%*"
            )
        except Exception as e:
            txt = f"Ошибка: {e}"
        _edit_message(chat_id, message_id, txt)

    elif data == "adm_fullreport":
        _edit_admin_text(
            chat_id, message_id,
            "📦 *Полный отчёт*\n\nС какой даты считать?\n"
            "_Настройки фильтров менялись — данные до и после смешивать нельзя, "
            "иначе цифры ни о чём._",
            {"inline_keyboard": [
                [{"text": "7 дней",  "callback_data": "adm_rep_7"},
                 {"text": "30 дней", "callback_data": "adm_rep_30"}],
                [{"text": "📅 Своя дата", "callback_data": "adm_rep_date"},
                 {"text": "🗄 Всё время",  "callback_data": "adm_rep_all"}],
                [{"text": "« Назад", "callback_data": "adm_sec_analytics"}],
            ]},
        )

    elif data == "adm_rep_date":
        _pending_report_date[chat_id] = message_id
        try:
            _requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                json={"chat_id": chat_id,
                      "text": "📅 С какой даты считать отчёт?\n"
                              "Формат: `26.07` или `26.07.2026`\n"
                              "Возьму всё, что есть с этой даты по сейчас.",
                      "parse_mode": "Markdown"},
                timeout=10,
            )
        except Exception as e:
            log.warning(f"report date prompt failed: {e}")

    elif data in ("adm_rep_7", "adm_rep_30", "adm_rep_all"):
        _answer_callback(callback_id, "Собираю отчёт…")
        _since, _label = {
            "adm_rep_7":   (time.time() - 7 * 86400,  "последние 7 дней"),
            "adm_rep_30":  (time.time() - 30 * 86400, "последние 30 дней"),
            "adm_rep_all": (0.0, "всё время"),
        }[data]
        _build_and_send_report(chat_id, message_id, _since, _label)

    elif data == "adm_ai_acc":
        try:
            since7  = time.time() - 7 * 86400
            since30 = time.time() - 30 * 86400

            def _acc_block(label, since):
                a = get_setup_accuracy(since)
                s, r = a["sent"], a["rejected"]
                lines = [f"*{label}:*"]
                lines.append(
                    f"  📤 Отправлено: {s['n']}"
                    + (f" · TP1 {s['tp1_pct']:.0f}% · SL {s['sl_pct']:.0f}%" if s['n'] else " · нет данных")
                )
                lines.append(
                    f"  🚫 Отклонено: {r['n']}"
                    + (f" · TP1 {r['tp1_pct']:.0f}% · SL {r['sl_pct']:.0f}%" if r['n'] else " · нет данных")
                )
                # Mirror shadow experiment: flip the rejected setups (levels swapped)
                m_dec = r.get('mirror_wins', 0) + r.get('mirror_losses', 0)
                if m_dec:
                    m_r = r.get('mirror_r', 0.0)
                    icon = "🟢" if m_r > 0 else ("🔴" if m_r < 0 else "➖")
                    lines.append(
                        f"  🔄 Отклон. перевёрнутые: {m_dec}"
                        f" · WR {r.get('mirror_wr', 0):.0f}%"
                        f" · {icon} {m_r:+.1f}R ({r.get('mirror_r_avg', 0):+.2f}/сд)"
                    )
                else:
                    lines.append("  🔄 Отклон. перевёрнутые: нет данных")
                # Verdict: is the rejected bucket actually worse?
                if s['n'] >= 10 and r['n'] >= 10:
                    gap = s['tp1_pct'] - r['tp1_pct']
                    if gap >= 8:
                        lines.append(f"  ✅ ИИ режет хуже на {gap:.0f}пп TP1 — фильтр работает")
                    elif gap <= -8:
                        lines.append(f"  ⚠️ Отклонённые доходят до TP1 на {-gap:.0f}пп *чаще* — ИИ слишком строг")
                    else:
                        lines.append(f"  ➖ Разница {gap:+.0f}пп — ИИ почти не отделяет")
                return "\n".join(lines)

            txt = (
                "🎯 *ТОЧНОСТЬ ИИ*\n"
                "_Сравнение исхода отправленных vs отклонённых сетапов "
                "(теневой трекинг по реальным котировкам)._\n\n"
                f"{_acc_block('За 7 дней', since7)}\n\n"
                f"{_acc_block('За 30 дней', since30)}\n\n"
                "_TP1% = доля дошедших до первого тейка._\n"
                "_🔄 перевёрнутые = эксперимент: если бы зеркалили отклонённые "
                "(стоп↔тейк). +R = зеркало в плюс. Нужна выборка ≥20-30 и пару "
                "недель, прежде чем верить._"
            )

            # SL-wick diagnostic: how many live stops were thin-X-Perp noise
            def _wick_line(label, since):
                w = get_sl_wick_stats(since)
                if not w["n"]:
                    return f"*{label}:* нет данных"
                return (f"*{label}:* стопов {w['n']} · "
                        f"🌊 X-Perp шум {w['xperp_only']} ({w['xperp_only_pct']:.0f}%) · "
                        f"📉 реальный разворот {w['confirmed']}")
            txt += (
                "\n\n━━━━━━━━━\n"
                "🎯 *Природа стопов* (X-Perp фитиль vs реальный разворот)\n"
                f"{_wick_line('7 дней', since7)}\n"
                f"{_wick_line('30 дней', since30)}\n"
                "_🌊 = стоп задело только на тонком X-Perp, глубокий рынок нет "
                "(шум исполнения). Если доля высокая — стоит расширить буфер стопа._"
            )
        except Exception as e:
            txt = f"Ошибка: {e}"
        _edit_message(chat_id, message_id, txt)

    elif data == "adm_open":
        try:
            sigs = get_open_signals()
            if not sigs:
                txt = "📋 *Открытые сделки*\n\nНет активных позиций."
            else:
                lines = ["📋 *Открытые сделки*\n"]
                for s in sigs:
                    lines.append(_format_open_signal(s))
                txt = "\n\n".join(lines)
        except Exception as e:
            txt = f"Ошибка: {e}"
        _edit_message(chat_id, message_id, txt)

    elif data == "adm_blocks":
        try:
            blocks = get_active_symbol_blocks()
            if not blocks:
                txt = "🚫 *Авто-блок*\n\nЗаблокированных тикеров нет."
                _edit_message(chat_id, message_id, txt)
            else:
                import time as _t
                lines = ["🚫 *Авто-блок*\n"]
                keyboard_rows = []
                for b in blocks:
                    until = datetime.fromtimestamp(b["blocked_until"], tz=_riga_tz()).strftime("%d.%m %H:%M")
                    lines.append(f"• *{_disp_sym(b['symbol'])}* до {until}\n  _{b['reason']}_")
                    keyboard_rows.append([{
                        "text": f"✅ Разблокировать {_disp_sym(b['symbol'])}",
                        "callback_data": f"adm_unblock_{b['symbol']}",
                    }])
                # add back-row with main buttons
                keyboard_rows.append([
                    {"text": "« Назад", "callback_data": _sec_back_cb(chat_id)}
                ])
                _edit_with_keyboard(chat_id, message_id, "\n".join(lines), keyboard_rows)
        except Exception as e:
            _edit_message(chat_id, message_id, f"Ошибка: {e}")

    elif data.startswith("adm_unblock_"):
        symbol = data[len("adm_unblock_"):]
        try:
            unblock_symbol(symbol)
            txt = f"✅ *{_disp_sym(symbol)}* разблокирована.\n\nНажми 🚫 Авто-блок чтобы обновить список."
        except Exception as e:
            txt = f"Ошибка разблокировки: {e}"
        _edit_message(chat_id, message_id, txt)

    elif data == "adm_top" or data.startswith("adm_top_d"):
        try:
            _now_riga = datetime.now(_riga_tz())
            _midnight = _now_riga.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
            period_map = {
                "adm_top":    (30, None,      "30д"),
                "adm_top_d0": (30, _midnight, "сегодня"),
                "adm_top_d7": (7,  None,      "7д"),
                "adm_top_d30":(30, None,      "30д"),
            }
            days, since_ts, label = period_map.get(data, (30, None, "30д"))
            perfs = get_symbols_performance(days=days, since_ts=since_ts)
            top = [p for p in perfs if p["total_r"] > 0][:8]
            if not top:
                txt = f"🏆 *Топ тикеров ({label})*\n\nНет прибыльных тикеров с данными."
            else:
                lines = [f"🏆 *Топ тикеров ({label})*\n"]
                for i, p in enumerate(top, 1):
                    lines.append(
                        f"{i}. *{_disp_sym(p['symbol'])}*  {p['total_r']:+.2f}R  "
                        f"win {p['win_rate']}%  ({p['trades']} сд)"
                    )
                txt = "\n".join(lines)
            period_kb = [[
                {"text": "📅 Сегодня", "callback_data": "adm_top_d0"},
                {"text": "📅 7 дней",  "callback_data": "adm_top_d7"},
                {"text": "📅 30 дней", "callback_data": "adm_top_d30"},
            ], [{"text": "« Назад", "callback_data": _sec_back_cb(chat_id)}]]
            _edit_with_keyboard(chat_id, message_id, txt, period_kb)
        except Exception as e:
            _edit_message(chat_id, message_id, f"Ошибка: {e}")

    elif data == "adm_worst" or data.startswith("adm_worst_d"):
        try:
            _now_riga = datetime.now(_riga_tz())
            _midnight = _now_riga.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
            period_map = {
                "adm_worst":    (30, None,      "30д"),
                "adm_worst_d0": (30, _midnight, "сегодня"),
                "adm_worst_d7": (7,  None,      "7д"),
                "adm_worst_d30":(30, None,      "30д"),
            }
            days, since_ts, label = period_map.get(data, (30, None, "30д"))
            perfs = get_symbols_performance(days=days, since_ts=since_ts)
            worst = [p for p in reversed(perfs) if p["trades"] >= 1 and p["total_r"] < 0][:8]
            if not worst:
                txt = f"💀 *Худшие тикеры ({label})*\n\nНедостаточно данных или убытков нет."
            else:
                lines = [f"💀 *Худшие тикеры ({label})*\n"]
                for i, p in enumerate(worst, 1):
                    lines.append(
                        f"{i}. *{_disp_sym(p['symbol'])}*  {p['total_r']:+.2f}R  "
                        f"win {p['win_rate']}%  ({p['trades']} сд)"
                    )
                txt = "\n".join(lines)
            period_kb = [[
                {"text": "📅 Сегодня", "callback_data": "adm_worst_d0"},
                {"text": "📅 7 дней",  "callback_data": "adm_worst_d7"},
                {"text": "📅 30 дней", "callback_data": "adm_worst_d30"},
            ], [{"text": "« Назад", "callback_data": _sec_back_cb(chat_id)}]]
            _edit_with_keyboard(chat_id, message_id, txt, period_kb)
        except Exception as e:
            _edit_message(chat_id, message_id, f"Ошибка: {e}")

    elif data == "adm_users" or data == "adm_users_p0":
        _render_users_page(chat_id, message_id, 0)

    elif data.startswith("adm_users_p"):
        try:
            page = int(data[len("adm_users_p"):])
        except ValueError:
            page = 0
        _render_users_page(chat_id, message_id, page)

    elif data.startswith("adm_users_q_"):
        # Format: adm_users_q_{query}_p{N}
        rest = data[len("adm_users_q_"):]
        try:
            q_part, p_part = rest.rsplit("_p", 1)
            page = int(p_part)
        except ValueError:
            q_part, page = rest, 0
        _render_users_page(chat_id, message_id, page, query=q_part)

    elif data == "adm_users_search":
        _pending_users_search[chat_id] = True
        try:
            _requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                json={
                    "chat_id": chat_id,
                    "text": "🔍 Введи @ник или Telegram ID для поиска:",
                    "parse_mode": "Markdown",
                },
                timeout=10,
            )
        except Exception as e:
            log.warning(f"users_search prompt failed: {e}")

    elif data == "adm_admins":
        try:
            dynamic  = get_dynamic_admins()
            mods     = [a for a in dynamic if a.get("role") == "moderator"]
            fulls    = [a for a in dynamic if a.get("role") != "moderator"]
            lines = ["👮 *Управление админами*\n"]
            lines.append("🔒 *Супер-админы (config):*")
            for aid in sorted(ADMIN_IDS):
                lines.append(f"  `{aid}`")
            if fulls:
                lines.append("\n➕ *Добавленные админы:*")
                for a in fulls:
                    fn   = a.get("first_name") or ""
                    un   = f" @{a['username']}" if a.get("username") else ""
                    lines.append(f"  • {fn}{un} `{a['user_id']}`")
            if mods:
                lines.append("\n🛡 *Модераторы:*")
                for a in mods:
                    fn   = a.get("first_name") or ""
                    un   = f" @{a['username']}" if a.get("username") else ""
                    lines.append(f"  • {fn}{un} `{a['user_id']}`")
            if not fulls and not mods:
                lines.append("\n_Добавленных админов/модераторов нет._")

            kb_rows = []
            if _is_super_admin(user_id):
                for a in dynamic:
                    label = a.get("first_name") or str(a["user_id"])
                    kb_rows.append([{
                        "text": f"❌ Удалить {label}",
                        "callback_data": f"adm_rm_admin_{a['user_id']}",
                    }])
                kb_rows.append([{
                    "text": "➕ Добавить администратора",
                    "callback_data": "adm_add_admin",
                }])
                kb_rows.append([{
                    "text": "➕ Добавить модератора",
                    "callback_data": "adm_add_moderator",
                }])
            kb_rows.append([{"text": "« Назад", "callback_data": _sec_back_cb(chat_id)}])
            _edit_with_keyboard(chat_id, message_id, "\n".join(lines), kb_rows)
        except Exception as e:
            _edit_message(chat_id, message_id, f"Ошибка: {e}")

    elif data.startswith("adm_rm_admin_"):
        if not _is_super_admin(user_id):
            _edit_message(chat_id, message_id, "⛔ Только супер-администратор может удалять.")
            return
        try:
            rm_id = int(data[len("adm_rm_admin_"):])
            remove_dynamic_admin(rm_id)
            txt = f"✅ Удалено: `{rm_id}`.\n\nНажми 👮 Админы чтобы обновить список."
        except Exception as e:
            txt = f"Ошибка удаления: {e}"
        _edit_message(chat_id, message_id, txt)

    elif data in ("adm_add_admin", "adm_add_moderator"):
        if not _is_super_admin(user_id):
            _edit_message(chat_id, message_id, "⛔ Только супер-администратор может добавлять.")
            return
        role = "moderator" if data == "adm_add_moderator" else "admin"
        _pending_add_admin[chat_id] = role
        label = "модератора" if role == "moderator" else "администратора"
        try:
            _requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                json={
                    "chat_id": chat_id,
                    "text": f"Отправь *Telegram ID* нового {label}.\nПример: `123456789`",
                    "parse_mode": "Markdown",
                },
                timeout=10,
            )
        except Exception as e:
            log.warning(f"add_admin prompt failed: {e}")

    elif data == "adm_autotrade":
        try:
            rows = at_all_allowed()
            lines = ["🤖 *Автотрейдинг — доступ*\n"]
            if rows:
                for r in rows:
                    u_info = get_user_by_id(r["user_id"]) or {}
                    un = f"@{u_info['username']}" if u_info.get("username") else ""
                    st = "🟢 активен" if r.get("active") else ("🟡 ждёт настройки" if r.get("allowed") else "⚪")
                    lines.append(f"• {un} `{r['user_id']}` — {st}")
            else:
                lines.append("_Никто не добавлен._")
            if not keystore_ready():
                lines.append("\n⚠️ *AUTOTRADE\\_ENC\\_KEY не настроен* — подключение ключей не заработает.")
            kb_rows = []
            for r in rows:
                if r.get("allowed"):
                    kb_rows.append([{
                        "text": f"❌ Убрать {r['user_id']}",
                        "callback_data": f"adm_at_rm_{r['user_id']}",
                    }])
            kb_rows.append([{"text": "➕ Добавить в автотрейдинг", "callback_data": "adm_at_add"}])
            kb_rows.append([{"text": "« Назад", "callback_data": _sec_back_cb(chat_id)}])
            _edit_with_keyboard(chat_id, message_id, "\n".join(lines), kb_rows)
        except Exception as e:
            _edit_message(chat_id, message_id, f"Ошибка: {e}")

    elif data == "adm_at_add":
        _pending_add_autotrade[chat_id] = True
        try:
            _requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                json={
                    "chat_id": chat_id,
                    "text": "Отправь *Telegram ID* пользователя для автотрейдинга.\nПример: `123456789`",
                    "parse_mode": "Markdown",
                },
                timeout=10,
            )
        except Exception as e:
            log.warning(f"at_add prompt failed: {e}")

    elif data.startswith("adm_at_rm_"):
        try:
            rm_id = int(data[len("adm_at_rm_"):])
            at_remove(rm_id)
            txt = (f"✅ Пользователь `{rm_id}` убран из автотрейдинга.\n"
                   f"Его API-ключи удалены из базы. Открытые позиции (если есть) "
                   f"остаются на бирже — управлять ими бот больше не будет.")
            try:
                autotrader._dm(rm_id, "🤖 Автотрейдинг отключён администратором. "
                                      f"Вопросы — {AUTOTRADE_CONTACT}.")
            except Exception:
                pass
        except Exception as e:
            txt = f"Ошибка: {e}"
        _edit_message(chat_id, message_id, txt)

    elif data == "adm_deals" or data == "adm_deals_p0":
        _render_deals_page(chat_id, message_id, 0)

    elif data.startswith("adm_deals_p"):
        try:
            page = int(data[len("adm_deals_p"):])
        except ValueError:
            page = 0
        _render_deals_page(chat_id, message_id, page)

    elif data == "adm_deals_search":
        _render_deals_symbol_picker(chat_id, message_id)

    elif data.startswith("adm_dsym_"):
        # Format: adm_dsym_{SYMBOL}_p{N}
        rest = data[len("adm_dsym_"):]
        try:
            sym_part, page_part = rest.rsplit("_p", 1)
            page = int(page_part)
        except ValueError:
            sym_part, page = rest, 0
        _render_deals_page(chat_id, message_id, page, symbol=sym_part)

    elif data.startswith("adm_del_sig_"):
        try:
            sig_id = int(data[len("adm_del_sig_"):])
            delete_signal(sig_id)
        except Exception as e:
            log.warning(f"delete signal failed: {e}")
        # Refresh the deals list so the deleted row disappears in place
        _render_deals_page(chat_id, message_id, 0)

    elif data == "adm_manblock":
        _render_manual_block_panel(chat_id, message_id)

    elif data.startswith("adm_mb_block_"):
        symbol = data[len("adm_mb_block_"):]
        try:
            set_symbol_block(symbol, days=1, reason="Manual block by admin")
            _answer_callback(callback_id, f"🚫 {_disp_sym(symbol)} заблокирована на 24ч")
        except Exception as e:
            _answer_callback(callback_id, f"Ошибка: {e}")
        _render_manual_block_panel(chat_id, message_id)

    elif data.startswith("adm_mb_unblock_"):
        symbol = data[len("adm_mb_unblock_"):]
        try:
            unblock_symbol(symbol)
            _answer_callback(callback_id, f"✅ {_disp_sym(symbol)} разблокирована")
        except Exception as e:
            _answer_callback(callback_id, f"Ошибка: {e}")
        _render_manual_block_panel(chat_id, message_id)

    elif data == "adm_mb_input":
        _pending_block_chat[chat_id] = 1  # days
        try:
            _requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                json={
                    "chat_id": chat_id,
                    "text": (
                        "Введи тикер акции для блокировки на 24ч.\n"
                        "Пример: `BTCUSDC` или `BTC`"
                    ),
                    "parse_mode": "Markdown",
                },
                timeout=10,
            )
        except Exception as e:
            log.warning(f"adm_mb_input prompt failed: {e}")

    elif data == "adm_budget":
        try:
            from config import CLAUDE_DAILY_BUDGET_USD
            s = get_claude_spend_stats()
            remaining = max(0.0, round(CLAUDE_DAILY_BUDGET_USD - s["today_usd"], 4))
            bar_filled = int((s["today_usd"] / CLAUDE_DAILY_BUDGET_USD) * 10) if CLAUDE_DAILY_BUDGET_USD else 0
            bar_filled = min(bar_filled, 10)
            bar = "█" * bar_filled + "░" * (10 - bar_filled)
            txt = (
                f"💰 *Бюджет Claude*\n\n"
                f"Лимит: ${CLAUDE_DAILY_BUDGET_USD:.2f}/день\n"
                f"[{bar}] ${s['today_usd']:.4f}\n"
                f"Осталось сегодня: *${remaining:.4f}*\n\n"
                f"*За сегодня:* {s['today_calls']} вызовов · ${s['today_usd']:.4f}\n"
                f"*За 7 дней:* {s['week_calls']} вызовов · ${s['week_usd']:.4f}\n"
                f"*Всего:* {s['total_calls']} вызовов · ${s['total_usd']:.4f}"
            )
        except Exception as e:
            txt = f"Ошибка: {e}"
        _edit_message(chat_id, message_id, txt)

    elif data == "adm_filters":
        try:
            from config import (
                EFF_RATIO_FILTER, EFF_RATIO_MIN,
                BEAR_TREND_HOT_VOL_GUARD, BEAR_TREND_HOT_VOL_MIN_RATIO,
                BEAR_TREND_SKIP_SESSIONS,
                DIRECTIONAL_RSI_MIDLINE_FILTER, RSI_LONG_MIN_MIDLINE, RSI_SHORT_MAX_MIDLINE,
                OVERLAP_BEARISH_1H_GUARD, MACD_CHOCH_NOISE_FILTER,
                DAILY_TREND_FILTER, DOUBLE_NEUTRAL_LONG_FILTER, DAILY_TREND_SHORT_FILTER,
                VOL_REGIME_FILTER, VOL_MIN_ATR_PCT, VOL_MIN_RATIO,
                SOURCE_EDGE_FILTER, LOW_EDGE_FVG_SYMBOLS,
                SYMBOL_EDGE_FILTER, LOW_EDGE_SYMBOLS,
                DIRECTION_EDGE_FILTER, LOW_EDGE_SHORT_SYMBOLS,
                LONG_RELATIVE_WEAKNESS_FILTER, LONG_RELATIVE_WEAKNESS_MAX_PCT,
                LONG_NY_COIN_MOMENTUM_FILTER,
                SHORT_FVG_COIN_MOMENTUM_FILTER,
                FVG_LONDON_BTC_UP_FILTER, FVG_LONDON_BTC_UP_MIN_PCT,
                QUALITY_RISK_OVERLAY, QUALITY_RISK_MULT,
                REL_STRENGTH_RISK_UP, REL_STRENGTH_RISK_UP_MULT,
                TREND_PAIR_RISK_UP, TREND_PAIR_RISK_UP_MULT,
                AUTO_BLOCK_ENABLED, AUTO_BLOCK_MIN_TRADES,
                AUTO_BLOCK_MAX_PROFIT_FACTOR, AUTO_BLOCK_MAX_WIN_RATE,
                ADAPTIVE_FILTER_PACKS, REQUIRE_STRICT_HTF,
            )
            # MTF_MIN_SCORE / TRAIL_RUNNER_ENABLED / TRAIL_ATR_MULT / SCAN_INTERVAL_MINUTES are
            # imported at module level and MUST NOT be re-imported here: a name
            # bound anywhere inside a function is local to ALL of it, so any
            # branch reading one of them BEFORE this line raises
            # UnboundLocalError. This exact trap fired on the sister crypto bot.
            import time as _t
            stats = _last_scan_stats
            if stats["ts"] > 0:
                age_min = int((_t.time() - stats["ts"]) / 60)
                scan_line = (
                    f"🕐 Последний скан: {age_min} мин назад\n"
                    f"  Тикеров в пуле: {stats['coins']}  →  "
                    f"SMC: {stats['setups']}  →  "
                    f"Claude: {stats['enriched']}  →  "
                    f"Отправлено: {stats['sent']}\n\n"
                )
            else:
                scan_line = "🕐 Скан ещё не запускался\n\n"

            def _f(on, label, hint=""):
                icon = "✅" if on else "⬜"
                return f"{icon} {label}" + (f"\n   _↳ {hint}_" if hint and on else "")
            fvg_skip = ", ".join(LOW_EDGE_FVG_SYMBOLS[:3]) or "нет"
            sym_skip = ", ".join(LOW_EDGE_SYMBOLS[:3]) or "нет"
            dir_skip = ", ".join(LOW_EDGE_SHORT_SYMBOLS[:3]) or "нет"
            txt = (
                f"📊 *Мониторинг фильтров*\n\n"
                f"{scan_line}"

                f"*🔍 Качество сетапа*\n"
                f"{_f(EFF_RATIO_FILTER, 'Чистота тренда', f'блокирует боковик — движение должно быть направленным (порог {EFF_RATIO_MIN})')}\n"
                f"{_f(VOL_REGIME_FILTER, 'Режим волатильности', f'пропускает только живые рынки — не спящие и не взрывные')}\n"
                f"{_f(DIRECTIONAL_RSI_MIDLINE_FILTER, 'RSI по направлению', f'лонг только при RSI≥{RSI_LONG_MIN_MIDLINE}, шорт при RSI<{RSI_SHORT_MAX_MIDLINE} — отсекает против-тренд')}\n"
                f"{_f(EFF_RATIO_FILTER, 'Мин. балл сетапа', f'нужно ≥{MTF_MIN_SCORE} подтверждений из разных таймфреймов')}\n"

                f"\n*📅 Тренд и направление*\n"
                f"{_f(DAILY_TREND_FILTER, 'Дневной тренд (лонг)', 'лонг запрещён если дневная свеча медвежья — не покупаем против дня')}\n"
                f"{_f(DAILY_TREND_SHORT_FILTER, 'Дневной тренд (шорт)', 'шорт запрещён если дневная свеча бычья — не шортим против дня')}\n"
                f"{_f(DOUBLE_NEUTRAL_LONG_FILTER, 'Двойной боковик', '4h + дневка оба нейтральны = полный боковик, лонги пропускаем')}\n"
                f"{_f(OVERLAP_BEARISH_1H_GUARD, 'Защита Overlap-сессии', 'лонг в перекрытие Лондон+Нью-Йорк при медвежьем 1h — пропускаем (опоздавшие входы давят цену)')}\n"
                f"{_f(BEAR_TREND_HOT_VOL_GUARD, 'Защита шорт-сквиза', f'медвежий тренд + объём ≥{BEAR_TREND_HOT_VOL_MIN_RATIO}x = переполненный шорт, пропускаем')}\n"
                f"{_f(MACD_CHOCH_NOISE_FILTER, 'Шум MACD/ChoCH', 'блокирует ложные развороты без подтверждения MACD')}\n"

                f"\n*⚡ Моментум тикера*\n"
                f"{_f(LONG_RELATIVE_WEAKNESS_FILTER, 'Слабость акции vs SPY', f'лонг пропускаем если акция слабее SPY (рынка) на ≥{abs(LONG_RELATIVE_WEAKNESS_MAX_PCT)}% за час — нет интереса покупателей')}\n"
                f"{_f(LONG_NY_COIN_MOMENTUM_FILTER, 'Лонг по моментуму', 'лонг на открытой сессии США только если акция уже растёт — не против моментума')}\n"
                f"{_f(SHORT_FVG_COIN_MOMENTUM_FILTER, 'Шорт FVG моментум', 'шорт по FVG-зоне только если акция уже падает — не против моментума')}\n"
                f"{_f(FVG_LONDON_BTC_UP_FILTER, 'FVG Лондон + BTC растёт', f'FVG-шорт в Лондон пропускаем если BTC вырос ≥{FVG_LONDON_BTC_UP_MIN_PCT}% за час — шортить против роста BTC опасно')}\n"

                f"\n*🚫 Заблокированные тикеры/стратегии*\n"
                f"{_f(SYMBOL_EDGE_FILTER, f'Тикеры без статистики: {sym_skip}', 'исторически плохие результаты — полностью исключены')}\n"
                f"{_f(SOURCE_EDGE_FILTER, f'FVG-зоны запрещены: {fvg_skip}', 'у этих тикеров FVG-сетапы не работают — только OB')}\n"
                f"{_f(DIRECTION_EDGE_FILTER, f'Шорты запрещены: {dir_skip}', 'у этих тикеров шорты исторически убыточны')}\n"

                f"\n*💰 Повышение размера позиции*\n"
                f"{_f(QUALITY_RISK_OVERLAY, f'Бонус за качество ×{QUALITY_RISK_MULT}', 'OB-вход + хороший RSI + объём + топ-тикер = увеличиваем риск на 15%')}\n"
                f"{_f(REL_STRENGTH_RISK_UP, f'Бонус за силу монеты ×{REL_STRENGTH_RISK_UP_MULT}', 'акция сильнее SPY (рынка) = больше шансов дойти до TP2, берём чуть больше')}\n"
                f"{_f(TREND_PAIR_RISK_UP, f'Бонус за тренд ×{TREND_PAIR_RISK_UP_MULT}', '1h и 4h оба в одну сторону = сильный тренд, увеличиваем')}\n"

                f"\n*⚙️ Управление сделкой*\n"
                f"{_f(TRAIL_RUNNER_ENABLED, f'Трейлинг-стоп ATR×{TRAIL_ATR_MULT}', 'после TP1 остаток ведётся скользящим стопом — не даём прибыли уйти в ноль')}\n"
                f"{_f(AUTO_BLOCK_ENABLED, 'Авто-блок убыточных тикеров', f'тикер с ≥{AUTO_BLOCK_MIN_TRADES} сделками и WR≤{AUTO_BLOCK_MAX_WIN_RATE}% автоматически блокируется')}\n"

                f"\n🕐 Скан каждые {SCAN_INTERVAL_MINUTES} мин  •  мин. балл сетапа: {MTF_MIN_SCORE}"
            )
        except Exception as e:
            txt = f"Ошибка: {e}"
        _edit_message(chat_id, message_id, txt)

    elif data in ("adm_setups", "adm_setups_today"):
        _riga = _riga_tz()
        date_str = datetime.now(_riga).strftime("%d.%m.%Y")
        rows = get_setups_by_date(date_str)
        txt, total_pages = _format_setups_page(rows, date_str, 0)
        _edit_with_keyboard(chat_id, message_id, txt,
                            _setups_kb(chat_id, date_str, 0, total_pages))

    elif data.startswith("adm_setups_pg_"):
        payload = data[len("adm_setups_pg_"):]
        date_str, _, page_s = payload.rpartition("_")
        try:
            page = int(page_s)
        except ValueError:
            page = 0
        rows = get_setups_by_date(date_str)
        txt, total_pages = _format_setups_page(rows, date_str, page)
        _edit_with_keyboard(chat_id, message_id, txt,
                            _setups_kb(chat_id, date_str, page, total_pages))

    elif data == "adm_setups_date":
        _pending_setups_date[chat_id] = message_id
        try:
            _requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                json={
                    "chat_id": chat_id,
                    "text": "📅 Введи дату в формате ДД.ММ или ДД.ММ.ГГГГ\nПример: `10.06` или `10.06.2026`",
                    "parse_mode": "Markdown",
                },
                timeout=10,
            )
        except Exception as e:
            log.warning(f"setups_date prompt failed: {e}")

    elif data == "adm_back":
        _admin_section.pop(chat_id, None)
        _edit_admin_text(chat_id, message_id, "🛠 *TUSA Admin Panel*\nВыбери раздел:", _ADMIN_KEYBOARD)

    elif data == "adm_open_new":
        # Sent after user-search results (new message, no message_id to edit) — open fresh panel
        _send_keyboard(chat_id, "🛠 *TUSA Admin Panel*\nВыбери раздел:")


_SETUPS_PER_PAGE = 10


def _setups_kb(chat_id, date_str: str, page: int, total_pages: int) -> list:
    """Nav keyboard for the paginated setup history."""
    nav = []
    if page > 0:
        nav.append({"text": "◀️", "callback_data": f"adm_setups_pg_{date_str}_{page - 1}"})
    if page < total_pages - 1:
        nav.append({"text": "▶️", "callback_data": f"adm_setups_pg_{date_str}_{page + 1}"})
    kb_rows = []
    if nav:
        kb_rows.append(nav)
    kb_rows.append([
        {"text": "📅 Другая дата", "callback_data": "adm_setups_date"},
        {"text": "« Назад",        "callback_data": _sec_back_cb(chat_id)},
    ])
    return kb_rows


def _format_setups_page(rows: list, date_str: str, page: int = 0,
                        per: int = _SETUPS_PER_PAGE) -> tuple:
    """Render one page of setup_log rows. Returns (text, total_pages)."""
    _riga = _riga_tz()
    if not rows:
        return f"🔍 *История сетапов за {date_str}*\n\nНет сетапов за эту дату.", 1

    total = len(rows)
    total_pages = (total + per - 1) // per
    page = max(0, min(page, total_pages - 1))
    page_rows = rows[page * per:(page + 1) * per]

    lines = [f"🔍 *История сетапов за {date_str}* — {total} шт. "
             f"(стр. {page + 1}/{total_pages})\n"]
    for r in page_rows:
        ts_str = datetime.fromtimestamp(r["ts"], tz=_riga).strftime("%H:%M")
        sym    = r.get("symbol", "?")
        direct = r.get("direction", "?")
        entry  = r.get("entry_price")
        tp1    = r.get("tp1")
        tp2    = r.get("tp2")
        dec    = r.get("decision") or "NO TRADE"
        conf   = r.get("confidence") or ""
        risk   = r.get("risk_score")
        reason = r.get("reason") or ""
        sent   = r.get("sent", 0)

        dir_icon = "📈" if direct == "LONG" else "📉"
        dec_icon = "✅" if dec in ("LONG", "SHORT") else "❌"
        sent_icon = "📤 отправлен" if sent else "🚫 отклонён"

        def _px(v):
            if v is None: return "?"
            return f"{v:,.0f}" if v >= 100 else f"{v:.4g}"

        entry_s = _px(entry)
        tp1_s   = _px(tp1)
        tp2_s   = _px(tp2)
        risk_s  = f" R{risk}" if risk is not None else ""
        conf_s  = f" {conf}" if conf else ""
        reason_safe  = (reason.replace("_", "\\_").replace("*", "\\*")
                              .replace("`", "\\`").replace("[", "\\["))[:60]
        reason_short = (reason_safe + "…") if len(reason) > 60 else reason_safe

        lines.append(
            f"{ts_str} {dir_icon} *{_disp_sym(sym)}* {direct}\n"
            f"  Вход: `{entry_s}` · TP1: `{tp1_s}` · TP2: `{tp2_s}`\n"
            f"  {dec_icon} Клод: {dec}{conf_s}{risk_s} — {sent_icon}\n"
            + (f"  _{reason_short}_\n" if reason_short else "")
        )

    return "\n".join(lines), total_pages


def _num(s):
    """Parse FF numeric string ('0.3%', '187K', '<0.1') → float or None."""
    if not s:
        return None
    t = s.strip().replace(",", "").replace("%", "").replace("<", "").replace(">", "")
    mult = 1.0
    if t and t[-1] in ("K", "k"):
        mult, t = 1e3, t[:-1]
    elif t and t[-1] in ("M", "m"):
        mult, t = 1e6, t[:-1]
    elif t and t[-1] in ("B", "b"):
        mult, t = 1e9, t[:-1]
    try:
        return float(t) * mult
    except Exception:
        return None


# (keyword substring, RU title, short RU explanation) — first match wins,
# so put more specific phrases before generic ones.
_RU_EVENTS = [
    ("ism manufacturing prices", "Цены в промышленности (ISM)",
     "ценовое давление у производителей — сигнал по инфляции"),
    ("ism manufacturing", "Деловая активность в промышленности (ISM)",
     "настроения производителей; выше 50 = рост экономики"),
    ("ism services", "Деловая активность в услугах (ISM)",
     "настроения в секторе услуг; выше 50 = рост"),
    ("non-farm", "Занятость вне сельского хозяйства (NFP)",
     "ключевой отчёт по рынку труда США"),
    ("nonfarm", "Занятость вне сельского хозяйства (NFP)",
     "ключевой отчёт по рынку труда США"),
    ("adp", "Занятость в частном секторе (ADP)",
     "предвестник NFP по найму в частном секторе"),
    ("unemployment rate", "Уровень безработицы", "доля безработных"),
    ("unemployment claims", "Заявки на пособие по безработице",
     "число новых заявок за неделю"),
    ("jobless claims", "Заявки на пособие по безработице",
     "число новых заявок за неделю"),
    ("core cpi", "Базовая инфляция (Core CPI)",
     "рост цен без еды и энергии"),
    ("cpi", "Инфляция (CPI)", "рост потребительских цен"),
    ("core ppi", "Базовые цены производителей (Core PPI)", "оптовая инфляция"),
    ("ppi", "Цены производителей (PPI)", "оптовая инфляция"),
    ("core pce", "Базовый PCE", "любимый показатель инфляции ФРС"),
    ("pce", "Расходы на личное потребление (PCE)", "инфляция и траты"),
    ("retail sales", "Розничные продажи", "потребительский спрос"),
    ("gdp", "ВВП", "темп роста экономики"),
    ("federal funds rate", "Решение ФРС по ставке",
     "главное событие — ставка ФРС"),
    ("interest rate decision", "Решение по процентной ставке",
     "уровень ключевой ставки"),
    ("fomc statement", "Заявление ФРС", "сопроводительный текст к ставке"),
    ("fomc meeting minutes", "Протокол заседания ФРС",
     "детали обсуждения ставки"),
    ("fomc economic projections", "Экономические прогнозы ФРС",
     "ожидания ФРС по ставке и инфляции"),
    ("powell", "Выступление главы ФРС Пауэлла",
     "намёки на курс по ставке"),
    ("bailey", "Выступление главы Банка Англии",
     "намёки на курс по ставке"),
    ("lagarde", "Выступление главы ЕЦБ", "намёки на курс по ставке"),
    ("fomc member", "Выступление члена ФРС", "намёки на курс по ставке"),
    ("fed chair", "Выступление главы ФРС", "намёки на курс по ставке"),
    ("member", "Выступление представителя ЦБ", "намёки на курс по ставке"),
    ("speaks", "Выступление представителя ЦБ", "намёки на курс по ставке"),
    ("consumer confidence", "Индекс доверия потребителей",
     "настроения покупателей"),
    ("consumer sentiment", "Индекс настроений потребителей",
     "настроения покупателей"),
    ("durable goods", "Заказы на товары длительного пользования",
     "спрос на дорогие товары"),
    ("trade balance", "Торговый баланс", "экспорт минус импорт"),
    ("building permits", "Разрешения на строительство",
     "активность в недвижимости"),
    ("crude oil inventories", "Запасы нефти", "влияет на цену нефти"),
    ("manufacturing pmi", "PMI в промышленности",
     "деловая активность; выше 50 = рост"),
    ("services pmi", "PMI в услугах", "деловая активность; выше 50 = рост"),
    ("flash manufacturing pmi", "Предв. PMI в промышленности",
     "ранняя оценка деловой активности"),
    ("flash services pmi", "Предв. PMI в услугах",
     "ранняя оценка деловой активности"),
    ("pmi", "Индекс деловой активности (PMI)",
     "выше 50 = рост, ниже = спад"),
    ("bank holiday", "Банковский выходной", "биржи/банки закрыты"),
]


def _ru_event(title: str):
    """Map an English FF title → (ru_title, ru_note). Falls back to original."""
    low = title.lower()
    for kw, ru, note in _RU_EVENTS:
        if kw in low:
            return ru, note
    return title, ""


# Market impact one-liners shown after actual result (better / worse than forecast).
# Tuple: (note_if_better, note_if_worse)
_MARKET_NOTES: dict = {
    "core cpi":        ("Базовая инфляция выше → ФРС не снижает ставку → крипта ↓",
                        "Базовая инфляция ниже → ФРС ближе к снижению → крипта ↑"),
    "cpi":             ("Инфляция выше ожиданий → ФРС жёстче → крипта/акции ↓",
                        "Инфляция ниже ожиданий → путь к снижению ставки → крипта ↑"),
    "core ppi":        ("Оптовая инфляция выше → давление сохраняется → осторожно",
                        "Оптовая инфляция ниже → хороший сигнал для рынков"),
    "ppi":             ("Цены производителей выше → инфляционное давление → крипта ↓",
                        "Цены производителей ниже → меньше инфляции → позитив"),
    "core pce":        ("PCE выше → ФРС не спешит со снижением → риски для крипты",
                        "PCE ниже → снижение ставки ближе → позитив для крипты"),
    "pce":             ("Расходы выше ожиданий → инфляционное давление",
                        "Расходы ниже → потребитель экономит → осторожно"),
    "non-farm":        ("Рынок труда силён → доллар ↑, крипта под давлением",
                        "Занятость слабее → доллар ↓, позитив для крипты"),
    "nonfarm":         ("Рынок труда силён → доллар ↑, крипта под давлением",
                        "Занятость слабее → доллар ↓, позитив для крипты"),
    "adp":             ("Частный найм активен → рынок труда силён → доллар ↑",
                        "Частный найм слабее → сигнал охлаждения экономики"),
    "unemployment rate": ("Безработица выше → экономика охлаждается",
                          "Безработица ниже → сильный рынок труда → доллар ↑"),
    "unemployment":    ("Заявки выросли → рынок труда слабеет",
                        "Заявки упали → рынок труда устойчив"),
    "jobless":         ("Заявки выросли → рынок труда слабеет",
                        "Заявки упали → рынок труда устойчив"),
    "gdp":             ("ВВП выше ожиданий → экономика сильнее → доллар ↑",
                        "ВВП ниже ожиданий → риски замедления → осторожно"),
    "retail sales":    ("Потребитель тратит больше → рост экономики",
                        "Розничные продажи слабее → потребитель экономит"),
    "federal funds":   ("Ставка выше ожиданий → доллар ↑, крипта ↓",
                        "Ставка ниже ожиданий → доллар ↓, крипта ↑"),
    "interest rate":   ("Ставка выше ожиданий → доллар ↑, крипта ↓",
                        "Ставка ниже ожиданий → доллар ↓, крипта ↑"),
    "ism manufacturing": ("Промышленность растёт → позитив для экономики",
                          "Промышленность сокращается → риски рецессии"),
    "ism services":    ("Сектор услуг растёт → позитив",
                        "Сектор услуг замедляется → осторожно"),
    "manufacturing pmi": ("Промышленность выше 50 → рост → позитив",
                          "Промышленность ниже 50 → сжатие → осторожно"),
    "services pmi":    ("Услуги выше 50 → рост экономики",
                        "Услуги ниже 50 → замедление"),
    "consumer confidence": ("Потребители оптимистичны → рост трат → позитив",
                            "Потребители пессимистичны → спад трат → осторожно"),
    "durable goods":   ("Спрос на товары высок → экономика активна",
                        "Спрос на товары слаб → инвестиции снижаются"),
}


def _market_note(event_title: str, is_better: bool) -> str:
    """Brief market impact note for a past event. Empty string if unknown."""
    low = event_title.lower()
    for kw, (note_b, note_w) in _MARKET_NOTES.items():
        if kw in low:
            return note_b if is_better else note_w
    return ""


# Market digest cache — get_daily_digest() hits Groq + RSS, cache 30 min so
# repeated button presses don't spam the API.
_digest_cache = {"at": 0.0, "items": []}


def _cached_digest() -> list:
    import time as _t
    now = _t.time()
    if now - _digest_cache["at"] > 1800:
        try:
            _digest_cache["items"] = get_daily_digest().get("items", [])
        except Exception:
            pass
        _digest_cache["at"] = now
    return _digest_cache["items"]


def _riga_tz():
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo("Europe/Riga")
    except Exception:
        from datetime import timezone as _tz, timedelta as _td
        return _tz(_td(hours=3))


def _format_day_news() -> str:
    """'📰 Новости на сегодня' — economic calendar + market headlines, ≤10 total."""
    RIGA = _riga_tz()

    data   = get_day_events(max_events=10)
    events = data["events"]
    d      = data["date"]
    crypto = _cached_digest()                      # ≤5 AI-picked market headlines

    # Budget: ≤10 total. Reserve up to 4 slots for headlines, grow if calendar small.
    n_crypto = min(len(crypto), 4)
    n_macro  = min(len(events), 10 - n_crypto)
    n_crypto = min(len(crypto), 10 - n_macro)
    events, crypto = events[:n_macro], crypto[:n_crypto]

    header = f"📰 *Новости на сегодня* ({d.strftime('%d.%m')})"
    if data["weekend_rolled"]:
        header += "\n_Выходной — календарь на понедельник._"
    lines = [header]

    # ── Economic calendar ──
    if events:
        lines.append("\n🗓 *Экономический календарь*")
        for e in events:
            flag = "🔴" if e["impact"] == "high" else "🟡"
            when = ("весь день" if e["all_day"] or not e["when_utc"]
                    else e["when_utc"].astimezone(RIGA).strftime("%H:%M по Риге"))
            cc       = f"{e['country']} " if e["country"] else ""
            ru, note = _ru_event(e["title"])
            lines.append(f"{flag} *{cc}{ru}* — {when}")
            if note:
                lines.append(f"   📖 {note}")
            f_, p_, a_ = e["forecast"], e["previous"], e["actual"]
            extra_prev = f" / пред {p_}" if p_ else ""
            if e["passed"] and a_:
                # Event passed AND actual value published
                af, ff = _num(a_), _num(f_)
                if af is not None and ff is not None:
                    is_better = af > ff
                    tag = ("📈 лучше прогноза" if is_better else
                           "📉 хуже прогноза"  if af < ff else "➡️ по прогнозу")
                else:
                    is_better = None
                    tag = "✅ вышло"
                lines.append(f"   факт *{a_}* / прогноз {f_ or '—'}{extra_prev} → {tag}")
                if is_better is not None:
                    impact = _market_note(e["title"], is_better)
                    if impact:
                        lines.append(f"   💡 _{impact}_")
            elif e["passed"]:
                # Event passed but actual not published yet
                lines.append(f"   ✅ прошло · прогноз {f_ or '—'}{extra_prev} · _факт не опубликован_")
            else:
                # Upcoming event
                lines.append(f"   🔮 прогноз {f_ or '—'}{extra_prev}")

    # ── Market headlines ──
    if crypto:
        dir_emoji = {"BULLISH": "📈", "BEARISH": "📉", "NEUTRAL": "➡️"}
        lines.append("\n📊 *Новости рынка*")
        for it in crypto:
            em = dir_emoji.get(it.get("direction", "NEUTRAL"), "➡️")
            lines.append(f"{em} *{it.get('title', '')}*")
            expl = it.get("explanation", "")
            if expl:
                lines.append(f"   {expl}")

    if not events and not crypto:
        return header + "\n\nВажных событий и новостей нет. Спокойно 🌤"

    lines.append("\n🔴 высокая важность  🟡 средняя")
    return "\n".join(lines)


# ── Telegram webhook — handles incoming messages ──────────────────────────────
# ── Autotrade onboarding (DM dialog) ──────────────────────────────────────────

def _at_delete_msg(chat_id: int, message_id: int):
    """Delete a user's message that contains a secret (API key / passphrase)."""
    try:
        _requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/deleteMessage",
            json={"chat_id": chat_id, "message_id": message_id},
            timeout=5,
        )
    except Exception as e:
        log.warning(f"_at_delete_msg failed: {e}")


def _at_reply_kb(chat_id: int, text: str, kb_rows: list):
    try:
        _requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown",
                  "reply_markup": {"inline_keyboard": kb_rows}},
            timeout=10,
        )
    except Exception as e:
        log.warning(f"_at_reply_kb failed: {e}")


def _at_ask_size(chat_id: int, balance: float, resize: bool = False):
    """Route to percent (≥$100) or fixed (<$100) size question."""
    prefix = "resize_" if resize else "size_"
    if balance >= AUTOTRADE_BALANCE_THRESHOLD:
        _at_onboarding[chat_id]["step"] = prefix + "percent"
        _reply(chat_id,
               f"💰 Твой баланс: *${balance:.2f}*.\n\n"
               f"Какой *процент депозита* ставить на каждую сделку?\n"
               f"Отправь число от *1 до 10* (например `2`).")
    else:
        _at_onboarding[chat_id]["step"] = prefix + "fixed"
        _reply(chat_id,
               f"💰 Твой баланс: *${balance:.2f}* (меньше ${AUTOTRADE_BALANCE_THRESHOLD:.0f}).\n\n"
               f"Какую *фиксированную сумму в $* ставить на каждую сделку?\n"
               f"Отправь число (например `5`). Это маржа сделки, позиция будет с плечом 10x.")


def _at_show_menu(chat_id: int, u: dict):
    """Status panel for an onboarded (active or paused) autotrader."""
    mode = u.get("size_mode")
    mode_str = (f"{u['size_value']:.0f}% от депозита" if mode == "percent"
                else f"${u['size_value']:.2f} на сделку" if mode == "fixed" else "—")
    bal = u.get("last_balance")
    bal_str = f"${bal:.2f}" if bal is not None else "—"
    state = "🟢 включён" if u.get("active") else "⏸ выключен"
    tp1_pct = float(u.get("tp1_close_pct") or 0)
    tp1_str = "не закрывать (весь объём под трейлинг)" if tp1_pct <= 0 else f"{tp1_pct:.0f}%"
    rows = [[{"text": ("⏸ Выключить" if u.get("active") else "▶️ Включить"),
              "callback_data": "at_toggle"}],
            [{"text": "⚙️ Изменить размер сделки", "callback_data": "at_resize"}],
            [{"text": f"🎯 % закрытия на TP1: {tp1_pct:.0f}%", "callback_data": "at_tp1pct"}],
            [{"text": "🔑 Заменить API-ключи", "callback_data": "at_rekey"}]]
    _at_reply_kb(chat_id,
                 f"🤖 *Автотрейдинг*\n\nСтатус: {state}\nРазмер: *{mode_str}*\n"
                 f"Закрытие на TP1: *{tp1_str}*\n"
                 f"Баланс (посл. известный): {bal_str}\nПлечо: 10x, изолированная маржа",
                 rows)


def _at_begin_keys(chat_id: int):
    """Start (or restart) the API-key part of onboarding."""
    _at_onboarding[chat_id] = {"step": "api_key", "data": {}}
    _reply(chat_id,
           "🔐 *Подключение OKX*\n\n"
           "Создай API-ключ на OKX: Профиль → API → *Торговля API*.\n"
           "Права: *Чтение + Торговля* (⚠️ БЕЗ вывода средств).\n\n"
           "Шаг 1/3 — отправь *API Key*.\n"
           "_Сообщения с ключами я сразу удаляю из чата._")


def _at_handle_text(chat_id: int, user_id: int, text_raw: str, message_id: int) -> bool:
    """Autotrade onboarding step machine. Returns True if the message was consumed."""
    st = _at_onboarding.get(chat_id)
    if not st:
        return False
    step = st["step"]
    val  = text_raw.strip()

    # A command or a menu button aborts the dialog instead of being eaten as input
    if val.startswith("/") or val.lower() in (
        "🤖 автотрейдинг", "❓ помощь", "📋 открытые сделки",
        "📈 результаты", "📰 новости на сегодня", "🛠 админ панель",
    ):
        _at_onboarding.pop(chat_id, None)
        return False

    if step == "api_key":
        _at_delete_msg(chat_id, message_id)
        st["data"]["api_key"] = val
        st["step"] = "api_secret"
        _reply(chat_id, "Шаг 2/3 — отправь *Secret Key*.")
        return True

    if step == "api_secret":
        _at_delete_msg(chat_id, message_id)
        st["data"]["api_secret"] = val
        st["step"] = "passphrase"
        _reply(chat_id, "Шаг 3/3 — отправь *пасс-фразу* (passphrase), которую ты указал при создании ключа.")
        return True

    if step == "passphrase":
        _at_delete_msg(chat_id, message_id)
        st["data"]["passphrase"] = val
        _reply(chat_id, "⏳ Проверяю ключи на OKX...")
        creds = dict(st["data"])
        ok, balance = _okx_trade.get_balance(creds)
        if not ok:
            _at_onboarding.pop(chat_id, None)
            _reply(chat_id,
                   f"❌ Не получилось подключиться к OKX:\n`{balance}`\n\n"
                   f"Проверь ключи и нажми «🤖 Автотрейдинг» чтобы попробовать заново.")
            return True
        try:
            at_set_keys(user_id,
                        encrypt_secret(creds["api_key"]),
                        encrypt_secret(creds["api_secret"]),
                        encrypt_secret(creds["passphrase"]))
            at_set_balance(user_id, balance)
        except Exception as e:
            _at_onboarding.pop(chat_id, None)
            log.warning(f"at_set_keys failed for {user_id}: {e}")
            _reply(chat_id, f"❌ Ошибка сохранения ключей. Напиши {AUTOTRADE_CONTACT}.")
            return True
        st["data"] = {}
        _reply(chat_id, "✅ Ключи работают и сохранены (в зашифрованном виде).")
        _at_ask_size(chat_id, balance)
        return True

    if step in ("size_percent", "resize_percent"):
        try:
            pct = float(val.replace(",", "."))
        except ValueError:
            _reply(chat_id, "Нужно число от 1 до 10. Например `2`.")
            return True
        if not (1 <= pct <= 10):
            _reply(chat_id, "Процент должен быть от *1 до 10*. Попробуй ещё раз.")
            return True
        at_set_mode(user_id, "percent", pct)
        _at_finish_size(chat_id, user_id, step, f"{pct:.0f}% от депозита")
        return True

    if step in ("size_fixed", "resize_fixed"):
        try:
            usd = float(val.replace(",", ".").lstrip("$"))
        except ValueError:
            _reply(chat_id, "Нужно число — сумма в $. Например `5`.")
            return True
        if usd <= 0:
            _reply(chat_id, "Сумма должна быть больше 0.")
            return True
        at_set_mode(user_id, "fixed", usd)
        _at_finish_size(chat_id, user_id, step, f"${usd:.2f} на сделку")
        return True

    if step == "tp1pct":
        try:
            pct = float(val.replace(",", ".").rstrip("%"))
        except ValueError:
            _reply(chat_id, "Нужно число от 0 до 100. Например `0` или `50`.")
            return True
        if not (0 <= pct <= 100):
            _reply(chat_id, "Процент должен быть от *0 до 100*.")
            return True
        at_set_tp1_close_pct(user_id, pct)
        mode_str = st["data"].get("mode_str", "")
        _at_onboarding.pop(chat_id, None)
        _at_show_final_confirm(chat_id, mode_str, pct)
        return True

    if step == "tp1pct_change":
        try:
            pct = float(val.replace(",", ".").rstrip("%"))
        except ValueError:
            _reply(chat_id, "Нужно число от 0 до 100. Например `0` или `50`.")
            return True
        if not (0 <= pct <= 100):
            _reply(chat_id, "Процент должен быть от *0 до 100*.")
            return True
        at_set_tp1_close_pct(user_id, pct)
        _at_onboarding.pop(chat_id, None)
        label = "не закрывать (весь объём под трейлинг)" if pct <= 0 else f"{pct:.0f}% на TP1"
        _reply(chat_id, f"✅ Обновлено: {label}.")
        return True

    return False


def _at_finish_size(chat_id: int, user_id: int, step: str, mode_str: str):
    if step.startswith("resize_"):
        _at_onboarding.pop(chat_id, None)
        _reply(chat_id, f"✅ Размер сделки обновлён: *{mode_str}*.")
        return
    # First-time onboarding → ask the TP1 partial-close preference before
    # the final confirmation (default stays 0 = full position on trailing).
    _at_onboarding[chat_id] = {"step": "tp1pct", "data": {"mode_str": mode_str}}
    _reply(chat_id,
           "🎯 *Закрытие части позиции на TP1*\n\n"
           "По умолчанию стратегия держит ВСЮ позицию после TP1 и ведёт трейлинг-стопом "
           "(это провалидированный на бэктестах режим).\n\n"
           "Хочешь вместо этого закрывать часть позиции сразу на TP1? Напиши процент от "
           "*0 до 100* (0 = не закрывать, оставить всё под трейлинг; например `50` = "
           "закрыть половину на TP1, остаток — под трейлинг).")


def _at_show_final_confirm(chat_id: int, mode_str: str, tp1_pct: float):
    tp1_line = ("держим всю позицию, трейлинг с самого TP1" if tp1_pct <= 0
                else f"закрываем {tp1_pct:.0f}% на TP1, остаток — трейлинг")
    _at_reply_kb(chat_id,
                 f"⚠️ *Последний шаг*\n\n"
                 f"Бот будет *сам открывать реальные сделки* на твоём OKX:\n"
                 f"• размер: *{mode_str}*\n"
                 f"• плечо: 10x, изолированная маржа\n"
                 f"• TP1: {tp1_line}\n"
                 f"• стоп-лосс и тейки ставятся автоматически\n\n"
                 f"Торговля с плечом = риск потерять депозит. Включаем?",
                 [[{"text": "✅ Включить автотрейдинг", "callback_data": "at_confirm"},
                   {"text": "❌ Отмена", "callback_data": "at_cancel"}]])


def _at_handle_callback(cb_id: str, chat_id: int, user_id: int, data: str) -> bool:
    """Autotrade inline-button presses (user side). Returns True if handled."""
    if not data.startswith("at_"):
        return False
    u = at_get(user_id)
    if not u or not u.get("allowed"):
        _answer_callback(cb_id, "Нет доступа.")
        return True

    if data == "at_confirm":
        at_set_active(user_id, True)
        _answer_callback(cb_id, "Автотрейдинг включён!")
        _reply(chat_id, "🟢 *Автотрейдинг включён.* Сделки будут открываться автоматически по сигналам бота.")
    elif data == "at_cancel":
        _answer_callback(cb_id)
        _reply(chat_id, "Ок, автотрейдинг не включён. Нажми «🤖 Автотрейдинг» когда будешь готов.")
    elif data == "at_toggle":
        newly_active = not u.get("active")
        at_set_active(user_id, newly_active)
        _answer_callback(cb_id)
        if newly_active:
            _reply(chat_id, "🟢 Автотрейдинг снова включён.")
        else:
            _reply(chat_id, "⏸ Автотрейдинг выключен. Новые сделки открываться не будут.\n"
                            "Уже открытые позиции бот продолжит вести до закрытия.")
    elif data == "at_resize":
        bal = u.get("last_balance")
        creds = autotrader._creds_of(u)
        if creds:
            ok, live_bal = _okx_trade.get_balance(creds)
            if ok:
                bal = live_bal
                at_set_balance(user_id, bal)
        _at_onboarding[chat_id] = {"step": "", "data": {}}
        _at_ask_size(chat_id, float(bal or 0), resize=True)
        _answer_callback(cb_id)
    elif data == "at_rekey":
        _answer_callback(cb_id)
        _at_begin_keys(chat_id)
    elif data == "at_tp1pct":
        _at_onboarding[chat_id] = {"step": "tp1pct_change", "data": {}}
        _answer_callback(cb_id)
        _reply(chat_id,
               "Какой % позиции закрывать на TP1? Напиши число *0-100* "
               "(0 = не закрывать, оставить всё под трейлинг).")
    elif data == "at_mode_keep":
        at_set_mode_prompt(user_id, False)
        _answer_callback(cb_id, "Ок, оставляем как есть.")
    elif data == "at_mode_switch":
        at_set_mode_prompt(user_id, False)
        bal = float(u.get("last_balance") or 0)
        # Switch = pick the mode matching the CURRENT balance side
        _at_onboarding[chat_id] = {"step": "", "data": {}}
        _at_ask_size(chat_id, bal, resize=True)
        _answer_callback(cb_id)
    else:
        _answer_callback(cb_id)
    return True


@app.route("/webhook", methods=["POST"])
def webhook():
    data = flask_request.get_json(silent=True)
    if not data:
        return "ok", 200

    # ── Inline button press ───────────────────────────────────────────────────
    cb = data.get("callback_query")
    if cb:
        user_id    = cb.get("from", {}).get("id")
        chat_id    = cb.get("message", {}).get("chat", {}).get("id")
        message_id = cb.get("message", {}).get("message_id")
        cb_data    = cb.get("data", "")
        cb_id      = cb.get("id")
        if cb.get("message", {}).get("photo"):
            _mark_photo_panel_message(chat_id, message_id)
        if cb_data == "prayer_commandments":
            _answer_callback(cb_id)
            send_commandments(chat_id)
        elif cb_data == "evening_ritual":
            _answer_callback(cb_id)
            send_evening_ritual(chat_id)
        elif cb_data.startswith("at_"):
            _at_handle_callback(cb_id, chat_id, user_id, cb_data)
        elif _is_admin(user_id):
            _handle_admin_callback(cb_id, chat_id, message_id, cb_data, user_id)
        else:
            _answer_callback(cb_id, "Нет доступа.")
        return "ok", 200

    message = data.get("message") or data.get("channel_post")
    if not message:
        return "ok", 200

    chat_id  = message.get("chat", {}).get("id")
    user_id  = message.get("from", {}).get("id")
    from_obj = message.get("from", {})
    text_raw = message.get("text", "").strip()
    text     = text_raw.lower()

    if not chat_id:
        return "ok", 200

    # Track every user who interacts with the bot
    if user_id:
        try:
            upsert_user(
                user_id,
                username=from_obj.get("username"),
                first_name=from_obj.get("first_name"),
                last_name=from_obj.get("last_name"),
            )
        except Exception as _ue:
            log.warning(f"upsert_user failed: {_ue}")

    if not text:
        return "ok", 200

    # DM = positive chat_id (private chat with bot)
    is_dm = isinstance(chat_id, int) and chat_id > 0

    # ── Autotrade onboarding dialog (DM only, consumes key/size inputs) ──────
    if is_dm and chat_id in _at_onboarding:
        try:
            msg_id = message.get("message_id")
            if _at_handle_text(chat_id, user_id, text_raw, msg_id):
                return "ok", 200
        except Exception as e:
            log.warning(f"autotrade onboarding failed for {user_id}: {e}")
            _at_onboarding.pop(chat_id, None)
            _reply(chat_id, f"❌ Что-то пошло не так. Нажми «🤖 Автотрейдинг» и попробуй заново.")
            return "ok", 200

    # ── Pending "add admin/moderator" state — super-admin just typed an ID ───
    if is_dm and _is_super_admin(user_id) and chat_id in _pending_add_admin:
        _pending_role = _pending_add_admin.pop(chat_id)
        raw_id = text_raw.strip()
        if raw_id.lstrip("-").isdigit():
            new_id = int(raw_id)
            # Try to look up name from users table (may already have interacted)
            u_info = get_user_by_id(new_id) or {}
            add_dynamic_admin(
                new_id,
                username=u_info.get("username"),
                first_name=u_info.get("first_name"),
                added_by=user_id,
                role=_pending_role,
            )
            label = "Модератор" if _pending_role == "moderator" else "Администратор"
            _reply(chat_id,
                   f"✅ {label} `{new_id}` добавлен.\n"
                   f"Ему нужно написать /start чтобы получить панель.")
        else:
            _reply(chat_id, "❌ Не похоже на Telegram ID. Нужно число, например `123456789`.")
        return "ok", 200

    # ── Pending "add autotrade" state — admin typed a user's ID ──────────────
    if is_dm and _is_admin(user_id) and _pending_add_autotrade.pop(chat_id, False):
        raw_id = text_raw.strip()
        if raw_id.lstrip("-").isdigit():
            new_id = int(raw_id)
            at_add_allowed(new_id, added_by=user_id)
            _reply(chat_id,
                   f"✅ Пользователь `{new_id}` добавлен в автотрейдинг.\n"
                   f"Теперь у него в личке с ботом заработает кнопка 🤖 Автотрейдинг.")
            try:
                autotrader._dm(new_id,
                               "🤖 Тебе открыт доступ к автотрейдингу!\n"
                               "Нажми /start и выбери кнопку «🤖 Автотрейдинг» для подключения.")
            except Exception:
                pass
        else:
            _reply(chat_id, "❌ Не похоже на Telegram ID. Нужно число, например `123456789`.")
        return "ok", 200

    # ── Pending "users search" state — admin typed a search query ────────────
    if is_dm and _is_admin(user_id) and _pending_users_search.pop(chat_id, False):
        query = text_raw.strip().lstrip("@")
        _reply(chat_id, f"🔍 Ищу: `{query}`...")
        _back_kb = {"inline_keyboard": [[{"text": "« К панели", "callback_data": "adm_open_new"}]]}
        try:
            total = get_users_count(query)
            users = get_all_users(limit=_USERS_PER_PAGE, offset=0, query=query)
            if not users:
                _send_admin_text(chat_id, f"👥 *Поиск `{query}`*\n\nНикто не найден.", _back_kb)
            else:
                lines = [f"👥 *Поиск `{query}`* — {total} чел.\n"]
                for u in users:
                    fn    = u.get("first_name") or ""
                    name  = fn.strip() or "—"
                    uname = f"@{u['username']}" if u.get("username") else f"`{u['user_id']}`"
                    last  = datetime.fromtimestamp(u["last_seen"], tz=_riga_tz()).strftime("%d.%m %H:%M")
                    lines.append(f"• {name} {uname} `{u['user_id']}` — {last}")
                _send_admin_text(chat_id, "\n".join(lines), _back_kb)
        except Exception as e:
            _send_admin_text(chat_id, f"Ошибка поиска: {e}", _back_kb)
        return "ok", 200

    # ── Setup-history date — only after pressing "Другая дата" (armed state) ──
    # Works in DM and in the signal group (no is_dm gate); a bare date typed
    # without arming the button is intentionally ignored.
    if _is_admin(user_id) and chat_id in _pending_setups_date:
        _pending_setups_date.pop(chat_id, None)
        _send_setups_for_date(chat_id, text_raw.strip())
        return "ok", 200

    # ── Full-report start date — armed by "Своя дата" ────────────────────────
    if _is_admin(user_id) and chat_id in _pending_report_date:
        _pending_report_date.pop(chat_id, None)
        _raw = text_raw.strip()
        _ts = _parse_date_to_ts(_raw)
        if _ts is None:
            _send_admin_text(
                chat_id,
                f"❌ Не понял дату *{_raw}*.\nФормат: `26.07` или `26.07.2026`",
                _KB_ANALYTICS,
            )
        else:
            _build_and_send_report(chat_id, None, _ts, f"с {_raw}")
        return "ok", 200

    # ── Pending "manual block" state — admin typed a symbol to block ──────────
    if is_dm and _is_admin(user_id) and chat_id in _pending_block_chat:
        days = _pending_block_chat.pop(chat_id)
        raw_sym = text_raw.strip().upper().replace("-", "").replace("/", "").replace("_", "")
        # Normalize any quote (USDC display / USDT internal / bare base) → internal USDT
        for q in ("USDC", "USDT"):
            if raw_sym.endswith(q):
                raw_sym = raw_sym[:-len(q)]
                break
        raw_sym = raw_sym + "USDT"
        try:
            set_symbol_block(raw_sym, days=days, reason="Manual block by admin")
            _reply(chat_id, f"🚫 *{_disp_sym(raw_sym)}* заблокирована на {days}ч (24ч).\nОткрой 🔒 Блок тикеров чтобы проверить.")
        except Exception as e:
            _reply(chat_id, f"Ошибка: {e}")
        return "ok", 200

    # /start → постоянное меню (у админов расширенное — ЛС и группа, группа
    # для этого бота admin-only, так что гейт по is_dm тут не нужен для админки;
    # автотрейдинг всё равно DM-only, is_dm решает какую клавиатуру слать)
    if text == "/start":
        _send_persistent_menu(chat_id, is_admin=_is_admin(user_id), is_dm=is_dm)

    # 🛠 Кнопка "Админ панель" → инлайн-панель
    elif text == "🛠 админ панель":
        if _is_admin(user_id):
            _send_keyboard(chat_id, "🛠 *TUSA Admin Panel*\nВыбери раздел:")
        else:
            _reply(chat_id, "Нет доступа.")

    # 🤖 Автотрейдинг — только в личке, только для допущенных админом
    elif text == "🤖 автотрейдинг":
        if not is_dm:
            _reply(chat_id, "🤖 Автотрейдинг настраивается только в личном чате с ботом.")
        else:
            u = at_get(user_id)
            if not u or not u.get("allowed"):
                _reply(chat_id,
                       f"⛔ У тебя нет права доступа к автотрейдингу.\n"
                       f"Чтобы подключиться — напиши `{AUTOTRADE_CONTACT}`.")
            elif not keystore_ready():
                _reply(chat_id, f"⚠️ Автотрейдинг временно недоступен (техническая настройка). Напиши `{AUTOTRADE_CONTACT}`.")
            elif u.get("api_key_enc") and u.get("size_mode"):
                _at_show_menu(chat_id, u)
            else:
                _at_begin_keys(chat_id)

    # 📋 Открытые сделки
    elif text == "📋 открытые сделки":
        try:
            sigs = get_open_signals()
            if not sigs:
                _reply(chat_id, "📋 *Открытые сделки*\n\nНет активных позиций.")
            else:
                lines = ["📋 *Открытые сделки*\n"]
                for s in sigs:
                    lines.append(_format_open_signal(s))
                _reply(chat_id, "\n\n".join(lines))
        except Exception as e:
            _reply(chat_id, f"Ошибка: {e}")

    # 📈 Результаты
    elif text == "📈 результаты":
        try:
            # "Today" = since midnight Europe/Riga (calendar day, not rolling 24h)
            _now_riga    = datetime.now(_riga_tz())
            _riga_midnight = _now_riga.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
            sd  = get_stats(since_ts=_riga_midnight)
            s7  = get_stats(days=7)
            s30 = get_stats(days=30)

            def _fmt_stats(s: dict, label: str) -> str:
                # R sign and emoji
                r_val  = s["total_r"]
                r_sign = "📈" if r_val > 0 else ("📉" if r_val < 0 else "➖")
                r_str  = f"{r_val:+.2f}R"
                rpt    = s["r_per_trade"]
                rpt_str = f"{rpt:+.3f}R"

                # Direction lines (only show if data exists)
                lo = s.get("long",  {})
                sh = s.get("short", {})
                dir_lines = ""
                if lo.get("total", 0) or sh.get("total", 0):
                    def _dir(d, name):
                        if not d.get("total"):
                            return ""
                        wr  = d["win_rate"]
                        dr  = d["total_r"]
                        dr_s = f"{dr:+.2f}R"
                        icon = "🟢" if dr > 0 else ("🔴" if dr < 0 else "⚪")
                        return f"  {icon} {name}: {d['total']} сд. → {wr}% win  {dr_s}\n"
                    dir_lines = "\n" + _dir(lo, "LONG") + _dir(sh, "SHORT")

                in_work = s["open"] + s["tp1_partial_open"]
                in_work_str = ""
                if in_work:
                    tp1p = s["tp1_partial_open"]
                    in_work_str = f"  В работе сейчас: *{in_work}*"
                    if tp1p:
                        in_work_str += f" ({tp1p} уже взяли TP1)"
                    in_work_str += "\n"

                # Premium 💎 breakdown (only when at least one premium signal exists)
                prem = s.get("premium", {})
                prem_str = ""
                if prem.get("total"):
                    pr_r = prem["total_r"]
                    pr_sign = "🟢" if pr_r > 0 else ("🔴" if pr_r < 0 else "⚪")
                    if prem.get("closed"):
                        prem_str = (
                            f"\n💎 Premium: {prem['total']} сд. "
                            f"({prem['closed']} закр.) → *{prem['win_rate']}%* win  "
                            f"{pr_sign} {pr_r:+.2f}R\n"
                        )
                    else:
                        prem_str = f"\n💎 Premium: {prem['total']} сд. (ещё в работе)\n"

                return (
                    f"*{label}*\n"
                    f"  Сигналов: {s['total']}  •  Закрыто: {s['closed']}\n"
                    f"{in_work_str}"
                    f"  TP1: {s['tp1_hit']}  TP2: {s['tp2_hit']}  "
                    f"BE: {s['breakeven']}  SL: {s['sl_hit']}\n"
                    f"  Win rate: *{s['win_rate']}%*  "
                    f"•  TP1 reach: {s['tp1_rate']}%\n"
                    f"{dir_lines}"
                    f"{prem_str}"
                    f"\n{r_sign} Итого: *{r_str}*  •  {rpt_str} за сделку"
                )

            def _fmt_streak(s: dict) -> str:
                if not s.get("streak"):
                    return ""
                icons = " ".join(s["streak"])
                run   = s["current_run"]
                first = s["streak"][0]
                if first == "❌" and run >= 2:
                    run_txt = f"  ⚠️ {run} SL подряд"
                elif first in ("✅", "🏆") and run >= 2:
                    run_txt = f"  🔥 {run} в прибыли подряд"
                else:
                    run_txt = ""
                return f"🕐 *Последние:* {icons}{run_txt}\n\n"

            text_out = (
                "📈 *Результаты бота*\n\n"
                + _fmt_streak(s30)
                + _fmt_stats(sd,  "Сегодня")
                + "\n\n"
                + _fmt_stats(s7,  "За 7 дней")
                + "\n\n"
                + _fmt_stats(s30, "За 30 дней")
            )
            _reply(chat_id, text_out)
        except Exception as e:
            _reply(chat_id, f"Ошибка: {e}")

    # 📰 Новости на сегодня
    elif text == "📰 новости на сегодня":
        try:
            _reply(chat_id, _format_day_news())
        except Exception as e:
            _reply(chat_id, f"Ошибка: {e}")

    # ❓ Помощь
    elif text == "❓ помощь":
        _reply(chat_id,
               "❓ *Как читать сигналы*\n\n"
               "*Направление:*\n"
               "  📈 LONG — ожидаем рост, покупаем\n"
               "  📉 SHORT — ожидаем падение, продаём\n\n"
               "*Уровни:*\n"
               "  🎯 *TP1* — первая цель. Позиция НЕ закрывается, включается трейлинг-стоп\n"
               "  🎯 *TP2* — вторая цель. Потолок для всей позиции\n"
               "  🛑 *SL* — стоп-лосс. Выход если цена пошла против\n\n"
               "*Как работает сделка:*\n"
               "  1. Сигнал открыт — бот ждёт TP1 или SL\n"
               "  2. TP1 взят — вся позиция остаётся открытой\n"
               "  3. Дальше сделка ведётся *трейлинг-стопом*: стоп ползёт\n"
               "     за ценой (сила трейла зависит от силы движения после TP1)\n"
               "     и никогда не опускается ниже входа\n"
               "  4. Закрывается по трейлингу или при достижении TP2\n\n"
               "*Исходы сделки:*\n"
               "  ✅ *TP1\\_TRAIL* — TP1 взят, раннер закрыт трейлингом \\(прибыль\\)\n"
               "  🏆 *TP2\\_HIT* — TP1 + TP2 оба взяты \\(максимум\\)\n"
               "  ⚖️ *BE* — TP1 взят, остаток закрыт в безубыток\n"
               "  ❌ *SL* — убыток \\(до TP1 не дошло\\)\n"
               "  ⏱ *Expired* — время вышло, закрыто без результата\n\n"
               "*Win rate* — % прибыльных от закрытых.\n"
               "Норма для SMC стратегии: 38–45% при высоком R\\:R.\n\n"
               "💎 *PREMIUM* — тройное совпадение зон \\(OB \\+ FVG \\+ свип\\),\n"
               "повышенная вероятность успеха.")

    # /admin works in groups too; group privacy mode still delivers commands.
    elif text.startswith("/admin") or text in ("admin", "admin panel", "panel"):
        if _is_admin(user_id):
            _send_keyboard(chat_id, "🛠 *TUSA Admin Panel*\nВыбери раздел:")
        else:
            _reply(chat_id, "Нет доступа.")

    # /status — подробный статус
    elif text in ("/status", "/старт"):
        _reply(chat_id,
               f"🤖 *TUSA STOCKS BOT*\n"
               f"✅ Работает\n"
               f"⏱ Интервал: {SCAN_INTERVAL_MINUTES} мин\n"
               f"📊 Сигналов в кэше: {len(_signal_cache)}\n"
               f"💾 Данные: OKX\n"
               f"🧠 AI: Claude Sonnet")

    # /stats — статистика побед/поражений
    elif text in ("/stats", "/статистика"):
        try:
            _now_riga      = datetime.now(_riga_tz())
            _riga_midnight = _now_riga.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
            sd  = get_stats(since_ts=_riga_midnight)
            s7  = get_stats(days=7)
            s30 = get_stats(days=30)
            blocks = get_active_symbol_blocks()
            blocks_line = ", ".join(b["symbol"] for b in blocks[:6]) if blocks else "нет"
            _reply(chat_id,
                   f"📈 *СТАТИСТИКА*\n\n"
                   f"*Сегодня:*\n"
                   f"  Сигналов: {sd['total']}  Закрыто: {sd['closed']}\n"
                   f"  TP1: {sd['tp1_hit']} ({sd['tp1_rate']}%)  TP2: {sd['tp2_hit']}\n"
                   f"  BE: {sd['breakeven']}  SL: {sd['sl_hit']}  Expired: {sd['expired']}\n"
                   f"  Win rate: *{sd['win_rate']}%*\n\n"
                   f"*За 7 дней:*\n"
                   f"  Сигналов: {s7['total']}  Закрыто: {s7['closed']}\n"
                   f"  TP1: {s7['tp1_hit']} ({s7['tp1_rate']}%)  TP2: {s7['tp2_hit']}\n"
                   f"  BE: {s7['breakeven']}  SL: {s7['sl_hit']}  Expired: {s7['expired']}\n"
                   f"  Win rate: *{s7['win_rate']}%*\n\n"
                   f"*За 30 дней:*\n"
                   f"  Сигналов: {s30['total']}  Закрыто: {s30['closed']}\n"
                   f"  TP1: {s30['tp1_hit']} ({s30['tp1_rate']}%)  TP2: {s30['tp2_hit']}\n"
                   f"  BE: {s30['breakeven']}  SL: {s30['sl_hit']}  Expired: {s30['expired']}\n"
                   f"  Win rate: *{s30['win_rate']}%*\n\n"
                   f"🚫 Авто-блок: {blocks_line}")
        except Exception as e:
            _reply(chat_id, f"Ошибка статистики: {e}")

    return "ok", 200


def _reply(chat_id: int, text: str):
    """Send a reply to a specific chat. Falls back to plain text when Telegram
    rejects the Markdown (e.g. an unbalanced `_` from a @username)."""
    try:
        resp = _requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
            timeout=10,
        )
        if resp.status_code != 200 and "parse" in _telegram_error(resp).lower():
            _requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                json={"chat_id": chat_id, "text": text},
                timeout=10,
            )
    except Exception as e:
        log.warning(f"Reply failed: {e}")


# ── Signal deduplication cache ────────────────────────────────────────────────
# Prevents sending the same signal for the same coin repeatedly.
# Format: { "BTCUSDT": ("LONG", 1714000000.0) }
_signal_cache: dict[str, tuple[str, float]] = {}

# ── News alert deduplication cache ────────────────────────────────────────────
# Prevents re-sending the same major event alert for 6 hours.
# Format: { "event name": timestamp_sent }
_news_alert_cache: dict[str, float] = {}
_NEWS_ALERT_COOLDOWN_HOURS = 6

# ── Scan-pause state ───────────────────────────────────────────────────────────
# Send "scan paused" only on the FIRST blocked scan, "scan resumed" when it lifts.
_scan_paused: bool = False

# ── Live price cache ───────────────────────────────────────────────────────────
# Updated every 1 min by _check_open_signals() from kline close data.
# Used by _format_open_signal() so "open trades" never makes a live API call.
_last_prices: dict[str, float] = {}  # symbol → last known close price


def _is_alert_duplicate(name: str) -> bool:
    if name in _news_alert_cache:
        age_hours = (time.time() - _news_alert_cache[name]) / 3600
        if age_hours < _NEWS_ALERT_COOLDOWN_HOURS:
            return True
    return False


def _cooldowns_load() -> None:
    """Restore both cooldown caches from the DB at boot.

    They used to live only in process memory, so every redeploy silently reset
    SIGNAL_COOLDOWN_HOURS and REJECT_COOLDOWN_HOURS to zero. The open-position
    guard covered the worst case, but a symbol whose trade had already CLOSED
    could be re-signalled right after a restart, and a setup Claude had just
    rejected could be re-asked on the very next scan.
    """
    global _signal_cache, _reject_cache
    try:
        raw = get_bot_state("cooldown_signals")
        if raw:
            _signal_cache = {k: (v[0], float(v[1])) for k, v in json.loads(raw).items()}
        raw = get_bot_state("cooldown_rejects")
        if raw:
            _reject_cache = {tuple(k.split("|", 1)): (float(v[0]), float(v[1]), float(v[2]))
                             for k, v in json.loads(raw).items()}
        log.info(f"Cooldowns restored: {len(_signal_cache)} signal, {len(_reject_cache)} reject")
    except Exception as e:
        log.warning(f"Cooldown restore failed (starting empty): {e}")


def _cooldowns_save() -> None:
    """Persist both caches after every mutation — they are tiny and a scan is
    minutes apart."""
    try:
        set_bot_state("cooldown_signals", json.dumps(
            {k: [v[0], v[1]] for k, v in _signal_cache.items()}))
        set_bot_state("cooldown_rejects", json.dumps(
            {f"{k[0]}|{k[1]}": [v[0], v[1], v[2]] for k, v in _reject_cache.items()}))
    except Exception as e:
        log.warning(f"Cooldown save failed: {e}")


def _is_duplicate(symbol: str, direction: str) -> bool:
    if symbol in _signal_cache:
        cached_dir, cached_ts = _signal_cache[symbol]
        age_hours = (time.time() - cached_ts) / 3600
        if cached_dir == direction and age_hours < SIGNAL_COOLDOWN_HOURS:
            return True
    return False


def _cache_signal(symbol: str, direction: str):
    _signal_cache[symbol] = (direction, time.time())
    _cooldowns_save()


# ── Reject cooldown ────────────────────────────────────────────────────────────
# Claude said NO TRADE → the setup used to come back on EVERY 5-min scan at the
# same price, and его non-determinism eventually approved one of the retries.
# Cooldown: same symbol + direction is not re-asked while price stays within
# 1 ATR of the rejected entry, for REJECT_COOLDOWN_HOURS.
# {(symbol, direction): (rejected_price, atr, ts)}
_reject_cache: dict = {}


def _is_reject_cooled(symbol: str, direction: str, price, atr) -> bool:
    ent = _reject_cache.get((symbol, direction))
    if not ent:
        return False
    r_price, r_atr, r_ts = ent
    if (time.time() - r_ts) / 3600 >= REJECT_COOLDOWN_HOURS:
        _reject_cache.pop((symbol, direction), None)
        _cooldowns_save()
        return False
    eff_atr = float(atr or 0) or r_atr
    try:
        if eff_atr > 0 and abs(float(price) - r_price) > eff_atr:
            _reject_cache.pop((symbol, direction), None)   # left the zone
            _cooldowns_save()
            return False
    except (TypeError, ValueError):
        pass
    return True


def _cache_rejection(symbol: str, direction: str, price, atr) -> None:
    try:
        _reject_cache[(symbol, direction)] = (float(price or 0), float(atr or 0), time.time())
        _cooldowns_save()
    except (TypeError, ValueError):
        pass


def _apply_knn_overlay(setup: dict, symbol: str) -> None:
    """
    k-NN price-shape analog risk overlay (Kronos-inspired, CPU-only).

    Deep-fetches a ~1000-candle 15m series (paginated OKX) for this already-
    qualified setup, scores how often the symbol's most-similar past windows
    moved in the trade direction, and folds the result into setup['risk_mult']
    as a position-size suggestion. Size up on strong analogs (≥0.55), down on
    weak ones (<0.50). Never gates — a failure leaves the setup untouched.
    """
    setup["knn_score"] = None
    if not KNN_RISK_OVERLAY:
        return
    try:
        deep = get_klines(symbol, limit=KNN_DEEP_CANDLES)
        n = len(deep.get("close", []))
        score = knn_direction_score(
            deep, n, setup["direction"],
            shape_len=KNN_SHAPE_LEN, horizon=KNN_HORIZON, k=KNN_K,
            min_history=KNN_MIN_HISTORY, max_history=KNN_MAX_HISTORY,
        )
        setup["knn_score"] = score
        mult, tag = knn_risk_mult(
            score,
            high_score=KNN_HIGH_SCORE, high_mult=KNN_HIGH_MULT,
            low_score=KNN_LOW_SCORE, low_mult=KNN_LOW_MULT,
        )
        if mult != 1.0:
            base = float(setup.get("risk_mult", 1.0) or 1.0)
            new  = max(KNN_RISK_MIN_MULT, min(KNN_RISK_MAX_MULT, base * mult))
            setup["risk_mult"] = round(new, 4)
            if tag:
                setup.setdefault("score_tags", []).append(tag)
                # Refresh the (display-only) "Risk x.." chip rather than duplicate it.
                sigs = [s for s in setup.get("signals", []) if not str(s).startswith("Risk x")]
                sigs.append(f"Risk x{new:.2f}")
                setup["signals"] = sigs
    except Exception as e:
        log.warning(f"  kNN overlay skipped {symbol}: {e}")


def _setup_rank(setup: dict) -> tuple:
    """Rank setups before Claude so only the strongest spend LLM tokens."""
    mtf_score    = int(setup.get("mtf_score", 0) or 0)
    confirmations = sum(1 for k in ("fvg", "order_block", "liq_sweep") if setup.get(k))
    volume_score  = float(setup.get("volume_ratio", 0.0))
    zone_bonus    = 1 if setup.get("entry_source") in ("OB", "FVG") else 0
    return (mtf_score, zone_bonus, confirmations, volume_score)


# ── Open-signal monitor (updates TP/SL hits in DB) ────────────────────────────
def _slice_candles_from_open(candles: dict, after_ts: float) -> dict:
    """Return only candles that opened after after_ts to avoid counting pre-entry moves."""
    idxs = [i for i, ts in enumerate(candles.get("time", [])) if float(ts) >= float(after_ts)]
    return {k: [v[i] for i in idxs] for k, v in candles.items()}


def _post_tp1_trail_mult(direction: str, entry: float, tp1: float, tp2: float,
                         high: float, low: float, close: float) -> float:
    """Context-aware runner trail chosen from the TP1-acceptance candle (post_tp1_v2).

    Strong follow-through past TP1 (close or wick well into the TP1→TP2 leg) → trail
    WIDE so the winner can run. Weak/rejected acceptance (closed back below TP1) →
    trail TIGHT to lock the gain. Falls back to the flat TRAIL_ATR_MULT for other
    exit profiles. Validated: lets winners run while bounding give-back.
    """
    base = max(0.0, float(TRAIL_ATR_MULT))
    if str(EXIT_PROFILE).lower() != "post_tp1_v2":
        return base
    leg = abs(float(tp2) - float(tp1))
    if leg <= 0:
        return base
    if str(direction).upper() == "LONG":
        close_progress = (float(close) - float(tp1)) / leg
        wick_progress  = (float(high) - float(tp1)) / leg
        failed_close   = float(close) < float(tp1)
    else:
        close_progress = (float(tp1) - float(close)) / leg
        wick_progress  = (float(tp1) - float(low)) / leg
        failed_close   = float(close) > float(tp1)
    if close_progress >= POST_TP1_STRONG_CLOSE_PROGRESS or wick_progress >= POST_TP1_STRONG_WICK_PROGRESS:
        return max(base, float(POST_TP1_STRONG_TRAIL_ATR_MULT))
    if failed_close or close_progress <= POST_TP1_WEAK_CLOSE_PROGRESS:
        return min(base, float(POST_TP1_WEAK_TRAIL_ATR_MULT))
    return base


def _check_open_signals():
    """For each OPEN signal in DB, fetch current price and update status."""
    active_signals = get_open_signals()
    if not active_signals:
        return

    now = time.time()

    for sig in active_signals:
        try:
            opened_at  = float(sig["opened_at"])
            age_hours  = (now - opened_at) / 3600
            # 15m candles = 4 per hour; fetch enough to cover the signal's age
            candle_lim = max(8, min(220, int(age_hours * 4) + 6))
            # Judge TP/SL on the X-Perp (user's actual market) — its wicks are
            # what the user's position really experiences. Global feed = fallback.
            # include_forming=True: catch SL/TP touches within the current still-
            # forming 15m candle instead of waiting up to ~15min (avg ~7.5min) for
            # it to close — was causing SL_HIT notifications to lag the real
            # exchange-side stop fill by minutes (found 2026-07-22 in the sister
            # crypto bot, same code pattern here — ported fix).
            df_all     = get_klines_xperp(sig["symbol"], limit=candle_lim, include_forming=True) \
                         or get_klines(sig["symbol"], limit=candle_lim)

            # Cache last close price — used by "open trades" display (no extra API call)
            if df_all.get("close"):
                _last_prices[sig["symbol"]] = df_all["close"][-1]

            status    = sig["status"]
            direction = sig["direction"]
            entry     = float(sig["entry_price"])
            tp1, tp2, sl = float(sig["tp1"]), float(sig["tp2"]), float(sig["sl"])

            # Inspect only candles that opened after signal time (OPEN)
            # or after TP1 was recorded (TP1_PARTIAL) to avoid pre-entry moves
            monitor_from = opened_at if status == "OPEN" else float(sig.get("tp1_hit_at") or opened_at)
            df = _slice_candles_from_open(df_all, monitor_from)

            new_status   = None
            realized_r   = None
            runner_trail_atr_mult = None   # frozen at TP1 candle, reused next cycles
            live_trail_stop = None         # current trail level → autotrade SL amend
            exit_px      = df_all["close"][-1] if df_all.get("close") else entry

            # Runner exit after TP1 (post_tp1_v2): keep TP1_CLOSE_FRAC of the position
            # closed at TP1, trail the rest by a context-chosen ATR multiple.
            atr        = float(sig.get("atr") or 0.0)
            risk       = abs(entry - sl)
            tp1_r      = (abs(tp1 - entry) / risk) if risk > 0 else 0.0
            tp2_r      = (abs(tp2 - entry) / risk) if risk > 0 else 0.0
            tp1_close_frac = max(0.0, min(1.0, float(TP1_CLOSE_FRAC)))
            runner_frac    = 1.0 - tp1_close_frac
            use_trail  = TRAIL_RUNNER_ENABLED and atr > 0 and risk > 0
            best_price = entry   # running peak since TP1
            # Trail multiple: reuse the one frozen at the TP1 candle; legacy rows
            # without it fall back to the flat base and recompute on first bar below.
            stored_trail_mult = sig.get("runner_trail_atr_mult")
            if stored_trail_mult not in (None, ""):
                try:
                    trail_atr_mult = max(0.0, float(stored_trail_mult))
                except (TypeError, ValueError):
                    trail_atr_mult = max(0.0, float(TRAIL_ATR_MULT))
            else:
                trail_atr_mult = max(0.0, float(TRAIL_ATR_MULT))

            _conf_flags = df.get("confirmed") or []
            for i in range(len(df.get("close", []))):
                high  = float(df["high"][i])
                low   = float(df["low"][i])
                close = float(df["close"][i])
                exit_px = close
                # Close-confirmed stop: the level must be breached by a CLOSED
                # candle's close, not a wick. Unconfirmed (still-forming)
                # candles never trigger it — their "close" is just the current
                # price. Falls back to wick-touch when the feed carries no
                # confirmed flags (global-feed fallback path).
                # Without flags we cannot know which bars are closed, and the
                # series always ENDS with the forming one. Defaulting to
                # "confirmed" there let a still-open bar's close — which is just
                # the current price — trigger the stop, so on the fallback feed
                # the rule silently became touch-the-level while backtest.py
                # still required a closed bar. Only get_klines_xperp() builds
                # these flags; the global-feed fallback next to it does not.
                if i < len(_conf_flags):
                    _is_confirmed = bool(_conf_flags[i])
                else:
                    _is_confirmed = i < len(df["close"]) - 1
                if STOP_CLOSE_CONFIRM:
                    _sl_breached = _is_confirmed and (
                        close <= sl if direction == "LONG" else close >= sl
                    )
                else:
                    _sl_breached = low <= sl if direction == "LONG" else high >= sl
                # Exit price is the real close when confirmed on a close —
                # booking it at the SL level would understate the loss. Matters
                # more here than on crypto: a gap can close far past the level.
                _sl_exit_px = close if STOP_CLOSE_CONFIRM else sl
                _sl_r = -1.0
                if STOP_CLOSE_CONFIRM and risk > 0:
                    _sl_r = round(
                        ((_sl_exit_px - entry) if direction == "LONG" else (entry - _sl_exit_px)) / risk,
                        4,
                    )

                if status == "OPEN":
                    if direction == "LONG":
                        if _sl_breached:          realized_r = _sl_r; new_status, exit_px = "SL_HIT", _sl_exit_px;  break
                        if high >= tp2:
                            realized_r = round(tp1_close_frac * tp1_r + runner_frac * tp2_r, 4)
                            new_status, exit_px = "TP2_HIT", tp2; break
                        if high >= tp1:
                            runner_trail_atr_mult = _post_tp1_trail_mult(direction, entry, tp1, tp2, high, low, close)
                            new_status, exit_px = "TP1_PARTIAL", tp1; break
                    else:
                        if _sl_breached:          realized_r = _sl_r; new_status, exit_px = "SL_HIT", _sl_exit_px;  break
                        if low <= tp2:
                            realized_r = round(tp1_close_frac * tp1_r + runner_frac * tp2_r, 4)
                            new_status, exit_px = "TP2_HIT", tp2; break
                        if low <= tp1:
                            runner_trail_atr_mult = _post_tp1_trail_mult(direction, entry, tp1, tp2, high, low, close)
                            new_status, exit_px = "TP1_PARTIAL", tp1; break

                elif status == "TP1_PARTIAL":
                    if use_trail:
                        # Recompute the trail on the first bar for legacy rows missing it.
                        if i == 0 and stored_trail_mult in (None, ""):
                            trail_atr_mult = _post_tp1_trail_mult(direction, entry, tp1, tp2, high, low, close)
                        if direction == "LONG":
                            best_price = max(best_price, high)
                            trail_stop = max(entry, best_price - atr * trail_atr_mult)
                            live_trail_stop = trail_stop
                            if low <= trail_stop:
                                runner_r   = max(0.0, (trail_stop - entry) / risk)
                                realized_r = round(tp1_close_frac * tp1_r + runner_frac * runner_r, 4)
                                new_status, exit_px = "TP1_TRAIL", trail_stop; break
                            if high >= tp2:
                                realized_r = round(tp1_close_frac * tp1_r + runner_frac * tp2_r, 4)
                                new_status, exit_px = "TP2_HIT", tp2; break
                        else:
                            best_price = min(best_price, low)
                            trail_stop = min(entry, best_price + atr * trail_atr_mult)
                            live_trail_stop = trail_stop
                            if high >= trail_stop:
                                runner_r   = max(0.0, (entry - trail_stop) / risk)
                                realized_r = round(tp1_close_frac * tp1_r + runner_frac * runner_r, 4)
                                new_status, exit_px = "TP1_TRAIL", trail_stop; break
                            if low <= tp2:
                                realized_r = round(tp1_close_frac * tp1_r + runner_frac * tp2_r, 4)
                                new_status, exit_px = "TP2_HIT", tp2; break
                    else:
                        # Legacy: SL moved to breakeven, fixed TP2 (no stored ATR).
                        if direction == "LONG":
                            if low <= entry:
                                realized_r = round(tp1_close_frac * tp1_r, 4)
                                new_status, exit_px = "BREAKEVEN", entry; break
                            if high >= tp2:
                                realized_r = round(tp1_close_frac * tp1_r + runner_frac * tp2_r, 4)
                                new_status, exit_px = "TP2_HIT", tp2; break
                        else:
                            if high >= entry:
                                realized_r = round(tp1_close_frac * tp1_r, 4)
                                new_status, exit_px = "BREAKEVEN", entry; break
                            if low <= tp2:
                                realized_r = round(tp1_close_frac * tp1_r + runner_frac * tp2_r, 4)
                                new_status, exit_px = "TP2_HIT", tp2; break

            # Expire on MARKET-OPEN hours, not wall-clock: a weekend must not
            # age out a Friday signal whose underlying cannot move. Measured on
            # the 2022-2026 deep set, wall-clock expiry killed 25.7% of Friday
            # entries vs 1.7-5.3% Mon-Thu (62% of ALL expiries were Fridays).
            # SIGNAL_EXPIRY_MAX_DAYS is a hard calendar ceiling on top, so a
            # Friday entry cannot ride ~12 calendar days into its own earnings.
            _session_age = session_hours_between(opened_at, now)
            if new_status is None and (
                _session_age > SIGNAL_EXPIRY_HOURS
                or (SIGNAL_EXPIRY_MAX_DAYS > 0 and age_hours > SIGNAL_EXPIRY_MAX_DAYS * 24)
            ):
                new_status = "TP1_EXPIRED" if status == "TP1_PARTIAL" else "EXPIRED"
                realized_r = round(tp1_close_frac * tp1_r, 4) if status == "TP1_PARTIAL" else 0.0

            if new_status:
                update_signal_status(sig["id"], new_status, exit_px, realized_r=realized_r,
                                     runner_trail_atr_mult=runner_trail_atr_mult)
                # SL-wick diagnostic: real reversal (deep global feed also breached
                # SL) or thin-X-Perp execution noise (only X-Perp wicked to it)?
                if new_status == "SL_HIT":
                    try:
                        g = get_klines(sig["symbol"], limit=candle_lim)  # global feed
                        gd = _slice_candles_from_open(g, monitor_from)
                        if direction == "LONG":
                            g_breached = any(float(x) <= sl for x in gd.get("low", []))
                        else:
                            g_breached = any(float(x) >= sl for x in gd.get("high", []))
                        set_sl_xperp_only(sig["id"], 0 if g_breached else 1)
                    except Exception as _we:
                        log.debug(f"  SL-wick check failed #{sig['id']}: {_we}")
                log.info(f"  Signal #{sig['id']} {sig['symbol']} → {new_status}")
                try:
                    send_signal_update(sig, new_status, exit_px)
                except Exception as _e:
                    log.warning(f"  Update notification failed #{sig['id']}: {_e}")
                # Autotrade: mirror the transition on the exchange (close /
                # start-trailing) for every user holding this signal live.
                try:
                    autotrader.mirror_transition(sig, new_status, exit_px)
                except Exception as _ae:
                    log.warning(f"  Autotrade mirror failed #{sig['id']}: {_ae}")
            elif status == "TP1_PARTIAL" and live_trail_stop is not None:
                # No transition, trail may have moved — sync users' exchange SL.
                try:
                    autotrader.update_trailing(sig, live_trail_stop)
                except Exception as _ae:
                    log.warning(f"  Autotrade trail sync failed #{sig['id']}: {_ae}")

        except Exception as e:
            log.warning(f"  Could not check signal #{sig['id']}: {e}")


# ── Shadow-outcome tracker (rejected + sent setups) ───────────────────────────
def _simulate_setup_outcome(direction: str, entry: float, tp1: float, tp2: float,
                            sl: float, highs: list, lows: list,
                            closes: list = None, require_fill: bool = True,
                            wait_bars: int = 4, atr: float = 0.0) -> tuple:
    """Replay a setup's bracket over forward candles (same order as the validated
    backtest: SL → TP2 → TP1 each bar). Returns (outcome|None, reached_tp1,
    reached_tp2). outcome is None while still live (no TP1/SL hit yet).

    After TP1 the stop moves to breakeven (mirrors live TP1=50%→SL-to-BE); the
    runner either reaches TP2 or exits flat at BE. We only need the categorical
    result (SL / TP1 / TP2) for the learning signal, not the runner's exact R.

    STOP_CLOSE_CONFIRM must mirror the live position monitor and backtest.py
    here too — this is a THIRD independent place the stop rule is implemented
    (setup_log / "Точность ИИ" / the mirror experiment). Currently harmless
    since STOP_CLOSE_CONFIRM defaults 0 on this bot (unmeasured — see
    portfolio_sim findings), but if it is ever enabled without this fix, a
    setup that is both sent (real position) and logged gets two different
    outcomes for the same trade.
    """
    tp1_reached = False
    risk = abs(entry - sl)
    if risk <= 0:
        return None, 0, 0
    # The bracket used to start at `entry` on bar 0 whether or not price ever
    # traded there — the same fantasy fill removed from backtest.py, and never
    # ported here. It inflates every shadow number: unsent setups resolved TP2
    # at 51.4% against 6.9% for the ones actually sent.
    start = 0
    if require_fill:
        if min(len(highs), len(lows)) < int(wait_bars) + 1:
            return None, 0, 0
        start = -1
        for i in range(min(int(wait_bars) + 1, len(highs), len(lows))):
            touched = (lows[i] <= entry) if direction == "LONG" else (highs[i] >= entry)
            if touched:
                start = i
                break
        if start < 0:
            return "NO_FILL", 0, 0
        highs, lows = highs[start:], lows[start:]
        if closes is not None:
            closes = closes[start:]

    use_close = STOP_CLOSE_CONFIRM and closes is not None

    # ATR for the post-TP1 trail, from the same candles.
    _rng = []
    for i in range(min(len(highs), len(lows))):
        r = highs[i] - lows[i]
        if closes is not None and i > 0 and i < len(closes):
            r = max(r, abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
        _rng.append(r)
    # The live monitor and backtest both trail on the ATR the strategy measured
    # at signal time. Estimating one from FORWARD candles instead put this
    # tracker 13% out of step with the model on synthetic series — always in the
    # same direction, TP2 in the model against TP1 here. Use the stored value
    # whenever setup_log has it; the estimate is only a fallback for rows
    # written before that column existed.
    _atr = float(atr) if atr and atr > 0 else (
        (sum(_rng[:14]) / len(_rng[:14])) if _rng else 0.0)
    _tmult = max(0.0, float(TRAIL_ATR_MULT))
    _peak = entry

    for idx, (h, l) in enumerate(zip(highs, lows)):
        if not tp1_reached:
            if direction == "LONG":
                _sl_hit = (closes[idx] <= sl) if use_close else (l <= sl)
                if _sl_hit:   return "SL", 0, 0
                if h >= tp2:  return "TP2", 1, 1
                if h >= tp1:  tp1_reached = True
                # The peak must run from the FILL, not from the TP1 bar. Starting
                # it at entry when TP1 lands leaves the trail sitting near entry,
                # so the runner survives pullbacks the real trail would have
                # closed — 24 of 400 synthetic cases resolved TP2 here against
                # TP1 in the model, all in that direction.
                _peak = max(_peak, h)
            else:
                _sl_hit = (closes[idx] >= sl) if use_close else (h >= sl)
                if _sl_hit:   return "SL", 0, 0
                if l <= tp2:  return "TP2", 1, 1
                if l <= tp1:  tp1_reached = True
                _peak = min(_peak, l)
        else:
            # Post-TP1 the live runner is TRAILED, not carried to TP2 or
            # breakeven; TP1_CLOSE_FRAC has been 0 since this exit profile
            # shipped. The trail is anchored to PRIOR bars only, mirroring
            # backtest.py's BT_TRAIL_LAG.
            _trail = (max(entry, _peak - _atr * _tmult) if direction == "LONG"
                      else min(entry, _peak + _atr * _tmult))
            if direction == "LONG":
                if h >= tp2:    return "TP2", 1, 1
                if l <= _trail: return "TP1", 1, 0
                _peak = max(_peak, h)
            else:
                if l <= tp2:    return "TP2", 1, 1
                if h >= _trail: return "TP1", 1, 0
                _peak = min(_peak, l)
    # No terminal hit within the candles seen so far.
    return (None, 1, 0) if tp1_reached else (None, 0, 0)


def _track_setup_outcomes():
    """Resolve shadow outcomes for logged setups (sent AND rejected) so the bot
    can later learn whether Claude's verdicts matched reality. Runs every 15 min.
    """
    try:
        max_age = SIGNAL_EXPIRY_HOURS * 3600
        pending = get_unresolved_setups(max_age_sec=max_age, limit=80)
        if not pending:
            return
        # Fetch each symbol's candles once per cycle.
        by_symbol: dict = {}
        for s in pending:
            by_symbol.setdefault(s["symbol"], []).append(s)

        resolved = 0
        for symbol, rows in by_symbol.items():
            try:
                df_all = get_klines(symbol, limit=220)
            except Exception as e:
                log.debug(f"  shadow: klines failed {symbol}: {e}")
                continue
            if not df_all.get("time"):
                continue
            for s in rows:
                df = _slice_candles_from_open(df_all, float(s["ts"]))
                highs  = [float(x) for x in df.get("high", [])]
                lows   = [float(x) for x in df.get("low", [])]
                closes = [float(x) for x in df.get("close", [])]
                outcome, r1, r2 = _simulate_setup_outcome(
                    s["direction"], float(s["entry_price"]),
                    float(s["tp1"]), float(s["tp2"]), float(s["sl"]),
                    highs, lows, closes,
                    atr=float(s["atr"] or 0.0) if "atr" in s.keys() else 0.0,
                )
                age_h = (time.time() - float(s["ts"])) / 3600
                if outcome is None:
                    # No terminal hit yet — finalise only once the window expired.
                    if age_h > SIGNAL_EXPIRY_HOURS:
                        outcome = "TP1" if r1 else "EXPIRED"
                    else:
                        continue   # still live, re-check next cycle
                mark_setup_resolved(s["id"], outcome, r1, r2)
                resolved += 1
        if resolved:
            log.info(f"Shadow tracker: resolved {resolved} setup outcome(s)")
    except Exception as e:
        log.warning(f"Shadow tracker failed: {e}")


# ── Main scanning function ────────────────────────────────────────────────────
_OI_MIN_DELTA_PCT = 0.3  # ignore OI moves below this (noise floor)


def _attach_oi(setup: dict) -> None:
    """Shadow feature: tag a setup with its Open-Interest regime. NO trade impact —
    written to setup_log so we can later correlate OI with reached_tp1.

    OI is paired with the setup's price direction (the BOS):
      rising OI  = new money behind the break → CONFIRMS the setup
      falling OI = positions unwinding (short-cover / long-liq) → WARNS
    Regime label differs by side for readability; oi_confirms is the learning bit.
    """
    series = get_open_interest(setup["symbol"])
    if len(series) < 2 or not series[0]:
        return
    delta = (series[-1] - series[0]) / series[0] * 100.0
    setup["oi_delta_pct"] = round(delta, 3)
    direction = setup.get("direction", "")
    if abs(delta) < _OI_MIN_DELTA_PCT:
        setup["oi_regime"], setup["oi_confirms"] = "flat", 0
        return
    if direction == "LONG":
        regime = "real_up" if delta > 0 else "short_cover"
    else:  # SHORT
        regime = "real_down" if delta > 0 else "long_liq"
    setup["oi_regime"]   = regime
    setup["oi_confirms"] = 1 if regime in ("real_up", "real_down") else 0


def run_scan():
    now_utc = datetime.now(timezone.utc)

    # TP/SL monitoring moved to dedicated 1-min job (_monitor_open_signals)

    # US market session gate — replaces the crypto bot's weekend/UTC-hours
    # filters (weekends, holidays, half-days and DST all handled inside).
    # Admin toggle "вне сессии" overrides for 24/7 scanning.
    from src.market_hours import is_market_open
    if not is_market_open() and not _off_session_enabled() and not _in_extended_window():
        log.info("US market closed — new-signal scan skipped (monitoring continues)")
        return

    # Daily kill-switch: N consecutive SL among today's closed signals →
    # stop generating new signals until the next Riga day. Monitoring of
    # already-open positions continues (separate 1-min job).
    if KILL_SWITCH_SL_STREAK > 0:
        try:
            _now_riga  = datetime.now(_riga_tz())
            _day_start = _now_riga.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
            _streak    = get_today_sl_streak(_day_start)
            if _streak >= KILL_SWITCH_SL_STREAK:
                _ks_key = f"kill_switch_notified_{_now_riga.strftime('%Y%m%d')}"
                if not get_bot_state(_ks_key):
                    set_bot_state(_ks_key, str(_streak))
                    send_status(
                        f"🛑 *Дневной стоп: {_streak} стопа подряд.*\n"
                        f"Новые сигналы приостановлены до завтра — рынок сегодня "
                        f"рубит стопы, пересидим. Открытые позиции ведутся как обычно."
                    )
                log.warning(f"Kill-switch: {_streak} consecutive SL today — scan skipped")
                return
        except Exception as e:
            log.warning(f"Kill-switch check failed (scan continues): {e}")

    log.info("=== Scan started (SMC mode) ===")

    try:
        # Step 0a: Global macro news (Groq free tier)
        news = get_market_news()
        log.info(
            f"News: {news['sentiment']} — {news['summary']} "
            f"({news['headline_count']} headlines)"
        )
        if news["pause"]:
            # Log only — no Telegram notification (user requested silence on pause/resume)
            trigger = news.get("trigger", "") or news.get("summary", "")
            log.warning(f"News agent: PAUSE — extreme market event, skipping scan ({trigger})")
            return

        # Step 0a-2: Detect and broadcast high-impact macro events
        try:
            headlines = fetch_recent_headlines()
            events = detect_major_events(headlines)
            for ev in events:
                if not _is_alert_duplicate(ev["name"]):
                    if send_news_alert(ev):
                        _news_alert_cache[ev["name"]] = time.time()
                        log.info(f"News alert sent: {ev['name']} ({ev['direction']} {ev['level']}x)")
        except Exception as e:
            log.warning(f"Major event check failed: {e}")

        # Step 0b: BTC 1h change for correlation filter + 1D for Claude macro context
        btc_change = get_btc_change_1h()
        btc_change_1d = get_btc_change_1d()
        log.info(f"BTC change: {btc_change:+.2f}% 1h, {btc_change_1d:+.2f}% 1D")

        # Auto-block symbols with consistently bad stats (local DB, no API calls)
        new_blocks = auto_block_bad_symbols()
        for b in new_blocks:
            log.info(f"  Auto-blocked: {b['reason']}")

        # Step 1: top 25 liquid coins (quality filtered)
        coins = get_top_coins()
        before_blocks = len(coins)
        coins = [s for s in coins if not is_symbol_auto_blocked(s)]
        if len(coins) != before_blocks:
            log.info(f"Auto-block: skipped {before_blocks - len(coins)} blocked symbol(s)")

        # OKX EU tradability gate: keep only coins with a live X-Perp contract —
        # a signal on a coin the user can't open (no X-Perp) is useless.
        xperp_bases = set(get_xperp_instruments())
        if xperp_bases:
            before_xp = len(coins)
            coins = [s for s in coins if s[:-len("USDT")] in xperp_bases]
            if len(coins) != before_xp:
                log.info(f"X-Perp gate: {before_xp} → {len(coins)} coins tradable on OKX EU")

        # Earnings blackout: no NEW signals on a ticker reporting today/tomorrow —
        # the report gaps the stock 5-10% overnight, no 15m SMC setup prices that.
        # Fail-open on calendar outage; open positions untouched (monitor handles).
        try:
            from src.earnings_calendar import is_earnings_blackout
            _kept = []
            for s in coins:
                blackout, why = is_earnings_blackout(s)
                if blackout:
                    log.info(f"Earnings blackout: {s} skipped — {why}")
                else:
                    _kept.append(s)
            coins = _kept
        except Exception as e:
            log.warning(f"Earnings blackout check failed (trading normally): {e}")

        mode = "whitelist" if ALLOWED_SYMBOLS else "auto top-volume"
        log.info(f"Fetched {len(coins)} coins ({mode})")

        setups = []
        smc_diag = {}  # funnel diagnostics: how many coins reached scoring + best score

        # Step 2: SMC filter — BOS + confirmation + 1h/4h trend + BTC correlation
        for symbol in coins:
            try:
                df_15m = get_klines(symbol)
                df_1h  = get_klines_1h(symbol)
                df_4h  = get_klines_4h(symbol)
                df_1d  = get_klines_1d(symbol)
                setup  = analyze_coin_smc(df_15m, df_1h, symbol, df_4h, btc_change, df_1d, diag=smc_diag)
                if setup:
                    _apply_knn_overlay(setup, symbol)
                    log.info(
                        f"  SMC setup: {symbol:12s}  {setup['direction']}  "
                        f"1d={setup.get('trend_1d','?')} 4h={setup['trend_4h']} 1h={setup['trend_1h']}  "
                        f"signals={setup['signals']}"
                    )
                    setups.append(setup)
                time.sleep(0.2)  # 2 API calls per coin — small delay
            except Exception as e:
                log.warning(f"  Skip {symbol}: {e}")

        log.info(f"SMC filter: {len(setups)} setups from {len(coins)} coins")
        # Funnel: how many coins survived ALL structural gates to reach scoring,
        # and the best score seen — distinguishes "strict gate" (close miss) from
        # "no structure today" (0 reach). best vs MTF_MIN_SCORE shows the gap.
        from config import MTF_MIN_SCORE as _MTF_MIN
        log.info(
            f"  SMC funnel: {smc_diag.get('reached_score', 0)}/{len(coins)} reached scoring · "
            f"best score {smc_diag.get('best_score', 0)}/{_MTF_MIN} needed "
            f"({smc_diag.get('best_symbol', '-')}) · "
            f"{smc_diag.get('score_fail', 0)} missed score gate"
        )
        _last_scan_stats["coins"]  = len(coins)
        _last_scan_stats["setups"] = len(setups)
        _last_scan_stats["ts"]     = time.time()

        # Step 3: remove duplicates
        # Also block any symbol that already has an OPEN or TP1_PARTIAL position in DB.
        # This prevents re-signalling a coin we're already trading (e.g. BNB hit TP1,
        # still waiting for TP2 — bot must not open a second trade on BNB).
        _active_now = {sig["symbol"] for sig in get_open_signals()}
        def _blocked(s):
            sym = s["symbol"]
            if sym in _active_now:
                log.info(f"  Skip {sym} — already have open position")
                return True
            if _is_reject_cooled(sym, s["direction"], s.get("current_price"), s.get("atr")):
                log.info(f"  Skip {sym} {s['direction']} — Claude rejected recently, price still in zone")
                return True
            return _is_duplicate(sym, s["direction"])
        fresh = [s for s in setups if not _blocked(s)]
        log.info(f"After dedup: {len(fresh)} fresh setups")
        _last_scan_stats["fresh"] = len(fresh)

        # Step 3b: news + funding enrichment
        enriched = []
        for s in fresh:
            # News check — block on bad news
            news = check_news_sentiment(s["symbol"])
            if not news["safe"]:
                log.info(f"  Skip {s['symbol']} — {news['reason']}")
                continue
            # Funding rate — fetch + hard filter crowded positions
            fr = get_funding_rate(s["symbol"])
            s["funding_rate"] = fr
            if fr is not None:
                if s["direction"] == "LONG"  and fr >  0.0005:   # >+0.05% = crowded longs
                    log.info(f"  Skip {s['symbol']} LONG — funding {fr*100:+.3f}% crowded")
                    continue
                if s["direction"] == "SHORT" and fr < -0.0005:   # <-0.05% = crowded shorts
                    log.info(f"  Skip {s['symbol']} SHORT — funding {fr*100:+.3f}% crowded")
                    continue
            enriched.append(s)

        # Sort by quality score, keep only top MAX_SETUPS_TO_CLAUDE (saves tokens)
        enriched.sort(key=_setup_rank, reverse=True)
        if len(enriched) > MAX_SETUPS_TO_CLAUDE:
            log.info(f"Token saver: top {MAX_SETUPS_TO_CLAUDE} of {len(enriched)} → Claude")
            enriched = enriched[:MAX_SETUPS_TO_CLAUDE]

        log.info(f"After news/funding/ranking: {len(enriched)} setups → sending to Claude")
        _last_scan_stats["enriched"] = len(enriched)

        # OI shadow feature — tag only the ≤7 setups that go to Claude (cheap).
        # Decision is NOT affected; we log oi_regime/oi_confirms to learn its edge.
        for _s in enriched:
            try:
                _attach_oi(_s)
            except Exception as _e:
                log.debug(f"  OI attach failed {_s.get('symbol','?')}: {_e}")

        if not enriched:
            log.info("=== Scan complete — 0 signal(s) sent ===\n")
            return

        # Step 4: LIGHT tier — ONE batch call to Claude (cached rules + news + BTC macro)
        claude_ctx = dict(news or {})
        claude_ctx["btc_1h"] = btc_change
        claude_ctx["btc_1d"] = btc_change_1d
        try:
            analyses = analyze_batch_with_claude(enriched, news_context=claude_ctx)
        except Exception as e:
            log.error(f"Claude LIGHT batch call failed: {e}")
            return

        # Step 4b: HEAVY tier — Sonnet second opinion on the strongest survivors.
        # Only setups the LIGHT gate approved (LONG/SHORT, not LOW) with a high
        # mtf_score qualify; capped per scan to protect the budget. Coin memory
        # (recent outcomes) is injected so Sonnet learns from this symbol's past.
        heavy_done = 0
        for analysis in analyses:
            if heavy_done >= CLAUDE_HEAVY_MAX_PER_SCAN:
                break
            decision = analysis.get("decision", "NO TRADE")
            conf     = analysis.get("confidence", "LOW").upper()
            score    = int(analysis.get("mtf_score", 0) or 0)
            if decision in ("LONG", "SHORT") and conf != "LOW" and score >= CLAUDE_HEAVY_MIN_SCORE:
                try:
                    history = get_recent_outcomes(analysis["symbol"], limit=CLAUDE_MEMORY_LIMIT)
                    heavy = analyze_heavy(analysis, news_context=claude_ctx, history=history)
                    for k in ("decision", "confidence", "risk_score", "trend_strength", "reason", "counter"):
                        if k in heavy:
                            analysis[k] = heavy[k]
                    heavy_done += 1
                    log.info(
                        f"  HEAVY: {analysis['symbol']} → {analysis['decision']} "
                        f"({analysis.get('confidence','?')}) risk={analysis.get('risk_score','?')} "
                        f"— {analysis.get('reason','')}"
                    )
                except Exception as e:
                    log.warning(f"  HEAVY check failed {analysis.get('symbol','?')}: {e}")

        # Book state at judgment time — how many positions in each direction were
        # already open when Claude saw these setups. Snapshotted once here so the
        # logged value is what Claude actually reasoned against, and reused as the
        # starting count for the correlation cap in the send loop below.
        _dir_open = {"LONG": 0, "SHORT": 0}
        for _s in get_open_signals():
            _d = str(_s.get("direction", "")).upper()
            if _d in _dir_open:
                _dir_open[_d] += 1

        # Log all Claude-evaluated setups (approved and rejected) for admin history
        for _a in analyses:
            _a["open_same_dir"] = _dir_open.get(
                str(_a.get("direction", "")).upper(), 0
            )
            try:
                _a["_setup_log_id"] = log_setup_candidate(_a)
            except Exception as _e:
                log.debug(f"setup_log insert failed: {_e}")
            # Reject cooldown: remember NO TRADEs so the next scans don't
            # re-ask the same setup at the same price ("ask until yes").
            try:
                if _a.get("decision", "NO TRADE") == "NO TRADE":
                    _cache_rejection(_a.get("symbol", ""), _a.get("direction", ""),
                                     _a.get("current_price"), _a.get("atr"))
            except Exception:
                pass

        # Upcoming high-impact macro events (CPI/FOMC/NFP) — warn on signals
        event_warning = ""
        try:
            events = get_upcoming_high_impact_events(EVENT_WARN_HOURS)
            if events:
                ev = events[0]
                cc = f"{ev['country']} " if ev.get("country") else ""
                event_warning = (
                    f"{cc}{ev['title']} через {ev['hours_until']}ч — "
                    f"высокая волатильность, осторожно"
                )
        except Exception as e:
            log.warning(f"Calendar check failed: {e}")

        # Step 5: Send signals to Telegram (hard cap: max 3 per scan)
        MAX_SIGNALS_PER_SCAN = 3
        sent_count = 0
        for analysis in analyses:
            try:
                # Attach news context to each analysis for Telegram message
                analysis["news_sentiment"] = news.get("sentiment", "")
                analysis["news_summary"]   = news.get("summary", "")
                analysis["event_warning"]  = event_warning

                log.info(
                    f"  Claude: {analysis['symbol']} → {analysis['decision']} "
                    f"({analysis.get('confidence','?')}) — {analysis.get('reason','')}"
                )
                decision   = analysis.get("decision", "NO TRADE")
                direction  = analysis.get("direction")
                confidence = analysis.get("confidence", "LOW").upper()

                # SHADOW MODE (CLAUDE_GATE_ENABLED=0): Claude still runs, is
                # still scored and logged, but does not withhold anything — the
                # rules filter alone decides. His verdict can then be scored
                # against REAL fills instead of against shadow-simulated ones,
                # which is the only way to settle whether he adds or subtracts.
                _gate = _claude_gate_enabled()

                # Guard: Claude must confirm setup direction, not flip it.
                # Kept even in shadow mode: a flipped side is not a weaker
                # opinion, it is a contradiction, and trading it would test
                # nothing about the filter.
                if decision in ("LONG", "SHORT") and decision != direction:
                    log.warning(f"  Skip {analysis['symbol']} — Claude flipped side blocked")
                    continue

                if confidence == "LOW":
                    if _gate:
                        log.info(f"  Skip {analysis['symbol']} — LOW confidence")
                        continue
                    log.info(f"  [shadow] {analysis['symbol']} — LOW confidence, trading anyway")

                if decision == "NO TRADE" and not _gate:
                    log.info(f"  [shadow] {analysis['symbol']} — Claude said NO TRADE, trading anyway")

                if decision != "NO TRADE" or not _gate:
                    # Both caps withhold a setup Claude APPROVED. Tag why, so it
                    # is not later counted as one of Claude's rejections and can
                    # be judged on its own outcome (get_cap_impact_stats).
                    if sent_count >= MAX_SIGNALS_PER_SCAN:
                        log.info(f"  Skip {analysis['symbol']} — scan cap {MAX_SIGNALS_PER_SCAN} reached")
                        try:
                            mark_setup_blocked(analysis.get("_setup_log_id"), "scan_cap")
                        except Exception:
                            pass
                        continue

                    if (MAX_SAME_DIRECTION_POSITIONS > 0
                            and _dir_open.get(decision, 0) >= MAX_SAME_DIRECTION_POSITIONS):
                        log.info(
                            f"  Skip {analysis['symbol']} — {_dir_open[decision]} {decision} "
                            f"already open, correlation cap {MAX_SAME_DIRECTION_POSITIONS} reached"
                        )
                        try:
                            mark_setup_blocked(analysis.get("_setup_log_id"), "dir_cap")
                        except Exception:
                            pass
                        continue

                    # Snapshot the live X-Perp price (the instrument the user
                    # actually trades on OKX EU) at publish moment — signal
                    # levels re-anchor to it so entry/TP/SL match the user's
                    # chart. Falls back to the analysis feed price if X-Perp
                    # ticker is unavailable.
                    #
                    # Drift guard is capped at LIVE_PRICE_MAX_DRIFT_PCT (= our
                    # own RISK_MAX_PCT, 1.5%) — NOT a flat 3% crypto leftover.
                    # A stock's SL band is 0.4-1.5%; a 3% "sanity" gap is
                    # 2-7x that, wide enough for a thin overnight/open-gap
                    # tick to already sit past TP1/TP2 and still get accepted
                    # as the anchor — the setup's structural zone is stale by
                    # then, not a valid entry (this is the bug behind signals
                    # that "opened" already resolved as an instant win/loss).
                    stale_move = False
                    try:
                        _xp_px = get_xperp_price(analysis["symbol"])
                        live_px = _xp_px or get_current_price(analysis["symbol"])
                        if live_px and live_px > 0:
                            # Structure is measured on the deep GLOBAL swap feed
                            # but the position is opened/monitored on the X-Perp,
                            # which trades at a persistent discount (measured
                            # 2026-07-31: -0.17%..-0.21%). Rescale global-derived
                            # levels into X-Perp space so entry and levels share
                            # one price space. NOTE: the basis is noisier here
                            # than on crypto (stdev up to 0.18pp vs 0.01-0.04pp),
                            # so the +/-1% sanity bound matters more.
                            _basis = 1.0
                            if _xp_px:
                                try:
                                    _gp = get_current_price(analysis["symbol"])
                                    if _gp and _gp > 0:
                                        _b = _xp_px / _gp
                                        if 0.99 <= _b <= 1.01:
                                            _basis = _b
                                        else:
                                            log.warning(
                                                f"  {analysis['symbol']}: implausible X-Perp basis "
                                                f"{(_b-1)*100:+.2f}% — levels left unscaled"
                                            )
                                except Exception as _be:
                                    log.debug(f"  basis calc failed {analysis['symbol']}: {_be}")
                            if _basis != 1.0:
                                for _k in ("recent_high", "recent_low", "tp1_level",
                                           "tp2_level", "entry_low", "entry_high", "atr"):
                                    try:
                                        _v = analysis.get(_k)
                                        if _v:
                                            analysis[_k] = float(_v) * _basis
                                    except (TypeError, ValueError):
                                        continue
                                log.info(f"  Levels rescaled to X-Perp space "
                                         f"(basis {(_basis-1)*100:+.3f}%)")
                            zone_px = float(analysis.get("current_price") or live_px)
                            drift   = abs(live_px - zone_px) / zone_px if zone_px else 0
                            if drift <= LIVE_PRICE_MAX_DRIFT_PCT:
                                analysis["zone_entry_price"] = zone_px   # keep zone for reference
                                analysis["current_price"]    = round(live_px, 8)
                                analysis["market_price"]     = round(live_px, 8)
                                log.info(
                                    f"  Entry price updated to live: {live_px} "
                                    f"(zone was {zone_px}, drift {drift*100:.2f}%)"
                                )
                            else:
                                log.warning(
                                    f"  Live price {live_px} vs zone {zone_px}: drift "
                                    f"{drift*100:.2f}% > {LIVE_PRICE_MAX_DRIFT_PCT*100:.1f}% "
                                    f"— setup stale, skipping signal"
                                )
                                stale_move = True
                    except Exception as e:
                        log.warning(f"  Live price fetch failed for {analysis['symbol']}: {e}")

                    if stale_move:
                        continue

                    # Second guard: even within the drift band, verify the live
                    # price hasn't ALREADY reached what TP1/SL would be at this
                    # entry — i.e. the move already happened before we could
                    # publish. Recompute the same levels send_signal() will use.
                    try:
                        # Anchor point for the bracket. current_price has by
                        # now been overwritten with the live quote, so levels
                        # computed from it follow the fill wherever it drifted —
                        # dragging the stop toward the noise the setup was meant
                        # to sit outside of. LEVELS_FROM_STRUCTURE=1 keeps them
                        # on the pre-drift price instead. See config for the
                        # measured trade-off; default is unchanged.
                        _price = analysis["current_price"]
                        if LEVELS_FROM_STRUCTURE:
                            _zp = analysis.get("zone_entry_price")
                            if _zp:
                                _price = float(_zp)
                        _atr   = analysis.get("atr", 0.0)
                        _rhi   = analysis.get("recent_high", _price * 1.03)
                        _rlo   = analysis.get("recent_low",  _price * 0.97)
                        _tp1, _tp2, _sl = calculate_tp_sl(
                            _price, direction, _atr, _rhi, _rlo,
                            tp1_level=analysis.get("tp1_level"),
                            tp2_level=analysis.get("tp2_level"),
                        )
                        already_resolved = (
                            (direction == "LONG"  and (_price >= _tp1 or _price <= _sl)) or
                            (direction == "SHORT" and (_price <= _tp1 or _price >= _sl))
                        )
                        if already_resolved:
                            log.warning(
                                f"  {analysis['symbol']}: live price {_price} already past "
                                f"TP1 {_tp1} / SL {_sl} at publish time — stale setup, skipping"
                            )
                            continue
                    except Exception as e:
                        log.warning(f"  Already-resolved check failed for {analysis['symbol']}: {e}")

                    if send_signal(analysis):
                        _cache_signal(analysis["symbol"], direction)
                        sent_count += 1
                        if direction in _dir_open:
                            _dir_open[direction] += 1
                        log.info(f"  Signal sent: {analysis['symbol']} {direction}")
                        try:
                            mark_setup_sent(analysis.get("_setup_log_id"))
                        except Exception as _me:
                            log.warning(f"  mark_setup_sent failed: {_me}")
                        # Keyed on the id log_signal returned, not "newest OPEN
                        # row for this symbol" — that was a guess that held only
                        # while one setup per symbol per scan is generated.
                        _sig_row = None
                        try:
                            _sid = analysis.get("_signal_id")
                            _sig_row = (get_signal_by_id(_sid) if _sid
                                        else get_latest_open_signal(analysis["symbol"]))
                        except Exception as _ge:
                            log.warning(f"  Could not fetch new signal row: {_ge}")
                        # Link so this setup's outcome comes from the real
                        # signal's close, not an independent shadow simulation
                        # (ported from crypto bot after the two disagreed there).
                        try:
                            if _sig_row:
                                link_setup_to_signal(analysis.get("_setup_log_id"), _sig_row["id"])
                        except Exception as _le:
                            # Must be loud: the row stays sent=1 with no
                            # signal_id and is then excluded from BOTH resolvers
                            # until backfill_setup_signal_links() repairs it.
                            log.warning(f"  link_setup_to_signal failed: {_le}")
                        # Autotrade: mirror the just-published signal into real
                        # OKX positions for onboarded users (async, fail-safe).
                        try:
                            autotrader.open_positions_for_signal(_sig_row)
                        except Exception as _ae:
                            log.warning(f"  Autotrade open hook failed: {_ae}")
                    else:
                        log.warning(
                            f"  Signal NOT sent: {analysis['symbol']} {direction} "
                            f"({analysis.get('confidence','?')}) — send_signal returned False"
                        )
                        # Ported from the crypto bot: a setup Claude APPROVED
                        # that failed to send otherwise sits at sent=0 with no
                        # block_reason, silently counted as a Claude rejection.
                        try:
                            mark_setup_blocked(analysis.get("_setup_log_id"), "send_failed")
                        except Exception:
                            pass
            except Exception as e:
                log.error(f"  Error sending {analysis.get('symbol','?')}: {e}")

        _last_scan_stats["sent"] = sent_count
        log.info(f"=== Scan complete — {sent_count} signal(s) sent ===\n")

    except Exception as e:
        log.error(f"Scan failed: {e}")


# ── Morning digest ────────────────────────────────────────────────────────────
def run_morning_digest():
    """Fetch last 18h headlines + today's high-impact econ calendar, send pre-market."""
    log.info("=== Morning digest started ===")
    try:
        digest = get_daily_digest()

        # Fold in today's HIGH-impact econ events (CPI/FOMC/NFP...) — the
        # single most actionable pre-market heads-up for equities, previously
        # only visible via the separate "📰 Новости на сегодня" button.
        try:
            day = get_day_events(max_events=10)
            high = [e for e in day["events"] if e["impact"] == "high" and not e["passed"]]
            cal_lines = []
            for e in high:
                ru, _note = _ru_event(e["title"])
                when = ("весь день" if e["all_day"] or not e["when_utc"]
                        else e["when_utc"].astimezone(_riga_tz()).strftime("%H:%M"))
                cc = f"{e['country']} " if e["country"] else ""
                cal_lines.append(f"🔴 {cc}{ru} — {when} (Рига)")
            digest["calendar"] = cal_lines
        except Exception as e:
            log.warning(f"Morning digest: calendar merge failed: {e}")

        send_morning_digest(digest)
        log.info(
            f"Morning digest sent — {len(digest.get('items', []))} items, "
            f"{len(digest.get('calendar', []))} high-impact events, "
            f"overall={digest.get('overall')}"
        )
    except Exception as e:
        log.error(f"Morning digest failed: {e}")


# ── Daily prayer (Mon–Fri 08:00 Riga) ────────────────────────────────────────
def run_daily_prayer():
    """Send the morning Dimoslav prayer with commandments button."""
    log.info("=== Daily prayer started ===")
    try:
        ok = send_daily_prayer()
        log.info(f"Daily prayer {'sent' if ok else 'FAILED'}")
    except Exception as e:
        log.error(f"Daily prayer failed: {e}")


# ── Evening prayer (Mon–Fri 23:50 Riga) ──────────────────────────────────────
def run_evening_prayer():
    """Send the evening Dimoslav prayer with ritual button."""
    log.info("=== Evening prayer started ===")
    try:
        ok = send_evening_prayer()
        log.info(f"Evening prayer {'sent' if ok else 'FAILED'}")
    except Exception as e:
        log.error(f"Evening prayer failed: {e}")


# ── Weekly digest (Sunday 22:00 Riga = 19:00 UTC summer) ─────────────────────
def run_weekly_digest():
    """Collect 7-day trade stats, generate Groq commentary, send to Telegram."""
    log.info("=== Weekly digest started ===")
    try:
        stats = get_weekly_stats()
        commentary = generate_weekly_commentary(stats)
        send_weekly_digest(stats, commentary)
        log.info(
            f"Weekly digest sent — {stats['n_total']} trades, "
            f"WR={stats['wr']}%, R={stats['total_r']:+.2f}"
        )
    except Exception as e:
        log.error(f"Weekly digest failed: {e}")


# ── Self-ping (keeps Render free tier awake) ──────────────────────────────────
def _app_url() -> str:
    """Return the public URL of this deployment from any known env var."""
    for key in ("APP_URL", "RENDER_EXTERNAL_URL", "RAILWAY_PUBLIC_DOMAIN"):
        val = os.environ.get(key, "").strip().rstrip("/")
        if val:
            # RAILWAY_PUBLIC_DOMAIN gives just the domain, add https://
            if not val.startswith("http"):
                val = f"https://{val}"
            return val
    return ""


def _self_ping():
    """Ping own health endpoint every 4 minutes to keep the service alive."""
    url = _app_url()
    if not url:
        log.info("No APP_URL set — self-ping disabled (local run)")
        return
    while True:
        time.sleep(240)  # 4 minutes
        try:
            _requests.get(f"{url}/", timeout=10)
            log.info("Self-ping OK")
        except Exception as e:
            log.warning(f"Self-ping failed: {e}")


# ── Webhook setup ────────────────────────────────────────────────────────────
def _setup_webhook():
    """Register Telegram webhook so bot can receive messages."""
    url = _app_url()
    if not url:
        log.info("No APP_URL set — webhook skipped (local run)")
        return
    webhook_url = f"{url}/webhook"
    try:
        resp = _requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/setWebhook",
            json={"url": webhook_url},
            timeout=10,
        )
        log.info(f"Webhook set: {webhook_url} → {resp.json().get('description', '?')}")
    except Exception as e:
        log.warning(f"Webhook setup failed: {e}")


# ── Startup ───────────────────────────────────────────────────────────────────
def _monitor_open_signals():
    """Lightweight 1-min job: check open trades for TP1/SL/BE hits. 24/7."""
    try:
        _check_open_signals()
    except Exception as e:
        log.warning(f"Open-signal monitor failed: {e}")
    # Real exchange position state, independent of the engine's own (slower,
    # kline-based) status detection — catches SL/TP/trail fires within one
    # cycle instead of waiting minutes for _check_open_signals to notice.
    try:
        autotrader.poll_exchange_closes()
    except Exception as e:
        log.warning(f"Autotrade exchange-close poll failed: {e}")


def _shadow_tracker_job():
    """15-min job: resolve outcomes for logged setups.

    Rejected/shadow setups (no real position) are simulated by
    _track_setup_outcomes(); sent setups have a real position, so
    resolve_sent_setups_from_signals() copies its actual close instead of
    running a second, independent simulation that could contradict it.
    """
    try:
        _track_setup_outcomes()
    except Exception as e:
        log.warning(f"Shadow tracker job failed: {e}")
    try:
        n = resolve_sent_setups_from_signals()
        if n:
            log.info(f"Shadow tracker: resolved {n} sent setup(s) from real signal outcomes")
    except Exception as e:
        log.warning(f"Sent-setup resolution failed: {e}")


# Each (csv, flag) seeds once. Add new batches as new tuples — already-seeded
# batches skip via their own bot_state flag, so redeploys never re-seed and new
# batches load independently of old ones.
_BT_SEED_DIR = os.path.dirname(os.path.abspath(__file__))
_BT_SEED_BATCHES = [
    ("backtest_seed_stocks.csv", "bt_seed_stocks_v5_done"),
]

# Bump whenever the seed CSV is regenerated under different trade geometry or a
# corrected cost/exit model. Priors are per-ticker base rates ("setups shaped
# like this resolved TP1 X% of the time"), so they are only meaningful while the
# geometry that produced them matches what the bot trades live — a mismatch
# feeds Claude a systematically wrong prior. On a generation change every
# source='backtest' row is wiped before the new batch loads, so a redeployed bot
# converges to exactly one generation instead of stacking contradictory ones.
#   v1: original run          (BACKTEST_TP_WINDOW unit bug, 12h not 48h)
#   v2: TP-window fixed       (WR 73.9%, +0.475R/tr)
#   v3: fee corrected to OKX's real 0.05% taker + TP1_R_MULT 1.0 -> 0.7
#   v4: expiry made session-aware + 5-day calendar ceiling, matching live
#   v5: TP1_R_MULT reverted 0.7 -> 1.0 (final geometry)
#       (1676tr, WR 73.9%, +0.591R/tr)
_BT_SEED_GENERATION = "v5"


def maybe_seed_backtest():
    """One-shot per batch: load backtest trades into setup_log as Claude memory
    priors (source='backtest'). Each batch gated by its own bot_state flag so
    redeploys never re-seed and new batches load on top of old ones.
    """
    import csv as _csv

    _prev_gen = get_bot_state("bt_seed_generation")
    if _prev_gen != _BT_SEED_GENERATION:
        try:
            n = delete_backtest_seed_rows()
            set_bot_state("bt_seed_generation", _BT_SEED_GENERATION)
            if n:
                log.info(
                    f"Backtest priors: wiped {n} rows from generation "
                    f"{_prev_gen or 'v1/unversioned'} before seeding {_BT_SEED_GENERATION}"
                )
        except Exception as e:
            log.warning(f"Backtest-seed generation purge failed: {e}")

    for fname, flag in _BT_SEED_BATCHES:
        try:
            if get_bot_state(flag):
                continue
            path = os.path.join(_BT_SEED_DIR, fname)
            if not os.path.exists(path):
                log.info(f"Seed CSV {fname} not found — skipping")
                continue
            with open(path, newline="", encoding="utf-8") as f:
                rows = list(_csv.DictReader(f))
            n = seed_backtest_outcomes(rows)
            set_bot_state(flag, str(n))
            log.info(f"Claude memory seeded: {n} trades from {fname} → setup_log[source=backtest]")
        except Exception as e:
            log.warning(f"Backtest seeding {fname} failed (will retry next boot): {e}")

    # One-shot backfill: rows seeded before the net_r column existed have
    # net_r=NULL → re-read the CSVs and fill it so expectancy (avg R) works
    # on the existing priors without a full re-seed. Gated once.
    if not get_bot_state("bt_net_r_backfilled"):
        total = 0
        for fname, _flag in _BT_SEED_BATCHES:
            try:
                path = os.path.join(_BT_SEED_DIR, fname)
                if not os.path.exists(path):
                    continue
                with open(path, newline="", encoding="utf-8") as f:
                    rows = list(_csv.DictReader(f))
                total += backfill_backtest_net_r(rows)
            except Exception as e:
                log.warning(f"net_r backfill {fname} failed (will retry next boot): {e}")
                return
        set_bot_state("bt_net_r_backfilled", str(total))
        log.info(f"Backtest net_r backfilled on {total} seeded rows")


def start_bot():
    log.info("Starting TUSA Stocks Bot...")
    # Data source diagnostics — OKX public API (EU region: no geoblock, no proxy)
    _okx_base = os.environ.get("OKX_BASE_URL", "")
    log.info(f"Data source: OKX{' via '+_okx_base if _okx_base else ' (default hosts)'}")

    # Initialise signal-tracking DB
    try:
        init_db()
        log.info("Database initialised")
    except Exception as e:
        log.warning(f"DB init failed: {e}")

    # One-shot Claude memory seeding from historical backtest (2024+)
    # Cooldowns survive redeploys (2026-08-03) — load before the first scan.
    _cooldowns_load()

    maybe_seed_backtest()

    # Self-healing backfill (ported 2026-07-31): link pre-fix sent setup_log
    # rows to their real signal so the next shadow-tracker tick corrects any
    # wrong independently-simulated outcome. Cheap, idempotent, self-limiting.
    try:
        _n = backfill_setup_signal_links()
        if _n:
            log.info(f"Linked {_n} pre-fix sent setup(s) to their real signal")
        # Resolve immediately rather than waiting for the 15-min tick.
        _r = resolve_sent_setups_from_signals()
        if _r:
            log.info(f"Corrected {_r} sent setup outcome(s) from real signals at boot")
    except Exception as e:
        log.warning(f"Setup-signal backfill failed: {e}")

    # Dedup guard: only send once per 60s per container (prevents
    # double-message during Render zero-downtime deploys where old + new
    # instances briefly overlap).
    _flag = "/tmp/tusa_started"
    try:
        skip = False
        if os.path.exists(_flag):
            if time.time() - os.path.getmtime(_flag) < 60:
                skip = True
        if not skip:
            open(_flag, "w").close()
            send_status(
                "🤖 *TUSA Stocks Bot Online*\n"
                f"Сканирую акции/ETF/товары OKX (X-Perps) каждые {SCAN_INTERVAL_MINUTES} мин, "
                f"только пока открыт рынок США (Пн-Пт, часы/праздники — кнопка 🕐 Рынок США в панели)."
            )
    except Exception as e:
        log.warning(f"Could not send startup message: {e}")

    scheduler = BackgroundScheduler(daemon=True)

    # Signal scan — every 5 min aligned to candle closes.
    # 15m candles close at :00/:15/:30/:45 → scan at :01/:16/:31/:46 (+1 min buffer).
    scheduler.add_job(
        run_scan, "cron",
        minute="1,6,11,16,21,26,31,36,41,46,51,56",
        timezone="UTC",
    )

    # TP/SL monitor — every 1 min, 24/7, lightweight (only price checks).
    scheduler.add_job(
        _monitor_open_signals, "cron",
        minute="*",
        timezone="UTC",
    )

    # Shadow-outcome tracker — every 15 min, resolves rejected+sent setup results
    # so the bot can learn whether Claude's verdicts matched reality.
    scheduler.add_job(
        _shadow_tracker_job, "cron",
        minute="3,18,33,48",
        timezone="UTC",
    )

    # 09:00 America/New_York = 30 min before the 9:30 bell — pre-market
    # briefing right when it's actionable. Fixed leftover from the crypto
    # bot's 10:00 Riga slot (6+ hours before this bot's session even opens).
    # NY timezone self-adjusts for US DST regardless of Riga's own DST dates.
    scheduler.add_job(
        run_morning_digest, "cron",
        day_of_week="mon-fri", hour=9, minute=0,
        timezone="America/New_York",
    )

    # Morning prayer — Mon–Fri 08:00 Riga
    scheduler.add_job(
        run_daily_prayer, "cron",
        day_of_week="mon-fri", hour=8, minute=0,
        timezone="Europe/Riga",
    )

    # Evening prayer — Mon–Fri 23:50 Riga
    scheduler.add_job(
        run_evening_prayer, "cron",
        day_of_week="mon-fri", hour=23, minute=50,
        timezone="Europe/Riga",
    )

    # Weekly digest — Sunday 22:00 Riga (19:00 UTC summer / 20:00 UTC winter, DST auto)
    scheduler.add_job(
        run_weekly_digest, "cron",
        day_of_week="sun", hour=22, minute=0,
        timezone="Europe/Riga",
    )
    scheduler.start()
    log.info("Scheduler: signal scan every 5 min (:01/:06/...), TP/SL monitor every 1 min")

    # Register Telegram webhook
    _setup_webhook()

    # First scan immediately
    threading.Thread(target=run_scan, daemon=True).start()

    # Self-ping — only needed on hosts that idle-sleep (e.g. Render free tier).
    # Off by default: Railway runs the container 24/7, so it's pointless there.
    # Set SELF_PING_ENABLED=1 to re-enable if moving back to a sleeping host.
    if os.getenv("SELF_PING_ENABLED", "0") != "0":
        threading.Thread(target=_self_ping, daemon=True).start()
        log.info("Self-ping enabled")
    else:
        log.info("Self-ping disabled (Railway does not idle-sleep)")


start_bot()  # runs at module load — works with gunicorn

def _warn_claude_shadow() -> None:
    """Say loudly which role Claude is in — the book differs by ~25% of trades."""
    if CLAUDE_GATE_ENABLED:
        return
    log.warning("=" * 70)
    log.warning("CLAUDE_GATE_ENABLED=0 — Claude is in SHADOW.")
    log.warning("  He is still called, scored and logged, but the rules filter")
    log.warning("  alone decides what trades. Approval here runs ~80%, so expect")
    log.warning("  about a quarter more trades than with him gating.")
    log.warning("  Purpose: score his verdict against REAL fills. Turn the gate")
    log.warning("  back on once that comparison has enough closed trades.")
    log.warning("=" * 70)


def _warn_stop_rule_coupling() -> None:
    """Shout if STOP_CLOSE_CONFIRM is off, because it silently changes TWO things.

    The obvious one is the monitor's stop rule. The one that costs money is in
    src/autotrader.py: moving the exchange stop out to STOP_EXCHANGE_BACKSTOP_R
    is gated on this same flag, so switching it off ALSO plants the exchange OCO
    directly on the signal's stop, where it fires on the first wick — exactly
    what the close-confirmed rule exists to prevent.

    Added 2026-08-29 after live data showed ten of twenty-four stops booked at
    the level rather than beyond it, and an average stop of -1.209R against the
    backtest's -1.534R. Both are signatures of touch-triggered exits. Whether
    the flag really is off in this deployment is the account owner's to check —
    this makes the answer visible in the logs instead of inferable from fills.
    """
    if STOP_CLOSE_CONFIRM:
        return
    log.warning("=" * 70)
    log.warning("STOP_CLOSE_CONFIRM=0 — stops fire on a WICK TOUCH, not a close.")
    log.warning("  This also stops the exchange SL being moved out to the %.1fR",
                float(STOP_EXCHANGE_BACKSTOP_R))
    log.warning("  backstop, so the exchange OCO sits ON the signal stop and")
    log.warning("  triggers first on any spike. The backtest models the OTHER")
    log.warning("  rule, so its figures will not describe this deployment.")
    log.warning("=" * 70)


_warn_stop_rule_coupling()
_warn_claude_shadow()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
