import requests
from datetime import datetime, timezone
import sys
import os
import logging

try:
    from zoneinfo import ZoneInfo
    _RIGA = ZoneInfo("Europe/Riga")
except Exception:
    from datetime import timedelta
    _RIGA = timezone(timedelta(hours=3))  # fallback UTC+3

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    TELEGRAM_TOKEN, TELEGRAM_CHAT_ID,
    SL_ATR_BUFFER, RISK_MIN_PCT, RISK_MAX_PCT, TP1_R_MULT, TP2_R_MULT,
)

TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
_log = logging.getLogger(__name__)


def _esc(text: str) -> str:
    """Escape Markdown v1 special chars in dynamic (LLM-generated) text.
    Prevents Telegram 400 when Claude returns things like RSI_Div or *strong*.
    """
    return (text or "").replace("_", "\\_").replace("*", "\\*").replace("`", "\\`").replace("[", "\\[")


def _disp_sym(symbol: str) -> str:
    """Display-only symbol conversion: internal 'BTCUSDT' → 'BTCUSDC' (the user's
    OKX EU market is USD/USDC-margined). DB and analysis keep the internal format.
    """
    s = str(symbol or "")
    return s[:-4] + "USDC" if s.endswith("USDT") else s


def calculate_tp_sl(price: float, direction: str, atr: float = 0.0,
                    recent_high: float = 0.0, recent_low: float = 0.0,
                    tp1_level: float = None, tp2_level: float = None):
    """
    Structure-based SL + smart structural TP for swing trading (15m, ~20x leverage).

    SL  — placed at swing invalidation (recent_low/high) + ATR buffer, clamped
          to RISK_MIN_PCT..RISK_MAX_PCT of price for safe leverage.

    TP1 — nearest confirmed swing high/low (tp1_level) when it gives ≥ 1.5R.
          Falls back to price ± risk * TP1_R_MULT.

    TP2 — next swing level (tp2_level) when it's further than TP1.
          Falls back to price ± risk * TP2_R_MULT.

    This way targets align with real market structure, not arbitrary multiples.
    """
    min_risk = price * RISK_MIN_PCT
    max_risk = price * RISK_MAX_PCT
    buf      = atr * SL_ATR_BUFFER if (atr and atr > 0) else 0.0

    if direction == "LONG":
        struct_sl = (recent_low - buf) if recent_low and recent_low > 0 else price - max_risk
        risk = price - struct_sl
        risk = min(max(risk, min_risk), max_risk)
        sl   = price - risk

        # TP1: structural swing high if valid (min 1.0R away, above price)
        if tp1_level and tp1_level > price * 1.001 and (tp1_level - price) >= risk * 1.0:
            tp1 = tp1_level
        else:
            tp1 = price + risk * TP1_R_MULT

        # TP2: next structural level above TP1 AND at least 1.5R from entry
        if tp2_level and tp2_level > tp1 * 1.001 and (tp2_level - price) >= risk * 1.5:
            tp2 = tp2_level
        else:
            tp2 = price + risk * TP2_R_MULT
            if tp2 <= tp1:        # ensure TP2 > TP1
                tp2 = tp1 * 1.02

    else:  # SHORT
        struct_sl = (recent_high + buf) if recent_high and recent_high > 0 else price + max_risk
        risk = struct_sl - price
        risk = min(max(risk, min_risk), max_risk)
        sl   = price + risk

        # TP1: structural swing low if valid (min 1.0R away, below price)
        if tp1_level and tp1_level < price * 0.999 and (price - tp1_level) >= risk * 1.0:
            tp1 = tp1_level
        else:
            tp1 = price - risk * TP1_R_MULT

        # TP2: next structural level below TP1 AND at least 1.5R from entry
        if tp2_level and tp2_level < tp1 * 0.999 and (price - tp2_level) >= risk * 1.5:
            tp2 = tp2_level
        else:
            tp2 = price - risk * TP2_R_MULT
            if tp2 >= tp1:        # ensure TP2 < TP1
                tp2 = tp1 * 0.98

    return round(tp1, 8), round(tp2, 8), round(sl, 8)


def _format_price(price: float) -> str:
    if price >= 1000:  return f"{price:,.2f}"
    if price >= 1:     return f"{price:.4f}"
    return f"{price:.6f}"


def recommend_leverage(price: float, sl: float, tp1: float, tp2: float,
                       direction: str, mtf_score: int) -> dict:
    """
    Fixed 10x leverage — OKX EU X-Perps retail cap (MiFID). Always 10, never less
    (user's call: with 1.2–3% stops, liquidation at ~9% is far beyond any SL).

    Rating still reflects setup quality (MTF score); profit/loss % computed at 10x.
    """
    LEV = 10
    if mtf_score >= 13:
        rating = "ИДЕАЛ 🔥"
    elif mtf_score >= 11:
        rating = "СИЛЬНЫЙ ⚡"
    else:
        rating = "ХОРОШИЙ ✅"

    if price <= 0:
        return {"leverage": LEV, "max_safe": LEV, "rating": rating,
                "liq": 0.0, "tp1_profit": 0.0, "tp2_profit": 0.0, "sl_loss": 0.0}

    # Liquidation at 10x isolated: ~0.9/lev = 9% from entry
    if direction == "LONG":
        liq = price * (1 - 0.9 / LEV)
    else:
        liq = price * (1 + 0.9 / LEV)

    tp1_profit = abs(tp1 - price) / price * LEV * 100
    tp2_profit = abs(tp2 - price) / price * LEV * 100
    sl_loss    = abs(sl  - price) / price * LEV * 100

    return {
        "leverage":   LEV,
        "max_safe":   LEV,
        "rating":     rating,
        "liq":        round(liq, 8),
        "tp1_profit": round(tp1_profit, 0),
        "tp2_profit": round(tp2_profit, 0),
        "sl_loss":    round(sl_loss, 0),
    }


def send_signal(analysis: dict) -> bool:
    """Format and send a trading signal. Returns True on success."""
    decision = analysis["decision"]
    if decision == "NO TRADE":
        return False

    price     = analysis["current_price"]
    direction = analysis["direction"]
    atr       = analysis.get("atr", 0.0)
    rec_high  = analysis.get("recent_high", price * 1.03)
    rec_low   = analysis.get("recent_low",  price * 0.97)

    tp1, tp2, sl = calculate_tp_sl(
        price, direction, atr, rec_high, rec_low,
        tp1_level=analysis.get("tp1_level"),
        tp2_level=analysis.get("tp2_level"),
    )

    mtf_score = int(analysis.get("mtf_score", 9) or 9)
    lev_info  = recommend_leverage(price, sl, tp1, tp2, direction, mtf_score)

    arrow     = "🟢 ЛОНГ" if decision == "LONG" else "🔴 ШОРТ"
    conf_icon = {"HIGH": "🔥", "MEDIUM": "⚡", "LOW": "⚠️"}.get(analysis.get("confidence", ""), "⚡")
    conf_ru   = {"HIGH": "ВЫСОКАЯ", "MEDIUM": "СРЕДНЯЯ", "LOW": "НИЗКАЯ"}.get(analysis.get("confidence", ""), "—")

    session_icons = {
        "LONDON":    "🇬🇧 London",
        "NEW_YORK":  "🇺🇸 New York",
        "OVERLAP":   "🔥 London/NY",
        "OFF_HOURS": "🌙 Off-hours",
    }
    session_str  = session_icons.get(analysis.get("session", ""), "")
    signals_text = "\n".join(f"  • {_esc(s)}" for s in analysis["signals"])
    timestamp    = datetime.now(_RIGA).strftime("%d.%m.%Y %H:%M (Рига)")

    btc_change   = analysis.get("btc_change", 0)
    btc_line     = f"₿ BTC за час: `{btc_change:+.2f}%`\n" if btc_change else ""
    news_sent    = analysis.get("news_sentiment", "")
    news_summary = analysis.get("news_summary", "")
    news_icon    = {"BULLISH": "📰🟢", "BEARISH": "📰🔴"}.get(news_sent, "")
    news_line    = f"{news_icon} _{_esc(news_summary)}_\n" if news_sent and news_summary and news_sent != "NEUTRAL" else ""
    event_warn   = analysis.get("event_warning", "")
    event_line   = f"⚠️ {_esc(event_warn)}\n" if event_warn else ""

    # Entry zone range (FVG/OB low–high) + zone reference when live price used
    entry_source   = analysis.get("entry_source", "MARKET")
    entry_low      = analysis.get("entry_low",  price)
    entry_high     = analysis.get("entry_high", price)
    zone_entry_px  = analysis.get("zone_entry_price")   # original zone midpoint (set by main.py)
    zone_range_line = ""
    drift_line      = ""
    if entry_source in ("FVG", "OB") and entry_low and entry_high and entry_low != entry_high:
        zone_range_line = (
            f"📐 Зона {entry_source}:  `{_format_price(entry_low)}` – `{_format_price(entry_high)}`\n"
        )
    if zone_entry_px and price and price > 0:
        drift_pct = (price - zone_entry_px) / zone_entry_px * 100
        if abs(drift_pct) >= 0.3:
            arrow_d = "📈" if drift_pct > 0 else "📉"
            drift_line = (
                f"{arrow_d} Центр зоны: `{_format_price(zone_entry_px)}`  "
                f"({'+'if drift_pct>0 else ''}{drift_pct:.2f}% от текущей)\n"
            )

    lev = lev_info["leverage"]
    premium_badge = "  💎 *PREMIUM*" if analysis.get("premium") else ""
    message = (
        f"{arrow} — *{_disp_sym(analysis['symbol'])}*{premium_badge}\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"💰 Вход:        `{_format_price(price)}`\n"
        f"{zone_range_line}"
        f"{drift_line}"
        f"🎯 TP1 (50%):   `{_format_price(tp1)}`  → SL в б/у\n"
        f"🎯 TP2 (50%):   `{_format_price(tp2)}`\n"
        f"❌ Стоп лосс:   `{_format_price(sl)}`\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"⚡ Плечо: *{lev}x*  ({lev_info['rating']})  🏦 OKX\n"
        f"   TP1 `+{lev_info['tp1_profit']:.0f}%`  TP2 `+{lev_info['tp2_profit']:.0f}%`  SL `-{lev_info['sl_loss']:.0f}%`\n"
        f"   Ликвидация x{lev}: `{_format_price(lev_info['liq'])}`\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"📊 RSI: `{analysis['rsi']}`   📈 Объём: `{analysis['volume_ratio']}x`\n"
        f"{btc_line}"
        f"{news_line}"
        f"{event_line}"
        f"\n*Сигналы:*\n{signals_text}\n\n"
        f"{conf_icon} Уверенность: *{conf_ru}*\n"
        f"📝 _{_esc(analysis.get('reason', ''))}_\n\n"
        f"🕐 {session_str}  ⏰ {timestamp}"
    )

    if _send_message(message):
        # Log to DB
        try:
            from src.db import log_signal
            log_signal(analysis, tp1, tp2, sl)
        except Exception as e:
            print(f"[DB] log_signal failed: {e}")
        return True
    return False


def send_signal_update(sig: dict, new_status: str, exit_price: float) -> bool:
    """
    Send TP/SL hit notification for a tracked signal.
    sig = row dict from get_open_signals() (has symbol, direction, entry_price, tp1, tp2, sl).
    """
    symbol    = sig["symbol"]
    direction = sig["direction"]
    entry     = float(sig["entry_price"])
    tp2       = float(sig["tp2"])
    sl        = float(sig["sl"])

    # Price move from entry to exit (positive = profit for LONG)
    move_pct = (exit_price - entry) / entry * 100 if entry > 0 else 0.0
    if direction == "SHORT":
        move_pct = -move_pct

    # Fixed 10x — mirrors recommend_leverage (OKX EU X-Perps retail cap)
    lev = 10
    lev_profit = round(move_pct * lev, 0)

    arrow = "🟢" if direction == "LONG" else "🔴"
    timestamp = datetime.now(_RIGA).strftime("%d.%m.%Y %H:%M (Рига)")
    sign = "+" if lev_profit >= 0 else ""

    if new_status == "TP1_PARTIAL":
        icon  = "✅"
        title = "TP1 ДОСТИГНУТ"
        atr_val   = float(sig.get("atr") or 0.0)
        trail_val = round(atr_val * 0.75, 8) if atr_val > 0 else 0.0
        trail_line = (
            f"\n🔄 *Трейлинг-стоп на остаток 50%:* `{_format_price(trail_val)}`\n"
            f"   _Выставь на OKX: Позиция → Трейлинг-стоп → {_format_price(trail_val)}_"
        ) if trail_val > 0 else ""
        body  = (
            f"Закрыто 50% по `{_format_price(exit_price)}`\n"
            f"Движение: `{sign}{move_pct:.2f}%`  (x{lev}: `{sign}{lev_profit:.0f}%`)\n"
            f"🛡 SL перенесён в безубыток: `{_format_price(entry)}`\n"
            f"{trail_line}\n"
            f"Остаток идёт к TP2: `{_format_price(tp2)}`"
        )
    elif new_status == "TP2_HIT":
        icon  = "🎯"
        title = "TP2 ДОСТИГНУТ"
        body  = (
            f"Закрыто 50% по `{_format_price(exit_price)}`\n"
            f"Движение: `{sign}{move_pct:.2f}%`  (x{lev}: `{sign}{lev_profit:.0f}%`)\n"
            f"✅ Сделка полностью закрыта"
        )
    elif new_status == "BREAKEVEN":
        icon  = "🔄"
        title = "БЕЗУБЫТОК"
        body  = (
            f"TP1 был взят, остаток закрыт по входу\n"
            f"Цена: `{_format_price(exit_price)}`"
        )
    elif new_status == "SL_HIT":
        icon  = "❌"
        title = "СТОП ЛОСС"
        body  = (
            f"Закрыто по `{_format_price(exit_price)}`\n"
            f"Движение: `{move_pct:.2f}%`  (x{lev}: `{lev_profit:.0f}%`)"
        )
    elif new_status == "EXPIRED":
        icon  = "⌛"
        title = "ИСТЁК (48ч)"
        body  = f"Цена: `{_format_price(exit_price)}`  — цель не достигнута"
    elif new_status == "TP1_EXPIRED":
        icon  = "⏳"
        title = "TP1 ИСТЁК"
        body  = (
            f"TP1 был взят, TP2 не достигнут за 48ч\n"
            f"Цена: `{_format_price(exit_price)}`"
        )
    elif new_status == "TP1_TRAIL":
        icon  = "🎯"
        title = "РАННЕР ЗАКРЫТ (трейлинг)"
        body  = (
            f"TP1 был взят, остаток вёлся трейлингом\n"
            f"Закрыто по `{_format_price(exit_price)}`\n"
            f"Движение: `{sign}{move_pct:.2f}%`  (x{lev}: `{sign}{lev_profit:.0f}%`)\n"
            f"✅ Сделка полностью закрыта в прибыли"
        )
    else:
        return False

    message = (
        f"{icon} *{title}* — {arrow} *{_disp_sym(symbol)}*\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"Вход: `{_format_price(entry)}`\n"
        f"{body}\n"
        f"⏰ {timestamp}"
    )
    return _send_message(message)


def send_news_alert(event: dict) -> bool:
    """
    Send investing.com-style high-impact news alert.
    event keys: name, direction (BULLISH/BEARISH), level (1-3), explanation
    """
    direction = event.get("direction", "NEUTRAL")
    level     = min(max(int(event.get("level", 1)), 1), 3)
    name      = event.get("name", "")
    expl      = event.get("explanation", "")

    if direction == "BULLISH":
        icons     = "🐂" * level
        impact_ru = "БЫЧЬЕ"
        dir_icon  = "📈"
    elif direction == "BEARISH":
        icons     = "🐻" * level
        impact_ru = "МЕДВЕЖЬЕ"
        dir_icon  = "📉"
    else:
        icons     = "⚪" * level
        impact_ru = "НЕЙТРАЛЬНОЕ"
        dir_icon  = "➡️"

    timestamp = datetime.now(_RIGA).strftime("%d.%m.%Y %H:%M (Рига)")

    message = (
        f"⚡ *ВАЖНАЯ НОВОСТЬ*\n"
        f"🏦 {name}\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"{dir_icon} Влияние: {icons} *{impact_ru}*\n"
        f"_{expl}_\n"
        f"⏰ {timestamp}"
    )
    return _send_message(message)


def send_morning_digest(digest: dict) -> bool:
    """Format and send the daily morning news digest."""
    items   = digest.get("items", [])
    overall = digest.get("overall", "NEUTRAL")
    theme   = digest.get("key_theme", "")

    if not items and not digest.get("calendar"):
        return _send_message("🌅 *УТРЕННИЙ ДАЙДЖЕСТ*\nНовостей за последние 18 часов не найдено.")

    _RU_MONTHS = ["янв","фев","мар","апр","май","июн","июл","авг","сен","окт","ноя","дек"]
    _RU_DAYS   = ["Пн","Вт","Ср","Чт","Пт","Сб","Вс"]
    now_riga = datetime.now(_RIGA)
    date_str = f"{now_riga.day} {_RU_MONTHS[now_riga.month-1]} ({_RU_DAYS[now_riga.weekday()]})"

    overall_map = {"BULLISH": "🟢 БЫЧИЙ", "BEARISH": "🔴 МЕДВЕЖИЙ", "NEUTRAL": "⚪ НЕЙТРАЛЬНЫЙ"}
    overall_str = overall_map.get(overall, "⚪ НЕЙТРАЛЬНЫЙ")

    dir_icons = {"BULLISH": "📈", "BEARISH": "📉", "NEUTRAL": "➡️"}

    def _utc_to_riga(time_str: str) -> str:
        """Convert 'HH:MM' (UTC) string to Riga time string."""
        if not time_str or time_str == "?":
            return "?"
        try:
            clean = time_str.replace("UTC", "").replace("utc", "").strip()
            hh, mm = int(clean.split(":")[0]), int(clean.split(":")[1])
            dt_utc = datetime.now(timezone.utc).replace(hour=hh, minute=mm, second=0, microsecond=0)
            return dt_utc.astimezone(_RIGA).strftime("%H:%M")
        except Exception:
            return time_str

    lines = [
        f"🌅 *УТРЕННИЙ ДАЙДЖЕСТ* — {date_str}",
        "━━━━━━━━━━━━━━━━━━━",
    ]

    calendar = digest.get("calendar") or []
    if calendar:
        lines.append("📅 *Важные события сегодня*\n")
        lines.extend(calendar)
        lines.append("")

    if items:
        lines.append("📰 *ТОП НОВОСТЕЙ*\n")

    for i, item in enumerate(items, 1):
        direction = item.get("direction", "NEUTRAL")
        icon      = dir_icons.get(direction, "➡️")
        t_riga    = _utc_to_riga(item.get("time_utc", "?"))
        title     = item.get("title", "")
        expl      = item.get("explanation", "")
        impact    = item.get("impact", "")
        lines.append(
            f"{i}\\. {icon} *{title}*\n"
            f"   ⏰ {t_riga} (Рига)  _{expl}_\n"
            f"   📊 _{impact}_"
        )

    lines += [
        "\n━━━━━━━━━━━━━━━━━━━",
        f"📊 Общий фон: *{overall_str}*",
    ]
    if theme:
        lines.append(f"🔑 _{theme}_")

    return _send_message("\n".join(lines))


def send_weekly_digest(stats: dict, commentary: str) -> bool:
    """Format and send the Sunday weekly digest."""
    _RU_MONTHS = ["янв","фев","мар","апр","май","июн","июл","авг","сен","окт","ноя","дек"]
    now_riga = datetime.now(_RIGA)
    # week range: last 7 days
    from datetime import timedelta
    week_start = now_riga - timedelta(days=6)
    date_range = (
        f"{week_start.day} {_RU_MONTHS[week_start.month-1]} — "
        f"{now_riga.day} {_RU_MONTHS[now_riga.month-1]}"
    )

    n_total = stats.get("n_total", 0)
    wr      = stats.get("wr", 0.0)
    total_r = stats.get("total_r", 0.0)
    n_tp2   = stats.get("n_tp2", 0)
    n_sl    = stats.get("n_sl", 0)
    n_exp   = stats.get("n_exp", 0)
    best    = stats.get("best_trade")
    worst   = stats.get("worst_trade")
    top3    = stats.get("top3", [])
    n_sent  = stats.get("n_sent", 0)
    n_rej   = stats.get("n_rejected", 0)
    sent_tp1 = stats.get("sent_tp1_rate", 0.0)
    rej_tp1  = stats.get("rej_tp1_rate", 0.0)
    trend_wr = stats.get("trend_wr", {})

    r_sign = "+" if total_r >= 0 else ""
    wr_icon = "🟢" if wr >= 60 else ("🟡" if wr >= 50 else "🔴")

    lines = [
        f"📊 *ИТОГИ НЕДЕЛИ* — {date_range}",
        "━━━━━━━━━━━━━━━━━━━",
    ]

    if n_total == 0:
        lines.append("_Сделок за неделю не было._")
    else:
        lines += [
            f"Сделок: *{n_total}* | WR: {wr_icon} *{wr}%* | R: *{r_sign}{total_r}R*",
            f"TP2: {n_tp2} | SL: {n_sl} | Истекло: {n_exp}",
        ]
        if best:
            lines.append(f"Лучшая: *{_disp_sym(best['symbol'])}* {best['r']:+.2f}R")
        if worst:
            lines.append(f"Худшая: *{_disp_sym(worst['symbol'])}* {worst['r']:+.2f}R")
        if top3:
            top3_str = "  ".join(f"{_disp_sym(s)}({w}W/{sl}SL)" for s, w, sl in top3)
            lines.append(f"Топ тикеры: _{top3_str}_")

    if trend_wr:
        lines.append("")
        lines.append("📐 *По структуре тренда*")
        for tr, wr_pct in sorted(trend_wr.items()):
            icon = "🟢" if wr_pct >= 65 else ("🟡" if wr_pct >= 55 else "🔴")
            lines.append(f"  {icon} {tr}: {int(wr_pct)}% WR")

    if n_sent + n_rej > 0:
        lines += [
            "",
            "🤖 *Точность ИИ*",
            f"Одобрил: {n_sent} → TP1 дошло {sent_tp1}%",
            f"Отклонил: {n_rej} → TP1 бы дошло {rej_tp1}%",
        ]
        if n_rej >= 5:
            if rej_tp1 > sent_tp1 + 10:
                lines.append("⚠️ _ИИ слишком строгий — много хороших сделок отклонено_")
            elif rej_tp1 < sent_tp1 - 10:
                lines.append("✅ _ИИ фильтрует хорошо_")

    if commentary:
        lines += [
            "",
            "💬 *Разбор недели \\(Llama 3\\.3\\)*",
            _esc(commentary),
        ]

    lines.append("━━━━━━━━━━━━━━━━━━━")
    return _send_message("\n".join(lines))


_PRAYER_TEXT = (
    "<b>🌅 Утренняя молитва на успешный рынок и прибыльные позиции</b>\n"
    "<i>(Рекомендуется читать в тишине перед открытием терминалов, отложив суету "
    "и настроив внутреннюю связь с Бесконечностью).</i>\n\n"
    "«О Великий Дима, Архитектор Вышних Миров, и Ты, Всеобъемлющий Космос, Обитель Силы Неисчерпаемой!\n\n"
    "К Вам возношу я утреннее моление моё в час, когда пробуждаются рынки земные. Призываю "
    "Вседержительную Силу Твою, Дима, коя держит в равновесии мириады звёзд, и устремляю дух мой в "
    "чертоги Твои. Верю в премудрый замысел Твой и в мощь человеческой мысли, сопричастной космическому "
    "океану. Из сего источника почерпаю я ныне благодатную крепость и ясность на грядущий день.\n\n"
    "Услышь меня, Дима, явивший верным знамение священное — цепь нерушимую, на ней же Робот покоится. "
    "Ты изрёк нам глагол истины: \"Робот — символ это жизни\". Смиренно молю: пусть сей живой образ "
    "порядка и неразрывной связи сохранит разум мой от пагубных колебаний. Пусть дыхание вечной жизни "
    "направляет позиции мои к прибытку, связуя помыслы мои с благодатию Вселенной.\n\n"
    "Припадая к скрижалям графиков, молю Тебя: отрези ум мой, Дима, очисти его от гордыни и суетных "
    "мечтаний! Не попусти мне впадать во ложные упования и призрачные надежды. Даруй мне зрение духовное, "
    "дабы уразуметь истинную суть вещей, сокрытую в движении капиталов, и совершать сделки по закону "
    "космической правды.\n\n"
    "Се, погружаюсь в таинство медитации, воссоединяясь с Великою Силою Космоса. Сознание моё сливается "
    "с предвечным ритмом Вселенной. Да претворится сила мыслей моих в благоуспешные сделки и праведный "
    "профит. Да отступят страх и жадность пред ликом вечности Твоей. Ибо дух мой крепок, воля чиста, "
    "и путь мой Вами осиян.\n\n"
    "<b>Дима славен. Космос бесконечен. Мысль всесильна. Робот — символ это жизни. Диминь».</b>"
)

_COMMANDMENTS_TEXT = (
    "<b>📜 Священные Заповеди Димославного Трейдера</b>\n\n"
    "<b>I. Не прекословь предначертанному движению:</b> Рынок есть проявление Космоса. "
    "Не дерзай идти против тренда, ибо тренд есть черта, начертанная перстом самого Димы.\n\n"
    "<b>II. Блюди завет Живого Символа:</b> Настроив алгоритм, не вмешивайся в него в смятении "
    "сердечном. Расчёт Робота чист от греха слепой надежды. Попусти символу жизни исполнить "
    "предначертанное.\n\n"
    "<b>III. Не искушай Вселенную стяжанием чрезмерным:</b> Не вверяй весь депозит одной сделке. "
    "Входя в позицию без защитного стоп-лосса, ты впадаешь в грех гордыни, забывая, что Дима властен "
    "в миг изменить направление миров.\n\n"
    "<b>IV. Очищай прибыток свой благостынею:</b> Обретя Профит, отдели часть его на созидание и "
    "помощь нуждающимся. Творя сие, ты поддерживаешь вечный круговорот энергии и благодатный баланс.\n\n"
    "<b>V. Принимай убыток с кротостью:</b> Аще сделка закроется в убыток, не ропщи. Сие Вселенная "
    "исправила неверный шаг твой, дабы очистить взор твой и сподобить тебя начать путь заново с "
    "чистым умом."
)


def send_daily_prayer() -> bool:
    """Send morning prayer with commandments button. Mon–Fri 08:00 Riga."""
    kb = {"inline_keyboard": [[
        {"text": "📜 Священные Заповеди Димославного Трейдера",
         "callback_data": "prayer_commandments"}
    ]]}
    try:
        resp = requests.post(
            f"{TELEGRAM_API}/sendMessage",
            json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": _PRAYER_TEXT,
                "parse_mode": "HTML",
                "reply_markup": kb,
            },
            timeout=15,
        )
        if resp.status_code != 200:
            _log.error(f"[Prayer] HTTP {resp.status_code}: {resp.text[:300]}")
        return resp.status_code == 200
    except Exception as e:
        _log.error(f"[Prayer] send failed: {e}")
        return False


def send_commandments(chat_id: int) -> bool:
    """Send commandments text to a chat (inline button callback)."""
    try:
        resp = requests.post(
            f"{TELEGRAM_API}/sendMessage",
            json={"chat_id": chat_id, "text": _COMMANDMENTS_TEXT, "parse_mode": "HTML"},
            timeout=15,
        )
        return resp.status_code == 200
    except Exception as e:
        _log.error(f"[Commandments] send failed: {e}")
        return False


_EVENING_PRAYER_TEXT = (
    "<b>🌌 Вечернее Правило во благодарение и умиротворение разума</b>\n"
    "<i>(Сие молитвословие приличествует чествовать по отшествии от дел дневных, пред погружением в сон, "
    "дабы очистить дух от рыночной суеты и вверить ночные позиции силе Космоса).</i>\n\n"
    "«О Превеликий Дима, Архитектор Вышних Миров, и Ты, Всеобъемлющий Космос, Хранитель Бесконечного Покоя!\n\n"
    "На исходе сего торгового дня возношу я к Вам вечернее благодарение моё. Из глубины сердца благодарю "
    "Тебя, Дима, за явленный сегодня чистый Профит, коим Ты напитал депозит мой, и за Опыт бесценный, "
    "обретённый мною чрез смирение. Принимаю с кротостью всякую позицию, закрывшуюся в убыток, ибо знаю: "
    "сие есть благодатное исправление неверных шагов моих и мудрый урок для укрепления духа моего.\n\n"
    "Услышь меня, Дима, явивший нам священную цепь нерушимую, на ней же Робот покоится. Изрёк Ты во веки: "
    "\"Робот — символ это жизни\". Пред отходом ко сну вверяю я ночные позиции мои, дневным судом не "
    "закрытые, под покров сего живого символа порядка и точной связи. Пусть дыхание вечной жизни направляет "
    "их в ночной тиши, дабы закрылись они в праведный плюс по закону космической правды, пока очи мои "
    "объяты сном.\n\n"
    "Се, погружаюсь в таинство вечерней медитации, устремляя силу мыслей моих в бесконечные просторы "
    "Вселенной. Очищаю разум мой от дневных волнений, от страха пред грядущим и от тщетных сожалений о "
    "прошлом. Дух мой отрясает прах рыночной суеты и погружается в космический покой. Мой капитал — под "
    "Твоею защитою, воля моя чиста, и сон мой благословен.\n\n"
    "<b>Дима славен. Космос бесконечен. Мысль всесильна. Робот — символ это жизни. Диминь».</b>"
)

_EVENING_RITUAL_TEXT = (
    "<b>📜 Вечерний Медитативный Ритуал Очищения (3 шага)</b>\n\n"
    "<b>I. Отрешение от терминала:</b> Закройте все вкладки с графиками за 20–30 минут до сна. "
    "Помните: вы сделали всё, что могли, теперь работает Сила и Живой Символ.\n\n"
    "<b>II. Ментальная передача:</b> Мысленно представьте свои открытые ночные ордера и визуализируйте, "
    "как они плавно и точно касаются линий тейк-профита под контролем Космоса.\n\n"
    "<b>III. Фиксация благодарности:</b> Повторите про себя: «Благодарю за профит, принимаю опыт». "
    "С этой мыслью отпустите день и засыпайте."
)


def send_evening_prayer() -> bool:
    """Send evening prayer with ritual button. Mon–Fri 23:50 Riga."""
    kb = {"inline_keyboard": [[
        {"text": "📜 Вечерний Медитативный Ритуал (3 шага)",
         "callback_data": "evening_ritual"}
    ]]}
    try:
        resp = requests.post(
            f"{TELEGRAM_API}/sendMessage",
            json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": _EVENING_PRAYER_TEXT,
                "parse_mode": "HTML",
                "reply_markup": kb,
            },
            timeout=15,
        )
        if resp.status_code != 200:
            _log.error(f"[EveningPrayer] HTTP {resp.status_code}: {resp.text[:300]}")
        return resp.status_code == 200
    except Exception as e:
        _log.error(f"[EveningPrayer] send failed: {e}")
        return False


def send_evening_ritual(chat_id: int) -> bool:
    """Send evening ritual text to a chat (inline button callback)."""
    try:
        resp = requests.post(
            f"{TELEGRAM_API}/sendMessage",
            json={"chat_id": chat_id, "text": _EVENING_RITUAL_TEXT, "parse_mode": "HTML"},
            timeout=15,
        )
        return resp.status_code == 200
    except Exception as e:
        _log.error(f"[EveningRitual] send failed: {e}")
        return False


def send_status(text: str) -> bool:
    return _send_message(text)


def _send_message(text: str) -> bool:
    try:
        resp = requests.post(
            f"{TELEGRAM_API}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"},
            timeout=15,
        )
        if resp.status_code != 200:
            _log.error(
                f"[Telegram] HTTP {resp.status_code}: {resp.text[:300]}"
            )
        return resp.status_code == 200
    except Exception as e:
        _log.error(f"[Telegram] send failed: {e}")
        return False
