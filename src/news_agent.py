"""
Global macro news agent.

Sources (all free, no API key needed):
  - Reuters Business RSS
  - CNBC Markets RSS
  - BBC Business RSS
  - CoinDesk RSS

AI: Groq free tier (llama-3.1-8b-instant) — 14 400 req/day, ~200ms latency.
Register free at https://groq.com → API Keys → Create Key → set GROQ_API_KEY in Render.

Runs once per scan. Returns:
  sentiment  : BULLISH | BEARISH | NEUTRAL
  summary    : one-line key event (max 15 words)
  pause      : True only on extreme events (war, total ban, major crash)
  headlines  : list of fetched titles
"""

import xml.etree.ElementTree as ET
import requests as _req
import sys
import os
from datetime import datetime, timezone, timedelta
from email.utils import parsedate_to_datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import GROQ_API_KEY, NEWS_LOOKBACK_HOURS

# RSS sources — all public, no registration
RSS_FEEDS = [
    ("Reuters",   "https://feeds.reuters.com/reuters/businessNews"),
    ("CNBC",      "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=15839135"),
    ("BBC Biz",   "https://feeds.bbci.co.uk/news/business/rss.xml"),
    ("CoinDesk",  "https://www.coindesk.com/arc/outboundfeeds/rss/"),
    ("Decrypt",   "https://decrypt.co/feed"),
    ("Cointelegraph", "https://cointelegraph.com/rss"),
    ("CryptoSlate",   "https://cryptoslate.com/feed/"),
    ("BTC Magazine",  "https://bitcoinmagazine.com/feed"),
]


def _fetch_rss(url: str, timeout: int = 8) -> list[dict]:
    """Fetch and parse a single RSS feed. Returns list of {title, published_utc}."""
    try:
        resp = _req.get(url, timeout=timeout,
                        headers={"User-Agent": "Mozilla/5.0 CryptoBot/1.0"})
        if resp.status_code != 200:
            return []
        root = ET.fromstring(resp.content)
        items = []
        for item in root.iter("item"):
            title = item.findtext("title", "").strip()
            pub   = item.findtext("pubDate", "")
            if not title:
                continue
            try:
                pub_dt = parsedate_to_datetime(pub).astimezone(timezone.utc) if pub else None
            except Exception:
                pub_dt = None
            items.append({"title": title, "published": pub_dt})
        return items
    except Exception:
        return []


def fetch_recent_headlines(hours: int = NEWS_LOOKBACK_HOURS) -> list[str]:
    """Collect headlines from all RSS sources published in last `hours` hours."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    result = []

    for name, url in RSS_FEEDS:
        items = _fetch_rss(url)
        for it in items[:15]:
            pub = it["published"]
            if pub and pub < cutoff:
                continue   # too old
            result.append(f"[{name}] {it['title']}")

    return result[:35]  # cap at 35 headlines to keep prompt small


def analyze_with_groq(headlines: list[str]) -> dict:
    """
    Send headlines to Groq llama-3.1-8b-instant.
    Free tier: 14 400 req/day, ~200ms.
    """
    if not headlines:
        return {"sentiment": "NEUTRAL", "summary": "No recent news", "pause": False}

    if not GROQ_API_KEY:
        return {"sentiment": "NEUTRAL", "summary": "GROQ_API_KEY not set", "pause": False}

    text = "\n".join(f"• {h}" for h in headlines)

    prompt = f"""You are a US equity market analyst. Read these recent global headlines and assess their impact on US stocks (S&P500/Nasdaq).

{text}

Reply in EXACTLY this format (4 lines, SUMMARY and TRIGGER must be in Russian):
SENTIMENT: BULLISH or BEARISH or NEUTRAL
PAUSE: YES or NO
SUMMARY: [на русском, макс 15 слов — главное рыночное событие]
TRIGGER: [на русском, точная причина если PAUSE=YES, иначе "нет"]

PAUSE=YES ONLY for: market-wide circuit breaker / trading halt, active major war start, Fed EMERGENCY intervention (unscheduled), US sovereign default event, 9/11-scale attack.
PAUSE=NO for EVERYTHING else including: general bearish sentiment, geopolitics, single-stock earnings misses, inflation fears, scheduled Fed meetings, rate hike talk, uncertainty, tariffs, trade wars, normal market corrections, crypto crashes."""

    try:
        resp = _req.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model":       "llama-3.1-8b-instant",
                "messages":    [{"role": "user", "content": prompt}],
                "max_tokens":  100,
                "temperature": 0.1,
            },
            timeout=12,
        )
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        return _parse_groq(raw)

    except Exception as e:
        return {"sentiment": "NEUTRAL", "summary": f"Groq unavailable: {e}", "pause": False}


# Hard-stop keywords that MUST appear in the trigger for PAUSE=True to be valid.
# Prevents LLM false-positives (general bearish news, geopolitics, uncertainty).
# Stock-market catastrophes, not crypto ones.
_PAUSE_REQUIRED_KEYWORDS = {
    "circuit breaker", "trading halt", "торги останов", "остановка торгов",
    "war", "война", "войну", "вторжение",
    "default", "дефолт",
    "emergency", "экстренн",  # Fed emergency intervention
    "collapse", "крах", "обвал рынка",
    "теракт", "attack", "атака",
    "биржа закрыт",
}


def _parse_groq(raw: str) -> dict:
    result = {"sentiment": "NEUTRAL", "summary": "", "pause": False, "trigger": ""}
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("SENTIMENT:"):
            val = line.split(":", 1)[1].strip().upper()
            if "BULLISH" in val:   result["sentiment"] = "BULLISH"
            elif "BEARISH" in val: result["sentiment"] = "BEARISH"
        elif line.startswith("PAUSE:"):
            result["pause"] = "YES" in line.upper()
        elif line.startswith("SUMMARY:"):
            result["summary"] = line.split(":", 1)[1].strip()
        elif line.startswith("TRIGGER:"):
            t = line.split(":", 1)[1].strip()
            if t.lower() not in ("нет", "no", "-", "none", ""):
                result["trigger"] = t

    # Hard-code guard: even if LLM says PAUSE=YES, override to False unless
    # the trigger contains a verified extreme-event keyword.
    # Prevents false positives on generic bearish/geopolitical headlines.
    if result["pause"]:
        trigger_low = result["trigger"].lower()
        if not any(kw in trigger_low for kw in _PAUSE_REQUIRED_KEYWORDS):
            result["pause"] = False

    return result


def detect_major_events(headlines: list[str]) -> list[dict]:
    """
    Detect HIGH IMPACT macro events from headlines.
    Returns list of {name, direction, level (1-3), explanation} dicts.
    Max 2 events per call.
    """
    if not headlines or not GROQ_API_KEY:
        return []

    text = "\n".join(f"• {h}" for h in headlines)

    prompt = f"""Analyze these headlines for HIGH IMPACT macro events that significantly move crypto markets.

HIGH IMPACT events only: Fed rate decision, ECB decision, US CPI release, NFP jobs report, GDP surprise, major crypto regulatory ban, exchange collapse or hack >$500M.

Headlines:
{text}

If HIGH IMPACT event found, reply one line per event (max 2 events):
EVENT|[event name in Russian, max 8 words]|BULLISH or BEARISH|[1 or 2 or 3]|[market effect in Russian, max 10 words]

Impact scale: 1=moderate, 2=significant, 3=major market mover

If NO high impact events found, reply only: NONE"""

    try:
        resp = _req.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model":       "llama-3.1-8b-instant",
                "messages":    [{"role": "user", "content": prompt}],
                "max_tokens":  150,
                "temperature": 0.1,
            },
            timeout=12,
        )
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        return _parse_events(raw)
    except Exception:
        return []


def _parse_events(raw: str) -> list[dict]:
    """Parse: EVENT|name|direction|level|explanation"""
    events = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.upper() == "NONE":
            continue
        if not line.startswith("EVENT|"):
            continue
        parts = line.split("|")
        if len(parts) < 5:
            continue
        try:
            direction = parts[2].strip().upper()
            if direction not in ("BULLISH", "BEARISH"):
                direction = "NEUTRAL"
            level = int(parts[3].strip())
            events.append({
                "name":        parts[1].strip(),
                "direction":   direction,
                "level":       level,
                "explanation": parts[4].strip(),
            })
        except (ValueError, IndexError):
            continue
    return events[:2]


def fetch_headlines_with_meta(hours: int = 18) -> list[dict]:
    """
    Return [{title, source, published_utc}] from last `hours` hours.
    Sorted newest-first. Used for morning digest.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    result = []
    for name, url in RSS_FEEDS:
        items = _fetch_rss(url)
        for it in items[:20]:
            pub = it["published"]
            if pub and pub < cutoff:
                continue
            result.append({
                "title":     it["title"],
                "source":    name,
                "published": pub,
            })
    result.sort(
        key=lambda x: x["published"] or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )
    # Drop duplicate titles (same story syndicated across feeds)
    seen, deduped = set(), []
    for it in result:
        key = it["title"].strip().lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(it)
    return deduped[:50]


def get_daily_digest() -> dict:
    """
    Morning digest: fetch last 18h headlines, ask Groq to select top 5 and explain.
    Returns {items: [...], overall: str, key_theme: str}.
    Each item: {title, time_utc, direction, explanation, impact}.
    """
    raw_items = fetch_headlines_with_meta(hours=18)
    if not raw_items:
        return {"items": [], "overall": "NEUTRAL", "key_theme": "Нет новостей за последние 18 часов"}

    if not GROQ_API_KEY:
        return {"items": [], "overall": "NEUTRAL", "key_theme": "GROQ_API_KEY не задан"}

    # Build headlines text with source + time
    lines = []
    for it in raw_items[:40]:
        pub = it["published"]
        t   = pub.strftime("%H:%M UTC") if pub else "??:??"
        lines.append(f"• [{it['source']}, {t}] {it['title']}")
    headlines_text = "\n".join(lines)

    prompt = f"""Ты — аналитик криптовалютного рынка. Вот заголовки новостей за последние 18 часов.

{headlines_text}

Выбери до 5 РАЗНЫХ самых важных для крипто/финансовых рынков. КАЖДАЯ новость уникальна — НЕ повторяй одну и ту же. Если важных меньше 5 — выведи меньше строк. Для каждой — одна строка:
ITEM|[название на рус., макс 8 слов]|[время HH:MM UTC из заголовка или ?]|BULLISH или BEARISH или NEUTRAL|[объяснение на рус., макс 12 слов]|[влияние на рынок на рус., макс 8 слов]

После строк ITEM добавь одну строку:
OVERALL|BULLISH или BEARISH или NEUTRAL|[ключевая тема дня на рус., макс 10 слов]"""

    try:
        resp = _req.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type":  "application/json",
            },
            json={
                "model":       "llama-3.1-8b-instant",
                "messages":    [{"role": "user", "content": prompt}],
                "max_tokens":  550,
                "temperature": 0.2,
            },
            timeout=25,
        )
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        return _parse_digest(raw)
    except Exception as e:
        return {"items": [], "overall": "NEUTRAL", "key_theme": f"Ошибка Groq: {e}"}


def _parse_digest(raw: str) -> dict:
    """Parse ITEM|...|...|...|...|... lines + OVERALL|...|..."""
    items     = []
    seen      = set()
    overall   = "NEUTRAL"
    key_theme = ""
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("ITEM|"):
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 6:
                try:
                    title_key = parts[1].strip().lower()
                    if not title_key or title_key in seen:
                        continue            # skip blank/duplicate titles
                    seen.add(title_key)
                    direction = parts[3].upper()
                    if direction not in ("BULLISH", "BEARISH", "NEUTRAL"):
                        direction = "NEUTRAL"
                    items.append({
                        "title":       parts[1],
                        "time_utc":    parts[2],
                        "direction":   direction,
                        "explanation": parts[4],
                        "impact":      parts[5],
                    })
                except (IndexError, ValueError):
                    continue
        elif line.startswith("OVERALL|"):
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 3:
                val = parts[1].upper()
                if val in ("BULLISH", "BEARISH", "NEUTRAL"):
                    overall = val
                key_theme = parts[2]
    return {"items": items[:5], "overall": overall, "key_theme": key_theme}


def get_market_news() -> dict:
    """
    Main entry point. Fetch headlines → analyze → return context dict.
    Always succeeds (errors return NEUTRAL).
    """
    headlines = fetch_recent_headlines()
    analysis  = analyze_with_groq(headlines)
    return {
        "sentiment":       analysis["sentiment"],
        "summary":         analysis["summary"],
        "pause":           analysis["pause"],
        "trigger":         analysis.get("trigger", ""),
        "headline_count":  len(headlines),
    }


def generate_weekly_commentary(stats: dict) -> str:
    """Send weekly trade stats to Groq Llama 3.3-70b, get Russian narrative commentary."""
    if not GROQ_API_KEY:
        return ""

    top3_str = ", ".join(
        f"{s}({w}W/{sl}SL)" for s, w, sl in stats.get("top3", [])
    ) or "нет данных"
    trend_str = " | ".join(
        f"{t}: {int(wr)}% WR" for t, wr in stats.get("trend_wr", {}).items()
    ) or "мало данных"
    best  = stats.get("best_trade")
    worst = stats.get("worst_trade")
    n_sent = stats.get("n_sent", 0)
    n_rej  = stats.get("n_rejected", 0)
    sent_tp1_rate = stats.get("sent_tp1_rate", 0)
    rej_tp1_rate  = stats.get("rej_tp1_rate", 0)
    best_str  = f"{best['symbol']} {best['r']:+.2f}R"  if best  else "нет"
    worst_str = f"{worst['symbol']} {worst['r']:+.2f}R" if worst else "нет"
    n_total   = stats.get("n_total", 0)
    wr        = stats.get("wr", 0)
    total_r   = stats.get("total_r", 0)
    n_tp2     = stats.get("n_tp2", 0)
    n_sl      = stats.get("n_sl", 0)
    n_exp     = stats.get("n_exp", 0)

    ai_verdict = ""
    if n_rej >= 5:
        if rej_tp1_rate > sent_tp1_rate + 10:
            ai_verdict = f"ИИ СЛИШКОМ СТРОГ — {rej_tp1_rate}% отклонённых сделок всё равно дошли до TP1."
        elif rej_tp1_rate < sent_tp1_rate - 10:
            ai_verdict = f"ИИ фильтрует хорошо — отклонённые дошли до TP1 только в {rej_tp1_rate}% против {sent_tp1_rate}% у одобренных."
        else:
            ai_verdict = f"ИИ работает нейтрально — отклонённые ({rej_tp1_rate}%) vs одобренные ({sent_tp1_rate}%) TP1 схожи."

    prompt = f"""Ты аналитик крипто-трейдинга. Вот статистика автоматического торгового бота за последние 7 дней.

ТОРГОВЛЯ:
- Завершённых сделок: {n_total}
- Win rate: {wr}%
- Чистый R за неделю: {total_r:+.1f}R
- TP2 достигнуто: {n_tp2} | SL: {n_sl} | Истекло: {n_exp}
- Лучшая сделка: {best_str}
- Худшая сделка: {worst_str}
- Топ тикеры: {top3_str}
- По структуре тренда: {trend_str}

AI ФИЛЬТР:
- Одобрено: {n_sent} сделок (TP1 дошло {sent_tp1_rate}%)
- Отклонено: {n_rej} сетапов (TP1 дошло бы {rej_tp1_rate}%)
- Вывод: {ai_verdict}

Напиши разбор недели на РУССКОМ. Максимум 5 предложений. Без воды. Что сработало, что нет, на что смотреть на следующей неделе."""

    try:
        resp = _req.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type":  "application/json",
            },
            json={
                "model":       "llama-3.3-70b-versatile",
                "messages":    [{"role": "user", "content": prompt}],
                "max_tokens":  350,
                "temperature": 0.4,
            },
            timeout=25,
        )
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"(комментарий недоступен: {e})"


# ── Economic calendar (ForexFactory weekly XML — free, no key) ─────────────────

_FF_CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"

try:
    from zoneinfo import ZoneInfo
    _ET_TZ = ZoneInfo("America/New_York")   # ForexFactory times are US Eastern
except Exception:
    _ET_TZ = timezone(timedelta(hours=-5))  # fallback EST

# 6-hour cache — calendar barely changes intraday, avoid re-fetching every scan.
_calendar_cache: dict = {"fetched_at": 0.0, "events": []}


def _parse_ff_event(ev) -> dict | None:
    """Parse one <event> node → {title, country, when_utc} or None if no usable time."""
    title   = (ev.findtext("title")   or "").strip()
    country = (ev.findtext("country") or "").strip()
    impact  = (ev.findtext("impact")  or "").strip()
    date_s  = (ev.findtext("date")    or "").strip()   # MM-DD-YYYY
    time_s  = (ev.findtext("time")    or "").strip()   # e.g. "8:30am"

    if impact.lower() != "high" or not title or not date_s:
        return None
    # Skip non-scheduled rows
    if not time_s or time_s.lower() in ("all day", "tentative", "day 1", "day 2"):
        return None

    try:
        t = time_s.lower().replace(" ", "")
        ampm = t[-2:]
        hm   = t[:-2]
        hh, mm = hm.split(":")
        hour, minute = int(hh), int(mm)
        if ampm == "pm" and hour != 12:
            hour += 12
        if ampm == "am" and hour == 12:
            hour = 0
        mo, da, yr = date_s.split("-")
        import datetime as _dt
        when_et = _dt.datetime(int(yr), int(mo), int(da), hour, minute, tzinfo=_ET_TZ)
        when_utc = when_et.astimezone(timezone.utc)
    except Exception:
        return None

    return {"title": title, "country": country, "when_utc": when_utc}


def get_upcoming_high_impact_events(within_hours: float = 3.0) -> list[dict]:
    """
    High-impact scheduled macro events in the next `within_hours`.
    Returns list of {title, country, when_utc, hours_until} sorted by soonest.
    Cached 6h. Never raises — returns [] on any failure.
    """
    import time as _t
    now = _t.time()
    if now - _calendar_cache["fetched_at"] > 6 * 3600:
        try:
            resp = _req.get(_FF_CALENDAR_URL, timeout=8,
                            headers={"User-Agent": "Mozilla/5.0 CryptoBot/1.0"})
            events = []
            if resp.status_code == 200:
                root = ET.fromstring(resp.content)
                for ev in root.iter("event"):
                    parsed = _parse_ff_event(ev)
                    if parsed:
                        events.append(parsed)
            _calendar_cache["events"] = events
            _calendar_cache["fetched_at"] = now
        except Exception:
            _calendar_cache["fetched_at"] = now  # don't hammer on failure

    now_utc = datetime.now(timezone.utc)
    upcoming = []
    for ev in _calendar_cache["events"]:
        delta_h = (ev["when_utc"] - now_utc).total_seconds() / 3600
        if 0 <= delta_h <= within_hours:
            upcoming.append({**ev, "hours_until": round(delta_h, 1)})
    upcoming.sort(key=lambda x: x["hours_until"])
    return upcoming


# ── "Новости на сегодня" — full day digest with forecast/result ───────────────

_day_cache: dict = {"fetched_at": 0.0, "raw": b""}


def _parse_ff_event_full(ev) -> dict | None:
    """Parse one <event> → rich dict (High/Medium impact, scheduled time)."""
    title    = (ev.findtext("title")    or "").strip()
    country  = (ev.findtext("country")  or "").strip()
    impact   = (ev.findtext("impact")   or "").strip()
    date_s   = (ev.findtext("date")     or "").strip()   # MM-DD-YYYY
    time_s   = (ev.findtext("time")     or "").strip()   # "8:30am"
    forecast = (ev.findtext("forecast") or "").strip()
    previous = (ev.findtext("previous") or "").strip()
    actual   = (ev.findtext("actual")   or "").strip()

    if impact.lower() not in ("high", "medium") or not title or not date_s:
        return None
    try:
        mo, da, yr = date_s.split("-")
        ev_date = datetime(int(yr), int(mo), int(da)).date()
    except Exception:
        return None

    when_utc, all_day = None, True
    ts = time_s.lower().replace(" ", "")
    if ts and ts not in ("allday", "tentative", "day1", "day2"):
        try:
            ampm, hm = ts[-2:], ts[:-2]
            hh, mm = hm.split(":")
            hour, minute = int(hh), int(mm)
            if ampm == "pm" and hour != 12:
                hour += 12
            if ampm == "am" and hour == 12:
                hour = 0
            import datetime as _dt
            when_et  = _dt.datetime(int(yr), int(mo), int(da), hour, minute, tzinfo=_ET_TZ)
            when_utc = when_et.astimezone(timezone.utc)
            all_day  = False
        except Exception:
            all_day = True

    return {
        "title": title, "country": country, "impact": impact.lower(),
        "ev_date": ev_date, "when_utc": when_utc, "all_day": all_day,
        "forecast": forecast, "previous": previous, "actual": actual,
    }


def get_day_events(max_events: int = 10) -> dict:
    """
    High/Medium-impact macro events for *today* (US Eastern calendar day).
    If today is Sat/Sun → rolls to next Monday.
    Returns {"date": <date>, "weekend_rolled": bool, "events": [...]} where each
    event = {title, country, impact, when_utc, all_day, forecast, previous,
             actual, passed}. Cached 1h. Never raises — returns empty on failure.
    """
    import time as _t
    now = _t.time()
    if now - _day_cache["fetched_at"] > 3600 or not _day_cache["raw"]:
        try:
            resp = _req.get(_FF_CALENDAR_URL, timeout=8,
                            headers={"User-Agent": "Mozilla/5.0 CryptoBot/1.0"})
            if resp.status_code == 200:
                _day_cache["raw"] = resp.content
            _day_cache["fetched_at"] = now
        except Exception:
            _day_cache["fetched_at"] = now

    # Use Riga timezone for "today" — bot users are in EU, not US Eastern.
    # ET is only used for parsing event times from the feed, not for date selection.
    try:
        from zoneinfo import ZoneInfo as _ZI
        _RIGA_TZ = _ZI("Europe/Riga")
    except Exception:
        _RIGA_TZ = timezone(timedelta(hours=3))
    now_local = datetime.now(_RIGA_TZ)
    target  = now_local.date()
    rolled  = False
    if now_local.weekday() >= 5:                    # Sat=5, Sun=6 → next Monday
        target += timedelta(days=7 - now_local.weekday())
        rolled  = True

    out = {"date": target, "weekend_rolled": rolled, "events": []}
    if not _day_cache["raw"]:
        return out

    try:
        root = ET.fromstring(_day_cache["raw"])
    except Exception:
        return out

    now_utc = datetime.now(timezone.utc)
    parsed = []
    for ev in root.iter("event"):
        p = _parse_ff_event_full(ev)
        if not p or p["ev_date"] != target:
            continue
        if p["all_day"] or p["when_utc"] is None:
            passed = (target < now_et.date())
        else:
            passed = p["when_utc"] < now_utc
        parsed.append({**p, "passed": passed})

    # Chronological — earliest first; all-day (no time) sink to the end
    far = datetime.max.replace(tzinfo=timezone.utc)
    parsed.sort(key=lambda e: e["when_utc"] or far)
    out["events"] = parsed[:max_events]
    return out
