"""
US stock market session gate (NYSE/Nasdaq).

X-Perps on OKX EU trade 24/7, but the UNDERLYING stock only moves during the
US session — off-session X-Perp candles are thin market-maker drift and would
poison SMC signals (same reason the crypto bot excludes stock swaps entirely).

Regular session: 09:30–16:00 America/New_York (DST-aware via zoneinfo).
Half days close at 13:00 ET. Exchange holidays computed by rule — no yearly
hardcoded list to go stale.

Public API:
    is_market_open(dt=None)  -> bool
    session_phase(dt=None)   -> "OPEN" | "MIDDAY" | "CLOSE" | "OFF"
    next_open(dt=None)       -> datetime (ET)
    next_close(dt=None)      -> datetime (ET)
    status_text()            -> ready-to-send Telegram status block (Russian)
"""

from datetime import datetime, date, time as dtime, timedelta
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

SESSION_OPEN = dtime(9, 30)
SESSION_CLOSE = dtime(16, 0)
HALF_DAY_CLOSE = dtime(13, 0)

# Intraday phases (classic intraday regimes — used as session labels in filters)
#   OPEN   09:30–11:00  high volume/volatility, gap plays
#   MIDDAY 11:00–14:30  lull, lower conviction
#   CLOSE  14:30–16:00  institutional rebalancing, volume returns
PHASE_OPEN_END = dtime(11, 0)
PHASE_CLOSE_START = dtime(14, 30)


# ── Holiday rules (NYSE) ──────────────────────────────────────────────────────

def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """n-th <weekday> (0=Mon) of a month; n=-1 → last."""
    if n > 0:
        d = date(year, month, 1)
        offset = (weekday - d.weekday()) % 7
        return d + timedelta(days=offset + 7 * (n - 1))
    # last weekday of month
    if month == 12:
        d = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        d = date(year, month + 1, 1) - timedelta(days=1)
    offset = (d.weekday() - weekday) % 7
    return d - timedelta(days=offset)


def _easter(year: int) -> date:
    """Anonymous Gregorian algorithm."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _observed(d: date) -> date:
    """NYSE observation: Sat holiday → Friday before, Sun → Monday after."""
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


def market_holidays(year: int) -> set:
    """Full NYSE closures for a year."""
    easter = _easter(year)
    return {
        _observed(date(year, 1, 1)),                # New Year's Day
        _nth_weekday(year, 1, 0, 3),                # MLK Day — 3rd Mon Jan
        _nth_weekday(year, 2, 0, 3),                # Presidents' Day — 3rd Mon Feb
        easter - timedelta(days=2),                 # Good Friday
        _nth_weekday(year, 5, 0, -1),               # Memorial Day — last Mon May
        _observed(date(year, 6, 19)),               # Juneteenth
        _observed(date(year, 7, 4)),                # Independence Day
        _nth_weekday(year, 9, 0, 1),                # Labor Day — 1st Mon Sep
        _nth_weekday(year, 11, 3, 4),               # Thanksgiving — 4th Thu Nov
        _observed(date(year, 12, 25)),              # Christmas
    }


def half_days(year: int) -> set:
    """13:00 ET early closes: July 3 (weekday, when Jul 4 is a weekday holiday),
    day after Thanksgiving, Christmas Eve (weekday)."""
    out = set()
    jul3 = date(year, 7, 3)
    # half-day only when Jul 4 itself is a weekday (else Jul 3 is the observed
    # full holiday or a weekend)
    if jul3.weekday() < 5 and date(year, 7, 4).weekday() < 5:
        out.add(jul3)
    out.add(_nth_weekday(year, 11, 3, 4) + timedelta(days=1))  # Black Friday
    dec24 = date(year, 12, 24)
    if dec24.weekday() < 5:
        out.add(dec24)
    return out


_HOLIDAY_NAMES = [
    ("Новый год", lambda y: _observed(date(y, 1, 1))),
    ("День Мартина Лютера Кинга", lambda y: _nth_weekday(y, 1, 0, 3)),
    ("День Президентов", lambda y: _nth_weekday(y, 2, 0, 3)),
    ("Страстная пятница", lambda y: _easter(y) - timedelta(days=2)),
    ("День поминовения", lambda y: _nth_weekday(y, 5, 0, -1)),
    ("Juneteenth", lambda y: _observed(date(y, 6, 19))),
    ("День независимости", lambda y: _observed(date(y, 7, 4))),
    ("День труда", lambda y: _nth_weekday(y, 9, 0, 1)),
    ("День благодарения", lambda y: _nth_weekday(y, 11, 3, 4)),
    ("Рождество", lambda y: _observed(date(y, 12, 25))),
]


def next_holiday(dt=None):
    """(name, date) of the nearest upcoming NYSE closure."""
    now = (dt or datetime.now(ET)).astimezone(ET).date()
    upcoming = []
    for year in (now.year, now.year + 1):
        for name, fn in _HOLIDAY_NAMES:
            d = fn(year)
            if d >= now:
                upcoming.append((d, name))
    d, name = min(upcoming)
    return name, d


# ── Session checks ────────────────────────────────────────────────────────────

def _close_time_for(d: date) -> dtime:
    return HALF_DAY_CLOSE if d in half_days(d.year) else SESSION_CLOSE


def _is_trading_day(d: date) -> bool:
    return d.weekday() < 5 and d not in market_holidays(d.year)


def is_market_open(dt=None) -> bool:
    now = (dt or datetime.now(ET)).astimezone(ET)
    d = now.date()
    if not _is_trading_day(d):
        return False
    return SESSION_OPEN <= now.time() < _close_time_for(d)


def session_phase(dt=None) -> str:
    """Intraday phase label — the stock-market analogue of crypto's
    ASIA/LONDON/NY session tags. 'OFF' outside the session."""
    now = (dt or datetime.now(ET)).astimezone(ET)
    if not is_market_open(now):
        return "OFF"
    t = now.time()
    if t < PHASE_OPEN_END:
        return "OPEN"
    if t < PHASE_CLOSE_START:
        return "MIDDAY"
    return "CLOSE"


def next_open(dt=None) -> datetime:
    now = (dt or datetime.now(ET)).astimezone(ET)
    d = now.date()
    # today, if the bell hasn't rung yet
    if _is_trading_day(d) and now.time() < SESSION_OPEN:
        return datetime.combine(d, SESSION_OPEN, tzinfo=ET)
    d += timedelta(days=1)
    for _ in range(15):  # longest gap is a long weekend + holiday
        if _is_trading_day(d):
            return datetime.combine(d, SESSION_OPEN, tzinfo=ET)
        d += timedelta(days=1)
    raise RuntimeError("no trading day found in 15 days — holiday rules broken")


def next_close(dt=None) -> datetime:
    now = (dt or datetime.now(ET)).astimezone(ET)
    if is_market_open(now):
        return datetime.combine(now.date(), _close_time_for(now.date()), tzinfo=ET)
    nxt = next_open(now)
    return datetime.combine(nxt.date(), _close_time_for(nxt.date()), tzinfo=ET)


def _fmt_delta(td: timedelta) -> str:
    total_min = int(td.total_seconds() // 60)
    h, m = divmod(total_min, 60)
    if h >= 24:
        d, h = divmod(h, 24)
        return f"{d} д {h} ч"
    return f"{h} ч {m:02d} мин"


def status_text() -> str:
    """Telegram-ready market status block (Russian, HTML parse mode)."""
    now = datetime.now(ET)
    hol_name, hol_date = next_holiday(now)
    if is_market_open(now):
        close_at = next_close(now)
        phase = {"OPEN": "открытие (высокая волатильность)",
                 "MIDDAY": "середина дня (затишье)",
                 "CLOSE": "закрытие (объёмы растут)"}[session_phase(now)]
        early = " (короткий день)" if now.date() in half_days(now.year) else ""
        lines = [
            "🟢 <b>Рынок США: ОТКРЫТ</b>" + early,
            f"Фаза: {phase}",
            f"До закрытия: {_fmt_delta(close_at - now)}",
        ]
    else:
        open_at = next_open(now)
        lines = [
            "🔴 <b>Рынок США: ЗАКРЫТ</b>",
            f"Откроется: {open_at.strftime('%d.%m %H:%M')} ET",
            f"До открытия: {_fmt_delta(open_at - now)}",
        ]
    lines.append(f"Ближайший праздник: {hol_name} — {hol_date.strftime('%d.%m.%Y')}")
    return "\n".join(lines)
