"""
Earnings-date blackout — the one gap SMC can't price.

An earnings report gaps a stock 5-10% overnight; no 15m structure setup
survives that coin flip. Rule: no NEW signals for a ticker that reports
TODAY or TOMORROW (ET). Covers both after-hours reports (gap hits next
open) and pre-market reports (gap already brewing). Open positions are
NOT touched — the 24/7 monitor manages those.

Source: Nasdaq public earnings calendar (no API key). One request per
date, covers every US ticker at once; cached 6h. Fail-open: if the API
is down we trade normally and log a warning — a missing blackout is a
risk, but freezing the whole bot on a calendar outage is worse.

Non-equity pool members (SPY/QQQ/EWY ETFs, XAU/XAG/CL/BZ commodities)
never report earnings → never blacked out. Foreign listings the Nasdaq
calendar misses (SAMSUNG/SKHYNIX) are a known gap.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import requests

_log = logging.getLogger(__name__)
_ET = ZoneInfo("America/New_York")

# tickers that can never have earnings — skip the lookup entirely
# Funds and commodities only — these have no earnings date, so the calendar
# lookup is skipped for them. A ticker must NOT be listed here unless it is
# genuinely not a company: anything left in by mistake trades straight through
# its own earnings report, which is the exact gap risk this module exists to
# avoid. Verify with Nasdaq's quote API before adding
# (api.nasdaq.com/api/quote/<SYM>/info?assetclass=etf -> "...ETF" name).
# 2026-07-22: CBRS (Cerebras Systems) and SPCX (Space Exploration Technologies)
# were wrongly listed here and have been removed — both are real companies.
_NON_EQUITY_BASES = {
    "SPY",    # SPDR S&P 500 ETF
    "QQQ",    # Invesco QQQ Trust
    "EWY",    # iShares MSCI South Korea ETF
    "SOXL",   # Direxion Daily Semiconductor Bull 3X ETF
    "DRAM",   # Roundhill Memory ETF
    "XAU", "XAG", "CL", "BZ",   # gold, silver, WTI, Brent
}

_CACHE_TTL = 6 * 3600
_cache: dict[str, tuple[float, set]] = {}  # date_str -> (fetched_at, {symbols})


def _base_of(symbol: str) -> str:
    s = str(symbol or "").upper()
    for q in ("USDT", "USDC"):
        if s.endswith(q):
            return s[: -len(q)]
    return s


def _reporters_on(date_str: str) -> set:
    """Set of US tickers reporting on date_str (YYYY-MM-DD). Cached; empty on failure."""
    now = time.time()
    hit = _cache.get(date_str)
    if hit and now - hit[0] < _CACHE_TTL:
        return hit[1]
    try:
        resp = requests.get(
            "https://api.nasdaq.com/api/calendar/earnings",
            params={"date": date_str},
            headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
            timeout=12,
        )
        rows = ((resp.json().get("data") or {}).get("rows")) or []
        symbols = {str(r.get("symbol", "")).upper() for r in rows if r.get("symbol")}
        _cache[date_str] = (now, symbols)
        return symbols
    except Exception as e:
        _log.warning(f"Earnings calendar fetch failed for {date_str}: {e}")
        stale = _cache.get(date_str)
        return stale[1] if stale else set()


def is_earnings_blackout(symbol: str) -> tuple[bool, str]:
    """(blackout?, reason). True when the ticker reports today or tomorrow (ET)."""
    base = _base_of(symbol)
    if base in _NON_EQUITY_BASES:
        return False, ""
    today = datetime.now(_ET).date()
    for label, d in (("сегодня", today), ("завтра", today + timedelta(days=1))):
        if base in _reporters_on(d.isoformat()):
            return True, f"отчёт {label} ({d.strftime('%d.%m')})"
    return False, ""
