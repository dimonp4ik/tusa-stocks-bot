"""
Dukascopy historical data client — DEEP history for backtests only.

Free public feed, no API key/registration (dukascopy-python package).
Covers 16 of the 26 pool tickers with YEARS of 15m history (AAPL from 2020),
vs ~4 months on the OKX swaps (listed 2026-03). The live bot still trades and
analyses the OKX swap — this source exists purely to give filter validation
enough data for train/test splits.

Known basis differences vs the OKX feed (acceptable for SMC structure work):
  - stock/ETF CFDs carry bid-side pricing and a small basis (AAPL ~0.2%)
  - volume units differ (CFD lots vs swap contracts) — all volume filters in
    signal_filter are RATIOS (current/median), so units cancel out
  - equity CFD bars exist only during exchange hours; metals/oil nearly 24/5

Interface mirrors backtest.fetch_history: returns dict-of-lists candles
(time seconds, oldest-first) with a local pickle cache.
"""

from __future__ import annotations

import pickle
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import dukascopy_python as _dp
from dukascopy_python import instruments as _di

CACHE_DIR = Path(__file__).resolve().parent.parent / "backtest_cache_duka"
CACHE_TTL_SEC = 24 * 3600  # historical data doesn't churn — 1 day is plenty

# internal symbol → (dukascopy instrument, class)
# class drives the overshoot multiplier when converting "N bars" → date range:
# equities only print ~26 15m bars/session-day (~19% of wall-clock 15m slots),
# metals/energy ~23h/5d (~68%).
INSTRUMENT_MAP: dict[str, tuple[str, str]] = {
    "AAPLUSDT":  (_di.INSTRUMENT_US_AAPL_US_USD, "equity"),
    "AMZNUSDT":  (_di.INSTRUMENT_US_AMZN_US_USD, "equity"),
    "GOOGLUSDT": (_di.INSTRUMENT_US_GOOGL_US_USD, "equity"),
    "METAUSDT":  (_di.INSTRUMENT_US_FB_US_USD, "equity"),   # META kept as FB at Dukascopy
    "MSFTUSDT":  (_di.INSTRUMENT_US_MSFT_US_USD, "equity"),
    "NVDAUSDT":  (_di.INSTRUMENT_US_NVDA_US_USD, "equity"),
    "TSLAUSDT":  (_di.INSTRUMENT_US_TSLA_US_USD, "equity"),
    "INTCUSDT":  (_di.INSTRUMENT_US_INTC_US_USD, "equity"),
    "MRVLUSDT":  (_di.INSTRUMENT_US_MRVL_US_USD, "equity"),
    "MUUSDT":    (_di.INSTRUMENT_US_MU_US_USD, "equity"),
    "QQQUSDT":   (_di.INSTRUMENT_ETF_CFD_US_QQQ_US_USD, "equity"),
    "SPYUSDT":   (_di.INSTRUMENT_ETF_CFD_US_SPY_US_USD, "equity"),
    "XAUUSDT":   (_di.INSTRUMENT_FX_METALS_XAU_USD, "cmd"),
    "XAGUSDT":   (_di.INSTRUMENT_FX_METALS_XAG_USD, "cmd"),
    "CLUSDT":    (_di.INSTRUMENT_CMD_ENERGY_E_LIGHT, "cmd"),  # WTI — NOT CL.US (Colgate!)
    "BZUSDT":    (_di.INSTRUMENT_CMD_ENERGY_E_BRENT, "cmd"),
}

_INTERVAL_MAP = {
    "15min": (_dp.INTERVAL_MIN_15, 15 * 60),
    "1hour": (_dp.INTERVAL_HOUR_1, 3600),
    "4hour": (_dp.INTERVAL_HOUR_4, 4 * 3600),
    "1d":    (_dp.INTERVAL_DAY_1, 86400),
}

# wall-clock overshoot so a "last N bars" request spans enough calendar days
_OVERSHOOT = {"equity": 6.5, "cmd": 1.8}


def covers(symbol: str) -> bool:
    return symbol.upper() in INSTRUMENT_MAP


def _cache_path(symbol: str, interval: str, count: int, end_date_ms: int | None) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_end{end_date_ms}" if end_date_ms else ""
    return CACHE_DIR / f"{symbol}_{interval}_{count}{suffix}.pkl"


def fetch_history_duka(
    symbol: str,
    interval: str,
    interval_sec: int,
    count: int,
    *,
    refresh_cache: bool = False,
    end_date_ms: int | None = None,
) -> dict[str, list]:
    """Last `count` candles for `symbol` from Dukascopy, oldest-first."""
    symbol = symbol.upper()
    if symbol not in INSTRUMENT_MAP:
        raise ValueError(f"{symbol} not covered by Dukascopy map")

    path = _cache_path(symbol, interval, count, end_date_ms)
    if not refresh_cache and path.exists():
        if time.time() - path.stat().st_mtime < CACHE_TTL_SEC:
            try:
                with path.open("rb") as f:
                    cached = pickle.load(f)
                if cached and cached.get("close"):
                    return cached
            except Exception:
                pass

    instrument, klass = INSTRUMENT_MAP[symbol]
    dp_interval, base_sec = _INTERVAL_MAP.get(interval, (None, None))
    if dp_interval is None:
        raise ValueError(f"unsupported interval {interval}")

    end = (
        datetime.fromtimestamp(end_date_ms / 1000, tz=timezone.utc)
        if end_date_ms else datetime.now(timezone.utc)
    )
    span_sec = count * interval_sec * _OVERSHOOT[klass]
    start = end - timedelta(seconds=span_sec)

    df = _dp.fetch(instrument, dp_interval, _dp.OFFER_SIDE_BID, start, end)
    if df is None or df.empty:
        raise ValueError(f"No Dukascopy data for {symbol} {interval}")

    df = df.tail(count)
    data = {
        "time":   [int(ts.timestamp()) for ts in df.index],
        "open":   [float(x) for x in df["open"]],
        "high":   [float(x) for x in df["high"]],
        "low":    [float(x) for x in df["low"]],
        "close":  [float(x) for x in df["close"]],
        "volume": [float(x) for x in df["volume"]],
    }

    with path.open("wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    return data
