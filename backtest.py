"""
Fast SMC backtest runner.

Usage:
    python backtest.py
    python backtest.py --symbols BTC-USDT,ETH-USDT --workers 4
    python backtest.py --stride 4 --export-trades backtest_trades.csv

The strategy logic still comes from src.signal_filter.analyze_coin_smc.
This file speeds up the runner around it:
  - process-level parallelism by symbol
  - zero-copy candle windows
  - exact cheap prefilter for BOS + volume before the expensive SMC stack
  - time-aligned 1h/4h snapshots
  - direct bracket simulation without per-bar future dict copies
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import pickle
import sys
import time
import types
from bisect import bisect_right
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable
from urllib.parse import urlencode  # noqa: F401 (kept for potential future use)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Backtests should run in a clean research environment even when optional app
# dependencies are not installed. The real bot still uses python-dotenv when
# present; this only lets config.py import with a no-op load_dotenv fallback.
if importlib.util.find_spec("dotenv") is None:
    dotenv_stub = types.ModuleType("dotenv")
    dotenv_stub.load_dotenv = lambda *args, **kwargs: None
    sys.modules["dotenv"] = dotenv_stub

from config import (
    EXTENSION_FRESH_THRESHOLD, EXTENSION_FRESH_SIZE_MULT,  # noqa: E402
    OPEN_SESSION_SIZE_MULT, OPEN_VOL_MIN,  # noqa: E402
    OFF_SESSION_SIZE_MULT,  # noqa: E402
    ORDERLY_EFF_MIN, ORDERLY_ATR_MAX, ORDERLY_EXT_MIN, ORDERLY_SIZE_MULT,  # noqa: E402
    SIZE_MULT_MAX,  # noqa: E402
    BACKTEST_CANDLES,
    BACKTEST_FEE_RATE,
    BACKTEST_SLIPPAGE_RATE,
    BACKTEST_TP_WINDOW,
    SIGNAL_EXPIRY_HOURS,
    SIGNAL_EXPIRY_MAX_DAYS,
    BLOCKED_SYMBOLS,
    BLOCK_STABLE_BASES,
    KLINES_1H_INTERVAL_SEC,
    KLINES_4H_INTERVAL_SEC,
    KLINES_INTERVAL_SEC,
    MAX_SAME_DIRECTION_POSITIONS,
    SIGNAL_COOLDOWN_HOURS,
    KILL_SWITCH_SL_STREAK,
    LEVERAGED_TOKEN_SUFFIXES,
    QUOTE_ASSET,
    RISK_MAX_PCT,
    RISK_MIN_PCT,
    SL_ATR_BUFFER,
    SMC_BOS_MIN_VOLUME,
    SMC_SWING_LOOKBACK,
    TIMEFRAME_1H_KUCOIN,
    TIMEFRAME_4H_KUCOIN,
    TIMEFRAME_KUCOIN,
    TP1_R_MULT,
    TP2_R_MULT,
    TRAIL_ATR_MULT,
    TP1_CLOSE_FRAC,
    EXIT_PROFILE,
    POST_TP1_STRONG_TRAIL_ATR_MULT,
    POST_TP1_WEAK_TRAIL_ATR_MULT,
    POST_TP1_STRONG_CLOSE_PROGRESS,
    POST_TP1_STRONG_WICK_PROGRESS,
    POST_TP1_WEAK_CLOSE_PROGRESS,
    MIN_24H_QUOTE_VOLUME_USDT,
    OFF_SESSION_SIGNALS,
    EXTENDED_SESSION_WINDOWS,
    MARKET_PROXY_SYMBOL,
)
from datetime import datetime as _datetime, timezone as _tz  # noqa: E402
from src.signal_filter import analyze_coin_smc  # noqa: E402
from src.knn_analog import knn_direction_score  # noqa: E402
from zoneinfo import ZoneInfo as _ZoneInfo  # noqa: E402
from src.market_hours import is_market_open as _is_market_open  # noqa: E402


def _in_extended_window_bt(dt_utc) -> bool:
    """Mirror of main._in_extended_window — see EXTENDED_SESSION_WINDOWS.

    Duplicated rather than imported because importing main from the backtest
    would drag in the scheduler and the Telegram client. Keep the two in sync.
    """
    if not EXTENDED_SESSION_WINDOWS:
        return False
    try:
        et = dt_utc.astimezone(_ZoneInfo("America/New_York"))
        if et.weekday() >= 5:
            return False
        return any(lo <= et.hour < hi for lo, hi in EXTENDED_SESSION_WINDOWS)
    except Exception:
        return False


PROJECT_DIR = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_DIR / "backtest_cache"
CACHE_TTL_SEC = 2 * 3600

# OKX API — SAME source the live bot analyses (stock/ETF/commodity USDT swaps).
# history-candles reaches back to ~2026-03 (when these swaps listed); ~96 15m
# candles/day (the perp trades 24/7, thin overnight). Session gating is applied
# at entry time so the backtest only takes trades the live bot could take.
OKX_HOSTS = ["https://www.okx.com", "https://aws.okx.com"]
OKX_PAGE_LIMIT = 300   # OKX max candles per history-candles request

# internal interval → OKX bar string (mirrors src.binance_client.TIMEFRAME_MAP)
OKX_INTERVAL_MAP = {
    "15min": "15m", "1hour": "1H", "4hour": "4H",
    "15m": "15m", "1H": "1H", "4H": "4H",
    "1d": "1Dutc", "1Dutc": "1Dutc",
}

WINDOW_15M = 300
WINDOW_1H = 90
WINDOW_4H = 50
DEFAULT_WARMUP = 50

# Fixed symbol set: reproducible A/B runs. Internal format (no dashes) — the
# 26 non-crypto X-Perp-tradable tickers, resolved to "<BASE>-USDT-SWAP" instIds
# for the OKX analysis feed. Override with --symbols or env BACKTEST_SYMBOLS.
BACKTEST_SYMBOLS = [
    "AAPLUSDT", "AMZNUSDT", "GOOGLUSDT", "METAUSDT", "MSFTUSDT",
    "NVDAUSDT", "TSLAUSDT", "INTCUSDT", "MRVLUSDT", "MSTRUSDT",
    "MUUSDT", "SNDKUSDT", "SOXLUSDT", "SPCXUSDT", "QQQUSDT",
    "SPYUSDT", "XAUUSDT", "XAGUSDT", "CLUSDT", "BZUSDT",
    "CRCLUSDT", "CBRSUSDT", "DRAMUSDT", "EWYUSDT", "SAMSUNGUSDT",
    "SKHYNIXUSDT",
]


def _inst_id(symbol: str) -> str:
    """Internal 'AAPLUSDT' → OKX analysis-feed instId 'AAPL-USDT-SWAP'."""
    s = symbol.upper()
    base = s[:-len(QUOTE_ASSET)] if s.endswith(QUOTE_ASSET) else s
    return f"{base}-{QUOTE_ASSET}-SWAP"


class Window:
    """Read-only list-like view over base[start:stop] without copying."""

    __slots__ = ("_base", "_start", "_stop")

    def __init__(self, base: list, start: int = 0, stop: int | None = None):
        self._base = base
        self._start = max(0, start)
        self._stop = len(base) if stop is None else max(self._start, min(stop, len(base)))

    def __len__(self) -> int:
        return self._stop - self._start

    def __iter__(self):
        base = self._base
        for i in range(self._start, self._stop):
            yield base[i]

    def __getitem__(self, idx):
        n = len(self)
        if isinstance(idx, slice):
            start, stop, step = idx.indices(n)
            base = self._base
            offset = self._start
            return [base[offset + i] for i in range(start, stop, step)]
        if idx < 0:
            idx += n
        if idx < 0 or idx >= n:
            raise IndexError(idx)
        return self._base[self._start + idx]

    def materialize(self) -> list:
        return self._base[self._start:self._stop]


def candle_window(candles: dict[str, list], start: int, stop: int) -> dict[str, Window]:
    return {k: Window(v, start, stop) for k, v in candles.items()}


def candle_slice(candles: dict[str, list], start: int, stop: int) -> dict[str, list]:
    return {k: v[start:stop] for k, v in candles.items()}


def parse_symbols(value: str | None) -> list[str]:
    if value:
        return [s.strip().upper() for s in value.split(",") if s.strip()]
    env_symbols = os.getenv("BACKTEST_SYMBOLS", "").strip()
    if env_symbols:
        return [s.strip().upper() for s in env_symbols.split(",") if s.strip()]
    return list(BACKTEST_SYMBOLS)


def _okx_get_bt(path: str, params: dict, timeout: int = 20, retries: int = 4):
    """OKX GET for backtest — host fallback + exponential backoff.

    Deep pagination across many symbols trips OKX rate limits and transient
    DNS failures; retry with backoff makes a cold-cache prefetch reliable.
    """
    import requests as _req
    base = os.getenv("OKX_BASE_URL", "").strip().rstrip("/")
    hosts = [base] if base else OKX_HOSTS
    last_exc = None
    for attempt in range(retries):
        for host in hosts:
            try:
                r = _req.get(f"{host}{path}", params=params, timeout=timeout)
                r.raise_for_status()
                return r
            except Exception as e:
                last_exc = e
                continue
        time.sleep(1.5 * (attempt + 1))  # 1.5s, 3s, 4.5s backoff
    raise RuntimeError(f"All OKX hosts failed for {path}: {last_exc}")


def fetch_top_symbols(limit: int) -> list[str]:
    """Top non-crypto (stock/ETF/commodity) OKX swaps by 24h USD turnover, in
    internal 'AAPLUSDT' form. Mirrors the live get_top_coins pool selection."""
    from src.binance_client import get_top_coins
    return list(get_top_coins())[:limit]


def choose_workers(symbol_count: int, candles: int, stride: int) -> int:
    """Pick a low-overhead default for the common pinned-symbol backtest."""
    if symbol_count <= 1:
        return 1

    cpu = os.cpu_count() or 2
    effective_bars = max(1, candles // max(1, stride))

    if symbol_count <= 24 and effective_bars <= 2_000:
        return max(1, min(4, cpu, symbol_count))
    return max(1, min(8, cpu, symbol_count))


def cache_path(symbol: str, interval: str, count: int, end_date_ms: int | None = None) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe = symbol.replace("/", "_").replace("-", "_")
    suffix = f"_end{end_date_ms}" if end_date_ms else ""
    return CACHE_DIR / f"{safe}_{interval}_{count}{suffix}.pkl"


def _normalize_cached_candles(obj) -> dict[str, list] | None:
    if not isinstance(obj, dict):
        return None
    required = ("time", "open", "high", "low", "close", "volume")
    if any(k not in obj for k in required):
        return None
    lengths = {len(obj[k]) for k in required}
    if len(lengths) != 1 or not next(iter(lengths), 0):
        return None
    return {k: list(obj[k]) for k in required}


# Data source switch: "okx" (default; the exact swap the bot trades, ~4mo) or
# "duka" (Dukascopy CFDs, YEARS of history for 16/26 tickers — deep filter
# validation). "duka" falls back to OKX for uncovered tickers.
# Read from env so ProcessPoolExecutor workers (fresh module import) inherit it.
#
# THESE TWO SOURCES DO NOT SELECT THE SAME TRADES. The HTF windows are counted
# in bars, and duka's session-only tape means 90x1h spans 16.9 days there vs 3.7
# on OKX — so the 4h/1h trend reads differ (opposite on some tickers) and the
# same 124 days yield 211 OKX trades vs 108 duka ones. Aggregate edge still
# matches (+0.599 vs +0.591 R/trade), so duka is fine for exit-geometry work and
# for confirming the edge exists across regimes; anything that decides WHICH
# trades to take must be re-checked with --source okx. Details in
# src/dukascopy_client.py.
DATA_SOURCE = os.getenv("BACKTEST_SOURCE", "okx").strip().lower()


def fetch_history(
    symbol: str,
    interval: str,
    interval_sec: int,
    count: int,
    *,
    refresh_cache: bool = False,
    end_date_ms: int | None = None,
) -> dict[str, list]:
    """Fetch historical candles (OKX by default, Dukascopy for deep runs)."""
    if DATA_SOURCE == "duka":
        try:
            from src.dukascopy_client import covers, fetch_history_duka
            if covers(symbol):
                return fetch_history_duka(
                    symbol, interval, interval_sec, count,
                    refresh_cache=refresh_cache, end_date_ms=end_date_ms,
                )
        except ImportError:
            pass  # dukascopy-python not installed — OKX fallback
    return _fetch_history_okx(
        symbol, interval, interval_sec, count,
        refresh_cache=refresh_cache, end_date_ms=end_date_ms,
    )


def _fetch_history_okx(
    symbol: str,
    interval: str,
    interval_sec: int,
    count: int,
    *,
    refresh_cache: bool = False,
    end_date_ms: int | None = None,
) -> dict[str, list]:
    """Fetch historical OKX candles with a local pickle cache.

    OKX history-candles format: [ts_ms, open, high, low, close, vol(contracts),
    volCcy(base), volCcyQuote(quote), confirm]. Returns newest-first — we
    reverse to oldest-first. Volume uses volCcy (index 6, base units) to match
    the live client exactly. Only closed candles (confirm == "1") are kept.
    Paginates backwards via `after` (records strictly older than the ts).

    end_date_ms anchors the window's newest candle to a specific past moment
    instead of "now".
    """
    path = cache_path(symbol, interval, count, end_date_ms)
    if not refresh_cache and path.exists():
        age = time.time() - path.stat().st_mtime
        if age < CACHE_TTL_SEC:
            try:
                with path.open("rb") as f:
                    cached = _normalize_cached_candles(pickle.load(f))
                if cached:
                    return cached
            except Exception:
                pass

    okx_bar = OKX_INTERVAL_MAP.get(str(interval), "15m")
    inst_id = _inst_id(symbol)
    anchor_ms = int(end_date_ms) if end_date_ms else None
    after = anchor_ms  # OKX 'after' = strictly older than this ts
    by_time: dict[int, list] = {}
    cutoff_ms = (anchor_ms if anchor_ms else int(time.time() * 1000)) - count * interval_sec * 1000

    while len(by_time) < count:
        params = {"instId": inst_id, "bar": okx_bar, "limit": OKX_PAGE_LIMIT}
        if after is not None:
            params["after"] = str(after)
        resp = _okx_get_bt("/api/v5/market/history-candles", params)
        raw = resp.json().get("data", [])
        if not raw:
            break

        for c in raw:
            if len(c) > 8 and c[8] != "1":
                continue  # unclosed candle — skip (no repaint)
            ts_s = int(float(c[0])) // 1000
            if ts_s not in by_time:
                by_time[ts_s] = c

        oldest_ts_ms = int(float(raw[-1][0]))
        if len(raw) < OKX_PAGE_LIMIT or oldest_ts_ms <= cutoff_ms:
            break
        after = oldest_ts_ms  # next page = strictly older

    candles = [by_time[ts] for ts in sorted(by_time)][-count:]
    if not candles:
        raise ValueError(f"No OKX data for {inst_id} {interval}")

    # OKX columns: [ts_ms, o, h, l, c, vol, volCcy, volCcyQuote, confirm]
    data = {
        "time":   [int(float(c[0])) // 1000 for c in candles],
        "open":   [float(c[1]) for c in candles],
        "high":   [float(c[2]) for c in candles],
        "low":    [float(c[3]) for c in candles],
        "close":  [float(c[4]) for c in candles],
        "volume": [float(c[6]) for c in candles],  # volCcy = base units (matches live)
    }

    with path.open("wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    return data


def calculate_tp_sl_local(
    price: float,
    direction: str,
    atr: float = 0.0,
    recent_high: float = 0.0,
    recent_low: float = 0.0,
    tp1_level: float | None = None,
    tp2_level: float | None = None,
) -> tuple[float, float, float]:
    """Local copy of telegram_notifier.calculate_tp_sl without requests import."""

    min_risk = price * RISK_MIN_PCT
    max_risk = price * RISK_MAX_PCT
    buf = atr * SL_ATR_BUFFER if atr and atr > 0 else 0.0

    if direction == "LONG":
        struct_sl = (recent_low - buf) if recent_low and recent_low > 0 else price - max_risk
        risk = min(max(price - struct_sl, min_risk), max_risk)
        sl = price - risk

        if tp1_level and tp1_level > price * 1.001 and (tp1_level - price) >= risk:
            tp1 = tp1_level
        else:
            tp1 = price + risk * TP1_R_MULT

        if tp2_level and tp2_level > tp1 * 1.001 and (tp2_level - price) >= risk * 1.5:
            tp2 = tp2_level
        else:
            tp2 = price + risk * TP2_R_MULT
            if tp2 <= tp1:
                tp2 = tp1 * 1.02
    else:
        struct_sl = (recent_high + buf) if recent_high and recent_high > 0 else price + max_risk
        risk = min(max(struct_sl - price, min_risk), max_risk)
        sl = price + risk

        if tp1_level and tp1_level < price * 0.999 and (price - tp1_level) >= risk:
            tp1 = tp1_level
        else:
            tp1 = price - risk * TP1_R_MULT

        if tp2_level and tp2_level < tp1 * 0.999 and (price - tp2_level) >= risk * 1.5:
            tp2 = tp2_level
        else:
            tp2 = price - risk * TP2_R_MULT
            if tp2 >= tp1:
                tp2 = tp1 * 0.98

    return round(tp1, 8), round(tp2, 8), round(sl, 8)


def _last_swing_high(highs: list[float], start: int, stop: int, lookback: int) -> float | None:
    for i in range(stop - lookback - 1, start + lookback - 1, -1):
        h = highs[i]
        if h == max(highs[i - lookback:i + lookback + 1]):
            return h
    return None


def _last_swing_low(lows: list[float], start: int, stop: int, lookback: int) -> float | None:
    for i in range(stop - lookback - 1, start + lookback - 1, -1):
        l = lows[i]
        if l == min(lows[i - lookback:i + lookback + 1]):
            return l
    return None


def cheap_prefilter_at(candles_15m: dict[str, list], end: int, window: int) -> bool:
    """
    Exact early reject for gates analyze_coin_smc also requires:
    enough candles, BOS present, and BOS-context volume threshold.
    """

    start = max(0, end - window)
    n = end - start
    if n < 30:
        return False

    volumes = candles_15m["volume"]
    if n >= 21:
        avg_vol = sum(volumes[end - 21:end - 1]) / 20
    else:
        avg_vol = sum(volumes[start:end]) / n
    volume_ratio = round(volumes[end - 1] / (avg_vol + 1e-10), 2)
    if volume_ratio < SMC_BOS_MIN_VOLUME:
        return False

    highs = candles_15m["high"]
    lows = candles_15m["low"]
    closes = candles_15m["close"]
    swing_lookback = SMC_SWING_LOOKBACK

    last_sh = _last_swing_high(highs, start, end, swing_lookback)
    if last_sh is None:
        return False
    last_sl = _last_swing_low(lows, start, end, swing_lookback)
    if last_sl is None:
        return False

    for i in range(max(start, end - 10), end - 1):
        c = closes[i]
        if c > last_sh or c < last_sl:
            return True
    return False


def aligned_slice_by_time(
    candles: dict[str, list],
    t_cur: int | None,
    lookback: int,
    fallback_end: int,
) -> dict[str, list]:
    if not candles or not candles.get("close"):
        return {}

    if t_cur is not None and candles.get("time"):
        end = bisect_right(candles["time"], t_cur)
    else:
        end = fallback_end

    end = max(1, min(end, len(candles["close"])))
    start = max(0, end - lookback)
    return candle_slice(candles, start, end)


_TP1_CLOSE_FRAC = max(0.0, min(1.0, float(TP1_CLOSE_FRAC)))
_RUNNER_FRAC = 1.0 - _TP1_CLOSE_FRAC


def _post_tp1_trail_mult_bt(direction: str, entry: float, tp1: float, tp2: float,
                            high: float, low: float, close: float) -> float:
    """Context-aware runner trail from the TP1 candle (mirrors live _post_tp1_trail_mult)."""
    base = max(0.0, float(TRAIL_ATR_MULT))
    if str(EXIT_PROFILE).lower() != "post_tp1_v2":
        return base
    leg = abs(float(tp2) - float(tp1))
    if leg <= 0:
        return base
    if str(direction).upper() == "LONG":
        close_progress = (float(close) - float(tp1)) / leg
        wick_progress = (float(high) - float(tp1)) / leg
        failed_close = float(close) < float(tp1)
    else:
        close_progress = (float(tp1) - float(close)) / leg
        wick_progress = (float(tp1) - float(low)) / leg
        failed_close = float(close) > float(tp1)
    if close_progress >= POST_TP1_STRONG_CLOSE_PROGRESS or wick_progress >= POST_TP1_STRONG_WICK_PROGRESS:
        return max(base, float(POST_TP1_STRONG_TRAIL_ATR_MULT))
    if failed_close or close_progress <= POST_TP1_WEAK_CLOSE_PROGRESS:
        return min(base, float(POST_TP1_WEAK_TRAIL_ATR_MULT))
    return base


# Close-confirmed stop — mirrors the live setting so backtest and live cannot
# drift apart (config already applies the STOP_CLOSE_CONFIRM env override).
from config import STOP_CLOSE_CONFIRM as _STOP_CLOSE_CONFIRM

# Anchor the runner trail to PRIOR bars only, so a trail exit can never be
# filled off the same bar that printed the peak. Ported from the crypto bot
# 2026-08-25, where the same code inflated headline profit by ~7%. Default ON:
# the honest convention is the default, BT_TRAIL_LAG=0 restores the old one for
# comparison.
_BT_TRAIL_LAG = os.getenv("BT_TRAIL_LAG", "1") == "1"

# Two-stage trail, ported 2026-08-26. It FAILED in the crypto bot, and the
# reason it failed is the reason to test it here: crypto has no runners to give
# room to (largest trail exit in the whole book +1.98R, median +0.65R), while
# this book does — median +0.93R, largest +4.10R, 8 trail exits past +2.0R and
# 100 TP2 hits against crypto's 23.
# Keep the tight trail until BT_TRAIL_STAGE_R, then widen to STAGE_MULT. 0=off.
_BT_TRAIL_STAGE_R    = float(os.getenv("BT_TRAIL_STAGE_R", "0") or 0)
_BT_TRAIL_STAGE_MULT = float(os.getenv("BT_TRAIL_STAGE_MULT", "0.35") or 0.35)


def _r_from_price(entry: float, exit_px: float, sl: float, direction: str) -> float:
    """Actual R of an exit at an arbitrary price (pre-TP1, full position open).

    Needed because gross_r_for_outcome() hardcodes SL = -1.0R, which only holds
    when the exit really happens AT the stop level. A close-confirmed stop exits
    at the candle CLOSE, which on a gapping instrument can be far beyond it —
    booking that as -1.0R would invent an edge that does not exist.
    """
    risk = abs(entry - sl)
    if risk <= 0:
        return 0.0
    return ((exit_px - entry) if direction == "LONG" else (entry - exit_px)) / risk


def gross_r_for_outcome(outcome: str, entry: float, tp1: float, tp2: float, sl: float) -> float:
    risk = abs(entry - sl)
    if risk <= 0:
        return 0.0

    tp1_r = abs(tp1 - entry) / risk
    tp2_r = abs(tp2 - entry) / risk

    if outcome == "TP2":
        return _TP1_CLOSE_FRAC * tp1_r + _RUNNER_FRAC * tp2_r
    if outcome == "TP1":
        return _TP1_CLOSE_FRAC * tp1_r
    if outcome == "SL":
        return -1.0
    return 0.0


def gross_r_for_trailing_exit(entry: float, tp1: float, trail_exit: float, sl: float, direction: str) -> float:
    risk = abs(entry - sl)
    if risk <= 0:
        return 0.0
    tp1_r = abs(tp1 - entry) / risk
    if direction == "LONG":
        trail_r = (trail_exit - entry) / risk
    else:
        trail_r = (entry - trail_exit) / risk
    return _TP1_CLOSE_FRAC * tp1_r + _RUNNER_FRAC * max(0.0, trail_r)


def execution_fill_price(
    direction: str,
    planned_entry: float,
    candles_15m: dict[str, list],
    entry_bar: int,
    delay_bars: int,
    adverse_bps: float,
) -> tuple[float, int]:
    fill_bar = min(max(entry_bar, entry_bar + max(0, delay_bars)), len(candles_15m["close"]) - 1)
    price = planned_entry if delay_bars <= 0 else float(candles_15m["close"][fill_bar])
    adverse = adverse_bps / 10_000.0
    if direction == "LONG":
        price *= 1.0 + adverse
    else:
        price *= 1.0 - adverse
    return price, fill_bar


def estimate_cost_r(entry: float, sl: float, fee_rate: float, slippage_rate: float) -> float:
    risk = abs(entry - sl)
    if entry <= 0 or risk <= 0:
        return 0.0
    round_trip_cost_pct = 2.0 * (fee_rate + slippage_rate)
    return round_trip_cost_pct * entry / risk


@dataclass
class TradeRecord:
    symbol: str
    entry_bar: int
    exit_bar: int
    entry_time: int | None
    exit_time: int | None
    direction: str
    outcome: str
    entry: float
    tp1: float
    tp2: float
    sl: float
    gross_r: float
    net_r: float
    cost_r: float
    mtf_score: int = 0
    volume_ratio: float = 0.0
    rsi: float = 0.0
    eff_ratio: float = 0.0
    vol_atr_pct: float = 0.0
    vol_ratio_regime: float = 0.0
    adaptive_pack: str = ""
    adaptive_reason: str = ""
    risk_mult: float = 1.0
    quality_score: float = 0.0
    trend_score: int = 0
    volatility_score: int = 0
    entry_quality_score: int = 0
    portfolio_risk_score: int = 0
    session: str = ""
    trend_1h: str = ""
    trend_4h: str = ""
    entry_source: str = ""
    signals: str = ""
    bos_extension_atr: float = 0.0
    bos_candles_ago: int = -1
    score_tags: str = ""
    premium: int = 0
    knn_score: float = -1.0
    swing_trend: str = ""  # 15m structure (bull/bear/range) — feeds Claude memory seeding


@dataclass
class SymbolResult:
    symbol: str
    bars: int = 0
    scanned: int = 0
    off_session: int = 0
    prefiltered: int = 0
    analyzed: int = 0
    trades: int = 0
    tp1: int = 0
    tp2: int = 0
    sl: int = 0
    expired: int = 0
    gross_r: float = 0.0
    net_r: float = 0.0
    elapsed_sec: float = 0.0
    error: str | None = None
    trade_records: list[TradeRecord] = field(default_factory=list)


def simulate_trade_direct(
    symbol: str,
    setup: dict,
    candles_15m: dict[str, list],
    entry_bar: int,
    window: int,
    fee_rate: float,
    slippage_rate: float,
    execution_delay_bars: int = 0,
    adverse_entry_bps: float = 0.0,
    exit_policy: str = "classic",
    trail_atr_mult: float = 0.75,
) -> TradeRecord:
    direction = setup["direction"]
    planned_entry = float(setup["current_price"])
    entry, fill_bar = execution_fill_price(
        direction,
        planned_entry,
        candles_15m,
        entry_bar,
        execution_delay_bars,
        adverse_entry_bps,
    )
    tp1, tp2, sl = calculate_tp_sl_local(
        entry,
        direction,
        atr=setup.get("atr", 0.0),
        recent_high=setup.get("recent_high", 0.0),
        recent_low=setup.get("recent_low", 0.0),
        tp1_level=setup.get("tp1_level"),
        tp2_level=setup.get("tp2_level"),
    )

    highs = candles_15m["high"]
    lows = candles_15m["low"]
    closes = candles_15m["close"]
    times = candles_15m.get("time") or []

    # Expiry clock counts IN-SESSION bars only: `window` bars of dead overnight
    # tape would otherwise expire a trade before the underlying ever moves
    # again (stocks bot: 48 session bars ≈ 2 trading days). SL/TP hits are
    # still checked on EVERY bar — the X-Perp trades 24/7 and the live monitor
    # watches open positions round the clock.
    #
    # PARITY with _check_open_signals: it ages a position on the SAME session
    # clock, with SIGNAL_EXPIRY_MAX_DAYS as a hard calendar ceiling so a Friday
    # entry cannot run ~12 calendar days into its own earnings report. Whichever
    # limit is reached first ends the trade, in both engines.
    _wall_cap_sec = SIGNAL_EXPIRY_MAX_DAYS * 86400 if SIGNAL_EXPIRY_MAX_DAYS > 0 else None
    if times:
        bar_indices: list[int] = []
        session_used = 0
        j = fill_bar
        n_all = len(highs)
        t_open = times[fill_bar] if fill_bar < len(times) else None
        while j < n_all and session_used < window:
            if _wall_cap_sec is not None and t_open is not None and j < len(times):
                if int(times[j]) - int(t_open) > _wall_cap_sec:
                    break
            bar_indices.append(j)
            try:
                if _is_market_open(_datetime.fromtimestamp(int(times[j]), tz=_tz.utc)):
                    session_used += 1
            except Exception:
                session_used += 1  # bad timestamp → count it, stay bounded
            j += 1
    else:
        bar_indices = list(range(fill_bar, min(fill_bar + window, len(highs))))

    outcome = "EXPIRED"
    tp1_reached = False
    closed = False
    exit_bar = bar_indices[-1] if bar_indices else fill_bar
    trailing_stop = entry
    trail_exit_price = entry
    best_price = entry
    trail_mult_eff = max(0.0, float(trail_atr_mult))  # context-frozen at TP1 candle

    stop_exit_price = None  # set when we exit at a price other than the SL level
    for j in bar_indices:
        h = highs[j]
        l = lows[j]
        if not tp1_reached:
            if direction == "LONG":
                if (closes[j] <= sl) if _STOP_CLOSE_CONFIRM else (l <= sl):
                    outcome = "SL"
                    if _STOP_CLOSE_CONFIRM:
                        stop_exit_price = closes[j]
                    exit_bar = j
                    closed = True
                    break
                if h >= tp2:
                    outcome = "TP2"
                    exit_bar = j
                    closed = True
                    break
                if h >= tp1:
                    outcome = "TP1"
                    tp1_reached = True
                    exit_bar = j
                    trail_mult_eff = _post_tp1_trail_mult_bt(direction, entry, tp1, tp2, h, l, closes[j])
                    continue
            else:
                if (closes[j] >= sl) if _STOP_CLOSE_CONFIRM else (h >= sl):
                    outcome = "SL"
                    if _STOP_CLOSE_CONFIRM:
                        stop_exit_price = closes[j]
                    exit_bar = j
                    closed = True
                    break
                if l <= tp2:
                    outcome = "TP2"
                    exit_bar = j
                    closed = True
                    break
                if l <= tp1:
                    outcome = "TP1"
                    tp1_reached = True
                    exit_bar = j
                    trail_mult_eff = _post_tp1_trail_mult_bt(direction, entry, tp1, tp2, h, l, closes[j])
                    continue
        else:
            if direction == "LONG":
                if exit_policy == "trail":
                    _ra = abs(entry - sl)
                    if _BT_TRAIL_STAGE_R > 0 and _ra > 0:
                        _g = (best_price - entry) / _ra
                        if _g >= _BT_TRAIL_STAGE_R:
                            trail_mult_eff = max(trail_mult_eff, _BT_TRAIL_STAGE_MULT)
                    # Ported from the crypto bot 2026-08-25. Anchoring the trail
                    # to THIS bar's own high and then testing THIS bar's low
                    # assumes the high printed first, which OHLC does not
                    # record. It pays out on every bar once the trail is
                    # narrower than the average bar range — worth ~7% of
                    # headline profit there.
                    if not _BT_TRAIL_LAG:
                        best_price = max(best_price, h)
                    trailing_stop = max(entry, best_price - max(0.0, float(setup.get("atr", 0.0) or 0.0)) * trail_mult_eff)
                    if _BT_TRAIL_LAG:
                        best_price = max(best_price, h)
                    if l <= trailing_stop:
                        outcome = "TRAIL"
                        trail_exit_price = trailing_stop
                        exit_bar = j
                        closed = True
                        break
                if l <= entry:
                    outcome = "TP1"
                    exit_bar = j
                    closed = True
                    break
                if h >= tp2:
                    outcome = "TP2"
                    exit_bar = j
                    closed = True
                    break
            else:
                if exit_policy == "trail":
                    _ra = abs(entry - sl)
                    if _BT_TRAIL_STAGE_R > 0 and _ra > 0:
                        _g = (entry - best_price) / _ra
                        if _g >= _BT_TRAIL_STAGE_R:
                            trail_mult_eff = max(trail_mult_eff, _BT_TRAIL_STAGE_MULT)
                    if not _BT_TRAIL_LAG:
                        best_price = min(best_price, l)
                    trailing_stop = min(entry, best_price + max(0.0, float(setup.get("atr", 0.0) or 0.0)) * trail_mult_eff)
                    if _BT_TRAIL_LAG:
                        best_price = min(best_price, l)
                    if h >= trailing_stop:
                        outcome = "TRAIL"
                        trail_exit_price = trailing_stop
                        exit_bar = j
                        closed = True
                        break
                if h >= entry:
                    outcome = "TP1"
                    exit_bar = j
                    closed = True
                    break
                if l <= tp2:
                    outcome = "TP2"
                    exit_bar = j
                    closed = True
                    break

    if tp1_reached and outcome == "TP1" and not closed:
        exit_bar = bar_indices[-1] if bar_indices else fill_bar

    if outcome == "TRAIL":
        gross_r = gross_r_for_trailing_exit(entry, tp1, trail_exit_price, sl, direction)
    elif stop_exit_price is not None:
        # Close-confirmed stop: full position still open, so R is the real move
        # to the exit price — NOT the -1.0R a level-touch stop would book. Can
        # be much worse than -1R when the instrument gaps.
        gross_r = _r_from_price(entry, stop_exit_price, sl, direction)
    else:
        gross_r = gross_r_for_outcome(outcome, entry, tp1, tp2, sl)
    cost_r = estimate_cost_r(entry, sl, fee_rate, slippage_rate)
    net_r = gross_r - cost_r
    # Fresh-break trim — see EXTENSION_FRESH_THRESHOLD in config.py. Entering
    # right at the break, before anything confirms it, is this bot's worst
    # bucket (53.2% WR against a 63.6% book).
    try:
        if _fld(setup, "bos_extension_atr", 99.0) <= EXTENSION_FRESH_THRESHOLD:
            _fm = float(EXTENSION_FRESH_SIZE_MULT)
            gross_r *= _fm; net_r *= _fm; cost_r *= _fm
    except (TypeError, ValueError):
        pass
    # Ceiling on the stacked product — see SIZE_MULT_MAX in config.py. Applied
    # as a correction after the boosts, since the multipliers here are folded
    # straight into gross/net/cost rather than accumulated in one variable.
    _stack = 1.0
    # Orderly trend rides bigger — see ORDERLY_SIZE_MULT in config.py.
    if ORDERLY_SIZE_MULT != 1.0:
        try:
            if (float(setup.get("eff_ratio") or 0.0) >= ORDERLY_EFF_MIN
                    and float(setup.get("vol_atr_pct") or 99.0) < ORDERLY_ATR_MAX
                    and _fld(setup, "bos_extension_atr", 0.0) >= ORDERLY_EXT_MIN):
                _sm = float(ORDERLY_SIZE_MULT)
                gross_r *= _sm; net_r *= _sm; cost_r *= _sm; _stack *= _sm
        except (TypeError, ValueError):
            pass
    # OFF session rides smaller — see OFF_SESSION_SIZE_MULT in config.py.
    if OFF_SESSION_SIZE_MULT != 1.0 and str(setup.get("session") or "") == "OFF":
        _fm = float(OFF_SESSION_SIZE_MULT)
        gross_r *= _fm; net_r *= _fm; cost_r *= _fm; _stack *= _fm
    # Opening bell rides bigger — see OPEN_SESSION_SIZE_MULT in config.py.
    if OPEN_SESSION_SIZE_MULT != 1.0 and str(setup.get("session") or "") == "OPEN":
        try:
            _vok = float(setup.get("volume_ratio") or 0.0) >= OPEN_VOL_MIN
        except (TypeError, ValueError):
            _vok = False
        if _vok:
            _om = float(OPEN_SESSION_SIZE_MULT)
            gross_r *= _om; net_r *= _om; cost_r *= _om; _stack *= _om

    if _stack > SIZE_MULT_MAX:
        _fix = SIZE_MULT_MAX / _stack
        gross_r *= _fix; net_r *= _fix; cost_r *= _fix

    return TradeRecord(
        symbol=symbol,
        entry_bar=fill_bar,
        exit_bar=exit_bar,
        entry_time=times[fill_bar - 1] if 0 <= fill_bar - 1 < len(times) else None,
        exit_time=times[exit_bar] if 0 <= exit_bar < len(times) else None,
        direction=direction,
        outcome=outcome,
        entry=entry,
        tp1=tp1,
        tp2=tp2,
        sl=sl,
        gross_r=gross_r,
        net_r=net_r,
        cost_r=cost_r,
        mtf_score=int(setup.get("mtf_score", 0) or 0),
        volume_ratio=float(setup.get("volume_ratio", 0.0) or 0.0),
        rsi=float(setup.get("rsi", 0.0) or 0.0),
        eff_ratio=float(setup.get("eff_ratio", 0.0) or 0.0),
        vol_atr_pct=float(setup.get("vol_atr_pct", 0.0) or 0.0),
        vol_ratio_regime=float(setup.get("vol_ratio_regime", 0.0) or 0.0),
        adaptive_pack=str(setup.get("adaptive_pack", "") or ""),
        adaptive_reason=str(setup.get("adaptive_reason", "") or ""),
        risk_mult=float(setup.get("risk_mult", 1.0) or 1.0),
        quality_score=float(setup.get("quality_score", 0.0) or 0.0),
        trend_score=int(setup.get("trend_score", 0) or 0),
        volatility_score=int(setup.get("volatility_score", 0) or 0),
        entry_quality_score=int(setup.get("entry_quality_score", 0) or 0),
        portfolio_risk_score=int(setup.get("portfolio_risk_score", 0) or 0),
        session=str(setup.get("session", "") or ""),
        trend_1h=str(setup.get("trend_1h", "") or ""),
        trend_4h=str(setup.get("trend_4h", "") or ""),
        entry_source=str(setup.get("entry_source", "") or ""),
        bos_extension_atr=round(_fld(setup, "bos_extension_atr", -1.0), 3),
        bos_candles_ago=int(_fld(setup, "bos_candles_ago", -1.0)),
        signals=" | ".join(setup.get("signals", [])),
        score_tags=" | ".join(setup.get("score_tags", [])),
        premium=int(bool(setup.get("premium"))),
        knn_score=float(setup.get("_knn_score", -1.0)),
        swing_trend=str(setup.get("swing_trend", "") or ""),
    )


def backtest_symbol(
    symbol: str,
    *,
    candles: int,
    tp_window: int,
    warmup: int,
    stride: int,
    window_15m: int,
    window_1h: int,
    window_4h: int,
    use_prefilter: bool,
    refresh_cache: bool,
    fee_rate: float,
    slippage_rate: float,
    execution_delay_bars: int,
    adverse_entry_bps: float,
    exit_policy: str,
    trail_atr_mult: float,
    end_date_ms: int | None = None,
) -> SymbolResult:
    started = time.perf_counter()
    result = SymbolResult(symbol=symbol)

    # HTF candle counts: the /4, /16, /96 ratios assume a 24/7 tape (crypto
    # clock). Dukascopy equity CFDs print only ~26 15m bars per trading day,
    # so N 15m bars span ~4x more CALENDAR days — HTF windows must stretch
    # accordingly or the oldest bars run with an empty 1d feed (daily-trend
    # filters silently off → not what the live bot does).
    if DATA_SOURCE == "duka":
        div_1h, div_4h, div_1d = 3, 10, 20
    else:
        div_1h, div_4h, div_1d = 4, 16, 96

    try:
        c15 = fetch_history(symbol, TIMEFRAME_KUCOIN, KLINES_INTERVAL_SEC, candles,
                            refresh_cache=refresh_cache, end_date_ms=end_date_ms)
        c1h = fetch_history(
            symbol,
            TIMEFRAME_1H_KUCOIN,
            KLINES_1H_INTERVAL_SEC,
            max(10, math.ceil(candles / div_1h) + 4),
            refresh_cache=refresh_cache,
            end_date_ms=end_date_ms,
        )
        c4h = fetch_history(
            symbol,
            TIMEFRAME_4H_KUCOIN,
            KLINES_4H_INTERVAL_SEC,
            max(10, math.ceil(candles / div_4h) + 4),
            refresh_cache=refresh_cache,
            end_date_ms=end_date_ms,
        )
        try:
            c1d = fetch_history(
                symbol, "1d", 86400,
                max(8, math.ceil(candles / div_1d) + 4),
                refresh_cache=refresh_cache,
                end_date_ms=end_date_ms,
            )
        except Exception:
            c1d = {}
        # Market-proxy (SPY) 1h series, so btc_change_pct can be REAL instead of
        # a constant 0.0. Passing 0.0 — as this did until 2026-07-31, same bug
        # the crypto bot had — silently handed every backtest trade the maximum
        # market-alignment bonus: the scorer does `LONG and chg >= 0 -> +2 /
        # SHORT and chg <= 0 -> +2 / else +1`, and 0.0 satisfies BOTH, so the
        # +1 branch never ran. It also disabled the market-move block filter
        # (BTC_BLOCK_THRESHOLD_PCT, which cannot trigger at 0.0) and fed a wrong
        # rel_strength to the momentum pack.
        try:
            proxy_1h = fetch_history(
                MARKET_PROXY_SYMBOL, TIMEFRAME_1H_KUCOIN, KLINES_1H_INTERVAL_SEC,
                max(10, math.ceil(candles / div_1h) + 4),
                refresh_cache=refresh_cache,
                end_date_ms=end_date_ms,
            )
        except Exception:
            proxy_1h = {}
    except Exception as exc:
        result.error = str(exc)
        result.elapsed_sec = time.perf_counter() - started
        return result

    n = len(c15["close"])
    result.bars = n
    if n < warmup + tp_window + 2:
        result.elapsed_sec = time.perf_counter() - started
        return result

    for i in range(warmup, n - tp_window, max(1, stride)):
        result.scanned += 1

        # Session gate — must match run_scan exactly: NYSE open, OR the
        # off-session toggle, OR an EXTENDED_SESSION_WINDOWS hour on a weekday.
        #
        # The window mechanism was added live on 2026-08-20 and this gate was
        # NOT updated with it, so for a day the backtest measured session-only
        # trading while production also scanned London and the overnight block —
        # the same live-vs-backtest divergence the live-gate work had just been
        # fixing. Judged on the just-closed candle (i-1), matching how the live
        # indicator labels the session.
        if not OFF_SESSION_SIGNALS:
            _ts = c15["time"][i - 1] if c15.get("time") and i > 0 else None
            if _ts is not None:
                _dt_utc = _datetime.fromtimestamp(int(_ts), tz=_tz.utc)
                if not _is_market_open(_dt_utc) and not _in_extended_window_bt(_dt_utc):
                    result.off_session += 1
                    continue

        if use_prefilter and not cheap_prefilter_at(c15, i, window_15m):
            result.prefiltered += 1
            continue

        snap_15 = candle_slice(c15, max(0, i - window_15m), i)
        t_cur = c15["time"][i - 1] if c15.get("time") and i > 0 else None
        snap_1h = aligned_slice_by_time(c1h, t_cur, window_1h, max(1, i // 4))
        snap_4h = aligned_slice_by_time(c4h, t_cur, window_4h, max(1, i // 16))
        snap_1d = aligned_slice_by_time(c1d, t_cur, 8, max(1, i // 96)) if c1d else None

        # Same definition the live bot uses (get_btc_change_1h): pct move of the
        # last CLOSED 1h proxy candle vs the one before, as of this scan bar.
        _mkt_chg = 0.0
        if proxy_1h:
            _psnap = aligned_slice_by_time(proxy_1h, t_cur, 3, max(1, i // div_1h))
            _pc = (_psnap or {}).get("close") or []
            if len(_pc) >= 2 and _pc[-2]:
                _mkt_chg = (_pc[-1] - _pc[-2]) / _pc[-2] * 100.0

        result.analyzed += 1
        setup = analyze_coin_smc(snap_15, snap_1h, symbol, snap_4h,
                                 btc_change_pct=_mkt_chg,
                                 candles_1d=snap_1d)
        if not setup:
            continue

        # k-NN price-shape analog score (research column, no look-ahead).
        # KNN_MAXHIST env caps the analog pool to test required live candle depth.
        _mh = os.getenv("KNN_MAXHIST", "").strip()
        knn = knn_direction_score(
            c15, i, setup["direction"],
            max_history=int(_mh) if _mh else None,
        )
        setup["_knn_score"] = -1.0 if knn is None else knn

        trade = simulate_trade_direct(
            symbol,
            setup,
            c15,
            i,
            tp_window,
            fee_rate,
            slippage_rate,
            execution_delay_bars=execution_delay_bars,
            adverse_entry_bps=adverse_entry_bps,
            exit_policy=exit_policy,
            trail_atr_mult=trail_atr_mult,
        )
        result.trade_records.append(trade)
        result.trades += 1
        result.gross_r += trade.gross_r
        result.net_r += trade.net_r

        if trade.outcome in ("TP1", "TRAIL"):
            result.tp1 += 1
        elif trade.outcome == "TP2":
            result.tp2 += 1
        elif trade.outcome == "SL":
            result.sl += 1
        else:
            result.expired += 1

    result.elapsed_sec = time.perf_counter() - started
    return result


def merge_results(results: Iterable[SymbolResult]) -> SymbolResult:
    total = SymbolResult(symbol="TOTAL")
    for r in results:
        total.bars += r.bars
        total.scanned += r.scanned
        total.off_session += r.off_session
        total.prefiltered += r.prefiltered
        total.analyzed += r.analyzed
        total.trades += r.trades
        total.tp1 += r.tp1
        total.tp2 += r.tp2
        total.sl += r.sl
        total.expired += r.expired
        total.gross_r += r.gross_r
        total.net_r += r.net_r
        total.elapsed_sec += r.elapsed_sec
        total.trade_records.extend(r.trade_records)
    return total


_LIVE_MAX_PER_SCAN = int(os.getenv("BT_LIVE_MAX_PER_SCAN", "3"))


def _fld(setup: dict, key: str, missing: float) -> float:
    """Read a numeric setup field, substituting ONLY when it is genuinely absent.

    Ported from the crypto bot 2026-08-28 to kill a live bug. The pattern
    `float(setup.get(k) or default)` also fires on a legitimate ZERO, and
    bos_extension_atr is exactly the field where zero carries meaning: it is
    None when no break level was found, and 0.0 when price sits right ON the
    break — the freshest entry there is. With `or 99.0` a zero became 99, failed
    the `<= EXTENSION_FRESH_THRESHOLD` test, and the fresh-break trim skipped
    precisely the trades it exists to catch. The same file read the same field
    with `or 0.0` ten lines later, which is how the inconsistency showed up.

    MEASURED before claiming it mattered, and it does not: on the current book
    bos_extension_atr is NEVER absent (0 of 1316 trades) and is exactly zero on
    4 trades (0.3%). Fixing it moved profit by -0.51R, 0.06%. Kept because it is
    correct and because a missing value becomes possible the moment the candle
    source changes — not because it bought anything.

    It also clears a suspicion worth recording: the fresh-break trim was fitted
    on an export that wrote missing values as 0.0, so its <=0.71 bucket looked
    like it could be padded with no-data trades. With zero absent values in the
    data, it was not. That justification stands.
    """
    v = setup.get(key)
    if v is None or v == "":
        return missing
    try:
        return float(v)
    except (TypeError, ValueError):
        return missing


def apply_live_gates(trades: list[TradeRecord]) -> list[TradeRecord]:
    """Trades that survive the live bot's throughput gates.

    Ported from the crypto bot 2026-08-20, where an audit found NINE gates
    run_scan applies and ZERO of them modelled in the backtest: funding, news,
    spread, stale-entry, kill-switch, auto-blocked symbols, reject cooldown, the
    per-scan signal cap and the per-coin signal cooldown. This file has the same
    gap — every headline it prints describes a book production cannot carry.
    There it cost 29% of the trade count and 28% of the profit.

    The four expressible from a trade list alone are replayed here in entry
    order. News, spread, funding and the Claude gates need live state, so even
    this figure is an upper bound.
    """
    ordered = sorted(trades, key=lambda t: (t.entry_time or 0, t.symbol, t.entry_bar))
    last_sig: dict = {}
    per_bar: dict = {}
    open_by_dir: dict = {}
    kept: list[TradeRecord] = []
    streak = 0
    cur_day = None
    blocked_day = None
    for t in ordered:
        raw = t.entry_time or 0
        ts = raw / 1000 if raw > 1e11 else raw
        day = int(ts // 86400)
        if day != cur_day:
            cur_day, streak, blocked_day = day, 0, None
        if KILL_SWITCH_SL_STREAK > 0 and blocked_day == day:
            continue
        key = (t.symbol, t.direction)
        if SIGNAL_COOLDOWN_HOURS > 0 and key in last_sig and (ts - last_sig[key]) / 3600 < SIGNAL_COOLDOWN_HOURS:
            continue
        bar = int(ts // (KLINES_INTERVAL_SEC or 900))
        if _LIVE_MAX_PER_SCAN > 0 and per_bar.get(bar, 0) >= _LIVE_MAX_PER_SCAN:
            continue
        if MAX_SAME_DIRECTION_POSITIONS > 0:
            live = [o for o in open_by_dir.get(t.direction, [])
                    if (o.exit_time or 0) > raw]
            if len(live) >= MAX_SAME_DIRECTION_POSITIONS:
                continue
            live.append(t)
            open_by_dir[t.direction] = live
        last_sig[key] = ts
        per_bar[bar] = per_bar.get(bar, 0) + 1
        kept.append(t)
        if KILL_SWITCH_SL_STREAK > 0:
            streak = streak + 1 if t.outcome == "SL" else 0
            if streak >= KILL_SWITCH_SL_STREAK:
                blocked_day = day
    return kept


def risk_profile(rs: list[float]) -> dict:
    """Downside measures that do NOT hang on a single week.

    Ported from the crypto bot, where max drawdown turned out to be one
    stretch: the whole -6.87R of a 922-trade book came from FIFTEEN trades
    across five days in April. Every equal-risk ranking divided by that, so
    thresholds jumped and halves disagreed on changes that were really noise.
    Without these two numbers the stocks bot had only Max DD to decide on,
    which is the measure that misled over there.

    worst_windows averages the k deepest rolling N-trade stretches, so several
    bad patches are needed to move it. ulcer is RMS of the underwater curve —
    it counts how LONG we sit below water, not only how deep. Both are
    downside-only: plain volatility punishes big wins too, which is not the
    risk being managed here.
    """
    import statistics as _st
    if not rs:
        return {"max_dd": 0.0, "worst_windows": 0.0, "ulcer": 0.0}
    cum = peak = 0.0
    worst = 0.0
    sq = []
    for x in rs:
        cum += x
        peak = max(peak, cum)
        worst = min(worst, cum - peak)
        sq.append((cum - peak) ** 2)
    win, k = 25, 5
    ww = 0.0
    if len(rs) >= win:
        sums = [sum(rs[i:i + win]) for i in range(len(rs) - win + 1)]
        ww = -_st.mean(sorted(sums)[:k])
    return {"max_dd": -worst,
            "worst_windows": ww,
            "ulcer": (sum(sq) / len(sq)) ** 0.5}


def max_drawdown_r(trades: list[TradeRecord], *, net: bool = True) -> float:
    equity = peak = 0.0
    max_dd = 0.0
    ordered = sorted(trades, key=lambda t: (t.entry_time or 0, t.symbol, t.entry_bar))
    for trade in ordered:
        equity += trade.net_r if net else trade.gross_r
        peak = max(peak, equity)
        max_dd = min(max_dd, equity - peak)
    return max_dd


def print_symbol_result(r: SymbolResult) -> None:
    if r.error:
        print(f"  {r.symbol:<13} ERROR {r.error}")
        return
    rate = r.scanned / r.elapsed_sec if r.elapsed_sec > 0 else 0.0
    print(
        f"  {r.symbol:<13} tr={r.trades:<4} "
        f"TP1={r.tp1:<3} TP2={r.tp2:<3} SL={r.sl:<3} EXP={r.expired:<3} "
        f"netR={r.net_r:+7.2f} "
        f"bars={r.scanned:<5} heavy={r.analyzed:<5} "
        f"{rate:7.0f} bars/s"
    )


def write_trades_csv(path: str, trades: list[TradeRecord]) -> None:
    fields = [
        "symbol", "entry_bar", "exit_bar", "entry_time", "exit_time",
        "direction", "outcome", "entry", "tp1", "tp2", "sl",
        "gross_r", "net_r", "cost_r", "mtf_score", "volume_ratio",
        "rsi", "eff_ratio", "vol_atr_pct", "vol_ratio_regime",
        "adaptive_pack", "adaptive_reason", "risk_mult",
        "quality_score", "trend_score", "volatility_score",
        "entry_quality_score", "portfolio_risk_score",
        "session", "trend_1h", "trend_4h", "entry_source",
        "bos_extension_atr", "bos_candles_ago",
        "signals", "score_tags", "premium", "knn_score", "swing_trend",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for trade in sorted(trades, key=lambda t: (t.entry_time or 0, t.symbol, t.entry_bar)):
            writer.writerow({name: getattr(trade, name) for name in fields})


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fast SMC backtest")
    p.add_argument("--symbols", default=None, help="Comma-separated symbols. Default: pinned set/env BACKTEST_SYMBOLS.")
    p.add_argument("--source", choices=("okx", "duka"), default=None,
                   help="Candle source: okx (live swap, ~4mo) or duka (Dukascopy CFD, years; OKX fallback for uncovered tickers).")
    p.add_argument("--top", type=int, default=0, help="Use current top N KuCoin USDT pairs by 24h volume.")
    p.add_argument("--candles", type=int, default=BACKTEST_CANDLES, help="15m candles per symbol.")
    p.add_argument(
        "--tp-window",
        type=int,
        default=BACKTEST_TP_WINDOW,
        help="Forward 15m candles for TP/SL simulation. Default mirrors SIGNAL_EXPIRY_HOURS.",
    )
    p.add_argument("--workers", type=int, default=0, help="Parallel worker processes. 0 = auto.")
    p.add_argument("--serial", action="store_true", help="Run without multiprocessing.")
    p.add_argument("--quiet", action="store_true", help="Print only the final summary.")
    p.add_argument("--stride", type=int, default=1, help="Scan every Nth candle. Use 4/8 for very fast rough sweeps.")
    p.add_argument("--warmup", type=int, default=DEFAULT_WARMUP, help="First scan bar.")
    p.add_argument("--window-15m", type=int, default=WINDOW_15M, help="15m lookback window passed to strategy.")
    p.add_argument("--window-1h", type=int, default=WINDOW_1H, help="1h lookback window passed to strategy.")
    p.add_argument("--window-4h", type=int, default=WINDOW_4H, help="4h lookback window passed to strategy.")
    p.add_argument("--no-prefilter", action="store_true", help="Disable exact BOS/volume early reject.")
    p.add_argument("--refresh-cache", action="store_true", help="Ignore cached candle files.")
    p.add_argument("--end-date", default=None,
                   help="ISO date (YYYY-MM-DD, UTC) to anchor the candle window's newest "
                        "bar to, instead of now. Lets --candles target an exact past range "
                        "(e.g. --end-date 2024-01-01 --candles 70080 = 2022-01-01..2024-01-01) "
                        "without re-downloading a range already covered by another batch.")
    p.add_argument("--fee-rate", type=float, default=BACKTEST_FEE_RATE, help="Per-side fee rate for net R estimate.")
    p.add_argument("--slippage-rate", type=float, default=BACKTEST_SLIPPAGE_RATE, help="Per-side slippage rate for net R estimate.")
    p.add_argument("--execution-delay-bars", type=int, default=0, help="Delay entry by N 15m bars for execution realism.")
    p.add_argument("--adverse-entry-bps", type=float, default=0.0, help="Extra adverse fill in basis points.")
    p.add_argument("--exit-policy", choices=["classic", "trail"], default="trail", help="Exit model after TP1 (default mirrors live TRAIL_RUNNER_ENABLED).")
    p.add_argument("--trail-atr-mult", type=float, default=TRAIL_ATR_MULT, help="ATR multiple for --exit-policy trail (default mirrors live config).")
    p.add_argument("--export-trades", default=None, help="Write trade list CSV.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.source:
        global DATA_SOURCE
        DATA_SOURCE = args.source
        os.environ["BACKTEST_SOURCE"] = args.source  # propagate to pool workers
    end_date_ms = None
    if args.end_date:
        from datetime import datetime as _dt, timezone as _tz
        end_date_ms = int(_dt.strptime(args.end_date, "%Y-%m-%d")
                          .replace(tzinfo=_tz.utc).timestamp() * 1000)
    if args.symbols:
        symbols = parse_symbols(args.symbols)
    elif args.top > 0:
        symbols = fetch_top_symbols(args.top)
    else:
        symbols = parse_symbols(None)
    worker_count = 1 if args.serial else (choose_workers(len(symbols), args.candles, args.stride) if args.workers <= 0 else args.workers)

    print(f"Fast backtest: {len(symbols)} symbols, {args.candles} candles, TP window {args.tp_window}")
    print(
        f"workers={worker_count}, stride={args.stride}, "
        f"prefilter={'off' if args.no_prefilter else 'on'}, cache={'refresh' if args.refresh_cache else 'ttl'}"
    )
    print()

    started = time.perf_counter()
    kwargs = dict(
        candles=args.candles,
        tp_window=args.tp_window,
        warmup=args.warmup,
        stride=max(1, args.stride),
        window_15m=args.window_15m,
        window_1h=args.window_1h,
        window_4h=args.window_4h,
        use_prefilter=not args.no_prefilter,
        refresh_cache=args.refresh_cache,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
        execution_delay_bars=max(0, args.execution_delay_bars),
        adverse_entry_bps=max(0.0, args.adverse_entry_bps),
        exit_policy=args.exit_policy,
        trail_atr_mult=max(0.0, args.trail_atr_mult),
        end_date_ms=end_date_ms,
    )

    results: list[SymbolResult] = []
    if worker_count == 1 or len(symbols) == 1:
        for symbol in symbols:
            r = backtest_symbol(symbol, **kwargs)
            results.append(r)
            if not args.quiet:
                print_symbol_result(r)
    else:
        workers = max(1, min(worker_count, len(symbols)))
        with ProcessPoolExecutor(max_workers=workers) as pool:
            future_map = {pool.submit(backtest_symbol, symbol, **kwargs): symbol for symbol in symbols}
            for fut in as_completed(future_map):
                r = fut.result()
                results.append(r)
                if not args.quiet:
                    print_symbol_result(r)

    wall_sec = time.perf_counter() - started
    total = merge_results(results)
    errors = [r for r in results if r.error]
    wins = total.tp1 + total.tp2
    win_rate = wins / total.trades * 100 if total.trades else 0.0
    gross_rpt = total.gross_r / total.trades if total.trades else 0.0
    net_rpt = total.net_r / total.trades if total.trades else 0.0
    total_rate = total.scanned / wall_sec if wall_sec > 0 else 0.0

    print("\n" + "=" * 72)
    print("BACKTEST RESULTS")
    print("=" * 72)
    print(f"Symbols:       {len(symbols)} ({len(errors)} errors)")
    print(f"Bars scanned:  {total.scanned} ({total_rate:,.0f} bars/s wall-clock)")
    _sess_note = "OFF (24/7)" if OFF_SESSION_SIGNALS else f"{total.off_session} bars skipped"
    print(f"US session gate: {_sess_note}")
    print(f"Heavy scans:   {total.analyzed}  skipped by prefilter: {total.prefiltered}")
    print(f"Trades:        {total.trades}")
    print(f"  TP1 hit:     {total.tp1}")
    print(f"  TP2 hit:     {total.tp2}")
    print(f"  SL hit:      {total.sl}")
    print(f"  Expired:     {total.expired}")
    print(f"Win rate:      {win_rate:.1f}%")
    print(f"Gross R:       {total.gross_r:+.2f}R total ({gross_rpt:+.3f}R/trade)")
    print(f"Net R est.:    {total.net_r:+.2f}R total ({net_rpt:+.3f}R/trade)")
    print(f"Max DD gross:  {max_drawdown_r(total.trade_records, net=False):+.2f}R")
    print(f"Max DD net:    {max_drawdown_r(total.trade_records, net=True):+.2f}R")

    # What the live bot could actually have carried — see apply_live_gates.
    if total.trade_records:
        gated = apply_live_gates(total.trade_records)
        if len(gated) != len(total.trade_records):
            g_net = sum(t.net_r for t in gated)
            g_wins = sum(1 for t in gated if t.outcome in ("TP1", "TP2", "TRAIL"))
            g_dd = max_drawdown_r(gated, net=True)
            print()
            print(
                # Candle count on the verdict line, not just the header —
                # a run that fell back to the small live default prints an
                # otherwise normal-looking summary. Ported from the crypto
                # bot, where exactly that cost a comparison.
                f"[{args.candles} candles"
                f"{' to ' + args.end_date if args.end_date else ' — ОКНО ПОЛЗЁТ'}] "
                f"With live gates (cooldown {SIGNAL_COOLDOWN_HOURS}h, "
                f"{_LIVE_MAX_PER_SCAN}/scan, {MAX_SAME_DIRECTION_POSITIONS}/dir, "
                f"kill {KILL_SWITCH_SL_STREAK}): "
                f"{len(gated)} trades "
                f"({len(total.trade_records) - len(gated)} refused), "
                f"WR {g_wins / len(gated) * 100:.1f}%, "
                f"net {g_net:+.2f}R, "
                f"Max DD {g_dd:+.2f}R, "
                f"profit/DD {g_net / abs(g_dd):.1f}"
            )
            _ordered = sorted(gated, key=lambda t: (t.entry_time or 0, t.symbol))
            _rp = risk_profile([t.net_r for t in _ordered])
            # worst_windows is the mean of the 5 most negative 25-trade sums,
            # negated. On a book where even the WORST 25-trade stretch made
            # money it therefore comes out NEGATIVE, and the ratio then prints
            # as a large negative number that reads like a catastrophe while
            # meaning the opposite. Seen for real on 2026-08-28 (-0.32R giving
            # a ratio of -703.8) on ~300-trade slices. Say so instead.
            _ww = _rp['worst_windows']
            _ulc = (f"ulcer {_rp['ulcer']:.2f} (прибыль/ulcer "
                    f"{g_net / _rp['ulcer']:.1f})" if _rp['ulcer'] > 0
                    else "ulcer 0 (не применим)")
            if _ww > 0:
                print(f"   устойчивый риск: худшие окна {_ww:.2f}R "
                      f"(прибыль/риск {g_net / _ww:.1f})   {_ulc}")
            else:
                print(f"   устойчивый риск: худшие окна НЕ ПРИМЕНИМЫ "
                      f"(ни один отрезок из 25 сделок не убыточен; сырое {_ww:.2f})"
                      f"   {_ulc}")
    print(f"Elapsed:       {wall_sec:.2f}s wall-clock")

    # Rejection funnel. Only meaningful with --serial: the workers have their
    # own REJECT_COUNTS and nothing collects them across processes.
    try:
        from src.signal_filter import REJECT_COUNTS as _RC
        if _RC:
            _tot = sum(_RC.values())
            print("")
            print(f"Воронка отказов ({_tot} шт., только --serial):")
            for _k, _v in sorted(_RC.items(), key=lambda kv: -kv[1])[:15]:
                print(f"  {_k:<38}{_v:>7}  {100*_v/_tot:>5.1f}%")
    except Exception:
        pass

    if args.export_trades:
        # Export the GATED book by default — an ungated dump cannot be compared
        # against the headline numbers, which are all post-gate, and its halves
        # describe a book production never trades. BT_EXPORT_RAW=1 restores the
        # full pre-gate list (used for Claude prior seeding in the crypto bot).
        _export = total.trade_records if os.getenv("BT_EXPORT_RAW") == "1"             else apply_live_gates(total.trade_records)
        write_trades_csv(args.export_trades, _export)
        print(f"Trades CSV:    {args.export_trades}")

    return 1 if errors and len(errors) == len(symbols) else 0


if __name__ == "__main__":
    raise SystemExit(main())
