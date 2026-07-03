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

from config import (  # noqa: E402
    BACKTEST_CANDLES,
    BACKTEST_FEE_RATE,
    BACKTEST_SLIPPAGE_RATE,
    BACKTEST_TP_WINDOW,
    BLOCKED_SYMBOLS,
    BLOCK_STABLE_BASES,
    KLINES_1H_INTERVAL_SEC,
    KLINES_4H_INTERVAL_SEC,
    KLINES_INTERVAL_SEC,
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
)
from datetime import datetime as _datetime, timezone as _tz  # noqa: E402
from src.signal_filter import analyze_coin_smc  # noqa: E402
from src.knn_analog import knn_direction_score  # noqa: E402
from src.market_hours import is_market_open as _is_market_open  # noqa: E402


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


def fetch_history(
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
    if times:
        bar_indices: list[int] = []
        session_used = 0
        j = fill_bar
        n_all = len(highs)
        while j < n_all and session_used < window:
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

    for j in bar_indices:
        h = highs[j]
        l = lows[j]
        if not tp1_reached:
            if direction == "LONG":
                if l <= sl:
                    outcome = "SL"
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
                if h >= sl:
                    outcome = "SL"
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
                    best_price = max(best_price, h)
                    trailing_stop = max(entry, best_price - max(0.0, float(setup.get("atr", 0.0) or 0.0)) * trail_mult_eff)
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
                    best_price = min(best_price, l)
                    trailing_stop = min(entry, best_price + max(0.0, float(setup.get("atr", 0.0) or 0.0)) * trail_mult_eff)
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
    else:
        gross_r = gross_r_for_outcome(outcome, entry, tp1, tp2, sl)
    cost_r = estimate_cost_r(entry, sl, fee_rate, slippage_rate)
    net_r = gross_r - cost_r

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

    try:
        c15 = fetch_history(symbol, TIMEFRAME_KUCOIN, KLINES_INTERVAL_SEC, candles,
                            refresh_cache=refresh_cache, end_date_ms=end_date_ms)
        c1h = fetch_history(
            symbol,
            TIMEFRAME_1H_KUCOIN,
            KLINES_1H_INTERVAL_SEC,
            max(10, math.ceil(candles / 4) + 4),
            refresh_cache=refresh_cache,
            end_date_ms=end_date_ms,
        )
        c4h = fetch_history(
            symbol,
            TIMEFRAME_4H_KUCOIN,
            KLINES_4H_INTERVAL_SEC,
            max(10, math.ceil(candles / 16) + 4),
            refresh_cache=refresh_cache,
            end_date_ms=end_date_ms,
        )
        try:
            c1d = fetch_history(
                symbol, "1d", 86400,
                max(8, math.ceil(candles / 96) + 4),
                refresh_cache=refresh_cache,
                end_date_ms=end_date_ms,
            )
        except Exception:
            c1d = {}
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

        # US session gate — the live scanner only fires while NYSE is open
        # (unless OFF_SESSION_SIGNALS). Entering off-session here would test
        # trades the bot never takes. Judge on the just-closed candle (i-1),
        # matching how the live indicator labels the session.
        if not OFF_SESSION_SIGNALS:
            _ts = c15["time"][i - 1] if c15.get("time") and i > 0 else None
            if _ts is not None and not _is_market_open(
                _datetime.fromtimestamp(int(_ts), tz=_tz.utc)
            ):
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

        result.analyzed += 1
        setup = analyze_coin_smc(snap_15, snap_1h, symbol, snap_4h, btc_change_pct=0.0,
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
        "signals", "score_tags", "premium", "knn_score", "swing_trend",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for trade in sorted(trades, key=lambda t: (t.entry_time or 0, t.symbol, t.entry_bar)):
            writer.writerow({name: getattr(trade, name) for name in fields})


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fast SMC backtest")
    p.add_argument("--symbols", default=None, help="Comma-separated KuCoin symbols. Default: pinned set/env BACKTEST_SYMBOLS.")
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
    print(f"Elapsed:       {wall_sec:.2f}s wall-clock")

    if args.export_trades:
        write_trades_csv(args.export_trades, total.trade_records)
        print(f"Trades CSV:    {args.export_trades}")

    return 1 if errors and len(errors) == len(symbols) else 0


if __name__ == "__main__":
    raise SystemExit(main())
