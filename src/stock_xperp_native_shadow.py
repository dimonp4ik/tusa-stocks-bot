"""Frozen 24/7 native stock X-Perp cross-sectional shadow route.

The route reads confirmed 15-minute candles, measures the last two-bar move
against the simultaneous universe, and proposes next-bar market entries.  It
does not choose size, send Telegram messages, or place exchange orders.
"""
from __future__ import annotations

import math
import statistics
import time


AUDITED_SYMBOLS = frozenset({
    "AAOIUSDT", "AAPLUSDT", "AMDUSDT", "AMZNUSDT", "ARMUSDT", "AVGOUSDT",
    "AXTIUSDT", "BEUSDT", "BMNRUSDT", "CBRSUSDT", "COINUSDT", "CRCLUSDT",
    "CRWVUSDT", "DELLUSDT", "DRAMUSDT", "EWYUSDT", "GOOGLUSDT", "HOODUSDT",
    "INTCUSDT", "KOUSDT", "LITEUSDT", "METAUSDT", "MRNAUSDT", "MRVLUSDT",
    "MSFTUSDT", "MSTRUSDT", "MUUSDT", "NBISUSDT", "NVDAUSDT", "OKTAUSDT",
    "PLTRUSDT", "QCOMUSDT", "RKLBUSDT", "SAMSUNGUSDT", "SKHYNIXUSDT",
    "SKHYUSDT", "SNDKUSDT", "SNXXUSDT", "SOFTBANKUSDT", "SOXLUSDT",
    "SOXSUSDT", "SPCXUSDT", "TSLAUSDT", "WDCUSDT",
})
LOOKBACK_BARS = 2
MINIMUM_UNIVERSE = 10
MINIMUM_MOVE_FRACTION = .004
MINIMUM_ABS_Z = 2.0
MAXIMUM_ATR14_FRACTION = .01
STOP_FRACTION = .01
TARGET_FRACTION = .02
HOLD_BARS = 64
MAX_SIGNAL_AGE_SECONDS = 900


def _finite_series(candles: dict, key: str) -> list[float] | None:
    try:
        values = [float(value) for value in candles[key]]
    except (KeyError, TypeError, ValueError):
        return None
    return values if values and all(math.isfinite(value) for value in values) else None


def normalized_atr14(candles: dict, index: int) -> float | None:
    """ATR from 14 fully closed bars ending at ``index``, divided by close."""
    if index < 14:
        return None
    high = _finite_series(candles, "high")
    low = _finite_series(candles, "low")
    close = _finite_series(candles, "close")
    if not high or not low or not close or min(len(high), len(low), len(close)) <= index:
        return None
    values = []
    for cursor in range(index - 13, index + 1):
        values.append(max(
            high[cursor] - low[cursor],
            abs(high[cursor] - close[cursor - 1]),
            abs(low[cursor] - close[cursor - 1]),
        ))
    atr = sum(values) / len(values)
    value = atr / close[index] if close[index] > 0 else math.nan
    return value if math.isfinite(value) and value >= 0 else None


def _latest_rows(candles_by_symbol: dict[str, dict]) -> tuple[int, list[dict]] | None:
    by_time: dict[int, list[dict]] = {}
    for symbol in sorted(AUDITED_SYMBOLS & set(candles_by_symbol)):
        candles = candles_by_symbol[symbol]
        times = candles.get("time") or []
        close = _finite_series(candles, "close")
        if close is None or len(times) != len(close) or len(close) < LOOKBACK_BARS + 1:
            continue
        index = len(close) - 1
        try:
            signal_time = int(times[index])
            if index > 0 and signal_time - int(times[index - 1]) != 900:
                continue
        except (TypeError, ValueError):
            continue
        move = close[index] / close[index - LOOKBACK_BARS] - 1
        if not math.isfinite(move):
            continue
        by_time.setdefault(signal_time, []).append({
            "symbol": symbol,
            "signal_index": index,
            "signal_time": signal_time,
            "signal_return": move,
            "candles": candles,
        })
    usable = [(stamp, rows) for stamp, rows in by_time.items()
              if len(rows) >= MINIMUM_UNIVERSE]
    return max(usable, key=lambda item: item[0]) if usable else None


def native_shadow_setups(candles_by_symbol: dict[str, dict],
                         now: float | None = None) -> list[dict]:
    """Return all qualifying rows in causal quality order for portfolio gating."""
    found = _latest_rows(candles_by_symbol)
    if found is None:
        return []
    signal_time, rows = found
    entry_time = signal_time + 900
    current_time = time.time() if now is None else float(now)
    if current_time < entry_time or current_time - entry_time > MAX_SIGNAL_AGE_SECONDS:
        return []
    moves = [float(row["signal_return"]) for row in rows]
    if len(moves) < MINIMUM_UNIVERSE:
        return []
    mean = statistics.fmean(moves)
    deviation = statistics.stdev(moves)
    if not math.isfinite(deviation) or deviation <= 0:
        return []
    output = []
    for row in rows:
        move = float(row["signal_return"])
        z_score = (move - mean) / deviation
        atr_fraction = normalized_atr14(row["candles"], row["signal_index"])
        if (abs(z_score) < MINIMUM_ABS_Z
                or abs(move) < MINIMUM_MOVE_FRACTION
                or atr_fraction is None
                or atr_fraction > MAXIMUM_ATR14_FRACTION):
            continue
        direction = "SHORT" if move > 0 else "LONG"
        close = float(row["candles"]["close"][row["signal_index"]])
        output.append({
            "symbol": row["symbol"],
            "direction": direction,
            "entry": close,
            "atr": close * STOP_FRACTION,
            "target_r": TARGET_FRACTION / STOP_FRACTION,
            "signal_bar_ts": float(signal_time),
            "entry_time": float(entry_time),
            "cross_z": float(z_score),
            "signal_return": move,
            "cross_mean": mean,
            "cross_std": deviation,
            "universe_n": len(rows),
            "atr14_fraction": atr_fraction,
            "hold_bars": HOLD_BARS,
            "module": "native_xperp_quality_reversal",
            "_quality_rank": -abs(float(z_score)),
            "_native_xperp_shadow": True,
        })
    return sorted(output, key=lambda row: (
        row["_quality_rank"], row["symbol"]))
