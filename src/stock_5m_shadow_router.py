"""Frozen shadow-only stock X-Perp five-minute regime supplement.

The module proposes market brackets and never places orders.  It consumes only
closed five-minute candles, requires exact QQQ time alignment, and fails closed
outside the US regular session or when history is incomplete.
"""
from __future__ import annotations

from bisect import bisect_right
from datetime import datetime
import math
from statistics import median
from zoneinfo import ZoneInfo


NY = ZoneInfo("America/New_York")
RANGE_MODULE = "shadow_stock_5m_range_reversion_short"
PULLBACK_MODULE = "shadow_stock_5m_trend_pullback_short"
AUDITED_SYMBOLS = frozenset({
    "AAPLUSDT", "AMZNUSDT", "METAUSDT", "MRVLUSDT",
    "MSFTUSDT", "MUUSDT", "NVDAUSDT", "QQQUSDT",
})
RANGE_SIGNAL_MINUTES_MAX = 200
RANGE_VOLUME_RATIO_MAX = 1.77
RANGE_STOP_ATR = 1.5
RANGE_TARGET_R = .5
PULLBACK_QQQ_TREND_DIR_ATR_MAX = 2.750618811881214
PULLBACK_VOLUME_RATIO_MAX = .6818181818181818
PULLBACK_STOP_ATR = 2.0
PULLBACK_TARGET_R = .25


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _atr(candles: dict, index: int, length: int = 14) -> float | None:
    if index < length or index >= len(candles.get("close", [])):
        return None
    values = []
    for current in range(index - length + 1, index + 1):
        high = float(candles["high"][current])
        low = float(candles["low"][current])
        previous = float(candles["close"][current - 1])
        values.append(max(high - low, abs(high - previous), abs(low - previous)))
    result = _mean(values)
    return result if math.isfinite(result) and result > 0 else None


def _features(candles: dict, index: int) -> dict | None:
    if index < 96:
        return None
    close = [float(value) for value in candles["close"]]
    unit = _atr(candles, index)
    if unit is None:
        return None
    recent = close[index - 19:index + 1]
    ma20 = _mean(recent)
    ma96 = _mean(close[index - 95:index + 1])
    path = sum(abs(recent[pos] - recent[pos - 1]) for pos in range(1, 20))
    efficiency = abs(recent[-1] - recent[0]) / path if path else 0.0
    std = _mean([(value - ma20) ** 2 for value in recent]) ** .5
    zscore = (close[index] - ma20) / std if std else 0.0
    volumes = [float(value) for value in candles["volume"][index - 19:index + 1]]
    volume_ratio = volumes[-1] / (median(volumes[:-1]) or 1.0)
    return {
        "atr": unit,
        "ma20": ma20,
        "ma96": ma96,
        "efficiency": efficiency,
        "zscore": zscore,
        "volume_ratio": volume_ratio,
        "current": close[index],
        "previous": close[index - 1],
    }


def _range_short(candles: dict, index: int) -> bool:
    values = _features(candles, index)
    return bool(values and values["efficiency"] < .3 and values["zscore"] >= 2)


def _pullback_short(candles: dict, index: int) -> bool:
    values = _features(candles, index)
    return bool(values
                and values["ma20"] < values["ma96"]
                and values["current"] > values["ma20"] + .5 * values["atr"]
                and values["current"] < values["ma96"]
                and values["current"] < values["previous"])


def _aligned_index(candles: dict, timestamp: int) -> int | None:
    times = [int(value) for value in candles.get("time", [])]
    index = bisect_right(times, int(timestamp)) - 1
    return index if index >= 0 and times[index] == int(timestamp) else None


def candidate_context(candles: dict, qqq_candles: dict,
                      symbol: str) -> dict | None:
    required = ("time", "open", "high", "low", "close", "volume")
    if symbol not in AUDITED_SYMBOLS or any(
            len(candles.get(key, [])) < 97 for key in required):
        return None
    index = len(candles["close"]) - 1
    signal_time = int(candles["time"][index])
    local = datetime.fromtimestamp(signal_time, NY)
    minute = local.hour * 60 + local.minute
    if local.weekday() >= 5 or not 570 <= minute <= 950:
        return None
    qqq_index = _aligned_index(qqq_candles, signal_time)
    if qqq_index is None or qqq_index < 96:
        return None
    values = _features(candles, index)
    qqq_values = _features(qqq_candles, qqq_index)
    if values is None or qqq_values is None:
        return None
    signal_minutes = minute - 570

    range_fresh = (_range_short(candles, index)
                   and not _range_short(candles, index - 1))
    if (range_fresh and signal_minutes <= RANGE_SIGNAL_MINUTES_MAX
            and values["volume_ratio"] <= RANGE_VOLUME_RATIO_MAX):
        return {
            "module": RANGE_MODULE,
            "family": "range_reversion",
            "direction": "SHORT",
            "source_atr": values["atr"],
            "stop_atr_mult": RANGE_STOP_ATR,
            "target_r": RANGE_TARGET_R,
            "volume_ratio": values["volume_ratio"],
            "signal_minutes": signal_minutes,
            "signal_bar_ts": signal_time,
            "qqq_trend_dir_atr": -(qqq_values["ma20"] - qqq_values["ma96"])
            / qqq_values["atr"],
        }

    pullback_fresh = (_pullback_short(candles, index)
                      and not _pullback_short(candles, index - 1))
    qqq_trend_dir = -(qqq_values["ma20"] - qqq_values["ma96"]) / qqq_values["atr"]
    if (pullback_fresh
            and qqq_trend_dir <= PULLBACK_QQQ_TREND_DIR_ATR_MAX
            and values["volume_ratio"] <= PULLBACK_VOLUME_RATIO_MAX):
        return {
            "module": PULLBACK_MODULE,
            "family": "trend_pullback",
            "direction": "SHORT",
            "source_atr": values["atr"],
            "stop_atr_mult": PULLBACK_STOP_ATR,
            "target_r": PULLBACK_TARGET_R,
            "volume_ratio": values["volume_ratio"],
            "signal_minutes": signal_minutes,
            "signal_bar_ts": signal_time,
            "qqq_trend_dir_atr": qqq_trend_dir,
        }
    return None


def shadow_setup(candles: dict, qqq_candles: dict, *, entry_time: float,
                 market_price: float, symbol: str) -> dict | None:
    context = candidate_context(candles, qqq_candles, symbol)
    entry = float(market_price)
    if context is None or not math.isfinite(entry) or entry <= 0:
        return None
    risk = context["source_atr"] * context["stop_atr_mult"]
    if not math.isfinite(risk) or risk <= 0 or risk >= entry:
        return None
    return {
        **context,
        "symbol": symbol,
        "entry": entry,
        "atr": risk,
        "entry_time": float(entry_time),
        "qqq_regime": "5m_regime",
        "eff_ratio": _features(candles, len(candles["close"]) - 1)["efficiency"],
        "_shadow_experiment": True,
        "_five_minute_shadow": True,
    }
