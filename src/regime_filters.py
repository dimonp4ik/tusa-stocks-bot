"""Independent causal cash-session filters validated outside legacy SMC."""
from __future__ import annotations

import math
from datetime import datetime
from zoneinfo import ZoneInfo


NY = ZoneInfo("America/New_York")


def _atr_before(candles: dict, index: int, period: int = 14) -> float | None:
    if index < period + 1:
        return None
    closes = candles["close"]
    values = [max(
        float(candles["high"][j]) - float(candles["low"][j]),
        abs(float(candles["high"][j]) - float(closes[j - 1])),
        abs(float(candles["low"][j]) - float(closes[j - 1])),
    ) for j in range(index - period, index)]
    result = sum(values) / period
    return result if math.isfinite(result) and result > 0 else None


def stock_opening_breakout_short(candles: dict, market_price: float | None = None,
                                 target_r: float = 0.5) -> dict | None:
    """Return a short only when the latest closed bar is today's first OR break.

    The previous day must not already have fallen by more than one signal-time
    ATR, which avoids entering an exhausted continuation.
    """
    times = candles.get("time", [])
    if len(times) < 20 or target_r <= 0:
        return None
    groups: dict[str, list[int]] = {}
    for index, ts in enumerate(times):
        local = datetime.fromtimestamp(float(ts), NY)
        minute = local.hour * 60 + local.minute
        if local.weekday() < 5 and 570 <= minute < 960:
            groups.setdefault(local.date().isoformat(), []).append(index)
    days = sorted(groups)
    if len(days) < 2:
        return None
    bars = groups[days[-1]]
    if len(bars) < 3 or bars[-1] != len(times) - 1:
        return None
    if datetime.fromtimestamp(float(times[bars[0]]), NY).strftime("%H:%M") != "09:30":
        return None
    opening = bars[:2]
    opening_high = max(float(candles["high"][j]) for j in opening)
    opening_low = min(float(candles["low"][j]) for j in opening)
    first_cross = None
    for j in bars[2:]:
        if datetime.fromtimestamp(float(times[j]), NY).hour >= 14:
            break
        close = float(candles["close"][j])
        previous = float(candles["close"][j - 1])
        if close > opening_high and previous <= opening_high:
            first_cross = ("LONG", j)
            break
        if close < opening_low and previous >= opening_low:
            first_cross = ("SHORT", j)
            break
    if first_cross != ("SHORT", bars[-1]):
        return None
    signal_atr = _atr_before(candles, bars[-1])
    entry_atr = _atr_before(candles, bars[-1] + 1)
    prior = groups[days[-2]]
    if not signal_atr or not entry_atr or not prior:
        return None
    prior_move = -(float(candles["close"][prior[-1]])
                   - float(candles["open"][prior[0]])) / signal_atr
    if prior_move > 1:
        return None
    entry = float(market_price if market_price is not None else candles["close"][-1])
    if not math.isfinite(entry) or entry <= 0 or entry_atr >= entry:
        return None
    return {
        "family": "opening_breakout_short",
        "direction": "SHORT",
        "entry": entry,
        "sl": entry + entry_atr,
        "tp": entry - target_r * entry_atr,
        "target_r": target_r,
        "atr": entry_atr,
        "prior_day_directional_atr": prior_move,
        "opening_low": opening_low,
    }
