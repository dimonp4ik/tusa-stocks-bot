"""Current stock X-Perp session router, disabled until explicitly configured.

The module proposes market-entry brackets only.  It never selects client size
or places an order.  Inputs must contain closed 15-minute X-Perp candles.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from statistics import median
from zoneinfo import ZoneInfo


NY = ZoneInfo("America/New_York")
AUDITED_SYMBOLS = frozenset({
    "AAPLUSDT", "AMZNUSDT", "GOOGLUSDT", "INTCUSDT", "METAUSDT", "MRVLUSDT",
    "MSFTUSDT", "MUUSDT", "NVDAUSDT", "QQQUSDT", "SPYUSDT", "TSLAUSDT",
})


def atr_at(candles: dict, index: int, period: int = 14) -> float | None:
    if index < period + 1:
        return None
    closes = candles["close"]
    values = [max(
        float(candles["high"][j]) - float(candles["low"][j]),
        abs(float(candles["high"][j]) - float(closes[j - 1])),
        abs(float(candles["low"][j]) - float(closes[j - 1])),
    ) for j in range(index - period, index)]
    value = sum(values) / period
    return value if math.isfinite(value) and value > 0 else None


def core_groups(candles: dict) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for index, timestamp in enumerate(candles["time"]):
        local = datetime.fromtimestamp(timestamp, NY)
        minute = local.hour * 60 + local.minute
        if local.weekday() < 5 and 570 <= minute < 960:
            groups.setdefault(local.date().isoformat(), []).append(index)
    return groups


def _day_context(candles: dict, groups: dict, days: list[str], position: int,
                 signal_bar: int, direction: int) -> dict | None:
    if position < 20:
        return None
    bars = groups[days[position]]
    previous = groups[days[position - 1]]
    atr = atr_at(candles, signal_bar)
    if not atr:
        return None
    closes = [float(candles["close"][groups[days[p]][-1]])
              for p in range(position - 20, position)]
    ranges = [float(max(candles["high"][j] for j in groups[days[p]])
                    - min(candles["low"][j] for j in groups[days[p]]))
              for p in range(position - 20, position)]
    previous_open = float(candles["open"][previous[0]])
    previous_close = float(candles["close"][previous[-1]])
    previous_high = max(float(candles["high"][j]) for j in previous)
    previous_low = min(float(candles["low"][j]) for j in previous)
    day_open = float(candles["open"][bars[0]])
    signal_open = float(candles["open"][signal_bar])
    signal_close = float(candles["close"][signal_bar])
    signal_high = float(candles["high"][signal_bar])
    signal_low = float(candles["low"][signal_bar])
    sma5, sma20 = sum(closes[-5:]) / 5, sum(closes) / 20
    normal_range = median(ranges) or atr
    local = datetime.fromtimestamp(candles["time"][signal_bar], NY)
    return {
        "signal_minutes": local.hour * 60 + local.minute - 570,
        "gap_dir_atr": direction * (day_open - previous_close) / atr,
        "prior_day_dir_atr": direction * (previous_close - previous_open) / atr,
        "prior_close_sma20_dir_atr": direction * (previous_close - sma20) / atr,
        "trend_5_20_dir_atr": direction * (sma5 - sma20) / atr,
        "prior_range_ratio": (previous_high - previous_low) / normal_range,
        "signal_body_dir_atr": direction * (signal_close - signal_open) / atr,
        "signal_range_atr": (signal_high - signal_low) / atr,
        "opening_move_dir_atr": direction * (signal_close - day_open) / atr,
        "atr_pct": atr / signal_close if signal_close > 0 else None,
        "previous_high": previous_high, "previous_low": previous_low, "atr": atr,
    }


def market_context_by_day(candles: dict) -> dict[str, dict]:
    groups = core_groups(candles)
    days = sorted(groups)
    result = {}
    for position, day in enumerate(days):
        if position < 20:
            continue
        bars, previous = groups[day], groups[days[position - 1]]
        atr = atr_at(candles, bars[0])
        if not atr:
            continue
        closes = [float(candles["close"][groups[days[p]][-1]])
                  for p in range(position - 20, position)]
        ranges = [float(max(candles["high"][j] for j in groups[days[p]])
                        - min(candles["low"][j] for j in groups[days[p]]))
                  for p in range(position - 20, position)]
        previous_open = float(candles["open"][previous[0]])
        previous_close = float(candles["close"][previous[-1]])
        day_open = float(candles["open"][bars[0]])
        sma5, sma20 = sum(closes[-5:]) / 5, sum(closes) / 20
        trend = (sma5 - sma20) / atr
        regime = "bull" if trend >= .25 else "bear" if trend <= -.25 else "range"
        result[day] = {
            "qqq_trend_5_20_atr": trend,
            "qqq_distance_sma20_atr": (previous_close - sma20) / atr,
            "qqq_prior_day_atr": (previous_close - previous_open) / atr,
            "qqq_gap_atr": (day_open - previous_close) / atr,
            "qqq_vol_ratio": (max(candles["high"][j] for j in previous)
                              - min(candles["low"][j] for j in previous))
                              / (median(ranges) or atr),
            "qqq_regime": regime,
        }
    return result


def attach_market_context(rows: list[dict], context: dict[str, dict]) -> None:
    for row in rows:
        day = datetime.fromtimestamp(row["entry_time"], NY).date().isoformat()
        market = context.get(day)
        if not market:
            continue
        sign = 1 if row["direction"] == "LONG" else -1
        row.update(market)
        row["qqq_trend_dir_atr"] = sign * market["qqq_trend_5_20_atr"]
        row["qqq_distance_dir_atr"] = sign * market["qqq_distance_sma20_atr"]
        row["qqq_prior_day_dir_atr"] = sign * market["qqq_prior_day_atr"]
        row["qqq_gap_dir_atr"] = sign * market["qqq_gap_atr"]


def market_intraday_context_by_time(candles: dict, prefix: str) -> dict[int, dict]:
    groups = core_groups(candles)
    result = {}
    for bars in groups.values():
        if not bars:
            continue
        day_open = float(candles["open"][bars[0]])
        for offset, bar in enumerate(bars):
            atr = atr_at(candles, bar)
            close = float(candles["close"][bar])
            if not atr or close <= 0:
                continue
            opening = bars[:min(2, offset + 1)]
            high = max(float(candles["high"][j]) for j in opening)
            low = min(float(candles["low"][j]) for j in opening)
            result[int(candles["time"][bar])] = {
                f"{prefix}_intraday_move_atr": (close - day_open) / atr,
                f"{prefix}_signal_body_atr": (
                    float(candles["close"][bar]) - float(candles["open"][bar])) / atr,
                f"{prefix}_signal_range_atr": (
                    float(candles["high"][bar]) - float(candles["low"][bar])) / atr,
                f"{prefix}_opening_range_atr": (high - low) / atr,
                f"{prefix}_opening_position_atr": (close - (high + low) / 2) / atr,
            }
    return result


def attach_intraday_market_context(rows: list[dict], context: dict[int, dict],
                                   prefix: str) -> None:
    for row in rows:
        market = context.get(int(row["entry_time"]) - 900)
        if not market:
            continue
        sign = 1 if row["direction"] == "LONG" else -1
        row.update(market)
        for suffix in ("intraday_move_atr", "signal_body_atr", "opening_position_atr"):
            stem = suffix.removesuffix("_atr")
            row[f"{prefix}_{stem}_dir_atr"] = sign * market[f"{prefix}_{suffix}"]


@dataclass(frozen=True)
class Module:
    name: str
    family: str
    direction: str
    qqq_regime: str
    session_bucket: str
    target_r: float
    conditions: tuple[tuple[str, str, float], ...]


PROFIT_MODULES = (
    Module("drive_short_bear_open", "opening_drive", "SHORT", "bear", "00_30", 1.0,
           (("trend_5_20_dir_atr", "ge", 1.0),)),
    Module("reclaim_long_bear_45_90", "opening_reclaim", "LONG", "bear", "45_90", 1.0,
           (("spy_opening_range_atr", "ge", 1.5), ("signal_body_dir_atr", "ge", .1))),
    Module("gap_fade_short_bear_open", "gap_fade", "SHORT", "bear", "00_30", 1.0,
           (("qqq_gap_dir_atr", "le", 1.0),)),
    Module("breakout_long_bear_45_90", "opening_breakout", "LONG", "bear", "45_90", 1.0,
           (("gap_dir_atr", "le", .5),)),
    Module("breakout_long_bear_open", "opening_breakout", "LONG", "bear", "00_30", .75,
           (("prior_range_ratio", "ge", 1.0),
            ("spy_opening_position_dir_atr", "ge", 0.0))),
    Module("drive_long_bear_open", "opening_drive", "LONG", "bear", "00_30", 1.0,
           (("prior_close_sma20_dir_atr", "le", -2.0),
            ("signal_body_dir_atr", "ge", 1.0))),
)
PRECISION_MODULES = PROFIT_MODULES[:1]
BALANCED_MODULES = (
    Module("drive_short_bear_open", "opening_drive", "SHORT", "bear", "00_30", 1.0,
           (("trend_5_20_dir_atr", "ge", 1.0),)),
    Module("reclaim_long_bear_45_90", "opening_reclaim", "LONG", "bear", "45_90", .25,
           (("spy_opening_range_atr", "ge", 1.5), ("signal_body_dir_atr", "ge", .1))),
    Module("breakout_long_bear_45_90", "opening_breakout", "LONG", "bear", "45_90", .25,
           (("gap_dir_atr", "le", .5),)),
    Module("breakout_long_bear_open", "opening_breakout", "LONG", "bear", "00_30", .5,
           (("prior_range_ratio", "ge", 1.0),
            ("spy_opening_position_dir_atr", "ge", 0.0))),
    Module("drive_long_bear_open", "opening_drive", "LONG", "bear", "00_30", .25,
           (("prior_close_sma20_dir_atr", "le", -2.0),
            ("signal_body_dir_atr", "ge", 1.0))),
)
FREQUENCY_MODULES = (
    Module("drive_short_bear_open", "opening_drive", "SHORT", "bear", "00_30", 1.0,
           (("trend_5_20_dir_atr", "ge", 1.0),)),
    Module("reclaim_long_bear_45_90", "opening_reclaim", "LONG", "bear", "45_90", .25,
           (("spy_opening_range_atr", "ge", 1.5), ("signal_body_dir_atr", "ge", .1))),
    Module("gap_fade_short_bear_open", "gap_fade", "SHORT", "bear", "00_30", 1.0,
           (("qqq_gap_dir_atr", "le", 1.0),)),
    Module("breakout_long_bear_45_90", "opening_breakout", "LONG", "bear", "45_90", .25,
           (("gap_dir_atr", "le", .5),)),
    Module("breakout_long_bear_open", "opening_breakout", "LONG", "bear", "00_30", .5,
           (("prior_range_ratio", "ge", 1.0),
            ("spy_opening_position_dir_atr", "ge", 0.0))),
    Module("drive_long_bear_open", "opening_drive", "LONG", "bear", "00_30", .25,
           (("prior_close_sma20_dir_atr", "le", -2.0),
            ("signal_body_dir_atr", "ge", 1.0))),
)
LOW_DRAWDOWN_MODULES = (
    Module("drive_short_bear_open", "opening_drive", "SHORT", "bear", "00_30", 1.0,
           (("trend_5_20_dir_atr", "ge", 1.0),)),
    Module("breakout_long_bear_45_90", "opening_breakout", "LONG", "bear", "45_90", .75,
           (("gap_dir_atr", "le", .5),)),
    Module("breakout_long_bear_open", "opening_breakout", "LONG", "bear", "00_30", .5,
           (("prior_range_ratio", "ge", 1.0),
            ("spy_opening_position_dir_atr", "ge", 0.0))),
)
ROBUST_FREQUENCY_EXCLUDED_SYMBOLS = frozenset({
    "GOOGLUSDT", "INTCUSDT", "SPYUSDT", "TSLAUSDT",
})
PROFILE_MODULES = {
    "profit": PROFIT_MODULES,
    "precision": PRECISION_MODULES,
    "balanced": BALANCED_MODULES,
    "frequency": FREQUENCY_MODULES,
    "low_drawdown": LOW_DRAWDOWN_MODULES,
    "robust_frequency": FREQUENCY_MODULES,
    "robust_dynamic": FREQUENCY_MODULES,
    "robust_dynamic_entry_filtered": FREQUENCY_MODULES,
}
DYNAMIC_TARGET_RULES = {
    "breakout_long_bear_45_90": ("gap_dir_atr", "le", -1.0, 1.0),
    "breakout_long_bear_open": ("signal_range_atr", "ge", 2.0, 1.0),
    "drive_long_bear_open": ("prior_day_dir_atr", "le", 1.0, 1.0),
}
ENTRY_FILTER_RULES = {
    "gap_fade_short_bear_open": ("qqq_gap_dir_atr", "le", -1.0),
    "breakout_long_bear_45_90": ("opening_range_atr", "ge", 2.0),
    "drive_long_bear_open": ("prior_range_ratio", "le", 1.5),
}


def _session_bucket(signal_minutes: float) -> str:
    if signal_minutes <= 30:
        return "00_30"
    if signal_minutes <= 90:
        return "45_90"
    if signal_minutes <= 150:
        return "105_150"
    return "165_plus"


def _latest_family_rows(candles: dict) -> list[dict]:
    groups = core_groups(candles)
    days = sorted(groups)
    if len(days) < 21:
        return []
    position = len(days) - 1
    bars = groups[days[position]]
    if not bars or bars[-1] != len(candles["time"]) - 1:
        return []
    if datetime.fromtimestamp(candles["time"][bars[0]], NY).strftime("%H:%M") != "09:30":
        return []
    signal_bar = bars[-1]
    entry_time = int(candles["time"][signal_bar]) + 900
    result = []

    def add(family: str, direction: int, boundary: float | None = None) -> None:
        context = _day_context(candles, groups, days, position, signal_bar, direction)
        if context is None:
            return
        opening = bars[:min(2, len(bars))]
        opening_high = max(float(candles["high"][index]) for index in opening)
        opening_low = min(float(candles["low"][index]) for index in opening)
        signal_close = float(candles["close"][signal_bar])
        row = {
            "family": family, "direction": "LONG" if direction == 1 else "SHORT",
            "entry_time": entry_time,
            "opening_range_atr": (opening_high - opening_low) / context["atr"],
            "extension_atr": (direction * (signal_close - boundary) / context["atr"]
                              if boundary is not None else None),
            **{key: value for key, value in context.items()
               if key not in {"previous_high", "previous_low", "atr"}},
        }
        row["session_bucket"] = _session_bucket(row["signal_minutes"])
        result.append(row)

    if len(bars) == 1 and position >= 20:
        previous = groups[days[position - 1]]
        previous_close = float(candles["close"][previous[-1]])
        day_open = float(candles["open"][bars[0]])
        first_close = float(candles["close"][bars[0]])
        gap_direction = 1 if day_open > previous_close else -1 if day_open < previous_close else 0
        first_direction = 1 if first_close > day_open else -1 if first_close < day_open else 0
        if gap_direction and first_direction and gap_direction != first_direction:
            add("gap_fade", first_direction, previous_close)

    if len(bars) == 2:
        drive = float(candles["close"][bars[1]]) - float(candles["open"][bars[0]])
        if drive:
            add("opening_drive", 1 if drive > 0 else -1)

    if len(bars) >= 3:
        opening_high = max(float(candles["high"][index]) for index in bars[:2])
        opening_low = min(float(candles["low"][index]) for index in bars[:2])
        first_breakout = None
        first_reclaim = None
        for index in bars[2:]:
            close = float(candles["close"][index])
            previous = float(candles["close"][index - 1])
            if first_breakout is None:
                if close > opening_high and previous <= opening_high:
                    first_breakout = (index, 1, opening_high)
                elif close < opening_low and previous >= opening_low:
                    first_breakout = (index, -1, opening_low)
            if first_reclaim is None:
                if float(candles["low"][index]) < opening_low < close:
                    first_reclaim = (index, 1, opening_low)
                elif float(candles["high"][index]) > opening_high > close:
                    first_reclaim = (index, -1, opening_high)
        if first_breakout and first_breakout[0] == signal_bar:
            add("opening_breakout", first_breakout[1], first_breakout[2])
        if first_reclaim and first_reclaim[0] == signal_bar:
            add("opening_reclaim", first_reclaim[1], first_reclaim[2])
    return result


def _matches(module: Module, row: dict) -> bool:
    if (module.family != row["family"] or module.direction != row["direction"]
            or module.qqq_regime != row.get("qqq_regime")
            or module.session_bucket != row["session_bucket"]):
        return False
    for field, operator, threshold in module.conditions:
        value = row.get(field)
        if value is None or not math.isfinite(float(value)):
            return False
        if operator == "ge" and value < threshold:
            return False
        if operator == "le" and value > threshold:
            return False
    return True


def _target_for(module: Module, row: dict, profile: str) -> float:
    if profile not in {"robust_dynamic", "robust_dynamic_entry_filtered"} \
            or module.name not in DYNAMIC_TARGET_RULES:
        return module.target_r
    field, operator, threshold, upgraded = DYNAMIC_TARGET_RULES[module.name]
    value = row.get(field)
    if value is None or not math.isfinite(float(value)):
        return module.target_r
    matches = value >= threshold if operator == "ge" else value <= threshold
    return upgraded if matches else module.target_r


def _entry_filter_matches(module: Module, row: dict, profile: str) -> bool:
    if profile != "robust_dynamic_entry_filtered" \
            or module.name not in ENTRY_FILTER_RULES:
        return True
    field, operator, threshold = ENTRY_FILTER_RULES[module.name]
    value = row.get(field)
    if value is None or not math.isfinite(float(value)):
        return False
    return value >= threshold if operator == "ge" else value <= threshold


def stock_venue_setups(candles: dict, qqq_candles: dict, spy_candles: dict, *,
                       market_price: float, profile: str = "profit",
                       symbol: str | None = None) -> list[dict]:
    if profile in {"robust_frequency", "robust_dynamic",
                   "robust_dynamic_entry_filtered"} and (
            not symbol or symbol in ROBUST_FREQUENCY_EXCLUDED_SYMBOLS):
        return []
    modules = PROFILE_MODULES.get(profile, PROFIT_MODULES)
    rows = _latest_family_rows(candles)
    if not rows:
        return []
    attach_market_context(rows, market_context_by_day(qqq_candles))
    attach_intraday_market_context(
        rows, market_intraday_context_by_time(qqq_candles, "qqq"), "qqq")
    attach_intraday_market_context(
        rows, market_intraday_context_by_time(spy_candles, "spy"), "spy")
    entry = float(market_price)
    risk = atr_at(candles, len(candles["time"]))
    if not risk or not math.isfinite(entry) or entry <= 0 or risk >= entry:
        return []
    for module in modules:
        for row in rows:
            if not _matches(module, row) or not _entry_filter_matches(
                    module, row, profile):
                continue
            target_r = _target_for(module, row, profile)
            sign = 1 if row["direction"] == "LONG" else -1
            stop = entry - sign * risk
            target = entry + sign * risk * target_r
            if min(stop, target) <= 0:
                continue
            return [{**row, "module": module.name, "entry_source": "MARKET",
                     "entry": entry, "sl": stop, "tp": target,
                     "atr": risk, "target_r": target_r,
                     "profile": profile}]
    return []
