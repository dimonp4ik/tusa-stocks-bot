"""Frozen 24/7 native stock X-Perp cross-sectional shadow route.

The route reads confirmed 15-minute candles, measures the last two-bar move
against the simultaneous universe, and proposes next-bar market entries.  It
does not choose size, send Telegram messages, or place exchange orders.
"""
from __future__ import annotations

import math
import statistics
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo


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
TARGET_FRACTION = .025
EXHAUSTED_TARGET_FRACTION = .02
EXPANDED_TARGET_FRACTION = .03
HOLD_BARS = 64
BACKING_LEVEL_LOOKBACK_BARS = 32
# Historical replay enters at the next 15-minute bar open.  The production
# scheduler normally sees that closed signal bar at +60 seconds.  Allow a small
# delivery buffer, but never let the +6/+11 minute retry ticks turn the same
# setup into a materially later entry that the replay never measured.
MAX_SIGNAL_AGE_SECONDS = 180
BASKET_TREND_BARS = 96
LONG_MIN_BASKET_TREND = -0.005626823134606253
SHORT_EXHAUSTION_BARS = 16
# Promoted in shadow after the pre-declared Oct-30 robustness grid was replayed
# through Nov-3.  The -0.75% variant improved aggregate return and reduced the
# fresh stop cluster while every complete month stayed profitable.  Real order
# enablement remains separately disabled.
SHORT_MIN_BASKET_TREND = -.0075
TARGET_EXHAUSTION_BARS = 32
TARGET_EXHAUSTION_THRESHOLD = -.04
TARGET_EXPANSION_START_HOUR = 6
TARGET_EXPANSION_END_HOUR = 12
PREMARKET_REGIME_TIMEZONE = "America/New_York"
PREMARKET_REGIME_BASKET_BARS = 64
PREMARKET_REGIME_TRIGGER_START_MINUTE = 3 * 60 + 15
PREMARKET_REGIME_TRIGGER_END_MINUTE = 4 * 60
PREMARKET_REGIME_LIQUIDITY_LOOKBACK_BARS = 8
PREMARKET_REGIME_MIN_ACTIVE_BARS = 6
PREMARKET_REGIME_MAX_LIQUID_UNIVERSE_RATIO = .55
PREMARKET_REGIME_MAX_BASKET_TREND_64 = -.06
# A Railway restart after the 03:15-04:00 New York trigger window must not
# forget a stress regime that happened earlier in the same local day.  Keep a
# full day of extra 15-minute bars so every later scan can reconstruct that
# window from closed candles instead of depending only on in-memory state.
PREMARKET_REGIME_RECOVERY_BARS = 96
REQUIRED_HISTORY_BARS = max(
    BASKET_TREND_BARS + LOOKBACK_BARS + 1,
    PREMARKET_REGIME_BASKET_BARS + LOOKBACK_BARS + 1
    + PREMARKET_REGIME_RECOVERY_BARS,
)
_NEW_YORK = ZoneInfo(PREMARKET_REGIME_TIMEZONE)


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


def backing_level(candles: dict, index: int,
                  direction: str) -> float | None:
    """Return the prior closed-bar level backing a reversal entry."""
    if index < BACKING_LEVEL_LOOKBACK_BARS - 1:
        return None
    key = "low" if direction == "LONG" else "high"
    values = _finite_series(candles, key)
    if values is None or len(values) <= index:
        return None
    start = index - BACKING_LEVEL_LOOKBACK_BARS + 1
    window = values[start:index + 1]
    level = min(window) if direction == "LONG" else max(window)
    return level if math.isfinite(level) and level > 0 else None


def backing_level_allows_entry(setup: dict, market_price: float) -> bool:
    """Reject an entry whose market price has crossed its backing level."""
    price = float(market_price)
    level = float(setup["backing_level"])
    if not math.isfinite(price) or price <= 0 or not math.isfinite(level):
        return False
    return price >= level if setup["direction"] == "LONG" else price <= level


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


def basket_trends(candles_by_symbol: dict[str, dict], signal_time: int,
                  periods: tuple[int, ...]) -> dict[int, float] | None:
    """Return causal two-bar cross-sectional sums for each requested period."""
    wanted_periods = tuple(sorted(set(int(value) for value in periods)))
    if not wanted_periods or wanted_periods[0] <= 0:
        raise ValueError("basket trend periods must be positive")
    means = []
    output = {}
    for offset in range(wanted_periods[-1]):
        wanted = int(signal_time) - 900 * offset
        moves = []
        for symbol in AUDITED_SYMBOLS & set(candles_by_symbol):
            candles = candles_by_symbol[symbol]
            times = candles.get("time") or []
            close = _finite_series(candles, "close")
            if close is None or len(times) != len(close):
                continue
            index = len(times) - 1 - offset
            if index < LOOKBACK_BARS:
                continue
            try:
                if (int(times[index]) != wanted
                        or int(times[index]) - int(times[index - LOOKBACK_BARS])
                        != 900 * LOOKBACK_BARS):
                    continue
            except (TypeError, ValueError):
                continue
            move = close[index] / close[index - LOOKBACK_BARS] - 1
            if math.isfinite(move):
                moves.append(move)
        if len(moves) < MINIMUM_UNIVERSE:
            return None
        means.append(statistics.fmean(moves))
        length = offset + 1
        if length in wanted_periods:
            value = sum(means)
            if not math.isfinite(value):
                return None
            output[length] = value
    return output


def basket_trend(candles_by_symbol: dict[str, dict], signal_time: int,
                 bars: int = BASKET_TREND_BARS) -> float | None:
    """Sum causal two-bar cross-sectional means ending at ``signal_time``."""
    values = basket_trends(candles_by_symbol, signal_time, (bars,))
    return None if values is None else values[bars]


def active_volume_bars(candles: dict, index: int,
                       lookback: int = PREMARKET_REGIME_LIQUIDITY_LOOKBACK_BARS
                       ) -> int | None:
    """Count fully closed bars with contracts traded, ending at ``index``."""
    index = int(index)
    if lookback < 1 or index < lookback - 1:
        return None
    volumes = candles.get("volume") or []
    if index >= len(volumes):
        return None
    try:
        values = [float(volumes[offset])
                  for offset in range(index - lookback + 1, index + 1)]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) and value >= 0 for value in values):
        return None
    return sum(value > 0 for value in values)


def liquid_universe_ratio(rows: list[dict]) -> float | None:
    """Share of aligned symbols trading in at least six of eight closed bars."""
    counts = [active_volume_bars(row["candles"], row["signal_index"])
              for row in rows]
    available = [value for value in counts if value is not None]
    if len(available) < MINIMUM_UNIVERSE:
        return None
    liquid = sum(value >= PREMARKET_REGIME_MIN_ACTIVE_BARS
                 for value in available)
    return liquid / len(available)


def new_york_calendar_day(stamp: float) -> str:
    """Return the DST-aware New York calendar day for a UTC epoch."""
    value = float(stamp)
    if not math.isfinite(value):
        raise ValueError("timestamp must be finite")
    return datetime.fromtimestamp(value, tz=timezone.utc).astimezone(
        _NEW_YORK).date().isoformat()


def premarket_regime_trigger(entry_time: float, basket_trend_64: float,
                             liquidity_ratio: float | None) -> bool:
    """Detect the audited illiquid crash regime at US premarket discovery."""
    if liquidity_ratio is None:
        return False
    stamp = float(entry_time)
    trend = float(basket_trend_64)
    ratio = float(liquidity_ratio)
    if not all(math.isfinite(value) for value in (stamp, trend, ratio)):
        return False
    local = datetime.fromtimestamp(stamp, tz=timezone.utc).astimezone(_NEW_YORK)
    minute = local.hour * 60 + local.minute
    return (
        PREMARKET_REGIME_TRIGGER_START_MINUTE <= minute
        < PREMARKET_REGIME_TRIGGER_END_MINUTE
        and trend <= PREMARKET_REGIME_MAX_BASKET_TREND_64
        and ratio <= PREMARKET_REGIME_MAX_LIQUID_UNIVERSE_RATIO
    )


def recover_premarket_regime(candles_by_symbol: dict[str, dict],
                             now: float | None = None) -> dict | None:
    """Recover today's premarket trigger after a process restart.

    The persistent latch is written by ``main`` after this function reports a
    trigger.  Rebuilding it from closed candles makes the latch independent of
    whether the worker happened to be alive during the three trigger bars.
    Only the current New York calendar day is eligible; an old crash cannot
    pause the following day.
    """
    current_time = time.time() if now is None else float(now)
    if not math.isfinite(current_time):
        raise ValueError("now must be finite")
    current_day = new_york_calendar_day(current_time)
    candidate_signal_times: set[int] = set()
    for symbol in AUDITED_SYMBOLS & set(candles_by_symbol):
        for raw_stamp in candles_by_symbol[symbol].get("time") or []:
            try:
                signal_time = int(raw_stamp)
                entry_time = signal_time + 900
                local = datetime.fromtimestamp(
                    entry_time, tz=timezone.utc).astimezone(_NEW_YORK)
            except (TypeError, ValueError, OverflowError):
                continue
            minute = local.hour * 60 + local.minute
            if (local.date().isoformat() == current_day
                    and PREMARKET_REGIME_TRIGGER_START_MINUTE <= minute
                    < PREMARKET_REGIME_TRIGGER_END_MINUTE):
                candidate_signal_times.add(signal_time)

    for signal_time in sorted(candidate_signal_times, reverse=True):
        sliced = {}
        for symbol in AUDITED_SYMBOLS & set(candles_by_symbol):
            series = candles_by_symbol[symbol]
            times = series.get("time") or []
            try:
                end = times.index(signal_time) + 1
            except ValueError:
                continue
            sliced[symbol] = {
                key: (list(values[:end]) if isinstance(values, (list, tuple))
                      else values)
                for key, values in series.items()
            }
        found = _latest_rows(sliced)
        if found is None or int(found[0]) != signal_time:
            continue
        _, rows = found
        trends = basket_trends(
            sliced, signal_time, (PREMARKET_REGIME_BASKET_BARS,))
        if trends is None:
            continue
        trend = trends[PREMARKET_REGIME_BASKET_BARS]
        liquidity = liquid_universe_ratio(rows)
        entry_time = signal_time + 900
        if premarket_regime_trigger(entry_time, trend, liquidity):
            return {
                "signal_time": signal_time,
                "entry_time": entry_time,
                "premarket_regime_new_york_day": current_day,
                "basket_trend_64": trend,
                "liquid_universe_ratio": liquidity,
                "aligned_symbols": len(rows),
            }
    return None


def routed_target_fraction(basket_trend_32: float,
                           entry_time: float) -> float:
    """Choose the audited shadow target from causal basket state and UTC hour."""
    trend = float(basket_trend_32)
    stamp = float(entry_time)
    if not math.isfinite(trend) or not math.isfinite(stamp):
        raise ValueError("target router inputs must be finite")
    if trend <= TARGET_EXHAUSTION_THRESHOLD:
        return EXHAUSTED_TARGET_FRACTION
    hour = int(stamp // 3600) % 24
    if TARGET_EXPANSION_START_HOUR <= hour < TARGET_EXPANSION_END_HOUR:
        return EXPANDED_TARGET_FRACTION
    return TARGET_FRACTION


def native_shadow_setups(candles_by_symbol: dict[str, dict],
                         now: float | None = None,
                         diagnostics: dict | None = None) -> list[dict]:
    """Return all qualifying rows in causal quality order for portfolio gating."""
    diag = diagnostics if diagnostics is not None else {}
    diag.clear()
    diag.update({
        "required_history_bars": REQUIRED_HISTORY_BARS,
        "symbols_received": len(AUDITED_SYMBOLS & set(candles_by_symbol)),
        "thresholds": {
            "minimum_universe": MINIMUM_UNIVERSE,
            "minimum_move_pct": MINIMUM_MOVE_FRACTION * 100,
            "minimum_abs_z": MINIMUM_ABS_Z,
            "maximum_atr14_pct": MAXIMUM_ATR14_FRACTION * 100,
        },
    })
    found = _latest_rows(candles_by_symbol)
    if found is None:
        diag["result"] = "no_aligned_universe"
        return []
    signal_time, rows = found
    diag.update(signal_time=signal_time, aligned_symbols=len(rows))
    entry_time = signal_time + 900
    current_time = time.time() if now is None else float(now)
    recovered_regime = recover_premarket_regime(
        candles_by_symbol, now=current_time)
    if recovered_regime is not None:
        diag.update({
            "result": "premarket_liquidity_regime_trigger",
            "premarket_regime_trigger": True,
            "premarket_regime_recovered_from_history": (
                int(recovered_regime["signal_time"]) != int(signal_time)),
            "premarket_regime_new_york_day":
                recovered_regime["premarket_regime_new_york_day"],
            "premarket_regime_signal_time":
                recovered_regime["signal_time"],
            "basket_trend_64_pct":
                float(recovered_regime["basket_trend_64"]) * 100,
            "liquid_universe_ratio":
                recovered_regime["liquid_universe_ratio"],
            "liquid_universe_symbols":
                recovered_regime["aligned_symbols"],
        })
        return []
    if current_time < entry_time or current_time - entry_time > MAX_SIGNAL_AGE_SECONDS:
        diag.update(
            result="stale_or_unclosed_bar",
            entry_time=entry_time,
            age_seconds=current_time - entry_time,
        )
        return []
    moves = [float(row["signal_return"]) for row in rows]
    if len(moves) < MINIMUM_UNIVERSE:
        return []
    mean = statistics.fmean(moves)
    deviation = statistics.stdev(moves)
    if not math.isfinite(deviation) or deviation <= 0:
        diag["result"] = "zero_cross_section_deviation"
        return []
    trends = basket_trends(
        candles_by_symbol, signal_time,
        (SHORT_EXHAUSTION_BARS, TARGET_EXHAUSTION_BARS,
         PREMARKET_REGIME_BASKET_BARS, BASKET_TREND_BARS))
    if trends is None:
        diag["result"] = "insufficient_regime_history"
        return []
    short_trend = trends[SHORT_EXHAUSTION_BARS]
    target_trend = trends[TARGET_EXHAUSTION_BARS]
    premarket_trend = trends[PREMARKET_REGIME_BASKET_BARS]
    long_trend = trends[BASKET_TREND_BARS]
    liquidity_ratio = liquid_universe_ratio(rows)
    if premarket_regime_trigger(
            entry_time, premarket_trend, liquidity_ratio):
        diag.update({
            "result": "premarket_liquidity_regime_trigger",
            "premarket_regime_trigger": True,
            "premarket_regime_new_york_day": new_york_calendar_day(entry_time),
            "basket_trend_64_pct": premarket_trend * 100,
            "liquid_universe_ratio": liquidity_ratio,
            "liquid_universe_symbols": len(rows),
        })
        return []
    target_fraction = routed_target_fraction(target_trend, entry_time)
    rejected = {
        "abs_z": 0, "minimum_move": 0, "atr_missing": 0, "atr_too_high": 0,
        "long_basket_trend": 0, "short_basket_trend": 0, "backing_level": 0,
    }
    output = []
    for row in rows:
        move = float(row["signal_return"])
        z_score = (move - mean) / deviation
        atr_fraction = normalized_atr14(row["candles"], row["signal_index"])
        if abs(z_score) < MINIMUM_ABS_Z:
            rejected["abs_z"] += 1
            continue
        if abs(move) < MINIMUM_MOVE_FRACTION:
            rejected["minimum_move"] += 1
            continue
        if atr_fraction is None:
            rejected["atr_missing"] += 1
            continue
        if atr_fraction > MAXIMUM_ATR14_FRACTION:
            rejected["atr_too_high"] += 1
            continue
        direction = "SHORT" if move > 0 else "LONG"
        if direction == "LONG" and long_trend < LONG_MIN_BASKET_TREND:
            rejected["long_basket_trend"] += 1
            continue
        if direction == "SHORT" and short_trend <= SHORT_MIN_BASKET_TREND:
            rejected["short_basket_trend"] += 1
            continue
        level = backing_level(
            row["candles"], row["signal_index"], direction)
        if level is None:
            rejected["backing_level"] += 1
            continue
        close = float(row["candles"]["close"][row["signal_index"]])
        output.append({
            "symbol": row["symbol"],
            "direction": direction,
            "entry": close,
            "atr": close * STOP_FRACTION,
            "target_r": target_fraction / STOP_FRACTION,
            "signal_bar_ts": float(signal_time),
            "entry_time": float(entry_time),
            "cross_z": float(z_score),
            "signal_return": move,
            "cross_mean": mean,
            "cross_std": deviation,
            "universe_n": len(rows),
            "atr14_fraction": atr_fraction,
            "backing_level": level,
            "backing_level_lookback_bars": BACKING_LEVEL_LOOKBACK_BARS,
            "basket_trend_16": short_trend,
            "basket_trend_32": target_trend,
            "basket_trend_64": premarket_trend,
            "basket_trend_96": long_trend,
            "liquid_universe_ratio": liquidity_ratio,
            "hold_bars": HOLD_BARS,
            "module": "native_xperp_quality_reversal",
            "_quality_rank": -abs(float(z_score)),
            "_native_xperp_shadow": True,
        })
    diag.update({
        "result": "candidates" if output else "no_candidate_passed",
        "cross_mean_pct": mean * 100,
        "cross_std_pct": deviation * 100,
        "basket_trend_16_pct": short_trend * 100,
        "basket_trend_32_pct": target_trend * 100,
        "basket_trend_64_pct": premarket_trend * 100,
        "basket_trend_96_pct": long_trend * 100,
        "liquid_universe_ratio": liquidity_ratio,
        "target_pct": target_fraction * 100,
        "rejected": rejected,
        "candidates": len(output),
    })
    return sorted(output, key=lambda row: (
        row["_quality_rank"], row["symbol"]))
