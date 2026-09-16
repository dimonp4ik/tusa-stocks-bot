"""Research regime-aware US cash-session entries across 2019-2026.

The script deliberately keeps signal families simple and causal.  It selects
rules on whole calendar years 2020-2024, then reveals 2025-2026 and the recent
OKX X-Perp replay.  All entries are the next 15-minute bar open (a market-order
proxy); position size and fees are outside this filter experiment.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import itertools
import json
import math
import pickle
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from zoneinfo import ZoneInfo

import numpy as np

from filter_lab import gate, metrics
from index_session_hypotheses import atr_at, core_groups
from src.backtest_integrity import simulate_exit


NY = ZoneInfo("America/New_York")
FAMILIES = (
    "opening_breakout",
    "opening_reclaim",
    "opening_drive",
    "previous_day_breakout",
    "previous_day_reclaim",
    "gap_continuation",
    "gap_fade",
    "opening_breakout_retest",
)
TARGETS = (.25, .5, .75, 1.0)
DEVELOPMENT_YEARS = (2020, 2021, 2022, 2023, 2024)
HOLDOUT_YEARS = (2025, 2026)


def _directional(value: float | None, direction: int) -> float | None:
    return direction * value if value is not None else None


def _day_context(candles: dict, groups: dict, days: list[str], position: int,
                 signal_bar: int, direction: int) -> dict | None:
    if position < 20:
        return None
    bars = groups[days[position]]
    previous = groups[days[position - 1]]
    first = bars[0]
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
    day_open = float(candles["open"][first])
    signal_open = float(candles["open"][signal_bar])
    signal_close = float(candles["close"][signal_bar])
    signal_high = float(candles["high"][signal_bar])
    signal_low = float(candles["low"][signal_bar])
    sma5 = sum(closes[-5:]) / 5
    sma20 = sum(closes) / 20
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
        "previous_high": previous_high,
        "previous_low": previous_low,
        "atr": atr,
    }


def _candidate(family: str, direction: int, signal_bar: int, bars: list[int],
               context: dict, *, opening_high: float, opening_low: float,
               boundary: float | None = None) -> dict | None:
    try:
        offset = bars.index(signal_bar)
    except ValueError:
        return None
    if offset + 1 >= len(bars):
        return None
    atr = context["atr"]
    return {
        "family": family,
        "direction_int": direction,
        "signal_bar": signal_bar,
        "entry_bar": bars[offset + 1],
        "session_end": bars[-1],
        "opening_range_atr": (opening_high - opening_low) / atr,
        "extension_atr": direction * (
            float(context.get("signal_close", 0)) - boundary
        ) / atr if boundary is not None and context.get("signal_close") is not None else None,
        **{key: value for key, value in context.items()
           if key not in {"previous_high", "previous_low", "atr", "signal_close"}},
    }


def candidate_records(candles: dict) -> list[dict]:
    groups = core_groups(candles)
    days = sorted(groups)
    rows = []
    for position, day in enumerate(days):
        bars = groups[day]
        if len(bars) < 8:
            continue
        if datetime.fromtimestamp(candles["time"][bars[0]], NY).strftime("%H:%M") != "09:30":
            continue
        opening_bars = bars[:2]
        opening_high = max(float(candles["high"][j]) for j in opening_bars)
        opening_low = min(float(candles["low"][j]) for j in opening_bars)
        search = [j for j in bars[2:-1]
                  if datetime.fromtimestamp(candles["time"][j], NY).hour < 14]

        def add(family: str, direction: int, signal_bar: int,
                boundary: float | None = None) -> None:
            context = _day_context(candles, groups, days, position, signal_bar, direction)
            if context is None:
                return
            context["signal_close"] = float(candles["close"][signal_bar])
            # Gap families signal after the first 15-minute bar and enter at
            # the next bar open.  The second opening-range bar is future data
            # at that point, so only bars closed by signal time may define the
            # opening range.  Other families signal at/after the second bar.
            known_opening = [j for j in opening_bars if j <= signal_bar]
            known_high = max(float(candles["high"][j]) for j in known_opening)
            known_low = min(float(candles["low"][j]) for j in known_opening)
            item = _candidate(family, direction, signal_bar, bars, context,
                              opening_high=known_high, opening_low=known_low,
                              boundary=boundary)
            if item:
                rows.append(item)

        # First close outside the initial 30-minute range.
        for j in search:
            close, previous = float(candles["close"][j]), float(candles["close"][j - 1])
            if close > opening_high and previous <= opening_high:
                add("opening_breakout", 1, j, opening_high)
                break
            if close < opening_low and previous >= opening_low:
                add("opening_breakout", -1, j, opening_low)
                break

        # First failed probe of the opening range.
        for j in search:
            if float(candles["low"][j]) < opening_low < float(candles["close"][j]):
                add("opening_reclaim", 1, j, opening_low)
                break
            if float(candles["high"][j]) > opening_high > float(candles["close"][j]):
                add("opening_reclaim", -1, j, opening_high)
                break

        # Direction of the first 30-minute drive; strength is a searchable feature.
        drive = float(candles["close"][opening_bars[-1]]) - float(candles["open"][opening_bars[0]])
        if drive:
            add("opening_drive", 1 if drive > 0 else -1, opening_bars[-1])

        if position >= 20:
            previous = groups[days[position - 1]]
            previous_high = max(float(candles["high"][j]) for j in previous)
            previous_low = min(float(candles["low"][j]) for j in previous)
            previous_close = float(candles["close"][previous[-1]])
            day_open = float(candles["open"][bars[0]])

            # First close through yesterday's extreme.
            for j in bars[:-1]:
                close, prior = float(candles["close"][j]), float(candles["close"][j - 1])
                if close > previous_high and prior <= previous_high:
                    add("previous_day_breakout", 1, j, previous_high)
                    break
                if close < previous_low and prior >= previous_low:
                    add("previous_day_breakout", -1, j, previous_low)
                    break

            # First false break of yesterday's extreme.
            for j in bars[:-1]:
                if float(candles["low"][j]) < previous_low < float(candles["close"][j]):
                    add("previous_day_reclaim", 1, j, previous_low)
                    break
                if float(candles["high"][j]) > previous_high > float(candles["close"][j]):
                    add("previous_day_reclaim", -1, j, previous_high)
                    break

            gap_direction = 1 if day_open > previous_close else -1 if day_open < previous_close else 0
            first_close = float(candles["close"][bars[0]])
            first_direction = 1 if first_close > day_open else -1 if first_close < day_open else 0
            if gap_direction and first_direction:
                family = "gap_continuation" if gap_direction == first_direction else "gap_fade"
                add(family, first_direction, bars[0], previous_close)

        # Break, retest the range boundary, and close back in breakout direction.
        breakout = None
        for j in search:
            close = float(candles["close"][j])
            if breakout is None:
                if close > opening_high:
                    breakout = (1, opening_high)
                elif close < opening_low:
                    breakout = (-1, opening_low)
                continue
            direction, boundary = breakout
            touched = (float(candles["low"][j]) <= boundary if direction == 1
                       else float(candles["high"][j]) >= boundary)
            held = (close > boundary if direction == 1 else close < boundary)
            if touched and held:
                add("opening_breakout_retest", direction, j, boundary)
                break
    return rows


def market_context_by_day(candles: dict) -> dict[str, dict]:
    """Causal QQQ context known at the cash-session open."""
    groups = core_groups(candles)
    days = sorted(groups)
    result = {}
    for position, day in enumerate(days):
        if position < 20:
            continue
        bars = groups[day]
        previous = groups[days[position - 1]]
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
        normal_range = median(ranges) or atr
        trend = (sma5 - sma20) / atr
        if trend >= .25:
            regime = "bull"
        elif trend <= -.25:
            regime = "bear"
        else:
            regime = "range"
        result[day] = {
            "qqq_trend_5_20_atr": trend,
            "qqq_distance_sma20_atr": (previous_close - sma20) / atr,
            "qqq_prior_day_atr": (previous_close - previous_open) / atr,
            "qqq_gap_atr": (day_open - previous_close) / atr,
            "qqq_vol_ratio": (max(candles["high"][j] for j in previous)
                              - min(candles["low"][j] for j in previous)) / normal_range,
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
    """Index state at each closed 15-minute bar, using no later candles."""
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
            known_opening = bars[:min(2, offset + 1)]
            opening_high = max(float(candles["high"][j]) for j in known_opening)
            opening_low = min(float(candles["low"][j]) for j in known_opening)
            opening_mid = (opening_high + opening_low) / 2
            result[int(candles["time"][bar])] = {
                f"{prefix}_intraday_move_atr": (close - day_open) / atr,
                f"{prefix}_signal_body_atr": (
                    float(candles["close"][bar]) - float(candles["open"][bar])) / atr,
                f"{prefix}_signal_range_atr": (
                    float(candles["high"][bar]) - float(candles["low"][bar])) / atr,
                f"{prefix}_opening_range_atr": (opening_high - opening_low) / atr,
                f"{prefix}_opening_position_atr": (close - opening_mid) / atr,
            }
    return result


def attach_intraday_market_context(rows: list[dict], context: dict[int, dict],
                                   prefix: str) -> None:
    for row in rows:
        # Every family enters at the next bar open, so this is the exact last
        # fully closed index candle available when the decision is made.
        market = context.get(int(row["entry_time"]) - 900)
        if not market:
            continue
        sign = 1 if row["direction"] == "LONG" else -1
        row.update(market)
        for suffix in ("intraday_move_atr", "signal_body_atr",
                       "opening_position_atr"):
            stem = suffix.removesuffix("_atr")
            row[f"{prefix}_{stem}_dir_atr"] = sign * market[f"{prefix}_{suffix}"]


def replay(candles: dict, symbol: str, targets=TARGETS,
           execution: dict | None = None) -> list[dict]:
    execution = execution or candles
    lookup = {value: index for index, value in enumerate(execution["time"])}
    result_rows = []
    for item in candidate_records(candles):
        entry_time = candles["time"][item["entry_bar"]]
        end_time = candles["time"][item["session_end"]]
        if entry_time not in lookup or end_time not in lookup:
            continue
        first, last = lookup[entry_time], lookup[end_time]
        expected = list(range(entry_time, end_time + 900, 900))
        if execution["time"][first:last + 1] != expected:
            continue
        direction = item["direction_int"]
        signal_atr = atr_at(candles, item["entry_bar"])
        source_entry = float(candles["open"][item["entry_bar"]])
        entry = float(execution["open"][first])
        if not signal_atr or source_entry <= 0 or entry <= 0:
            continue
        risk = signal_atr * entry / source_entry
        side = "LONG" if direction == 1 else "SHORT"
        for target_r in targets:
            stop = entry - direction * risk
            take = entry + direction * risk * target_r
            if min(stop, take) <= 0:
                continue
            outcome = simulate_exit(
                execution, range(first, last + 1), direction=side, entry=entry,
                sl=stop, tp1=take, tp2=take, atr=risk, tp1_fraction=1,
                trail=False, trail_mult=0, stop_on_close=False, backstop_r=1,
                choose_trail=lambda *_: 0,
            )
            result_rows.append({
                "symbol": symbol, "family": item["family"], "direction": side,
                "target_r": target_r, "entry_time": entry_time,
                "exit_time": execution["time"][outcome.bar] + 900,
                "outcome": outcome.outcome, "net_r": outcome.gross_r,
                "risk_pct": risk / entry, "size_mult": 1,
                **{key: value for key, value in item.items()
                   if key not in {"family", "direction_int", "signal_bar", "entry_bar", "session_end"}},
            })
    return result_rows


LEVELS = {
    "signal_minutes": (15, 30, 45, 60, 90, 120, 180),
    "opening_range_atr": (.5, .75, 1, 1.25, 1.5, 2),
    "extension_atr": (0, .05, .1, .2, .35, .5),
    "gap_dir_atr": (-1, -.5, -.25, 0, .25, .5, 1),
    "prior_day_dir_atr": (-1, -.5, 0, .5, 1),
    "prior_close_sma20_dir_atr": (-2, -1, -.5, 0, .5, 1, 2),
    "trend_5_20_dir_atr": (-1, -.5, 0, .5, 1),
    "prior_range_ratio": (.5, .75, 1, 1.25, 1.5, 2),
    "signal_body_dir_atr": (-.5, 0, .1, .25, .5, .75, 1),
    "signal_range_atr": (.5, .75, 1, 1.5, 2),
    "opening_move_dir_atr": (-1, -.5, 0, .5, 1, 1.5),
    "atr_pct": (.0025, .005, .0075, .01, .015, .025),
    "qqq_trend_dir_atr": (-2, -1, -.5, 0, .5, 1, 2),
    "qqq_distance_dir_atr": (-3, -2, -1, 0, 1, 2, 3),
    "qqq_prior_day_dir_atr": (-1, -.5, 0, .5, 1),
    "qqq_gap_dir_atr": (-1, -.5, 0, .5, 1),
    "qqq_vol_ratio": (.5, .75, 1, 1.25, 1.5, 2),
}


def _matches(row: dict, rule: tuple) -> bool:
    for field, operator, value in rule:
        actual = row.get(field)
        if actual is None:
            return False
        if operator == "ge" and not actual >= value:
            return False
        if operator == "le" and not actual <= value:
            return False
        if operator == "eq" and actual != value:
            return False
    return True


def _year(row: dict) -> int:
    return datetime.fromtimestamp(row["entry_time"], timezone.utc).year


def _selection_score(by_year: dict[int, dict], aggregate: dict) -> tuple:
    positive = sum(by_year[y]["net_r"] > 0 for y in DEVELOPMENT_YEARS)
    worst = min(by_year[y]["net_r"] for y in DEVELOPMENT_YEARS)
    return (positive, worst, aggregate["daily_lcb"],
            aggregate["net_r"] / max(aggregate["dd_r"], .25), aggregate["net_r"])


def select_rules(rows: list[dict]) -> list[dict]:
    development = [row for row in rows if _year(row) in DEVELOPMENT_YEARS]
    atoms = [(field, operator, value) for field, values in LEVELS.items()
             for value in values for operator in ("ge", "le")]
    selected = []
    for family in FAMILIES:
        for direction in ("LONG", "SHORT"):
            for target in TARGETS:
                base = (('family', 'eq', family), ('direction', 'eq', direction),
                        ('target_r', 'eq', target))
                base_rows = [row for row in development if _matches(row, base)]
                if len(base_rows) < 100:
                    continue
                outcomes = np.asarray([row["net_r"] for row in base_rows], dtype=float)
                years = np.asarray([_year(row) for row in base_rows], dtype=int)
                symbol_ids = np.asarray([row["symbol"] for row in base_rows], dtype=object)
                value_arrays = {
                    field: np.asarray([
                        float(row[field]) if row.get(field) is not None else np.nan
                        for row in base_rows
                    ], dtype=float)
                    for field in LEVELS
                }

                def fast_metrics(mask: np.ndarray) -> dict:
                    values = outcomes[mask]
                    if not len(values):
                        return {"n": 0, "net_r": 0.0, "pf": None, "dd_r": 0.0,
                                "symbols": 0}
                    gains = values[values > 0].sum()
                    losses = -values[values < 0].sum()
                    curve = values.cumsum()
                    peaks = np.maximum.accumulate(np.maximum(curve, 0))
                    return {
                        "n": int(len(values)), "net_r": float(values.sum()),
                        "pf": float(gains / losses) if losses else None,
                        "dd_r": float(np.max(peaks - curve)),
                        "symbols": int(len(set(symbol_ids[mask]))),
                    }

                def atom_mask(atom: tuple) -> np.ndarray:
                    field, operator, value = atom
                    actual = value_arrays[field]
                    valid = np.isfinite(actual)
                    return valid & (actual >= value if operator == "ge" else actual <= value)

                atom_masks = {atom: atom_mask(atom) for atom in atoms}

                def evaluate(extra: tuple) -> tuple | None:
                    mask = np.ones(len(base_rows), dtype=bool)
                    for atom in extra:
                        mask &= atom_masks[atom]
                    aggregate = fast_metrics(mask)
                    yearly = {year: fast_metrics(mask & (years == year))
                              for year in DEVELOPMENT_YEARS}
                    if (aggregate["n"] < 70 or aggregate["symbols"] < 4
                            or any(yearly[y]["n"] < 7 for y in DEVELOPMENT_YEARS)):
                        return None
                    positive = sum(yearly[y]["net_r"] > 0 for y in DEVELOPMENT_YEARS)
                    worst = min(yearly[y]["net_r"] for y in DEVELOPMENT_YEARS)
                    score = (positive, worst,
                             aggregate["net_r"] / max(aggregate["dd_r"], .25),
                             aggregate["pf"] or 99.0, aggregate["net_r"])
                    return base + extra, yearly, aggregate, score

                evaluated_singles = [result for extra in ((), *((atom,) for atom in atoms))
                                     if (result := evaluate(extra))]
                evaluated_singles.sort(key=lambda item: item[3], reverse=True)
                qualified = [item for item in evaluated_singles
                    if (item[2]["n"] >= 80
                        and all(item[1][y]["n"] >= 8 for y in DEVELOPMENT_YEARS)
                        and sum(item[1][y]["net_r"] > 0 for y in DEVELOPMENT_YEARS) >= 4
                        and item[2]["net_r"] > 0 and (item[2]["pf"] or 0) >= 1.15)]
                top_atoms = [item[0][-1] for item in evaluated_singles
                             if len(item[0]) == len(base) + 1][:16]
                for first_atom, second_atom in itertools.combinations(top_atoms, 2):
                    if first_atom[0] == second_atom[0]:
                        continue
                    result = evaluate((first_atom, second_atom))
                    if result and (result[2]["n"] >= 70
                            and all(result[1][y]["n"] >= 7 for y in DEVELOPMENT_YEARS)
                            and sum(result[1][y]["net_r"] > 0 for y in DEVELOPMENT_YEARS) >= 4
                            and result[2]["net_r"] > 0 and (result[2]["pf"] or 0) >= 1.2):
                        qualified.append(result)
                if qualified:
                    best = max(qualified, key=lambda item: item[3])
                    rule = best[0]
                    exact = gate([row for row in base_rows if _matches(row, rule[len(base):])], 4)
                    exact_years = {year: metrics(gate(
                        [row for row in exact if _year(row) == year], 4))
                        for year in DEVELOPMENT_YEARS}
                    selected.append({"rule": rule, "development": metrics(exact),
                                     "development_years": exact_years,
                                     "selection_score": best[3]})
    return sorted(selected, key=lambda item: tuple(item["selection_score"]), reverse=True)


def _month_metrics(rows: list[dict]) -> dict:
    values = collections.defaultdict(list)
    for row in rows:
        key = datetime.fromtimestamp(row["entry_time"], timezone.utc).strftime("%Y-%m")
        values[key].append(row)
    return {key: metrics(gate(value, 4)) for key, value in sorted(values.items())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--symbols", required=True)
    parser.add_argument("--count", type=int, default=45000)
    parser.add_argument("--external-index-dir", type=Path)
    parser.add_argument("--external-execution-dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    symbols = tuple(value.strip() for value in args.symbols.split(",") if value.strip())
    rows, sources, candles_by_symbol = [], {}, {}
    for symbol in symbols:
        path = args.cache_dir / f"{symbol}_15min_{args.count}.pkl"
        raw = path.read_bytes()
        sources[str(path)] = hashlib.sha256(raw).hexdigest()
        candles_by_symbol[symbol] = pickle.loads(raw)
        rows.extend(replay(candles_by_symbol[symbol], symbol))
    if "QQQUSDT" in candles_by_symbol:
        attach_market_context(rows, market_context_by_day(candles_by_symbol["QQQUSDT"]))
    rows.sort(key=lambda row: (row["entry_time"], row["symbol"], row["family"], row["target_r"]))
    rules = select_rules(rows)

    external, external_signals = [], {}
    if args.external_index_dir and args.external_execution_dir:
        for symbol in symbols:
            signal_path = args.external_index_dir / f"{symbol}_15min_18000.pkl"
            execution_path = args.external_execution_dir / f"{symbol}_15min_18000.pkl"
            if not signal_path.exists() or not execution_path.exists():
                continue
            signal_raw, execution_raw = signal_path.read_bytes(), execution_path.read_bytes()
            sources[str(signal_path)] = hashlib.sha256(signal_raw).hexdigest()
            sources[str(execution_path)] = hashlib.sha256(execution_raw).hexdigest()
            external_signals[symbol] = pickle.loads(signal_raw)
            external.extend(replay(external_signals[symbol], symbol,
                                   execution=pickle.loads(execution_raw)))
    if "QQQUSDT" in external_signals:
        attach_market_context(external, market_context_by_day(external_signals["QQQUSDT"]))
    external.sort(key=lambda row: (row["entry_time"], row["symbol"], row["family"], row["target_r"]))

    for item in rules:
        rule = tuple(tuple(part) for part in item["rule"])
        held = gate([row for row in rows if _year(row) in HOLDOUT_YEARS and _matches(row, rule)], 4)
        item["holdout"] = metrics(held)
        item["holdout_years"] = {year: metrics(gate(
            [row for row in held if _year(row) == year], 4)) for year in HOLDOUT_YEARS}
        item["external_xperp"] = metrics(gate(
            [row for row in external if _matches(row, rule)], 4))
        item["all_years"] = {year: metrics(gate(
            [row for row in rows if _year(row) == year and _matches(row, rule)], 4))
            for year in (*DEVELOPMENT_YEARS, *HOLDOUT_YEARS)}
        item["all_months"] = _month_metrics([row for row in rows if _matches(row, rule)])
        item["regimes"] = {regime: metrics(gate([
            row for row in rows if row.get("qqq_regime") == regime and _matches(row, rule)
        ], 4)) for regime in ("bull", "bear", "range")}

    # A strict final shortlist requires both unseen years and the different OKX
    # execution feed to remain profitable.  These conditions are evaluated only
    # after each rule has already been chosen on 2020-2024.
    confirmed = [item for item in rules
                 if item["holdout"]["n"] >= 30
                 and item["holdout"]["net_r"] > 0
                 and (item["holdout"]["pf"] or 0) >= 1.15
                 and item["holdout_years"][2025]["net_r"] > 0
                 and item["holdout_years"][2026]["net_r"] > 0
                 and item["external_xperp"]["n"] >= 8
                 and item["external_xperp"]["net_r"] > 0]
    report = {
        "status": "RESEARCH_ONLY",
        "method": {
            "development_years": DEVELOPMENT_YEARS,
            "unseen_holdout_years": HOLDOUT_YEARS,
            "families": FAMILIES,
            "targets_r": TARGETS,
            "market_entry_proxy": "next 15-minute bar open",
            "position_size": "fixed externally by client",
            "fees": "excluded at user request",
        },
        "raw_candidates": len(rows), "selected_before_holdout": len(rules),
        "confirmed_after_holdout": len(confirmed),
        "confirmed": confirmed,
        "selected": rules,
        "sources": sources,
        "limitations": [
            "Dukascopy CFD candles validate the underlying session pattern, not exact OKX fills.",
            "Recent OKX X-Perp coverage is short and is used only as an execution-transfer check.",
            "Historical profitability cannot guarantee future profit or eliminate drawdowns.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    detail = args.out.with_suffix(".csv")
    if rows:
        with detail.open("w", newline="", encoding="utf-8") as handle:
            fields = list(dict.fromkeys(key for row in rows for key in row))
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({
        "raw_candidates": len(rows), "selected": len(rules),
        "confirmed": len(confirmed),
        "top": [{
            "rule": item["rule"],
            "development": {key: item["development"][key]
                            for key in ("n", "wr", "net_r", "pf", "dd_r")},
            "holdout": {key: item["holdout"][key]
                        for key in ("n", "wr", "net_r", "pf", "dd_r")},
            "external_xperp": {key: item["external_xperp"][key]
                               for key in ("n", "wr", "net_r", "pf", "dd_r")},
        } for item in confirmed[:5]],
    }, indent=2))


if __name__ == "__main__":
    main()
