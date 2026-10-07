"""Long-history opening relative-volume audit with literal price targets.

The signal is built only from closed 15-minute cash candles.  Relative volume
uses the same opening slot from the previous 20 sessions.  Development ends in
2023; 2024 is not inspected unless a frozen 2022/2023 portfolio passes.
"""
from __future__ import annotations

import hashlib
import heapq
import itertools
import json
import math
import pickle
from collections import defaultdict
from pathlib import Path

import numba as nb
import numpy as np
import pandas as pd

from stock_5m_family_oct3_audit import regular_groups


ROOT = Path("reports/audit_2026_09_08")
MANIFEST = ROOT / "stock_extended_history_manifest.json"
OUT = ROOT / "stock_cash_opening_relative_volume_oct7.json"
ROWS_OUT = ROOT / "stock_cash_opening_relative_volume_oct7.csv"
SYMBOLS = ("AAPLUSDT", "AMZNUSDT", "GOOGLUSDT", "INTCUSDT", "METAUSDT",
           "MSFTUSDT", "MUUSDT", "NVDAUSDT", "TSLAUSDT")
SLOTS = (0, 1, 2, 3)
TARGETS = (.5, 1.0, 2.0)
STOPS = (.5, 1.0)
HOLDS = (4, 8, 16)
MARGIN_FRACTION = .03
LEVERAGE = 10.0


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_sources() -> tuple[dict, dict]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    output = {}
    sources = {str(MANIFEST): digest(MANIFEST)}
    for symbol in (*SYMBOLS, "QQQUSDT"):
        item = manifest["symbols"][symbol]
        path = Path(item["path"])
        raw = path.read_bytes()
        found = hashlib.sha256(raw).hexdigest()
        if found != item["sha256"]:
            raise ValueError(f"hash mismatch: {symbol}")
        output[symbol] = pickle.loads(raw)
        sources[str(path)] = found
    return output, sources


def build_features(data: dict) -> pd.DataFrame:
    qqq = data["QQQUSDT"]
    qgroups = regular_groups(qqq)
    qlookup = {int(value): index for index, value in enumerate(qqq["time"])}
    rows = []
    for symbol in SYMBOLS:
        source = data[symbol]
        groups = regular_groups(source)
        days = sorted(set(groups) & set(qgroups))
        history: list[dict] = []
        for position, day in enumerate(days):
            bars = groups[day]
            qbars = qgroups[day]
            if len(bars) < 18 or len(qbars) < 5 or position == 0:
                continue
            previous = groups[days[position - 1]]
            if len(previous) < 18:
                continue
            prior_close = float(source["close"][previous[-1]])
            q_prior = float(qqq["close"][qgroups[days[position - 1]][-1]])
            session_open = float(source["open"][bars[0]])
            q_open = float(qqq["open"][qbars[0]])
            day_high = max(float(source["high"][i]) for i in previous)
            day_low = min(float(source["low"][i]) for i in previous)
            prior_range = (day_high - day_low) / prior_close
            old_closes = [item["close"] for item in history[-20:]]
            trend20 = prior_close / old_closes[0] - 1 if len(old_closes) == 20 else 0.0
            range20 = float(np.median([item["range"] for item in history[-20:]])) if len(history) >= 20 else prior_range
            for slot in SLOTS:
                if slot + 1 >= len(bars):
                    continue
                signal = bars[slot]
                entry = bars[slot + 1]
                timestamp = int(source["time"][signal])
                qindex = qlookup.get(timestamp)
                if qindex is None or len(history) < 20:
                    continue
                cumulative_volume = sum(float(source["volume"][i]) for i in bars[:slot + 1])
                past_cumulative = [item["cum_volume"][slot] for item in history[-20:]]
                past_slot = [item["slot_volume"][slot] for item in history[-20:]]
                slot_volume = float(source["volume"][signal])
                close = float(source["close"][signal])
                high = max(float(source["high"][i]) for i in bars[:slot + 1])
                low = min(float(source["low"][i]) for i in bars[:slot + 1])
                opening_move = close / session_open - 1
                opening_range = (high - low) / session_open
                close_location = (close - low) / (high - low) if high > low else .5
                qmove = float(qqq["close"][qindex]) / q_open - 1
                rows.append({
                    "symbol": symbol, "session_date": day, "slot": slot + 1,
                    "entry_index": entry, "entry_time": int(source["time"][entry]),
                    "gap": session_open / prior_close - 1,
                    "opening_move": opening_move, "opening_range": opening_range,
                    "close_location": close_location,
                    "cum_rvol": cumulative_volume / (float(np.median(past_cumulative)) or 1.0),
                    "slot_rvol": slot_volume / (float(np.median(past_slot)) or 1.0),
                    "qqq_gap": q_open / q_prior - 1, "qqq_move": qmove,
                    "trend20": trend20, "prior_range_ratio": prior_range / (range20 or prior_range or 1.0),
                })
            current_high = max(float(source["high"][i]) for i in bars)
            current_low = min(float(source["low"][i]) for i in bars)
            current_close = float(source["close"][bars[-1]])
            history.append({
                "close": current_close, "range": (current_high - current_low) / current_close,
                "cum_volume": [sum(float(source["volume"][i]) for i in bars[:slot + 1]) for slot in SLOTS],
                "slot_volume": [float(source["volume"][bars[slot]]) for slot in SLOTS],
            })
    frame = pd.DataFrame(rows)
    stamp = pd.to_datetime(frame.entry_time, unit="s", utc=True)
    frame["year"] = stamp.dt.year
    frame["month"] = stamp.dt.strftime("%Y-%m")
    grouped = frame.groupby(["entry_time", "slot"], sort=False)
    frame["breadth_move"] = grouped.opening_move.transform(lambda x: (x > 0).mean())
    frame["breadth_gap"] = grouped.gap.transform(lambda x: (x > 0).mean())
    return frame.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)


def specs() -> list[dict]:
    output = []
    for slot in SLOTS:
        for gap, move, rvol, breadth in itertools.product((.003, .006, .01), (.001, .002), (1.25, 1.75, 2.5), (.5, .6)):
            output.append({"family": "gap_continuation", "slot": slot + 1, "gap": gap, "move": move, "rvol": rvol, "breadth": breadth})
        for move, rvol, location, breadth in itertools.product((.002, .004, .006), (1.25, 1.75, 2.5), (.7, .85), (.5, .6)):
            output.append({"family": "opening_momentum", "slot": slot + 1, "move": move, "rvol": rvol, "location": location, "breadth": breadth})
        for gap, reversal, rvol in itertools.product((.006, .01, .015), (.001, .002), (1.5, 2.5)):
            output.append({"family": "climactic_gap_fade", "slot": slot + 1, "gap": gap, "reversal": reversal, "rvol": rvol})
        for gap, reversal, rvol in itertools.product((.004, .007, .01), (.001, .002), (.75, 1.0)):
            output.append({"family": "quiet_gap_fade", "slot": slot + 1, "gap": gap, "reversal": reversal, "rvol_max": rvol})
        for move, rvol, trend in itertools.product((.002, .004), (1.25, 1.75), (.01, .03)):
            output.append({"family": "trend_opening_break", "slot": slot + 1, "move": move, "rvol": rvol, "trend": trend})
    for index, item in enumerate(output):
        item["id"] = index
    return output


def apply_spec(frame: pd.DataFrame, spec: dict) -> pd.DataFrame:
    found = frame[frame.slot.eq(spec["slot"])].copy()
    family = spec["family"]
    if family == "gap_continuation":
        direction = np.sign(found.gap)
        directed_gap = found.gap * direction
        directed_move = found.opening_move * direction
        directed_qqq = found.qqq_move * direction
        breadth = np.where(direction > 0, found.breadth_gap, 1 - found.breadth_gap)
        mask = (directed_gap >= spec["gap"]) & (directed_move >= spec["move"]) & (found.cum_rvol >= spec["rvol"]) & (directed_qqq >= -.001) & (breadth >= spec["breadth"])
    elif family == "opening_momentum":
        direction = np.sign(found.opening_move)
        directed_move = found.opening_move * direction
        location = np.where(direction > 0, found.close_location, 1 - found.close_location)
        breadth = np.where(direction > 0, found.breadth_move, 1 - found.breadth_move)
        mask = (directed_move >= spec["move"]) & (found.cum_rvol >= spec["rvol"]) & (location >= spec["location"]) & (breadth >= spec["breadth"])
    elif family in ("climactic_gap_fade", "quiet_gap_fade"):
        direction = -np.sign(found.gap)
        directed_gap = -found.gap * direction
        directed_move = found.opening_move * direction
        mask = (directed_gap >= spec["gap"]) & (directed_move >= spec["reversal"])
        if family == "climactic_gap_fade":
            mask &= found.cum_rvol >= spec["rvol"]
        else:
            mask &= found.cum_rvol <= spec["rvol_max"]
    else:
        direction = np.sign(found.trend20)
        directed_move = found.opening_move * direction
        directed_trend = found.trend20 * direction
        location = np.where(direction > 0, found.close_location, 1 - found.close_location)
        mask = (directed_trend >= spec["trend"]) & (directed_move >= spec["move"]) & (found.cum_rvol >= spec["rvol"]) & (location >= .7)
    found["direction"] = direction.astype(np.int8)
    found = found[mask & (direction != 0)].copy()
    found["family"] = family
    found["spec_id"] = spec["id"]
    found["score"] = found.opening_move.abs() + found.gap.abs() + .002 * np.log1p(found.cum_rvol)
    return found


@nb.njit(cache=True)
def resolve(open_, high, low, close, times, entries, directions, target, stop, hold):
    n = len(entries)
    returns = np.empty(n, dtype=np.float64)
    exits = np.empty(n, dtype=np.int64)
    codes = np.zeros(n, dtype=np.int8)
    for row in range(n):
        entry_index = entries[row]
        direction = directions[row]
        entry = open_[entry_index]
        target_price = entry * (1 + direction * target / 100)
        stop_price = entry * (1 - direction * stop / 100)
        last = min(len(open_) - 1, entry_index + hold - 1)
        exit_price = close[last]
        exit_index = last
        for index in range(entry_index, last + 1):
            target_hit = high[index] >= target_price if direction > 0 else low[index] <= target_price
            stop_hit = low[index] <= stop_price if direction > 0 else high[index] >= stop_price
            if target_hit and stop_hit:
                exit_price = stop_price
                exit_index = index
                codes[row] = -1
                break
            if stop_hit:
                exit_price = stop_price
                exit_index = index
                codes[row] = -1
                break
            if target_hit:
                exit_price = target_price
                exit_index = index
                codes[row] = 1
                break
        returns[row] = direction * (exit_price / entry - 1)
        exits[row] = times[exit_index]
    return returns, exits, codes


def add_outcomes(frame: pd.DataFrame, data: dict, target: float, stop: float, hold: int) -> pd.DataFrame:
    parts = []
    for symbol, group in frame.groupby("symbol", sort=False):
        source = data[symbol]
        values, exits, codes = resolve(
            np.asarray(source["open"], float), np.asarray(source["high"], float),
            np.asarray(source["low"], float), np.asarray(source["close"], float),
            np.asarray(source["time"], np.int64), group.entry_index.to_numpy(np.int64),
            group.direction.to_numpy(np.int8), target, stop, hold)
        found = group.copy()
        found["signed_return"] = values
        found["exit_time"] = exits
        found["outcome_code"] = codes
        found["target_pct"], found["stop_pct"], found["hold_bars"] = target, stop, hold
        parts.append(found)
    return pd.concat(parts, ignore_index=True) if parts else frame.iloc[0:0].copy()


def portfolio_rows(frame: pd.DataFrame) -> list[dict]:
    chosen = []
    last_symbol = defaultdict(lambda: -10**18)
    for timestamp, group in frame.sort_values(["entry_time", "score"], ascending=[True, False]).groupby("entry_time", sort=True):
        count = 0
        for row in group.to_dict("records"):
            if count >= 2 or timestamp - last_symbol[row["symbol"]] < 3600:
                continue
            chosen.append(row)
            last_symbol[row["symbol"]] = timestamp
            count += 1
    return chosen


def evaluate(frame: pd.DataFrame, year: int) -> dict:
    rows = portfolio_rows(frame[frame.year.eq(year)])
    if not rows:
        return {"n": 0, "wr": 0.0, "literal_target_hit_rate": 0.0, "return_pct": 0.0, "dd_pct": 0.0, "trades_per_calendar_day": 0.0, "negative_months": 0}
    equity = peak = 1.0
    drawdown = 0.0
    pending = []
    sequence = 0
    completed = []
    def realize(until):
        nonlocal equity, peak, drawdown
        while pending and (until is None or pending[0][0] < until):
            exit_time, _, pnl, row = heapq.heappop(pending)
            equity += pnl
            peak = max(peak, equity)
            drawdown = max(drawdown, (peak - equity) / peak)
            completed.append((exit_time, pnl, row))
    for row in sorted(rows, key=lambda x: (x["entry_time"], -x["score"])):
        realize(row["entry_time"])
        pnl = equity * MARGIN_FRACTION * LEVERAGE * row["signed_return"]
        sequence += 1
        heapq.heappush(pending, (row["exit_time"], sequence, pnl, row))
    realize(None)
    monthly_start, monthly_end = {}, {}
    ledger = 1.0
    for exit_time, pnl, _ in sorted(completed, key=lambda item: item[0]):
        month = pd.Timestamp(exit_time, unit="s", tz="UTC").strftime("%Y-%m")
        monthly_start.setdefault(month, ledger)
        ledger += pnl
        monthly_end[month] = ledger
    monthly = {key: 100 * (monthly_end[key] / monthly_start[key] - 1) for key in monthly_start}
    first = pd.Timestamp(min(row["entry_time"] for row in rows), unit="s", tz="UTC")
    last = pd.Timestamp(max(row["entry_time"] for row in rows), unit="s", tz="UTC")
    days = max(1, (last.date() - first.date()).days + 1)
    return {
        "n": len(rows), "wr": round(100 * sum(row["signed_return"] > 0 for row in rows) / len(rows), 2),
        "literal_target_hit_rate": round(100 * sum(row["outcome_code"] == 1 for row in rows) / len(rows), 2),
        "return_pct": round(100 * (equity - 1), 4), "dd_pct": round(100 * drawdown, 4),
        "trades_per_calendar_day": round(len(rows) / days, 3),
        "negative_months": sum(value < 0 for value in monthly.values()),
        "monthly_return_pct": {key: round(value, 4) for key, value in monthly.items()},
    }


def main() -> None:
    data, sources = load_sources()
    features = build_features(data)
    all_specs = specs()
    # The first pass ranks entry quality with one neutral literal exit.  Only a
    # small, predeclared number of rules per family reaches the exit sweep.
    screening = defaultdict(list)
    entry_cache = {}
    for spec in all_specs:
        entries = apply_spec(features, spec)
        if len(entries[entries.year.eq(2022)]) < 120:
            continue
        entry_cache[spec["id"]] = entries
        score = evaluate(add_outcomes(entries, data, .5, 1.0, 8), 2022)
        if score["n"] >= 120 and score["literal_target_hit_rate"] >= 55 and score["return_pct"] > -2:
            screening[spec["family"]].append((score["return_pct"], score["literal_target_hit_rate"], spec))
    shortlisted = []
    for values in screening.values():
        values.sort(key=lambda item: (item[0], item[1]), reverse=True)
        shortlisted.extend(item[2] for item in values[:15])
    candidates = []
    for spec in shortlisted:
        entries = entry_cache[spec["id"]]
        for target, stop, hold in itertools.product(TARGETS, STOPS, HOLDS):
            key = (spec["id"], target, stop, hold)
            resolved = add_outcomes(entries, data, target, stop, hold)
            score22 = evaluate(resolved, 2022)
            if score22["n"] >= 120 and score22["return_pct"] > 0 and score22["dd_pct"] <= 8 and score22["literal_target_hit_rate"] >= 58:
                candidates.append({"spec": spec, "exit": {"target_pct": target, "stop_pct": stop, "hold_bars": hold}, "score_2022": score22, "key": key})
    by_family = defaultdict(list)
    for candidate in candidates:
        spec = candidate["spec"]
        exit_spec = candidate["exit"]
        resolved = add_outcomes(entry_cache[spec["id"]], data, exit_spec["target_pct"], exit_spec["stop_pct"], exit_spec["hold_bars"])
        candidate["score_2023"] = evaluate(resolved, 2023)
        by_family[candidate["spec"]["family"]].append(candidate)
    frozen = []
    diagnostics = {}
    for family, values in by_family.items():
        values.sort(key=lambda x: (x["score_2023"]["return_pct"], x["score_2023"]["literal_target_hit_rate"], -x["score_2023"]["dd_pct"]), reverse=True)
        diagnostics[family] = [{k: v for k, v in item.items() if k != "key"} for item in values[:3]]
        best = values[0]
        score = best["score_2023"]
        if score["n"] >= 100 and score["return_pct"] > 0 and score["dd_pct"] <= 8 and score["literal_target_hit_rate"] >= 58:
            frozen.append(best)
    frozen_rows = []
    for item in frozen:
        spec, exit_spec = item["spec"], item["exit"]
        frozen_rows.append(add_outcomes(entry_cache[spec["id"]], data, exit_spec["target_pct"], exit_spec["stop_pct"], exit_spec["hold_bars"]))
    portfolio = pd.concat(frozen_rows, ignore_index=True) if frozen_rows else features.iloc[0:0].copy()
    development = {str(year): evaluate(portfolio, year) for year in (2022, 2023)} if frozen else {}
    pass_before_test = bool(frozen and development["2022"]["return_pct"] > 0 and development["2023"]["return_pct"] > 0 and development["2022"]["dd_pct"] <= 7 and development["2023"]["dd_pct"] <= 7 and development["2022"]["trades_per_calendar_day"] >= 2.5 and development["2023"]["trades_per_calendar_day"] >= 2.5)
    test_2024 = evaluate(portfolio, 2024) if pass_before_test else None
    pass_2024 = bool(test_2024 and test_2024["return_pct"] > 0 and test_2024["dd_pct"] <= 7 and test_2024["literal_target_hit_rate"] >= 58 and test_2024["trades_per_calendar_day"] >= 2.5)
    hidden = {str(year): evaluate(portfolio, year) for year in (2025, 2026)} if pass_2024 else None
    status = "SHADOW_CANDIDATE" if hidden and all(item["return_pct"] > 0 and item["dd_pct"] <= 7 for item in hidden.values()) else "REJECTED"
    if not portfolio.empty:
        portfolio.to_csv(ROWS_OUT, index=False)
    result = {
        "status": status,
        "protocol": {"source": "Dukascopy regular-session 15m cash candles", "premarket_available": False, "entry": "next 15m market open", "money": "3% entry equity margin, fixed 10x gross", "literal_targets_pct": TARGETS, "stops_pct": STOPS, "fit_context": "rolling prior 20 sessions", "selection": 2022, "validation_and_ranking": 2023, "untouched_test": 2024, "hidden_if_test_passes": [2025, 2026]},
        "feature_rows": len(features), "candidate_grid": len(all_specs) * len(TARGETS) * len(STOPS) * len(HOLDS), "screened_entries": len(shortlisted), "qualified_2022": len(candidates),
        "family_diagnostics": diagnostics,
        "frozen_before_2024": [{k: v for k, v in item.items() if k != "key"} for item in frozen],
        "development_portfolio": development, "pass_before_2024": pass_before_test,
        "test_2024": test_2024, "hidden": hidden,
        "sources": sources,
        "limitations": ["Dukascopy volume is venue tick volume, so only causal within-symbol relative ranks are used.", "The archive contains regular-session bars and cannot test true premarket volume.", "Any cash-history pass still requires untouched X-Perp transfer and live shadow observation."],
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": status, "qualified_2022": len(candidates), "frozen": len(frozen), "development": development, "test_2024": test_2024, "hidden": hidden}, indent=2))


if __name__ == "__main__":
    main()
