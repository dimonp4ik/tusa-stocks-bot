"""Audit a causal re-entry after the frozen router's stop level is reclaimed.

The original setup is watched virtually.  No entry is made until its stop is
touched and that closed 15-minute candle finishes back inside the stop level.
The re-entry is the next regular-session bar open.
"""
from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path

import pandas as pd

import stock_stop_path_cross_era_oct12_audit as stop_audit
import stock_vwap_structure_oct15_audit as common


ROOT = Path("reports/audit_2026_09_08")
SOURCE = ROOT / "stock_stop_path_cross_era_oct12_rows.csv"
OUT = ROOT / "stock_stop_reclaim_reentry_oct15.json"
TARGETS = (.5, 1.0)
STOPS = (.5, .75, 1.0)
HOLDS = (1, 2)
MAX_OVERSHOOT_R = (.25, .50, 99.0)
MIN_REJECTION = (0.0, .30)
DEVELOPMENT = (2018, 2019, 2020)
VALIDATION = (2021, 2022)
TEST = (2023, 2024)
CONFIRMATION = (2025, 2026)


def regular_sessions(candle: dict) -> tuple[list[str], dict[str, list[int]]]:
    by_day = common.groups(candle)
    days = sorted(by_day)
    return days, by_day


def build_candidates(frame: pd.DataFrame, caches: dict[str, dict]) -> list[dict]:
    output = []
    session_cache = {symbol: regular_sessions(candle)
                     for symbol, candle in caches.items()}
    for row in frame.to_dict("records"):
        if str(row["outcome"]) != "SL" or not bool(row["stop_candle_closed_back_inside"]):
            continue
        symbol = str(row["symbol"])
        candle = caches[symbol]
        lookup = {int(value): i for i, value in enumerate(candle["time"])}
        stop_time = int(float(row["exit_time"])) - 900
        stop_i = lookup.get(stop_time)
        if stop_i is None:
            raise ValueError(f"missing mapped stop candle: {symbol} {stop_time}")
        days, by_day = session_cache[symbol]
        signal_day = pd.Timestamp(stop_time, unit="s", tz="UTC").tz_convert(
            common.NY).date().isoformat()
        if signal_day not in by_day or stop_i not in by_day[signal_day]:
            raise ValueError(f"stop outside regular session: {symbol} {stop_time}")
        day_position = days.index(signal_day)
        bars = by_day[signal_day]
        offset = bars.index(stop_i)
        later = bars[offset+1:]
        if not later and day_position+1 < len(days):
            later = by_day[days[day_position+1]][:]
            entry_day_position = day_position+1
        else:
            entry_day_position = day_position
        if not later:
            continue
        entry_i = later[0]
        first_execution = (bars[offset+1:] if offset+1 < len(bars) else [])
        if not first_execution and day_position+1 < len(days):
            first_execution = by_day[days[day_position+1]][:]
            first_end_position = day_position+1
        else:
            first_end_position = day_position
        second_execution = list(first_execution)
        if first_end_position+1 < len(days):
            second_execution += by_day[days[first_end_position+1]]

        direction = 1 if str(row["direction"]) == "LONG" else -1
        original_entry = float(row["entry_price_rebuilt"])
        original_stop = float(row["stop_price_rebuilt"])
        risk = abs(original_entry-original_stop)
        if risk <= 0:
            continue
        high = float(candle["high"][stop_i])
        low = float(candle["low"][stop_i])
        close = float(candle["close"][stop_i])
        adverse_extreme = low if direction > 0 else high
        overshoot = max(0.0, direction*(original_stop-adverse_extreme))/risk
        candle_range = max(high-low, 1e-12)
        rejection = ((close-low)/candle_range if direction > 0
                     else (high-close)/candle_range)
        output.append({
            "symbol": symbol, "family": str(row["family"]),
            "direction": str(row["direction"]), "direction_n": direction,
            "signal_time": stop_time, "entry_time": int(candle["time"][entry_i]),
            "entry_index": entry_i,
            "year": pd.Timestamp(stop_time, unit="s", tz="UTC").year,
            "session_date": signal_day, "overshoot_r": overshoot,
            "rejection_fraction": rejection,
            "reclaimed_original_entry": bool(
                close >= original_entry if direction > 0 else close <= original_entry),
            "volume_ratio_20": float(row.get("volume_ratio_20") or 0),
            "_execution_1": tuple(first_execution),
            "_execution_2": tuple(second_execution),
        })
    return output


def resolve(row: dict, candle: dict, target_pct: float,
            stop_pct: float, hold: int) -> dict:
    sign = int(row["direction_n"])
    entry = float(candle["open"][row["entry_index"]])
    target = entry*(1+sign*target_pct/100)
    stop = entry*(1-sign*stop_pct/100)
    execution = row[f"_execution_{hold}"]
    outcome, exit_i = "TIMEOUT", execution[-1]
    exit_price = float(candle["close"][exit_i])
    mfe = mae = 0.0
    for j in execution:
        open_ = float(candle["open"][j])
        high = float(candle["high"][j])
        low = float(candle["low"][j])
        mfe = max(mfe, 100*((high-entry) if sign > 0 else (entry-low))/entry)
        mae = max(mae, 100*((entry-low) if sign > 0 else (high-entry))/entry)
        stop_hit = low <= stop if sign > 0 else high >= stop
        target_hit = high >= target if sign > 0 else low <= target
        if stop_hit:
            outcome, exit_i = "SL", j
            exit_price = min(open_, stop) if sign > 0 else max(open_, stop)
            break
        if target_hit:
            outcome, exit_i, exit_price = "TP", j, target
            break
    signed = sign*(exit_price-entry)/entry
    return {
        **row, "exit_time": int(candle["time"][exit_i]), "outcome": outcome,
        "signed_return": signed, "mfe_pct": mfe, "mae_pct": mae,
        "target_pct": target_pct, "stop_pct": stop_pct,
        "hold_sessions": hold,
        "score": (1-float(row["overshoot_r"]))
                 +float(row["rejection_fraction"]),
    }


def select(rows: list[dict], config: dict) -> list[dict]:
    filtered = [row for row in rows
                if row["target_pct"] == config["target"]
                and row["stop_pct"] == config["stop"]
                and row["hold_sessions"] == config["hold"]
                and row["overshoot_r"] <= config["max_overshoot_r"]
                and row["rejection_fraction"] >= config["min_rejection"]
                and (not config["reclaim_entry"]
                     or row["reclaimed_original_entry"])]
    accepted, opened = [], {}
    per_scan = collections.Counter()
    for row in sorted(filtered, key=lambda item: (
            item["entry_time"], -item["score"], item["symbol"])):
        now = row["entry_time"]
        opened = {symbol: active for symbol, active in opened.items()
                  if active["exit_time"] >= now}
        if row["symbol"] in opened or per_scan[now] >= 2:
            continue
        accepted.append(row)
        opened[row["symbol"]] = row
        per_scan[now] += 1
    return accepted


def period_metrics(rows: list[dict], years: tuple[int, ...],
                   days: dict[int, set[str]]) -> dict:
    subset = [row for row in rows if row["year"] in years]
    count_days = len(set().union(*(days.get(year, set()) for year in years)))
    return common.metrics(subset, count_days)


def development_ok(value: dict, annual: dict[str, dict]) -> bool:
    return bool(value["n"] >= 60 and value["return_pct"] > 0
                and value["literal_target_hit_rate"] >= 65
                and value["dd_pct"] <= 8
                and value["trades_per_market_day"] >= .25
                and sum(item["return_pct"] > 0 for item in annual.values()) >= 2)


def validation_ok(value: dict, annual: dict[str, dict]) -> bool:
    return bool(value["n"] >= 50 and value["return_pct"] > 0
                and value["literal_target_hit_rate"] >= 65
                and value["dd_pct"] <= 6
                and value["trades_per_market_day"] >= .25
                and all(item["return_pct"] > 0 for item in annual.values()))


def main() -> None:
    caches, sources = stop_audit.load_history()
    frame = pd.read_csv(SOURCE)
    candidates = build_candidates(frame, caches)
    days = collections.defaultdict(set)
    for row in candidates:
        days[row["year"]].add(row["session_date"])
    resolved = [resolve(row, caches[row["symbol"]], target, stop, hold)
                for row in candidates for target in TARGETS for stop in STOPS
                for hold in HOLDS]

    frontier, qualified = [], []
    for overshoot in MAX_OVERSHOOT_R:
        for rejection in MIN_REJECTION:
            for reclaim_entry in (False, True):
                for target in TARGETS:
                    for stop in STOPS:
                        for hold in HOLDS:
                            config = {"max_overshoot_r": overshoot,
                                      "min_rejection": rejection,
                                      "reclaim_entry": reclaim_entry,
                                      "target": target, "stop": stop,
                                      "hold": hold}
                            accepted = select(resolved, config)
                            development = period_metrics(
                                accepted, DEVELOPMENT, days)
                            annual = {str(year): period_metrics(
                                          accepted, (year,), days)
                                      for year in DEVELOPMENT}
                            item = {"config": config,
                                    "development_2018_2020": development,
                                    "development_annual": annual}
                            frontier.append(item)
                            if development_ok(development, annual):
                                validation = period_metrics(
                                    accepted, VALIDATION, days)
                                validation_annual = {
                                    str(year): period_metrics(
                                        accepted, (year,), days)
                                    for year in VALIDATION}
                                item["validation_2021_2022"] = validation
                                item["validation_annual"] = validation_annual
                                if validation_ok(validation, validation_annual):
                                    item["_accepted"] = accepted
                                    qualified.append(item)

    frontier.sort(key=lambda item: (
        item["development_2018_2020"]["return_pct"],
        item["development_2018_2020"]["literal_target_hit_rate"],
        -item["development_2018_2020"]["dd_pct"]), reverse=True)
    qualified.sort(key=lambda item: (
        item["validation_2021_2022"]["return_pct"],
        item["validation_2021_2022"]["literal_target_hit_rate"]), reverse=True)
    frozen = qualified[0] if qualified else None
    accepted = frozen.get("_accepted") if frozen else None
    for item in qualified:
        item.pop("_accepted", None)
    test = confirmation = objective = None
    if frozen:
        test = period_metrics(accepted, TEST, days)
        test_annual = {str(year): period_metrics(accepted, (year,), days)
                       for year in TEST}
        test = {"combined": test, "annual": test_annual}
        if validation_ok(test["combined"], test["annual"]):
            confirmation = period_metrics(accepted, CONFIRMATION, days)
            confirmation_annual = {
                str(year): period_metrics(accepted, (year,), days)
                for year in CONFIRMATION}
            confirmation = {"combined": confirmation,
                            "annual": confirmation_annual}
            objective = bool(
                confirmation["combined"]["literal_target_hit_rate"] >= 75
                and confirmation["combined"]["dd_pct"] <= 5
                and confirmation["combined"]["negative_months"] == 0)

    source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    report = {
        "status": ("SUPPLEMENTAL_RESEARCH_CANDIDATE_AWAITING_DIRECT_TRANSFER"
                   if objective else "REJECTED"),
        "decision": ("Do not enable; causal gates or final objective failed."
                     if not objective else
                     "Do not enable; first verify untouched direct X-Perp transfer."),
        "protocol": {
            "mechanism": "virtual initial setup; enter after stop touch and closed reclaim",
            "development": list(DEVELOPMENT), "validation": list(VALIDATION),
            "test": list(TEST), "confirmation": list(CONFIRMATION),
            "entry": "next regular-session 15-minute bar open",
            "targets_pct": TARGETS, "stops_pct": STOPS, "holds": HOLDS,
            "ambiguity": "stop first if target and stop share a bar",
            "position_model": "3% current equity margin, fixed 10x gross",
        },
        "counts": {"source_rows": len(frame), "causal_reclaims": len(candidates),
                   "resolved_rows": len(resolved), "grid": len(frontier),
                   "development_qualified": sum(
                       "validation_2021_2022" in item for item in frontier),
                   "validation_qualified": len(qualified)},
        "frozen": frozen, "test": test, "confirmation": confirmation,
        "objective_passed": objective,
        "top_development_frontier": frontier[:20],
        "top_development_hit_rate": sorted(
            frontier, key=lambda item: (
                item["development_2018_2020"]["literal_target_hit_rate"],
                item["development_2018_2020"]["return_pct"]),
            reverse=True)[:20],
        "source_rows": {"path": str(SOURCE), "sha256": source_hash},
        "sources": sources,
        "limitations": [
            "The initial route and stop are virtual, but the reclaim event is causal.",
            "Cash CFD candles are structural proxies rather than OKX X-Perp fills.",
            "This is a supplemental low-frequency route, not a complete bot.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "status", "counts", "frozen", "test", "confirmation")}, indent=2))


if __name__ == "__main__":
    main()
