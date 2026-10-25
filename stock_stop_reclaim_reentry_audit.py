"""Causal market re-entry audit after a wick-sensitive stock stop.

The original stop is always honored.  At most one recovery entry is allowed,
only after a completed 15-minute candle confirms a reclaim.  The recovery is
entered at the next market open and reuses the original material target and
stop.  Rules are selected on 2018-2020 and validated on 2021-2022; later
years remain sealed unless validation passes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import stock_stop_path_cross_era_oct12_audit as base


ROOT = base.ROOT
SOURCE_ROWS = ROOT / "stock_stop_path_cross_era_oct12_rows.csv"
SOURCE_REPORT = ROOT / "stock_stop_path_cross_era_oct12.json"
OUT = ROOT / "stock_stop_reclaim_reentry.json"
ROWS_OUT = ROOT / "stock_stop_reclaim_reentry.csv"
RULES = (
    "wick_inside_next_bar",
    "entry_reclaim",
    "entry_reclaim_no_structure_conflict",
    "entry_reclaim_index_aligned",
)
MAX_HOLD_BARS = 52


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def allowed(row: dict, rule: str) -> bool:
    if rule == "wick_inside_next_bar":
        return bool(row["stop_candle_closed_back_inside"])
    if rule == "entry_reclaim":
        return True
    if rule == "entry_reclaim_no_structure_conflict":
        return not (bool(row["adverse_failed_breakout"])
                    or bool(row["opposing_order_block_proxy"])
                    or bool(row["target_path_has_prior_swing"]))
    if rule == "entry_reclaim_index_aligned":
        return not bool(row["both_indices_against"])
    raise KeyError(rule)


def recovery(row: dict, candle: dict, rule: str) -> dict | None:
    if str(row["outcome"]) != "SL" or not allowed(row, rule):
        return None
    times = np.asarray(candle["time"], dtype=np.int64)
    opens = np.asarray(candle["open"], dtype=float)
    highs = np.asarray(candle["high"], dtype=float)
    lows = np.asarray(candle["low"], dtype=float)
    closes = np.asarray(candle["close"], dtype=float)
    lookup = {int(value): index for index, value in enumerate(times)}
    stop_index = lookup.get(int(float(row["exit_time"]))-900)
    if stop_index is None:
        raise ValueError(f"missing stop bar: {row['symbol']} {row['exit_time']}")
    direction = 1.0 if str(row["direction"]) == "LONG" else -1.0
    original_entry = float(row["entry_price_rebuilt"])
    original_stop = float(row["stop_price_rebuilt"])
    original_target = float(row["target_price_rebuilt"])
    if rule == "wick_inside_next_bar":
        confirm = stop_index
    else:
        confirm = None
        for index in range(stop_index, min(len(times)-1, stop_index+MAX_HOLD_BARS)):
            if direction*(closes[index]-original_entry) >= 0:
                confirm = index
                break
        if confirm is None:
            return None
    entry_index = confirm+1
    if entry_index >= len(times):
        return None
    entry = float(opens[entry_index])
    target_move = abs(original_target-entry)/entry
    risk = abs(entry-original_stop)/entry
    if not (.005 <= target_move <= .02) or risk <= 0 or risk > .03:
        return None
    signed = 0.0
    outcome = "EXPIRED"
    exit_time = int(times[min(len(times)-1, entry_index+MAX_HOLD_BARS-1)])+900
    for index in range(entry_index, min(len(times), entry_index+MAX_HOLD_BARS)):
        stop_hit = lows[index] <= original_stop if direction > 0 else highs[index] >= original_stop
        target_hit = highs[index] >= original_target if direction > 0 else lows[index] <= original_target
        if stop_hit:
            fill = min(opens[index], original_stop) if direction > 0 else max(opens[index], original_stop)
            signed = direction*(float(fill)-entry)/entry
            outcome = "SL"
            exit_time = int(times[index])+900
            break
        if target_hit:
            signed = direction*(original_target-entry)/entry
            outcome = "TP2"
            exit_time = int(times[index])+900
            break
    else:
        final = min(len(times)-1, entry_index+MAX_HOLD_BARS-1)
        signed = direction*(float(closes[final])-entry)/entry
    value = dict(row)
    value.update({
        "entry_time": int(times[entry_index]), "exit_time": exit_time,
        "entry_price_rebuilt": entry, "risk_pct": risk,
        "net_r": signed/risk, "outcome": outcome,
        "target_r": target_move/risk, "module": f"stop_reclaim_{rule}",
        "recovery_rule": rule, "confirmation_delay_bars": confirm-stop_index,
        "signed_return": signed,
    })
    return value


def no_symbol_overlap(row: dict, baseline: list[dict]) -> bool:
    now = float(row["entry_time"])
    symbol = str(row["symbol"])
    return not any(
        str(other["symbol"]) == symbol
        and float(other["entry_time"]) <= now < float(other["exit_time"])
        for other in baseline)


def recoveries(rows: list[dict], caches: dict[str, dict], rule: str) -> list[dict]:
    output = []
    for row in rows:
        found = recovery(row, caches[str(row["symbol"])], rule)
        if found is not None and no_symbol_overlap(found, rows):
            output.append(found)
    return output


def dominates(value: dict, baseline: dict) -> bool:
    return bool(
        value["return_3pct_margin_pct"] > baseline["return_3pct_margin_pct"]
        and value["dd_pct"] <= baseline["dd_pct"]
        and value["negative_months"] <= baseline["negative_months"]
        and value["minimum_month_pct"] >= baseline["minimum_month_pct"]
        and value["max_loss_streak"] <= baseline["max_loss_streak"]
        and value["projected_22_market_day_pct"]
        > baseline["projected_22_market_day_pct"])


def main() -> None:
    caches, cache_sources = base.load_history()
    rows = pd.read_csv(SOURCE_ROWS).to_dict("records")
    development = base.period(rows, base.DEVELOPMENT)
    development_baseline = base.metrics(development)
    frontier = []
    stored = {}
    for rule in RULES:
        added = recoveries(development, caches, rule)
        combined = [*development, *added]
        value = base.metrics(combined)
        item = {"rule": rule, "recovery_rows": len(added),
                "development": value,
                "development_pass": dominates(value, development_baseline)}
        frontier.append(item)
        stored[rule] = added
    eligible = [item for item in frontier if item["development_pass"]]
    eligible.sort(key=lambda item: (
        item["development"]["return_3pct_margin_pct"],
        item["development"]["minimum_month_pct"],
        -item["development"]["dd_pct"]), reverse=True)

    frozen = validation = test = None
    status = "REJECTED_DEVELOPMENT"
    output_rows = []
    if eligible:
        frozen = eligible[0]
        validation_base_rows = base.period(rows, base.VALIDATION)
        validation_baseline = base.metrics(validation_base_rows)
        validation_added = recoveries(
            validation_base_rows, caches, frozen["rule"])
        validation_rows = [*validation_base_rows, *validation_added]
        validation_value = base.metrics(validation_rows)
        validation = {"baseline": validation_baseline,
                      "recovery_rows": len(validation_added),
                      "combined": validation_value,
                      "passes": dominates(validation_value, validation_baseline)}
        output_rows = [*development, *stored[frozen["rule"]], *validation_rows]
        status = "REJECTED_VALIDATION"
        if validation["passes"]:
            test_base_rows = base.period(rows, base.TEST)
            test_added = recoveries(test_base_rows, caches, frozen["rule"])
            test_value = base.metrics([*test_base_rows, *test_added])
            test = {"baseline": base.metrics(test_base_rows),
                    "recovery_rows": len(test_added), "combined": test_value,
                    "passes": dominates(test_value, base.metrics(test_base_rows))}
            output_rows.extend([*test_base_rows, *test_added])
            status = ("STOP_RECLAIM_RESEARCH_CANDIDATE" if test["passes"]
                      else "REJECTED_TEST")
    pd.DataFrame(output_rows).to_csv(ROWS_OUT, index=False)
    objective = bool(test and test["passes"]
                     and test["combined"]["projected_22_market_day_pct"] >= 30
                     and test["combined"]["negative_months"] == 0
                     and test["combined"]["trades_per_market_day"] >= 3)
    report = {
        "status": status, "ready_for_live_orders": False,
        "order_eligible": False,
        "protocol": {
            "development": list(base.DEVELOPMENT),
            "validation": list(base.VALIDATION), "test_if_validation_passes": list(base.TEST),
            "confirmation": "closed 15m reclaim; next-bar market entry",
            "original_stop_honored": True, "maximum_reentries_per_stop": 1,
            "recovery_hold_bars": MAX_HOLD_BARS,
            "target_move_bounds_pct": [0.5, 2.0],
            "later_years_revealed_after_failure": False,
        },
        "development_baseline": development_baseline,
        "development_frontier": frontier,
        "frozen_before_validation": frozen,
        "validation": validation, "test": test,
        "economic_objective_pass": objective,
        "sources": {
            str(SOURCE_ROWS): digest(SOURCE_ROWS),
            str(SOURCE_REPORT): digest(SOURCE_REPORT),
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
            **cache_sources,
        },
        "limitations": [
            "Cash-proxy candles are not X-Perp fills.",
            "A recovery trade increases trade count after an already realized stop.",
            "Gross results exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": status, "development_baseline": development_baseline,
        "frontier": frontier, "frozen": frozen,
        "validation": validation, "test": test,
        "economic_objective_pass": objective,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
