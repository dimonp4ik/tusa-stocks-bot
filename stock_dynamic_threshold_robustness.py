"""Stress the frozen dynamic target rules at nearby thresholds."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import pandas as pd

from stock_dynamic_target_candidate import (
    BASE_TARGETS, EXCLUDED_SYMBOLS, HOLDOUT_MONTH, MODULE_NAMES, metric, route_rows,
)
from stock_strict_causal_modules import add_context


GRIDS = {
    "breakout_long_bear_45_90": ("gap_dir_atr", "le", (-1.5, -1.0, -.5)),
    "breakout_long_bear_open": ("signal_range_atr", "ge", (1.5, 2.0, 2.5)),
    "drive_long_bear_open": ("prior_day_dir_atr", "le", (.5, 1.0, 1.5)),
}
FROZEN = (-1.0, 2.0, 1.0)
MONTHS = ("2026-07", "2026-08", HOLDOUT_MONTH)


def materialize(frame: pd.DataFrame, selected: list[dict], thresholds: tuple) -> pd.DataFrame:
    threshold_by_name = dict(zip(GRIDS, thresholds, strict=True))
    parts = []
    for priority, (candidate, name, base_target) in enumerate(
            zip(selected, MODULE_NAMES, BASE_TARGETS, strict=True)):
        rows = route_rows(frame, candidate, base_target)
        if name in GRIDS:
            field, operator, _ = GRIDS[name]
            threshold = threshold_by_name[name]
            high = route_rows(frame, candidate, 1.0).set_index(
                ["symbol", "entry_time"], drop=False)
            actual = pd.to_numeric(rows[field], errors="coerce")
            upgrade = actual.notna() & (
                actual.ge(threshold) if operator == "ge" else actual.le(threshold)
            )
            replacements = []
            for index in rows.index[upgrade]:
                item = rows.loc[index]
                key = (item["symbol"], item["entry_time"])
                if key not in high.index:
                    raise KeyError(f"missing 1R row for {name} {key}")
                replacements.append(high.loc[key])
            if replacements:
                rows = pd.concat(
                    [rows.loc[~upgrade].copy(), pd.DataFrame(replacements)],
                    ignore_index=True,
                )
        rows["module_priority"] = priority
        parts.append(rows)
    output = pd.concat(parts, ignore_index=True)
    output = output[~output["symbol"].isin(EXCLUDED_SYMBOLS)].copy()
    output = output.sort_values(
        ["module_priority", "family", "symbol", "entry_time"])
    output = output.drop_duplicates(["symbol", "entry_time"], keep="first")
    return output.sort_values(["entry_time", "symbol"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("module_report", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    dataset_raw, module_raw = args.dataset.read_bytes(), args.module_report.read_bytes()
    frame = add_context(pd.read_csv(args.dataset))
    selected = json.loads(module_raw)["selected"]

    combinations = []
    grids = [value[2] for value in GRIDS.values()]
    for thresholds in itertools.product(*grids):
        output = materialize(frame, selected, thresholds)
        months = {month: metric(output[output["month"] == month]) for month in MONTHS}
        all_metrics = metric(output)
        combinations.append({
            "thresholds": dict(zip(GRIDS, thresholds, strict=True)),
            "months": months,
            "all": all_metrics,
            "all_months_positive": all(item["net_r"] > 0 for item in months.values()),
            "all_months_wr_at_least_75": all(item["wr"] >= 75 for item in months.values()),
        })

    frozen = next(item for item in combinations
                  if tuple(item["thresholds"].values()) == FROZEN)
    report = {
        "status": "PASS" if all(
            item["all_months_positive"] and item["all_months_wr_at_least_75"]
            for item in combinations
        ) else "SENSITIVE",
        "purpose": "nearby-threshold stress only; no alternative is selected",
        "frozen_thresholds": dict(zip(GRIDS, FROZEN, strict=True)),
        "frozen": frozen,
        "combination_count": len(combinations),
        "all_months_positive_count": sum(
            item["all_months_positive"] for item in combinations),
        "all_months_wr_at_least_75_count": sum(
            item["all_months_wr_at_least_75"] for item in combinations),
        "ranges": {
            "overall_wr": [
                min(item["all"]["wr"] for item in combinations),
                max(item["all"]["wr"] for item in combinations),
            ],
            "overall_net_r": [
                min(item["all"]["net_r"] for item in combinations),
                max(item["all"]["net_r"] for item in combinations),
            ],
            "overall_dd_r": [
                min(item["all"]["dd_r"] for item in combinations),
                max(item["all"]["dd_r"] for item in combinations),
            ],
            "holdout_wr": [
                min(item["months"][HOLDOUT_MONTH]["wr"] for item in combinations),
                max(item["months"][HOLDOUT_MONTH]["wr"] for item in combinations),
            ],
            "holdout_net_r": [
                min(item["months"][HOLDOUT_MONTH]["net_r"] for item in combinations),
                max(item["months"][HOLDOUT_MONTH]["net_r"] for item in combinations),
            ],
        },
        "combinations": combinations,
        "sources": {
            str(args.dataset): hashlib.sha256(dataset_raw).hexdigest(),
            str(args.module_report): hashlib.sha256(module_raw).hexdigest(),
        },
        "limitations": [
            "This is a local perturbation stress, not an independent holdout.",
            "The same short direct X-Perp sample underlies every combination.",
            "No threshold is changed based on this report.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "status", "combination_count", "all_months_positive_count",
        "all_months_wr_at_least_75_count", "ranges",
    )}, indent=2))


if __name__ == "__main__":
    main()
