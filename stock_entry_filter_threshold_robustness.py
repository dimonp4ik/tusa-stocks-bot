"""Stress nearby thresholds for the frozen stock entry-filter candidate."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import pandas as pd

from stock_dynamic_target_candidate import metric
from stock_robust_regime_router import condition_mask


GRIDS = {
    2: ("qqq_gap_dir_atr", "le", (-.75, -1.0, -1.25)),
    3: ("opening_range_atr", "ge", (1.5, 2.0, 2.5)),
    5: ("prior_range_ratio", "le", (1.25, 1.5, 1.75)),
}
MONTHS = ("2026-07", "2026-08", "2026-09")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("signals", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    raw = args.signals.read_bytes()
    frame = pd.read_csv(args.signals)
    combinations = []
    priorities = tuple(GRIDS)
    for thresholds in itertools.product(*(GRIDS[p][2] for p in priorities)):
        rules = {
            priority: (GRIDS[priority][0], GRIDS[priority][1], threshold)
            for priority, threshold in zip(priorities, thresholds, strict=True)
        }
        parts = []
        for priority, rows in frame.groupby("module_priority", sort=True):
            priority = int(priority)
            parts.append(rows.loc[condition_mask(rows, (rules[priority],))]
                         if priority in rules else rows)
        output = pd.concat(parts, ignore_index=True).sort_values(
            ["entry_time", "symbol"])
        months = {month: metric(output[output["month"].eq(month)]) for month in MONTHS}
        combinations.append({
            "thresholds": {str(priority): threshold
                           for priority, threshold in zip(priorities, thresholds)},
            "all": metric(output), "months": months,
            "all_months_positive": all(value["net_r"] > 0 for value in months.values()),
        })

    def bounds(path: tuple[str, ...]) -> dict:
        values = []
        for item in combinations:
            value = item
            for key in path:
                value = value[key]
            values.append(float(value))
        return {"min": min(values), "max": max(values)}

    report = {
        "status": "PASS" if all(item["all_months_positive"] for item in combinations)
        else "FAIL",
        "candidate_thresholds": {
            "2": -1.0, "3": 2.0, "5": 1.5,
        },
        "grids": {
            str(priority): {"field": field, "operator": operator,
                            "values": list(values)}
            for priority, (field, operator, values) in GRIDS.items()
        },
        "combinations_tested": len(combinations),
        "combinations_all_months_positive": sum(
            item["all_months_positive"] for item in combinations),
        "ranges": {
            "trades": bounds(("all", "n")),
            "win_rate": bounds(("all", "wr")),
            "net_r": bounds(("all", "net_r")),
            "drawdown_r": bounds(("all", "dd_r")),
            "max_loss_streak": bounds(("all", "max_loss_streak")),
            **{
                f"{month}_net_r": bounds(("months", month, "net_r"))
                for month in MONTHS
            },
        },
        "september_invariant": {
            "same_trade_count": len({item["months"]["2026-09"]["n"]
                                     for item in combinations}) == 1,
            "same_net_r": len({round(item["months"]["2026-09"]["net_r"], 12)
                               for item in combinations}) == 1,
        },
        "combinations": combinations,
        "sources": {str(args.signals): hashlib.sha256(raw).hexdigest()},
        "interpretation": (
            "Nearby thresholds do not create a narrow performance cliff.  This is a "
            "local sensitivity check, not independent forward validation."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "status", "combinations_tested", "combinations_all_months_positive",
        "ranges", "september_invariant",
    )}, indent=2))


if __name__ == "__main__":
    main()
