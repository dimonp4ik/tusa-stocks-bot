"""Select a balanced direct-X-Perp portfolio on July/August, reveal September."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from stock_strict_causal_modules import add_context, mask_for


TARGETS = (.25, .5, .75, 1.0)


def metric(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {"n": 0, "wr": 0.0, "net_r": 0.0, "pf": 0.0, "dd_r": 0.0}
    values = frame.sort_values(["exit_time", "entry_time"])["net_r"].to_numpy(float)
    curve = np.cumsum(values)
    drawdown = np.maximum.accumulate(np.r_[0.0, curve])[1:] - curve
    wins, losses = values[values > 0].sum(), -values[values < 0].sum()
    return {
        "n": int(len(values)), "wr": float(100 * np.mean(values > 0)),
        "net_r": float(values.sum()),
        "pf": float(wins / losses) if losses else None,
        "dd_r": float(drawdown.max(initial=0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("module_report", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    dataset_raw, module_raw = args.dataset.read_bytes(), args.module_report.read_bytes()
    frame = add_context(pd.read_csv(args.dataset))
    selected = json.loads(module_raw)["selected"]
    module_rows, target_choices = {}, []

    for priority, candidate in enumerate(selected):
        broad = {key: value for key, value in candidate["route"].items()
                 if key != "target_r"}
        conditions = tuple(tuple(item) for item in candidate["conditions"])
        choices = []
        for target in TARGETS:
            route = {**broad, "target_r": target}
            route_mask = np.ones(len(frame), dtype=bool)
            for field, value in route.items():
                route_mask &= frame[field].astype(str).to_numpy() == str(value)
            rows = frame.loc[route_mask & mask_for(frame, conditions)].copy()
            july = metric(rows[rows["month"] == "2026-07"])
            august = metric(rows[rows["month"] == "2026-08"])
            score = (min(july["wr"], august["wr"]),
                     min(july["net_r"], august["net_r"]),
                     july["net_r"] + august["net_r"])
            choices.append((score, target, rows, july, august))
        score, target, rows, july, august = max(choices, key=lambda item: item[0])
        module_rows[priority] = rows
        target_choices.append({
            "priority": priority, "broad_route": broad,
            "conditions": [list(item) for item in conditions],
            "target_r": target, "july": july, "august": august,
        })

    def union(priorities: tuple[int, ...]) -> pd.DataFrame:
        parts = []
        for priority in priorities:
            part = module_rows[priority].copy()
            part["module_priority"] = priority
            parts.append(part)
        combined = pd.concat(parts).sort_values(["module_priority", "family"])
        return combined.drop_duplicates(["symbol", "entry_time"], keep="first")

    portfolios = []
    for count in range(1, len(selected) + 1):
        for priorities in itertools.combinations(range(len(selected)), count):
            rows = union(priorities)
            july = metric(rows[rows["month"] == "2026-07"])
            august = metric(rows[rows["month"] == "2026-08"])
            if (july["n"] < 90 or august["n"] < 55
                    or july["net_r"] <= 0 or august["net_r"] <= 0):
                continue
            score = (min(july["wr"], august["wr"]),
                     july["net_r"] + august["net_r"])
            portfolios.append((score, priorities, rows, july, august))
    if not portfolios:
        raise RuntimeError("No portfolio passed the frozen calibration policy")
    score, priorities, output, july, august = max(portfolios, key=lambda item: item[0])
    september_rows = output[output["month"] == "2026-09"]
    september = metric(september_rows)
    report = {
        "status": ("PAPER_CANDIDATE" if september["wr"] >= 75
                   and september["net_r"] > 0 else "REJECT"),
        "selection_policy": {
            "target": "maximize minimum July/August WR, then minimum and total R per module",
            "portfolio": "minimum July n=90, August n=55, both positive; maximize minimum WR then R",
            "september": "sealed chronological holdout, never used for target or subset selection",
        },
        "target_choices": target_choices,
        "selected_priorities": list(priorities),
        "selected_modules": [target_choices[index] for index in priorities],
        "july_discovery": july, "august_calibration": august,
        "september_holdout": september,
        "all": metric(output),
        "sources": {
            str(args.dataset): hashlib.sha256(dataset_raw).hexdigest(),
            str(args.module_report): hashlib.sha256(module_raw).hexdigest(),
        },
        "limitations": [
            "Direct stock X-Perp history is short and begins in June 2026.",
            "A future paper-forward sample is required before live activation.",
            "Gross R excludes costs and client-selected sizing at user request.",
        ],
    }
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    output.sort_values("entry_time").to_csv(args.out.with_suffix(".csv"), index=False)
    print(json.dumps({key: report[key] for key in (
        "status", "selected_priorities", "july_discovery",
        "august_calibration", "september_holdout", "all")}, indent=2))


if __name__ == "__main__":
    main()
