"""Exhaustively select stock X-Perp module targets on July/August only."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from stock_strict_causal_modules import add_context, mask_for


TARGETS = (None, .25, .5, .75, 1.0)


def metric(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {"n": 0, "wr": 0.0, "net_r": 0.0, "pf": 0.0, "dd_r": 0.0}
    values = frame.sort_values(["exit_time", "entry_time"])["net_r"].to_numpy(float)
    curve = np.cumsum(values)
    drawdown = np.maximum.accumulate(np.r_[0.0, curve])[1:] - curve
    wins, losses = values[values > 0].sum(), -values[values < 0].sum()
    return {
        "n": int(len(values)),
        "wr": float(100 * np.mean(values > 0)),
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
    dataset_raw = args.dataset.read_bytes()
    module_raw = args.module_report.read_bytes()
    frame = add_context(pd.read_csv(args.dataset))
    modules = json.loads(module_raw)["selected"]

    frame = frame.reset_index(drop=True)
    months = frame["month"].astype(str).to_numpy()
    entry_times = frame["entry_time"].to_numpy(float)
    exit_times = frame["exit_time"].to_numpy(float)
    net_values = frame["net_r"].to_numpy(float)
    trade_keys = list(zip(frame["symbol"].astype(str), frame["entry_time"].astype(str)))
    rows_by_choice: dict[tuple[int, float], list[int]] = {}
    for priority, module in enumerate(modules):
        broad = {key: value for key, value in module["route"].items()
                 if key != "target_r"}
        conditions = tuple(tuple(item) for item in module["conditions"])
        for target in TARGETS[1:]:
            route = {**broad, "target_r": target}
            selected = np.ones(len(frame), dtype=bool)
            for field, value in route.items():
                selected &= frame[field].astype(str).to_numpy() == str(value)
            rows_by_choice[(priority, target)] = np.flatnonzero(
                selected & mask_for(frame, conditions)).tolist()

    def union_indices(choice: tuple[float | None, ...]) -> list[int]:
        result, seen = [], set()
        for priority, target in enumerate(choice):
            if target is None:
                continue
            for index in rows_by_choice[(priority, target)]:
                key = trade_keys[index]
                if key in seen:
                    continue
                seen.add(key)
                result.append(index)
        return result

    def fast_metric(indices: list[int], month: str | None = None) -> dict:
        selected = np.asarray(indices, dtype=int)
        if month is not None and len(selected):
            selected = selected[months[selected] == month]
        if not len(selected):
            return {"n": 0, "wr": 0.0, "net_r": 0.0, "pf": 0.0, "dd_r": 0.0}
        order = np.lexsort((entry_times[selected], exit_times[selected]))
        values = net_values[selected[order]]
        curve = np.cumsum(values)
        drawdown = np.maximum.accumulate(np.r_[0.0, curve])[1:] - curve
        wins, losses = values[values > 0].sum(), -values[values < 0].sum()
        return {
            "n": int(len(values)), "wr": float(100 * np.mean(values > 0)),
            "net_r": float(values.sum()),
            "pf": float(wins / losses) if losses else None,
            "dd_r": float(drawdown.max(initial=0)),
        }

    candidates = []
    for choice in itertools.product(TARGETS, repeat=len(modules)):
        if all(target is None for target in choice):
            continue
        indices = union_indices(choice)
        july = fast_metric(indices, "2026-07")
        august = fast_metric(indices, "2026-08")
        candidates.append((choice, july, august))

    policies = {
        "balanced": {
            "minimum": {"july_n": 90, "august_n": 55, "wr": 75.0},
            "rank": "minimum monthly WR, minimum monthly R, total R, inverse drawdown",
            "eligible": lambda j, a: (j["n"] >= 90 and a["n"] >= 55
                                       and min(j["wr"], a["wr"]) >= 75
                                       and min(j["net_r"], a["net_r"]) > 0),
            "score": lambda j, a: (min(j["wr"], a["wr"]),
                                     min(j["net_r"], a["net_r"]),
                                     j["net_r"] + a["net_r"],
                                     -max(j["dd_r"], a["dd_r"])),
        },
        "frequency": {
            "minimum": {"july_n": 90, "august_n": 63, "wr": 72.0},
            "rank": "minimum monthly WR, minimum monthly R, total R, inverse drawdown",
            "eligible": lambda j, a: (j["n"] >= 90 and a["n"] >= 63
                                       and min(j["wr"], a["wr"]) >= 72
                                       and min(j["net_r"], a["net_r"]) > 0),
            "score": lambda j, a: (min(j["wr"], a["wr"]),
                                     min(j["net_r"], a["net_r"]),
                                     j["net_r"] + a["net_r"],
                                     -max(j["dd_r"], a["dd_r"])),
        },
        "low_drawdown": {
            "minimum": {"july_n": 60, "august_n": 35, "wr": 75.0},
            "rank": "inverse worst monthly drawdown, minimum WR, minimum R, total R",
            "eligible": lambda j, a: (j["n"] >= 60 and a["n"] >= 35
                                       and min(j["wr"], a["wr"]) >= 75
                                       and min(j["net_r"], a["net_r"]) > 0),
            "score": lambda j, a: (-max(j["dd_r"], a["dd_r"]),
                                     min(j["wr"], a["wr"]),
                                     min(j["net_r"], a["net_r"]),
                                     j["net_r"] + a["net_r"]),
        },
        "profit": {
            "minimum": {"july_n": 90, "august_n": 55, "wr": 70.0},
            "rank": "minimum monthly R, total R, minimum WR, inverse drawdown",
            "eligible": lambda j, a: (j["n"] >= 90 and a["n"] >= 55
                                       and min(j["wr"], a["wr"]) >= 70
                                       and min(j["net_r"], a["net_r"]) > 0),
            "score": lambda j, a: (min(j["net_r"], a["net_r"]),
                                     j["net_r"] + a["net_r"],
                                     min(j["wr"], a["wr"]),
                                     -max(j["dd_r"], a["dd_r"])),
        },
    }

    results = {}
    csv_parts = []
    for name, policy in policies.items():
        eligible = [item for item in candidates
                    if policy["eligible"](item[1], item[2])]
        if not eligible:
            results[name] = {"status": "NO_CANDIDATE"}
            continue
        choice, july, august = max(
            eligible, key=lambda item: policy["score"](item[1], item[2]))
        indices = union_indices(choice)
        rows = frame.iloc[indices].copy()
        september_rows = rows[rows["month"] == "2026-09"]
        targets = {modules[index]["route"]["family"] + ":" +
                   modules[index]["route"]["direction"] + ":" +
                   modules[index]["route"]["session_bucket"]: target
                   for index, target in enumerate(choice) if target is not None}
        results[name] = {
            "status": "CALIBRATED",
            "policy": {"minimum": policy["minimum"], "rank": policy["rank"]},
            "choice_by_priority": list(choice),
            "targets": targets,
            "july_calibration": july,
            "august_calibration": august,
            "september_holdout": metric(september_rows),
            "all": metric(rows),
            "eligible_configurations": len(eligible),
        }
        export = rows.copy()
        export["portfolio_profile"] = name
        csv_parts.append(export)
        export.to_csv(
            args.out.with_name(f"{args.out.stem}_{name}.csv"), index=False)

    report = {
        "status": "RESEARCH_ONLY",
        "configurations_tested": len(candidates),
        "selection_data": ["2026-07", "2026-08"],
        "sealed_holdout": "2026-09",
        "profiles": results,
        "sources": {
            str(args.dataset): hashlib.sha256(dataset_raw).hexdigest(),
            str(args.module_report): hashlib.sha256(module_raw).hexdigest(),
        },
        "limitations": [
            "All target/subset configurations are tested, increasing selection bias.",
            "The direct X-Perp calibration history is only two full months.",
            "September is reported after selection and never ranks a configuration.",
        ],
    }
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if csv_parts:
        pd.concat(csv_parts).to_csv(args.out.with_suffix(".csv"), index=False)
    print(json.dumps({name: value for name, value in results.items()}, indent=2))


if __name__ == "__main__":
    main()
