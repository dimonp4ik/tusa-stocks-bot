"""Causal per-module health gate for the H1-calibrated stock router."""
from __future__ import annotations

import argparse
import hashlib
import heapq
import itertools
import json
from pathlib import Path

import pandas as pd

from stock_2026_regime_calibration import CUTOFF, detail
from stock_strict_causal_modules import add_context, apply


BREAKEVEN_WR = {.25: 80.0, .5: 100 / 1.5, .75: 100 / 1.75, 1.0: 50.0}


def module_health(rows: pd.DataFrame, *, window: int, min_history: int,
                  minimum_mean_r: float, win_buffer: float) -> pd.DataFrame:
    accepted = []
    for _, part in rows.groupby("module_id"):
        history = []
        pending = []
        target = float(part["target_r"].iloc[0])
        minimum_wr = BREAKEVEN_WR[target] + win_buffer
        for serial, row in enumerate(part.sort_values("entry_time").to_dict("records")):
            now = float(row["entry_time"])
            while pending and pending[0][0] < now:
                _, _, closed = heapq.heappop(pending)
                history.append(float(closed["net_r"]))
            recent = history[-window:]
            enabled = len(recent) < min_history or (
                sum(recent) / len(recent) >= minimum_mean_r
                and 100 * sum(value > 0 for value in recent) / len(recent) >= minimum_wr
            )
            if enabled:
                accepted.append(row)
            heapq.heappush(pending, (float(row["exit_time"]), serial, row))
    if not accepted:
        return rows.iloc[0:0]
    result = pd.DataFrame(accepted).sort_values(["module_priority", "family"])
    return result.drop_duplicates(["symbol", "entry_time"], keep="first")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calibration_report", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report_raw = args.calibration_report.read_bytes()
    dataset_raw = args.dataset.read_bytes()
    report = json.loads(report_raw)
    frame = add_context(pd.read_csv(args.dataset))
    frame = frame[frame["year"] == 2026]
    parts = []
    for module_id, item in enumerate(report["selected"]):
        part = apply(frame, item["candidate"]).copy()
        part["module_id"] = module_id
        part["module_priority"] = module_id
        parts.append(part)
    candidates = pd.concat(parts, ignore_index=True)

    tested = []
    for window, min_history, minimum_mean_r, win_buffer in itertools.product(
            (5, 10, 20, 30), (5, 10), (-.05, 0, .05), (-5, 0, 5)):
        if min_history > window:
            continue
        output = module_health(candidates, window=window, min_history=min_history,
                               minimum_mean_r=minimum_mean_r, win_buffer=win_buffer)
        calibration = output[output["entry_time"] < CUTOFF]
        report_part = detail(calibration)
        metric = report_part["all"]
        positive_months = len(report_part["months"]) - len(report_part["negative_months"])
        if (metric["n"] < 100 or metric["net_r"] <= 0
                or positive_months < 5 or metric["dd_r"] > 12):
            continue
        score = (metric["net_r"] / max(metric["dd_r"], .25)
                 + metric["mean_r"] * 10 + positive_months)
        tested.append({"parameters": {"window": window, "min_history": min_history,
                                       "minimum_mean_r": minimum_mean_r,
                                       "win_buffer": win_buffer},
                       "calibration": report_part, "score": score})
    chosen = max(tested, key=lambda item: item["score"], default=None)
    if chosen:
        output = module_health(candidates, **chosen["parameters"])
    else:
        output = candidates.iloc[0:0]
    calibration = output[output["entry_time"] < CUTOFF]
    holdout = output[output["entry_time"] >= CUTOFF]
    result = {
        "status": "PAPER_RESEARCH_ONLY" if chosen else "NO_STABLE_HEALTH_GATE",
        "chosen": chosen,
        "qualified_parameter_sets": len(tested),
        "calibration_h1": detail(calibration),
        "holdout_h2": detail(holdout),
        "all_2026": detail(output),
        "sources": {
            str(args.calibration_report): hashlib.sha256(report_raw).hexdigest(),
            str(args.dataset): hashlib.sha256(dataset_raw).hexdigest(),
        },
        "causality": "Only outcomes with exit_time earlier than the next entry are visible; declined candidates remain shadow observations.",
        "limitations": [
            "Health parameters are selected on 2026 H1 and revealed on H2.",
            "Cash-feed results still require X-Perp execution replay.",
        ],
    }
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    output.to_csv(args.out.with_suffix(".csv"), index=False)
    print(json.dumps({"status": result["status"], "qualified": len(tested),
                      "chosen": chosen["parameters"] if chosen else None,
                      "calibration": result["calibration_h1"]["all"],
                      "holdout": result["holdout_h2"]["all"],
                      "holdout_months": result["holdout_h2"]["months"]}, indent=2))


if __name__ == "__main__":
    main()
