"""Calibrate historically stable stock modules on 2026 H1, reveal H2."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import pandas as pd

from stock_robust_regime_router import compact
from stock_strict_causal_modules import add_context, apply


CUTOFF = pd.Timestamp("2026-07-01", tz="UTC").timestamp()


def months(frame: pd.DataFrame) -> dict:
    return {str(key): compact(part) for key, part in frame.groupby("month")}


def threshold(target: float) -> float:
    return {.25: 82.0, .5: 70.0, .75: 61.0, 1.0: 54.0}[target]


def detail(frame: pd.DataFrame) -> dict:
    monthly = months(frame)
    return {"all": compact(frame), "months": monthly,
            "negative_months": {key: value for key, value in monthly.items()
                                if value["net_r"] < -.01}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("module_report", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report_raw = args.module_report.read_bytes()
    dataset_raw = args.dataset.read_bytes()
    report = json.loads(report_raw)
    frame = add_context(pd.read_csv(args.dataset))
    year = frame[frame["year"] == 2026]
    calibration = year[year["entry_time"] < CUTOFF]
    holdout = year[year["entry_time"] >= CUTOFF]

    eligible = []
    for candidate in report["calibration_pool"]:
        part = apply(calibration, candidate)
        metric = compact(part)
        monthly = months(part)
        positive = sum(item["net_r"] > .01 for item in monthly.values())
        worst = min((item["net_r"] for item in monthly.values()), default=-999)
        target = float(candidate["route"]["target_r"])
        qualifies = (
            metric["n"] >= 12 and metric["net_r"] > 0
            and metric["wr"] >= threshold(target)
            and len(monthly) >= 3
            and positive >= math.ceil(len(monthly) * 2 / 3)
            and worst >= -1.0
        )
        if qualifies:
            eligible.append({"candidate": candidate, "calibration": metric,
                             "calibration_months": monthly,
                             "calibration_score": metric["mean_r"] * 10
                             + metric["net_r"] / max(metric["dd_r"], .25)})

    # Candidate variants for one broad regime overlap heavily. Select the one
    # with the strongest H1 expectancy, using historical score only as a tie-break.
    best = {}
    for item in eligible:
        route = item["candidate"]["route"]
        broad = tuple((key, route[key]) for key in
                      ("family", "direction", "qqq_regime", "session_bucket"))
        rank = (item["calibration_score"], item["candidate"]["score"])
        if broad not in best or rank > best[broad][0]:
            best[broad] = (rank, item)
    selected = [value[1] for value in best.values()]
    selected.sort(key=lambda item: (item["calibration_score"],
                                    item["candidate"]["score"]), reverse=True)
    selected = selected[:24]

    def union(source: pd.DataFrame) -> pd.DataFrame:
        parts = []
        for priority, item in enumerate(selected):
            part = apply(source, item["candidate"]).copy()
            part["module_priority"] = priority
            parts.append(part)
        if not parts:
            return source.iloc[0:0]
        combined = pd.concat(parts).sort_values(["module_priority", "family"])
        return combined.drop_duplicates(["symbol", "entry_time"], keep="first")

    calibration_output = union(calibration)
    holdout_output = union(holdout)
    combined = pd.concat([calibration_output, holdout_output]).sort_values("entry_time")
    result = {
        "status": "PAPER_RESEARCH_ONLY",
        "cutoff": pd.Timestamp(CUTOFF, unit="s", tz="UTC").isoformat(),
        "selection_policy": {
            "minimum_n": 12, "minimum_win_rate_by_target": {
                "0.25": 82, "0.5": 70, "0.75": 61, "1.0": 54},
            "minimum_active_months": 3,
            "minimum_positive_month_fraction": "2/3",
            "worst_active_month_r": -1.0,
            "one_variant_per_family_direction_regime_session": True,
        },
        "pool_count": len(report["calibration_pool"]),
        "eligible_count": len(eligible), "selected_count": len(selected),
        "selected": selected,
        "calibration_h1": detail(calibration_output),
        "holdout_h2": detail(holdout_output),
        "all_2026": detail(combined),
        "sources": {
            str(args.module_report): hashlib.sha256(report_raw).hexdigest(),
            str(args.dataset): hashlib.sha256(dataset_raw).hexdigest(),
        },
        "limitations": [
            "The H2 holdout is not used by the mechanical calibration policy.",
            "Cash-feed outcomes still require exact X-Perp venue replay.",
            "Gross outcomes exclude fees and client sizing at user request.",
        ],
    }
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    combined.to_csv(args.out.with_suffix(".csv"), index=False)
    print(json.dumps({"eligible_count": len(eligible), "selected_count": len(selected),
                      "calibration": result["calibration_h1"]["all"],
                      "holdout": result["holdout_h2"]["all"],
                      "holdout_months": result["holdout_h2"]["months"]}, indent=2))


if __name__ == "__main__":
    main()
