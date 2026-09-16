"""Calibrate historical causal modules on July/August X-Perp, reveal September."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from stock_2026_regime_calibration import detail, threshold
from stock_strict_causal_modules import add_context, apply


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("module_report", type=Path)
    parser.add_argument("xperp_csv", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    module_raw = args.module_report.read_bytes()
    xperp_raw = args.xperp_csv.read_bytes()
    report = json.loads(module_raw)
    frame = add_context(pd.read_csv(args.xperp_csv))
    july = frame[frame["month"] == "2026-07"]
    august = frame[frame["month"] == "2026-08"]
    september = frame[frame["month"] == "2026-09"]

    july_eligible = []
    for candidate in report["calibration_pool"]:
        part = apply(july, candidate)
        metric = detail(part)["all"]
        target = float(candidate["route"]["target_r"])
        if metric["n"] >= 4 and metric["net_r"] > 0 and metric["wr"] >= threshold(target):
            july_eligible.append({"candidate": candidate, "july": metric,
                                  "july_score": metric["mean_r"]
                                  + metric["net_r"] / max(metric["dd_r"], .25)})

    best = {}
    for item in july_eligible:
        route = item["candidate"]["route"]
        broad = tuple((key, route[key]) for key in
                      ("family", "direction", "qqq_regime", "session_bucket"))
        rank = (item["july_score"], item["candidate"]["score"])
        if broad not in best or rank > best[broad][0]:
            best[broad] = (rank, item)

    august_validated = []
    for _, item in best.values():
        part = apply(august, item["candidate"])
        metric = detail(part)["all"]
        target = float(item["candidate"]["route"]["target_r"])
        if metric["n"] >= 4 and metric["net_r"] > 0 and metric["wr"] >= threshold(target):
            august_validated.append({**item, "august": metric})
    august_validated.sort(key=lambda item: (
        min(item["july"]["mean_r"], item["august"]["mean_r"]),
        item["july"]["net_r"] + item["august"]["net_r"]), reverse=True)

    def union(source: pd.DataFrame) -> pd.DataFrame:
        parts = []
        for priority, item in enumerate(august_validated):
            part = apply(source, item["candidate"]).copy()
            part["module_priority"] = priority
            parts.append(part)
        if not parts:
            return source.iloc[0:0]
        combined = pd.concat(parts).sort_values(["module_priority", "family"])
        return combined.drop_duplicates(["symbol", "entry_time"], keep="first")

    july_output, august_output, september_output = map(union, (july, august, september))
    combined = pd.concat([july_output, august_output, september_output]).sort_values("entry_time")
    result = {
        "status": "PAPER_CANDIDATE" if august_validated else "NO_VENUE_MODULES",
        "sequence": ["Historical discovery 2017-23", "cash validation 2024",
                     "cash confirmation 2025", "X-Perp July calibration",
                     "X-Perp August validation", "X-Perp September holdout"],
        "july_eligible_count": len(july_eligible),
        "july_broad_routes": len(best),
        "selected_count": len(august_validated), "selected": august_validated,
        "xperp_july_calibration": detail(july_output),
        "xperp_august_validation": detail(august_output),
        "xperp_september_holdout": detail(september_output),
        "xperp_all": detail(combined),
        "sources": {str(args.module_report): hashlib.sha256(module_raw).hexdigest(),
                    str(args.xperp_csv): hashlib.sha256(xperp_raw).hexdigest()},
        "limitations": [
            "The X-Perp sample spans only July through mid-September.",
            "September is not read by the mechanical selection sequence.",
            "Gross outcomes exclude fees and client-selected position sizing.",
        ],
    }
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    combined.to_csv(args.out.with_suffix(".csv"), index=False)
    print(json.dumps({"selected_count": len(august_validated),
                      "july": result["xperp_july_calibration"]["all"],
                      "august": result["xperp_august_validation"]["all"],
                      "september": result["xperp_september_holdout"]["all"],
                      "all": result["xperp_all"]["all"]}, indent=2))


if __name__ == "__main__":
    main()
