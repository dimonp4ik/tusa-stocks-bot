"""Choose the stock per-scan cap before revealing the fresh forward slice."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from filter_lab import metrics
from stock_forward_monitor import portfolio_gate


def window(rows: list[dict], start: str, end: str) -> list[dict]:
    lower = pd.Timestamp(start, tz="UTC").timestamp()
    upper = pd.Timestamp(end, tz="UTC").timestamp()
    return [row for row in rows if lower <= row["entry_time"] < upper]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--fresh-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    reference_raw = args.reference.read_bytes()
    fresh_raw = args.fresh_report.read_bytes()
    reference = pd.read_csv(args.reference).to_dict("records")
    fresh_report = json.loads(fresh_raw)
    fresh = fresh_report["fresh_raw"]

    reports = {}
    for scan_cap in (1, 2, 3):
        replayed = portfolio_gate(reference, 4, scan_cap)
        calibration = window(replayed, "2026-07-01", "2026-09-01")
        validation = window(replayed, "2026-09-01", "2026-09-16")
        forward = portfolio_gate(fresh, 4, scan_cap)
        reports[str(scan_cap)] = {
            "calibration_july_august": metrics(calibration),
            "validation_sep_1_15": metrics(validation),
            "fresh_sep_16_18": metrics(forward),
        }

    baseline = reports["3"]["calibration_july_august"]
    best_r = max(item["calibration_july_august"]["net_r"]
                 for item in reports.values())
    eligible = [
        cap for cap, item in reports.items()
        if item["calibration_july_august"]["net_r"] >= 0.97 * best_r
        and item["calibration_july_august"]["wr"] >= baseline["wr"]
        and item["calibration_july_august"]["dd_r"] <= baseline["dd_r"]
    ]
    selected = min(map(int, eligible)) if eligible else 3
    report = {
        "status": "SELECTED_ON_JULY_AUGUST",
        "selection_rule": (
            "smallest cap retaining >=97% of best July-August net R, with win "
            "rate no lower and drawdown no higher than cap=3"
        ),
        "selected_scan_cap": selected,
        "validation_and_fresh_not_used_for_selection": True,
        "caps": reports,
        "sources": {
            str(args.reference): hashlib.sha256(reference_raw).hexdigest(),
            str(args.fresh_report): hashlib.sha256(fresh_raw).hexdigest(),
        },
        "limitations": [
            "Only three prespecified integer caps were compared.",
            "The reference strategy itself was developed on 2026 venue history.",
            "The fresh slice contains only three completed trading days.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "selected_scan_cap": selected,
        "caps": reports,
    }, indent=2))


if __name__ == "__main__":
    main()
