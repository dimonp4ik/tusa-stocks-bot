"""Freeze X-Perp-compatible stock families on July and reveal August 2026."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from filter_lab import gate, metrics
from stock_walk_forward_filter_lab import (
    CATEGORICAL,
    NUMERIC,
    load_external,
    rows_at_probability,
)


THRESHOLD = .84  # selected on deep-feed 2025 validation, before venue split


def compact(value: dict) -> dict:
    return {key: value[key] for key in
            ("n", "wr", "wr_lower_95", "net_r", "mean_r", "pf", "dd_r")}


def month(row: dict) -> str:
    return datetime.fromtimestamp(row["entry_time"], timezone.utc).strftime("%Y-%m")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--deep-csv", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--execution-dir", type=Path, required=True)
    parser.add_argument("--symbols", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    bundle = joblib.load(args.model)
    model = bundle["model"]
    symbols = tuple(value.strip() for value in args.symbols.split(",") if value.strip())
    external, external_sources = load_external(args.index_dir, args.execution_dir, symbols)
    probability = model.predict_proba(external[list(NUMERIC + CATEGORICAL)])[:, 1]
    eligible = rows_at_probability(external, probability, THRESHOLD)
    july = [row for row in eligible if month(row) == "2026-07"]
    august = [row for row in eligible if month(row) == "2026-08"]
    families = sorted({row["family"] for row in july})
    july_family = {family: metrics(gate(
        [row for row in july if row["family"] == family], 4)) for family in families}
    # Venue compatibility is frozen using July only.  Four observations are
    # required so a single lucky setup cannot admit a family.
    allowed = sorted(family for family, value in july_family.items()
                     if value["n"] >= 4 and value["net_r"] > 0
                     and value["wr"] >= 85)
    july_allowed = gate([row for row in july if row["family"] in allowed], 4)
    august_allowed = gate([row for row in august if row["family"] in allowed], 4)
    all_allowed = gate([row for row in eligible if row["family"] in allowed], 4)

    deep = pd.read_csv(args.deep_csv)
    deep = deep[deep["target_r"] == .25].copy()
    deep["year"] = pd.to_datetime(deep["entry_time"], unit="s", utc=True).dt.year
    deep_probability = model.predict_proba(deep[list(NUMERIC + CATEGORICAL)])[:, 1]
    deep_eligible = rows_at_probability(deep, deep_probability, THRESHOLD)
    report = {
        "status": "PAPER_CANDIDATE",
        "target_r": .25, "stop_atr": 1, "model_threshold": THRESHOLD,
        "venue_family_rule": {
            "calibration_month": "2026-07",
            "minimum_family_trades": 4,
            "minimum_family_win_rate": 85,
            "allowed": allowed,
            "july_all_families": {key: compact(value)
                                  for key, value in july_family.items()},
        },
        "xperp_calibration_july": compact(metrics(july_allowed)),
        "xperp_unseen_august": compact(metrics(august_allowed)),
        "xperp_all": compact(metrics(all_allowed)),
        "deep_feed": {
            str(year): compact(metrics(gate([
                row for row in deep_eligible
                if datetime.fromtimestamp(row["entry_time"], timezone.utc).year == year
                and row["family"] in allowed
            ], 4))) for year in (2025, 2026)
        },
        "sources": {
            "model": {"path": str(args.model),
                      "sha256": hashlib.sha256(args.model.read_bytes()).hexdigest()},
            "deep": {"path": str(args.deep_csv),
                     "sha256": hashlib.sha256(args.deep_csv.read_bytes()).hexdigest()},
            **external_sources,
        },
        "limitations": [
            "The August venue holdout contains only one month and eleven accepted trades.",
            "Family compatibility must be confirmed by forward paper outcomes before live use.",
            "Gross R excludes fees and client-selected position size.",
        ],
    }
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "venue_family_rule", "xperp_calibration_july", "xperp_unseen_august",
        "xperp_all", "deep_feed")}, indent=2))


if __name__ == "__main__":
    main()
