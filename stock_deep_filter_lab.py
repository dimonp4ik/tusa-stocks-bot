"""Select an interpretable stock regime on 2022-24, then reveal later data."""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

from filter_lab import gate, matches, metrics


LEVELS = {
    "minutes_after_open": [30, 45, 60, 90, 120, 180],
    "opening_range_atr": [.5, .75, 1, 1.25, 1.5, 2],
    "breakout_extension_atr": [.05, .1, .15, .25, .5],
    "signal_body_atr": [0, .1, .25, .5, .75, 1],
    "signal_range_atr": [.5, .75, 1, 1.5, 2],
    "pre_signal_move_atr": [-.5, 0, .5, 1],
    "gap_directional_atr": [-1, -.5, 0, .5, 1],
    "prior_day_directional_atr": [-1, -.5, 0, .5, 1],
}


def load(path: Path) -> list[dict]:
    rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    for row in rows:
        for field in {"entry_time", "exit_time", "net_r", "risk_pct", "size_mult", *LEVELS}:
            try:
                row[field] = float(row[field])
            except (KeyError, TypeError, ValueError):
                row[field] = None
    return rows


def year(row: dict) -> int:
    return datetime.fromtimestamp(row["entry_time"], timezone.utc).year


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("--external", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = load(args.raw)
    external = load(args.external)
    atoms = []
    for field, values in LEVELS.items():
        for value in values:
            atoms.extend(((field, "ge", value), (field, "le", value)))
    rules = [(atom,) for atom in atoms]
    rules += [pair for pair in itertools.combinations(atoms, 2)
              if pair[0][0] != pair[1][0]]
    seen = set()
    qualified = []
    for rule in rules:
        by_year = {}
        duplicate_key = None
        for value in (2022, 2023, 2024):
            selected = gate([row for row in rows if year(row) == value and matches(row, rule)], 4)
            by_year[str(value)] = metrics(selected)
            if value == 2022:
                duplicate_key = tuple((row["symbol"], row["entry_time"]) for row in selected)
        if duplicate_key in seen:
            continue
        seen.add(duplicate_key)
        if (all(by_year[str(value)]["n"] >= 30 and by_year[str(value)]["net_r"] > 0
                and (by_year[str(value)]["pf"] or 0) >= 1.15 for value in (2022, 2023))
                and by_year["2024"]["n"] >= 30 and by_year["2024"]["net_r"] > 0
                and (by_year["2024"]["pf"] or 0) >= 1.2):
            qualified.append({"rule": rule, "development": by_year})
    chosen = max(qualified, key=lambda item: (
        item["development"]["2024"]["net_r"]
        / max(item["development"]["2024"]["dd_r"], .25),
        item["development"]["2024"]["net_r"],
        item["development"]["2023"]["net_r"]), default=None)
    if chosen:
        for value in (2025, 2026):
            chosen.setdefault("holdout", {})[str(value)] = metrics(gate(
                [row for row in rows if year(row) == value and matches(row, chosen["rule"])], 4))
        chosen["xperp_external"] = metrics(gate(
            [row for row in external if matches(row, chosen["rule"])], 4))
    report = {
        "status": "FROZEN_FOR_HOLDOUT" if chosen else "NO_STABLE_RULE",
        "tested_unique_rules": len(seen), "qualified_rules": len(qualified),
        "chosen": chosen,
        "top_development": sorted(qualified, key=lambda item: (
            item["development"]["2024"]["net_r"]
            / max(item["development"]["2024"]["dd_r"], .25),
            item["development"]["2024"]["net_r"]), reverse=True)[:20],
        "sources": {
            "deep": {"path": str(args.raw), "sha256": hashlib.sha256(args.raw.read_bytes()).hexdigest()},
            "xperp": {"path": str(args.external),
                       "sha256": hashlib.sha256(args.external.read_bytes()).hexdigest()},
        },
        "limitations": [
            "Rules use 2022-23 for training and 2024 for validation.",
            "2025-26 and the recent X-Perp rows are revealed only after selection.",
            "Deep-feed OHLC validates the pattern; X-Perp OHLC validates recent execution transfer.",
        ],
    }
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "qualified": len(qualified),
                      "chosen": chosen}, indent=2))


if __name__ == "__main__":
    main()
