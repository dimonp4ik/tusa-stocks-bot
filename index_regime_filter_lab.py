"""Freeze a small opening-breakout regime filter on June/July and reveal August."""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

from filter_lab import gate, matches, metrics, timestamp


NUMERIC_LEVELS = {
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
    numeric = {"entry_time", "exit_time", "net_r", "risk_pct", "size_mult", *NUMERIC_LEVELS}
    for row in rows:
        for field in numeric:
            try:
                row[field] = float(row[field])
            except (KeyError, TypeError, ValueError):
                row[field] = None
    return rows


def atoms(rows: list[dict]) -> list[tuple]:
    result = []
    for field, levels in NUMERIC_LEVELS.items():
        for value in levels:
            result.extend(((field, "ge", value), (field, "le", value)))
    for value in sorted({row["direction"] for row in rows}):
        result.append(("direction", "eq", value))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = load(args.raw)
    july = timestamp("2026-07-01")
    august = timestamp("2026-08-01")
    train = [row for row in rows if row["entry_time"] < july and row["exit_time"] < july]
    validation = [row for row in rows if july <= row["entry_time"] < august
                  and row["exit_time"] < august]
    holdout = [row for row in rows if row["entry_time"] >= august]
    available = atoms(train)
    rules = [(atom,) for atom in available]
    rules += [pair for pair in itertools.combinations(available, 2)
              if pair[0][0] != pair[1][0]]
    seen = set()
    candidates = []
    for rule in rules:
        train_rows = gate([row for row in train if matches(row, rule)], 4)
        key = tuple((row["symbol"], row["entry_time"]) for row in train_rows)
        if key in seen or len(train_rows) < 12:
            continue
        seen.add(key)
        tm = metrics(train_rows)
        validation_rows = gate([row for row in validation if matches(row, rule)], 4)
        vm = metrics(validation_rows)
        item = {"rule": rule, "train": tm, "validation": vm}
        if (tm["net_r"] > 0 and (tm["pf"] or 0) >= 1.2
                and tm["dd_r"] <= 5 and vm["n"] >= 15
                and vm["net_r"] > 0 and (vm["pf"] or 0) >= 1.2):
            candidates.append(item)
    chosen = max(candidates, key=lambda item: (
        item["validation"]["net_r"] / max(item["validation"]["dd_r"], .25),
        item["validation"]["net_r"], item["train"]["net_r"]), default=None)
    if chosen:
        chosen["holdout"] = metrics(gate(
            [row for row in holdout if matches(row, chosen["rule"])], 4))
    report = {
        "status": "FROZEN_FOR_HOLDOUT" if chosen else "NO_STABLE_RULE",
        "objective": "positive gross R and restrained drawdown; no fees or position sizing",
        "source": str(args.raw),
        "source_sha256": hashlib.sha256(args.raw.read_bytes()).hexdigest(),
        "baseline": {
            "train": metrics(gate(train, 4)),
            "validation": metrics(gate(validation, 4)),
            "holdout": metrics(gate(holdout, 4)),
        },
        "tested_unique_rules": len(seen),
        "qualified_rules": len(candidates),
        "chosen": chosen,
        "top_qualified": sorted(candidates, key=lambda item: (
            item["validation"]["net_r"] / max(item["validation"]["dd_r"], .25),
            item["validation"]["net_r"]), reverse=True)[:20],
        "limitations": [
            "Only June selects thresholds, July validates them, and August is revealed last.",
            "The three-month X-Perp sample is short and needs forward paper confirmation.",
            "Only causal index features available before the next-bar market entry are used.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "qualified": len(candidates),
                      "baseline": report["baseline"], "chosen": chosen}, indent=2))


if __name__ == "__main__":
    main()
