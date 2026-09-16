"""Audit a frozen stock rule by month and symbol without selecting new thresholds."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from filter_lab import gate, matches, metrics
from index_regime_filter_lab import load


RULES = {
    "short_only": (("direction", "eq", "SHORT"),),
    "short_avoid_exhaustion": (
        ("prior_day_directional_atr", "le", 1),
        ("direction", "eq", "SHORT"),
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = load(args.raw)
    report = {"source": str(args.raw),
              "source_sha256": hashlib.sha256(args.raw.read_bytes()).hexdigest(),
              "rules": {}}
    for name, rule in RULES.items():
        selected = gate([row for row in rows if matches(row, rule)], 4)
        months = sorted({datetime.fromtimestamp(
            row["entry_time"], timezone.utc).strftime("%Y-%m")
            for row in selected})
        report["rules"][name] = {
            "rule": rule,
            "all": metrics(selected),
            "months": {month: metrics([row for row in selected if
                datetime.fromtimestamp(
                    row["entry_time"], timezone.utc
                ).strftime("%Y-%m") == month]) for month in months},
            "symbols": {symbol: metrics([row for row in selected if row["symbol"] == symbol])
                        for symbol in sorted({row["symbol"] for row in selected})},
        }
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["rules"], indent=2))


if __name__ == "__main__":
    main()
