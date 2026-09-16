"""Exact paper audit for the three stock rules positive across both feeds."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from stock_2025_validation_target_router import (
    detailed, load_external_all_targets, route,
)


SOURCE_RULE_INDICES = (5, 15, 30)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("developed_rules", type=Path)
    parser.add_argument("deep_csv", type=Path)
    parser.add_argument("--external-index-dir", type=Path, required=True)
    parser.add_argument("--external-execution-dir", type=Path, required=True)
    parser.add_argument("--symbols", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rules_raw = args.developed_rules.read_bytes()
    source = json.loads(rules_raw)["selected"]
    selected = [{"source_rule_index": index, "rule": source[index]["rule"]}
                for index in SOURCE_RULE_INDICES]
    deep_raw = args.deep_csv.read_bytes()
    deep = pd.read_csv(args.deep_csv).to_dict("records")
    historical = route(deep, selected)

    symbols = tuple(value.strip() for value in args.symbols.split(",") if value.strip())
    external, external_sources = load_external_all_targets(
        args.external_index_dir, args.external_execution_dir, symbols)
    venue = route(external, selected)

    per_rule = {}
    for item in selected:
        key = str(item["source_rule_index"])
        per_rule[key] = {
            "rule": item["rule"],
            "deep": detailed(route(deep, [item])),
            "external_xperp": detailed(route(external, [item])),
        }
    report = {
        "status": "PAPER_ONLY_POSTHOC_TRANSFER",
        "method": "three rules developed on 2020-24 and positive in 2025, 2026 and X-Perp",
        "selected": selected,
        "portfolio": {
            "deep": detailed(historical),
            "external_xperp": detailed(venue),
        },
        "per_rule": per_rule,
        "sources": {
            str(args.developed_rules): hashlib.sha256(rules_raw).hexdigest(),
            str(args.deep_csv): hashlib.sha256(deep_raw).hexdigest(),
            **external_sources,
        },
        "limitations": [
            "The three-rule intersection was identified after viewing both historical holdouts and X-Perp.",
            "This is a forward-paper candidate, not independent proof for live activation.",
            "Two rules have only three X-Perp observations each.",
            "Gross R excludes fees and client-selected position sizing at user request.",
        ],
    }
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "deep": report["portfolio"]["deep"]["all"],
        "deep_years": report["portfolio"]["deep"]["years"],
        "deep_negative_months": len(report["portfolio"]["deep"]["negative_months"]),
        "external_xperp": report["portfolio"]["external_xperp"]["all"],
        "rules": {key: value["external_xperp"]["all"]
                  for key, value in per_rule.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
