"""Select previously developed stock rules on 2025 and reveal 2026/X-Perp."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import pandas as pd

from filter_lab import gate, metrics
from stock_session_family_lab import (
    _matches, attach_market_context, market_context_by_day, replay,
)


THRESHOLDS = {
    .25: {"minimum_n": 40, "minimum_wr": 84, "minimum_r": 2, "minimum_pf": 1.2},
    .5: {"minimum_n": 30, "minimum_wr": 68, "minimum_r": 2, "minimum_pf": 1.15},
    .75: {"minimum_n": 20, "minimum_wr": 60, "minimum_r": 2, "minimum_pf": 1.15},
    1.0: {"minimum_n": 50, "minimum_wr": 51, "minimum_r": 2, "minimum_pf": 1.05},
}


def compact(rows: list[dict]) -> dict:
    value = metrics(rows)
    return {key: value[key] for key in (
        "n", "wr", "wr_lower_95", "net_r", "mean_r", "pf", "dd_r",
        "active_days", "symbols")}


def year(row: dict) -> int:
    return pd.Timestamp(row["entry_time"], unit="s", tz="UTC").year


def route(rows: list[dict], selected: list[dict], cap: int = 4) -> list[dict]:
    candidates = []
    for priority, item in enumerate(selected):
        rule = tuple(tuple(part) for part in item["rule"])
        for row in rows:
            if _matches(row, rule):
                candidates.append({**row, "route_priority": priority,
                                   "route_id": item["source_rule_index"]})
    candidates.sort(key=lambda row: (row["route_priority"], row["family"]))
    unique = {}
    for row in candidates:
        unique.setdefault((row["symbol"], row["entry_time"]), row)
    return gate(sorted(unique.values(), key=lambda row: (
        row["entry_time"], row["symbol"], row["route_priority"])), cap)


def detailed(rows: list[dict]) -> dict:
    years = sorted({year(row) for row in rows})
    month_keys = sorted({pd.Timestamp(row["entry_time"], unit="s", tz="UTC").strftime("%Y-%m")
                         for row in rows})
    monthly = {month: compact([row for row in rows if pd.Timestamp(
        row["entry_time"], unit="s", tz="UTC").strftime("%Y-%m") == month])
        for month in month_keys}
    return {
        "all": compact(rows),
        "years": {str(value): compact([row for row in rows if year(row) == value])
                  for value in years},
        "months": monthly,
        "negative_months": {key: value for key, value in monthly.items()
                            if value["net_r"] < -.01},
    }


def load_external_all_targets(index_dir: Path, execution_dir: Path,
                              symbols: tuple[str, ...]) -> tuple[list[dict], dict]:
    rows, signal_candles, sources = [], {}, {}
    for symbol in symbols:
        signal_path = index_dir / f"{symbol}_15min_18000.pkl"
        execution_path = execution_dir / f"{symbol}_15min_18000.pkl"
        if not signal_path.exists() or not execution_path.exists():
            continue
        signal_raw = signal_path.read_bytes()
        execution_raw = execution_path.read_bytes()
        sources[str(signal_path)] = hashlib.sha256(signal_raw).hexdigest()
        sources[str(execution_path)] = hashlib.sha256(execution_raw).hexdigest()
        signal_candles[symbol] = pickle.loads(signal_raw)
        rows.extend(replay(signal_candles[symbol], symbol,
                           targets=tuple(THRESHOLDS),
                           execution=pickle.loads(execution_raw)))
    if "QQQUSDT" in signal_candles:
        attach_market_context(rows, market_context_by_day(signal_candles["QQQUSDT"]))
    return rows, sources


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
    developed = json.loads(rules_raw)["selected"]
    deep_raw = args.deep_csv.read_bytes()
    deep = pd.read_csv(args.deep_csv).to_dict("records")
    validation = [row for row in deep if year(row) == 2025]
    selected = []
    audit = []
    for index, item in enumerate(developed):
        rule = tuple(tuple(part) for part in item["rule"])
        rows = gate([row for row in validation if _matches(row, rule)], 4)
        value = compact(rows)
        target = float(rule[2][2])
        threshold = THRESHOLDS[target]
        accepted = (
            value["n"] >= threshold["minimum_n"]
            and value["wr"] >= threshold["minimum_wr"]
            and value["net_r"] >= threshold["minimum_r"]
            and (value["pf"] or 0) >= threshold["minimum_pf"]
        )
        record = {
            "source_rule_index": index,
            "rule": item["rule"],
            "development": item["development"],
            "validation_2025": value,
            "selected": accepted,
        }
        audit.append(record)
        if accepted:
            selected.append(record)
    selected.sort(key=lambda item: (
        item["validation_2025"]["net_r"] / max(item["validation_2025"]["dd_r"], .25),
        item["validation_2025"]["mean_r"],
    ), reverse=True)

    symbols = tuple(value.strip() for value in args.symbols.split(",") if value.strip())
    external_rows, external_sources = load_external_all_targets(
        args.external_index_dir, args.external_execution_dir, symbols)
    layers = {
        "high_win_0_25": [item for item in selected if float(item["rule"][2][2]) == .25],
        "profit_0_5_to_1_0": [item for item in selected if float(item["rule"][2][2]) > .25],
        "combined": selected,
    }
    results = {}
    for name, layer in layers.items():
        historical = route(deep, layer)
        venue = route(external_rows, layer)
        results[name] = {
            "selected_count": len(layer),
            "selected_rule_indices": [item["source_rule_index"] for item in layer],
            "deep": detailed(historical),
            "validation_2025": compact([row for row in historical if year(row) == 2025]),
            "test_2026": compact([row for row in historical if year(row) == 2026]),
            "external_xperp": detailed(venue),
        }

    report = {
        "status": "RESEARCH_ONLY",
        "method": "rules developed on 2020-24; target-specific eligibility selected on 2025; 2026 and X-Perp revealed afterward",
        "target_thresholds": {str(key): value for key, value in THRESHOLDS.items()},
        "selected": selected,
        "rule_audit": audit,
        "layers": results,
        "sources": {
            str(args.developed_rules): hashlib.sha256(rules_raw).hexdigest(),
            str(args.deep_csv): hashlib.sha256(deep_raw).hexdigest(),
            **external_sources,
        },
        "limitations": [
            "This audit was run after earlier aggregate inspection of 2026; it is diagnostic, not pristine evidence.",
            "X-Perp coverage is short and does not participate in selection.",
            "Gross R excludes fees and client-selected position sizing at user request.",
        ],
    }
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"selected_count": len(selected), "layers": {
        name: {"selected_count": value["selected_count"],
               "validation_2025": value["validation_2025"],
               "test_2026": value["test_2026"],
               "external_xperp": value["external_xperp"]["all"]}
        for name, value in results.items()}}, indent=2))


if __name__ == "__main__":
    main()
