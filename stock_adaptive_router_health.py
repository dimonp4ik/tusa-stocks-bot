"""Causal shadow-health repair for the frozen stock regime router."""
from __future__ import annotations

import argparse
import collections
import hashlib
import heapq
import itertools
import json
from pathlib import Path

import pandas as pd

from filter_lab import gate
from stock_robust_regime_router import (
    DEVELOPMENT, add_context, compact, detailed, routed_candidates,
)
from stock_walk_forward_filter_lab import load_external


VALIDATION = 2025
TEST = 2026


def health_filter(frame: pd.DataFrame, *, key_mode: str, window: int,
                  min_history: int, minimum_wr: float,
                  minimum_r: float, cap: int = 4) -> pd.DataFrame:
    history = collections.defaultdict(lambda: collections.deque(maxlen=window))
    pending = []
    eligible = []
    serial = 0
    for row in frame.sort_values(["entry_time", "symbol", "route_priority"]).to_dict("records"):
        now = float(row["entry_time"])
        while pending and pending[0][0] < now:
            _, _, key, outcome = heapq.heappop(pending)
            history[key].append(outcome)
        key = ((int(row["route_id"]), row["symbol"])
               if key_mode == "route_symbol" else int(row["route_id"]))
        recent = history[key]
        enabled = (len(recent) < min_history or
                   (sum(recent) >= minimum_r
                    and sum(value > 0 for value in recent) / len(recent) >= minimum_wr))
        heapq.heappush(pending, (float(row["exit_time"]), serial,
                                key, float(row["net_r"])))
        serial += 1
        if enabled:
            eligible.append(row)
    if not eligible:
        return frame.iloc[0:0]
    accepted = gate(eligible, cap)
    return pd.DataFrame(accepted) if accepted else frame.iloc[0:0]


def negative_month_count(frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    month = pd.to_datetime(frame["entry_time"], unit="s", utc=True).dt.strftime("%Y-%m")
    return int(sum(part["net_r"].sum() < -.01 for _, part in frame.groupby(month)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("router", type=Path)
    parser.add_argument("deep_csv", type=Path)
    parser.add_argument("--external-index-dir", type=Path, required=True)
    parser.add_argument("--external-execution-dir", type=Path, required=True)
    parser.add_argument("--symbols", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    router_raw = args.router.read_bytes()
    selected = json.loads(router_raw)["selected"]
    deep_raw = args.deep_csv.read_bytes()
    deep = pd.read_csv(args.deep_csv)
    deep = add_context(deep[deep["target_r"] == .25])
    candidates = routed_candidates(deep, selected)

    tested = []
    specs = []
    specs.extend(itertools.product(
        ("route",), (20, 40, 60), (10, 20), (.80, .825, .85), (-.5, 0, 1)))
    specs.extend(itertools.product(
        ("route_symbol",), (10, 20, 40), (5, 10), (.80, .825, .85), (-.5, 0)))
    for key_mode, window, min_history, minimum_wr, minimum_r in specs:
        if min_history > window:
            continue
        params = {
            "key_mode": key_mode, "window": window, "min_history": min_history,
            "minimum_wr": minimum_wr, "minimum_r": minimum_r,
        }
        chosen = health_filter(candidates, **params)
        development = chosen[chosen["year"].isin(DEVELOPMENT)]
        validation = chosen[chosen["year"] == VALIDATION]
        development_metrics = compact(development)
        validation_metrics = compact(validation)
        yearly = {year: compact(development[development["year"] == year])
                  for year in DEVELOPMENT}
        eligible = (
            development_metrics["n"] >= 1200
            and validation_metrics["n"] >= 120
            and development_metrics["wr"] >= 84
            and validation_metrics["wr"] >= 81
            and validation_metrics["net_r"] > 0
            and all(value["net_r"] > 0 for value in yearly.values())
        )
        score = (-negative_month_count(validation) * 10
                 + validation_metrics["net_r"] / max(validation_metrics["dd_r"], .25)
                 + development_metrics["net_r"] / max(development_metrics["dd_r"], .25) * .1)
        tested.append({
            "parameters": params, "eligible": eligible, "score": score,
            "development": development_metrics,
            "validation_2025": validation_metrics,
            "development_years": yearly,
            "validation_negative_months": negative_month_count(validation),
        })

    chosen_spec = max((item for item in tested if item["eligible"]),
                      key=lambda item: item["score"], default=None)
    final = None
    external_final = None
    external_sources = {}
    if chosen_spec:
        final = health_filter(candidates, **chosen_spec["parameters"])
        chosen_spec["test_2026"] = compact(final[final["year"] == TEST])
        chosen_spec["full"] = detailed(final)

        symbols = tuple(value.strip() for value in args.symbols.split(",") if value.strip())
        external, external_sources = load_external(
            args.external_index_dir, args.external_execution_dir, symbols)
        external = add_context(external[external["target_r"] == .25])
        external["source"] = "xperp"
        historical_prefix = deep[deep["year"] <= VALIDATION].copy()
        historical_prefix["source"] = "deep"
        transfer = pd.concat([historical_prefix, external], ignore_index=True)
        transfer_candidates = routed_candidates(transfer, selected)
        transfer_final = health_filter(transfer_candidates, **chosen_spec["parameters"])
        external_final = transfer_final[transfer_final["source"] == "xperp"]
        chosen_spec["external_xperp"] = compact(external_final)

    baseline_rows = gate(candidates.sort_values(
        ["entry_time", "symbol", "route_priority"]).to_dict("records"), 4)
    baseline = pd.DataFrame(baseline_rows)
    report = {
        "status": "FROZEN_FOR_2026" if chosen_spec else "NO_STABLE_HEALTH_GATE",
        "tested_count": len(tested),
        "chosen": chosen_spec,
        "baseline": detailed(baseline),
        "top_validation": sorted(tested, key=lambda value: value["score"], reverse=True)[:12],
        "sources": {
            str(args.router): hashlib.sha256(router_raw).hexdigest(),
            str(args.deep_csv): hashlib.sha256(deep_raw).hexdigest(),
            **external_sources,
        },
        "limitations": [
            "Router rules were selected on 2020-2024 before this health audit.",
            "Health parameters use 2025 as validation; 2026 is revealed afterward in this audit.",
            "Disabled routes continue producing causal shadow outcomes observed only after exit.",
            "Gross R excludes fees and client-selected position sizing at user request.",
        ],
    }
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if final is not None:
        final.to_csv(args.out.with_suffix(".csv"), index=False)
    print(json.dumps({
        "status": report["status"], "tested_count": len(tested),
        "baseline_2025": compact(baseline[baseline["year"] == VALIDATION]),
        "baseline_2026": compact(baseline[baseline["year"] == TEST]),
        "chosen": chosen_spec,
    }, indent=2))


if __name__ == "__main__":
    main()
