"""Development-only subset selection for the fixed stock calendar families."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

import stock_15m_multisession_regime_oct6_audit as base
import stock_calendar_material_oct10_audit as cal


ROOT = Path("reports/audit_2026_09_08")
OUT = ROOT / "stock_calendar_subset_oct10.json"
DEV_YEARS = cal.DEV_YEARS
VALIDATION_YEARS = cal.VALIDATION_YEARS


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def combine(parts: list[pd.DataFrame], subset: tuple[str, ...]) -> pd.DataFrame:
    value = pd.concat([parts[name] for name in subset], ignore_index=True)
    keys = ["symbol", "entry_time"]
    consensus = value.groupby(keys).direction_n.transform("nunique").eq(1)
    value = value.loc[consensus].copy()
    value["family_count"] = value.groupby(keys).family.transform("nunique")
    value["score"] = value.groupby(keys).score.transform("max") + value.family_count
    return value.sort_values(
        ["score", "symbol"], ascending=[False, True]
    ).drop_duplicates(keys, keep="first")


def development_candidate(value: dict, annual: dict) -> bool:
    return bool(value["n"] >= 250 and value["return_pct"] > 0
                and value["dd_pct"] <= 10
                and value["trades_per_market_day"] >= .3
                and sum(item["return_pct"] > 0 for item in annual.values()) >= 4)


def development_score(value: dict, annual: dict) -> float:
    worst_year = min(item["return_pct"] for item in annual.values())
    return round(
        2*worst_year + value["median_month_pct"]
        + .05*value["return_pct"] - value["dd_pct"]
        - .5*value["negative_months"], 8)


def validation_pass(value: dict, annual: dict) -> bool:
    return bool(value["n"] >= 100 and value["return_pct"] > 0
                and value["dd_pct"] <= 5 and value["negative_months"] == 0
                and all(item["return_pct"] > 0 for item in annual.values())
                and 75 <= value["literal_target_hit_rate"] <= 95
                and value["economic_gate"]["passes"])


def compact(item: dict) -> dict:
    return {key: value for key, value in item.items() if key != "_rows"}


def main() -> None:
    candles, sources = base.load_history()
    raw_features, metadata = base.build_features(candles)
    features = cal.add_calendar(raw_features)
    resolved: dict[tuple[str, str], dict[str, pd.DataFrame]] = {}
    for profile, exit_spec in itertools.product(cal.PROFILES, cal.EXITS):
        by_family = {}
        for family in cal.BASE_FAMILIES:
            proposed = cal.entries(features, family, profile)
            value = base.add_outcomes(
                proposed, candles, metadata, exit_spec["target"],
                exit_spec["stop"], exit_spec["sessions"])
            value["module"] = f"stock_calendar_{family}"
            by_family[family] = value
        resolved[(profile, exit_spec["name"])] = by_family

    candidates = []
    for profile, exit_spec in itertools.product(cal.PROFILES, cal.EXITS):
        parts = resolved[(profile, exit_spec["name"])]
        for size in range(1, len(cal.BASE_FAMILIES)+1):
            for subset in itertools.combinations(cal.BASE_FAMILIES, size):
                rows = combine(parts, subset)
                development = cal.evaluate(rows, features, DEV_YEARS)
                annual = {str(year): cal.evaluate(rows, features, (year,))
                          for year in DEV_YEARS}
                if not development_candidate(development, annual):
                    continue
                candidates.append({
                    "route_id": f"{profile}_{exit_spec['name']}_"
                                + "__".join(subset),
                    "profile": profile, "exit": exit_spec,
                    "families": list(subset),
                    "development_2017_2021": development,
                    "development_years": annual,
                    "selection_score": development_score(development, annual),
                    "_rows": rows,
                })
    candidates.sort(key=lambda item: (
        item["selection_score"],
        item["development_2017_2021"]["return_pct"]), reverse=True)
    frozen = candidates[0] if candidates else None
    validation = validation_years = test_2024 = confirmation_2025 = None
    if frozen:
        validation = cal.evaluate(frozen["_rows"], features, VALIDATION_YEARS)
        validation_years = {
            str(year): cal.evaluate(frozen["_rows"], features, (year,))
            for year in VALIDATION_YEARS}
        if validation_pass(validation, validation_years):
            test_2024 = cal.evaluate(frozen["_rows"], features, (2024,))
        if test_2024 and validation_pass(test_2024, {"2024": test_2024}):
            confirmation_2025 = cal.evaluate(frozen["_rows"], features, (2025,))
    status = ("RESEARCH_CANDIDATE_AWAITING_XPERP_FORWARD"
              if confirmation_2025 and validation_pass(
                  confirmation_2025, {"2025": confirmation_2025})
              else "REJECTED")
    report = {
        "status": status, "ready_for_live_orders": False,
        "protocol": {
            "family_library": list(cal.BASE_FAMILIES),
            "subsets": 2**len(cal.BASE_FAMILIES)-1,
            "profiles": list(cal.PROFILES), "exits": list(cal.EXITS),
            "selection": "development score penalizes worst year, drawdown and losing months",
            "development": "2017-2021", "validation": "2022-2023",
            "test": "2024", "confirmation": "2025",
            "orders": "market entries only",
            "money": "3% current-equity margin, fixed 10x leverage",
            "fees": "excluded by user instruction",
        },
        "eligible_development_candidates": len(candidates),
        "frozen_before_validation": compact(frozen) if frozen else None,
        "validation_2022_2023": validation,
        "validation_years": validation_years,
        "validation_pass": bool(validation and validation_years
                                and validation_pass(validation, validation_years)),
        "test_2024": test_2024, "confirmation_2025": confirmation_2025,
        "development_frontier": [compact(item) for item in candidates[:20]],
        "sources": {**sources,
                    str(Path(cal.__file__)): digest(Path(cal.__file__)),
                    str(Path(base.__file__)): digest(Path(base.__file__)),
                    str(Path(__file__)): digest(Path(__file__))},
        "limitations": [
            "Subset selection increases development selection risk despite sealed validation.",
            "Only one frozen subset is revealed on 2022-2023.",
            "Dukascopy cash CFDs establish structural history, not X-Perp fills.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": status, "eligible": len(candidates),
        "frozen": compact(frozen) if frozen else None,
        "validation": validation, "validation_years": validation_years,
        "validation_pass": report["validation_pass"],
        "test_2024": test_2024, "confirmation_2025": confirmation_2025,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
