"""Causal cross-source agreement audit for material stock moves."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

import stock_15m_multisession_regime_oct6_audit as base
import stock_cboe_vol_surface_material_oct9_audit as cboe
import stock_finra_short_volume_material_oct9_audit as finra
import stock_fred_rates_material_oct10_audit as fred
import stock_qqq_spy_factor_router_oct9_audit as factor
from src.research_profit_gate import economic_gate, minimum_binary_hit_rate_pct


ROOT = Path("reports/audit_2026_09_08")
OUT = ROOT / "stock_cross_source_agreement_oct10.json"
EXITS = (
    {"name": "balanced", "target": 1.0, "stop": .75, "sessions": 1},
    {"name": "swing", "target": 1.5, "stop": .75, "sessions": 2},
    {"name": "runner", "target": 2.0, "stop": 1.0, "sessions": 3},
)
PROFILES = ("broad", "strict")
MIN_SOURCES = (1, 2, 3)
DEV_YEARS = (2019, 2020, 2021)
VALIDATION_YEARS = (2022, 2023)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_source(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    value = frame.copy()
    value["source"] = source
    value["source_family"] = value.family.astype(str)
    keys = ["source", "symbol", "entry_time"]
    consensus = value.groupby(keys).direction_n.transform("nunique").eq(1)
    value = value.loc[consensus].copy()
    value["source_score_pct"] = value.groupby(
        ["source", "session_date"]).score.rank(pct=True, method="average")
    return value.sort_values(
        ["source_score_pct", "symbol"], ascending=[False, True]
    ).drop_duplicates(keys, keep="first")


def source_entries(features: pd.DataFrame, profile: str) -> tuple[
        pd.DataFrame, dict, dict[str, str]]:
    pieces: list[pd.DataFrame] = []
    counts = {}
    input_sources: dict[str, str] = {}
    factor_level = "loose" if profile == "broad" else "strict"
    factor_parts = [factor.entries(features, item) for item in factor.PROFILES
                    if item["level"] == factor_level]
    factor_rows = normalize_source(pd.concat(factor_parts, ignore_index=True),
                                   "qqq_spy_factor")
    pieces.append(factor_rows)
    counts["qqq_spy_factor"] = len(factor_rows)

    raw_surface, _, found_sources = cboe.load_surface()
    input_sources.update(found_sources)
    surface_features = cboe.attach_surface(
        features, cboe.surface_features(raw_surface))
    cboe_parts = [cboe.entries(surface_features, family, profile)
                  for family in cboe.FAMILIES]
    cboe_rows = normalize_source(pd.concat(cboe_parts, ignore_index=True),
                                 "cboe_surface")
    pieces.append(cboe_rows)
    counts["cboe_surface"] = len(cboe_rows)

    raw_rates, _, found_sources = fred.load_rates()
    input_sources.update(found_sources)
    rate_features = fred.attach_rates(features, fred.rate_features(raw_rates))
    fred_parts = [fred.entries(rate_features, family, profile)
                  for family in fred.BASE_FAMILIES]
    fred_rows = normalize_source(pd.concat(fred_parts, ignore_index=True),
                                 "fred_rates")
    pieces.append(fred_rows)
    counts["fred_rates"] = len(fred_rows)

    raw_short, _, found_sources = finra.load_short()
    input_sources.update(found_sources)
    short_features = finra.attach_short(features, finra.short_features(raw_short))
    finra_parts = [finra.entries(short_features, family, profile)
                   for family in finra.BASE_FAMILIES]
    finra_rows = normalize_source(pd.concat(finra_parts, ignore_index=True),
                                  "finra_short_volume")
    pieces.append(finra_rows)
    counts["finra_short_volume"] = len(finra_rows)
    return pd.concat(pieces, ignore_index=True), counts, input_sources


def agreement_entries(frame: pd.DataFrame, minimum: int) -> pd.DataFrame:
    keys = ["symbol", "entry_time"]
    direction_count = frame.groupby(keys).direction_n.transform("nunique")
    source_count = frame.groupby(keys).source.transform("nunique")
    value = frame.loc[direction_count.eq(1) & source_count.ge(minimum)].copy()
    value["source_count"] = value.groupby(keys).source.transform("nunique")
    value["agreement_score"] = value.groupby(keys).source_score_pct.transform("mean")
    value["score"] = value.source_count + value.agreement_score
    representatives = value.sort_values(
        ["score", "source_score_pct", "symbol"], ascending=[False, False, True]
    ).drop_duplicates(keys, keep="first")
    source_sets = value.groupby(keys).source.apply(
        lambda values: "+".join(sorted(set(values))))
    representatives["source_set"] = [
        source_sets.loc[(row.symbol, row.entry_time)]
        for row in representatives.itertuples()]
    representatives["family"] = f"cross_source_min{minimum}"
    return representatives


def accepted_rows(frame: pd.DataFrame) -> list[dict]:
    return base.portfolio_gate(
        frame.to_dict("records"), None,
        rank_key=lambda row: (-float(row["score"]), str(row["symbol"])))


def evaluate(frame: pd.DataFrame, features: pd.DataFrame,
             years: tuple[int, ...]) -> dict:
    selected = frame[frame.year.isin(years)].copy()
    market_days = features[features.year.isin(years)].session_date.nunique()
    result = base.account_metrics(selected, market_days)
    rows = accepted_rows(selected)
    hits = sum(int(row["outcome_code"]) == 1 for row in rows)
    result["literal_target_hits"] = hits
    result["literal_target_hit_rate"] = round(
        100*hits/len(rows), 4) if rows else 0.0
    result["source_set_count"] = dict(sorted(
        pd.Series([str(row["source_set"]) for row in rows]).value_counts().to_dict().items()))
    result["economic_gate"] = (economic_gate(
        [float(row["signed_return"]) for row in rows],
        float(result["trades_per_market_day"])) if rows else {
            "n": 0, "observed_trades_per_day": 0.0,
            "projected_monthly_return_pct": 0.0,
            "frequency_pass": False, "return_pass": False, "passes": False})
    values = list(result.get("monthly_return_pct", {}).values())
    result["minimum_month_pct"] = round(min(values), 4) if values else 0.0
    result["median_month_pct"] = round(float(np.median(values)), 4) if values else 0.0
    return result


def development_ok(value: dict, annual: dict) -> bool:
    return bool(value["n"] >= 200 and value["return_pct"] > 0
                and value["dd_pct"] <= 12
                and sum(item["return_pct"] > 0 for item in annual.values()) >= 2
                and value["trades_per_market_day"] >= .5)


def validation_ok(value: dict, annual: dict) -> bool:
    return bool(value["n"] >= 100 and value["return_pct"] > 0
                and value["dd_pct"] <= 8
                and all(item["return_pct"] > 0 for item in annual.values())
                and value["economic_gate"]["passes"])


def hidden_ok(value: dict) -> bool:
    return bool(value["n"] >= 100 and value["return_pct"] > 0
                and value["dd_pct"] <= 5 and value["negative_months"] == 0
                and 75 <= value["literal_target_hit_rate"] <= 95
                and value["economic_gate"]["passes"])


def public(item: dict | None) -> dict | None:
    return ({key: value for key, value in item.items() if key != "_rows"}
            if item else None)


def required_hit_rate(target: float, stop: float) -> float | None:
    value = minimum_binary_hit_rate_pct(target, stop)
    return round(value, 6) if value is not None else None


def main() -> None:
    candles, price_sources = factor.load()
    features, metadata = factor.build_features(candles)
    proposals, proposal_counts, external_sources = {}, {}, {}
    for profile in PROFILES:
        source_rows, counts, found_sources = source_entries(features, profile)
        if external_sources and found_sources != external_sources:
            raise ValueError("source hashes changed during audit")
        external_sources = found_sources
        proposal_counts[profile] = counts
        for minimum in MIN_SOURCES:
            proposals[(profile, minimum)] = agreement_entries(source_rows, minimum)

    frontier, survivors, validation_failures = [], [], []
    for (profile, minimum), exit_spec in itertools.product(proposals, EXITS):
        proposed = proposals[(profile, minimum)]
        if proposed.empty:
            continue
        resolved = base.add_outcomes(
            proposed, candles, metadata, exit_spec["target"],
            exit_spec["stop"], exit_spec["sessions"])
        resolved["module"] = f"stock_cross_source_min{minimum}"
        development = evaluate(resolved, features, DEV_YEARS)
        annual_dev = {str(year): evaluate(resolved, features, (year,))
                      for year in DEV_YEARS}
        item = {
            "route_id": f"min{minimum}_{profile}_{exit_spec['name']}",
            "minimum_sources": minimum, "profile": profile,
            "exit": exit_spec, "development_2019_2021": development,
            "development_years": annual_dev,
        }
        frontier.append(item)
        if not development_ok(development, annual_dev):
            continue
        validation = evaluate(resolved, features, VALIDATION_YEARS)
        annual_validation = {str(year): evaluate(resolved, features, (year,))
                             for year in VALIDATION_YEARS}
        item["validation_2022_2023"] = validation
        item["validation_years"] = annual_validation
        if not validation_ok(validation, annual_validation):
            validation_failures.append(item)
            continue
        item["score"] = round(
            validation["economic_gate"]["projected_monthly_return_pct"]
            - 2*validation["dd_pct"], 8)
        item["_rows"] = resolved
        survivors.append(item)

    survivors.sort(key=lambda item: item["score"], reverse=True)
    frozen = survivors[0] if survivors else None
    test_2024 = confirmation_2025 = None
    if frozen:
        test_2024 = evaluate(frozen["_rows"], features, (2024,))
        if hidden_ok(test_2024):
            confirmation_2025 = evaluate(frozen["_rows"], features, (2025,))
    frontier.sort(key=lambda item: (
        item["development_2019_2021"]["economic_gate"]["projected_monthly_return_pct"],
        item["development_2019_2021"]["literal_target_hit_rate"],
        item["development_2019_2021"]["return_pct"]), reverse=True)
    validation_failures.sort(key=lambda item: (
        item["validation_2022_2023"]["economic_gate"]["projected_monthly_return_pct"],
        item["validation_2022_2023"]["return_pct"]), reverse=True)
    status = ("RESEARCH_CANDIDATE_AWAITING_XPERP_FORWARD"
              if confirmation_2025 and hidden_ok(confirmation_2025)
              else "REJECTED")
    source_hashes = {
        str(Path(module.__file__)): digest(Path(module.__file__))
        for module in (cboe, finra, fred, factor)}
    report = {
        "status": status, "ready_for_live_orders": False,
        "protocol": {
            "sources": ["Cboe volatility surface", "FRED rates",
                        "FINRA short volume", "QQQ-SPY factor"],
            "agreement": "conflicts dropped; same-direction distinct sources counted",
            "minimum_sources": list(MIN_SOURCES), "profiles": list(PROFILES),
            "exits": list(EXITS), "entry": "next regular-session market bar",
            "development": "2019-2021", "validation": "2022-2023",
            "test": "2024", "confirmation": "2025",
            "money": "3% current-equity margin, fixed 10x leverage",
            "orders": "market entries only", "fees": "excluded by user instruction",
        },
        "rows": {"features": len(features), "proposal_counts": proposal_counts,
                 "agreement_counts": {
                     f"{profile}_min{minimum}": len(proposals[(profile, minimum)])
                     for profile in PROFILES for minimum in MIN_SOURCES}},
        "minimum_ideal_hit_rate_for_30pct_month_at_3_trades_day": {
            spec["name"]: required_hit_rate(spec["target"], spec["stop"])
            for spec in EXITS},
        "qualified_before_2024": len(survivors),
        "frozen_before_2024": public(frozen), "test_2024": test_2024,
        "confirmation_2025": confirmation_2025,
        "validation_failures": [public(item) for item in validation_failures[:12]],
        "development_frontier": [public(item) for item in frontier[:18]],
        "sources": {**price_sources, **external_sources, **source_hashes,
                    str(Path(__file__)): digest(Path(__file__))},
        "limitations": [
            "Agreement can reduce frequency without increasing conditional edge.",
            "Daily sources are delayed according to their individual causal audits.",
            "Dukascopy cash CFDs establish structural history, not X-Perp fills.",
            "Any historical survivor still requires untouched direct X-Perp evidence.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": status, "rows": report["rows"],
        "qualified_before_2024": len(survivors),
        "frozen_before_2024": public(frozen), "test_2024": test_2024,
        "confirmation_2025": confirmation_2025,
        "best_development": report["development_frontier"][:5],
        "best_validation_failures": report["validation_failures"][:3],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
