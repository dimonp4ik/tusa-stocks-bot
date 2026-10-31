"""Test official CFTC Nasdaq positioning as a causal stock regime filter.

The candidate set is deliberately small and semantic.  CFTC reports dated
Tuesday become usable only seven days later.  Development (2017-2021) and
validation (2022-2023) select one route before 2024 is inspected.  Later
years remain sealed whenever the preceding holdout fails.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

import stock_15m_multisession_regime_oct6_audit as base
from src.research_profit_gate import economic_gate


ROOT = Path("reports/audit_2026_09_08")
CFTC = ROOT / "stock_cftc_tff_history_oct31.json"
OUT = ROOT / "stock_cftc_tff_material_oct31.json"
ROWS_OUT = ROOT / "stock_cftc_tff_material_oct31.csv"
DEVELOPMENT = (2017, 2018, 2019, 2020, 2021)
VALIDATION = (2022, 2023)

# Each rule was fixed before outcomes were evaluated.  Values are expanding
# 52-report z-scores of net positions divided by open interest.
RULES = (
    {"name": "asset_manager_confirms", "am_min": 0.0},
    {"name": "asset_manager_strong", "am_min": 0.5},
    {"name": "leveraged_not_crowded", "lev_max": 1.0},
    {"name": "leveraged_contrarian", "lev_max": 0.0},
    {"name": "dealer_hedge_confirms", "dealer_max": 0.0},
    {"name": "asset_confirm_and_not_crowded", "am_min": 0.0, "lev_max": 1.0},
    {"name": "institutional_spread_confirms", "spread_min": 0.0},
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_cftc() -> pd.DataFrame:
    payload = json.loads(CFTC.read_text(encoding="utf-8"))
    if payload.get("status") != "IMMUTABLE_OFFICIAL_CFTC_RESEARCH_INPUT":
        raise ValueError("unexpected CFTC input status")
    rows = [row for row in payload["records"]
            if row["contract"] == "nasdaq_100_consolidated"]
    frame = pd.DataFrame(rows).sort_values("available_at_utc").reset_index(drop=True)
    frame["available_time"] = pd.to_datetime(
        frame.available_at_utc, utc=True).astype("int64") // 10**9
    oi = frame.open_interest.astype(float)
    for prefix, long_col, short_col in (
            ("am", "asset_manager_long", "asset_manager_short"),
            ("lev", "leveraged_money_long", "leveraged_money_short"),
            ("dealer", "dealer_long", "dealer_short")):
        frame[f"{prefix}_net"] = (
            frame[long_col].astype(float)-frame[short_col].astype(float))/oi
        expanding_mean = frame[f"{prefix}_net"].shift(1).rolling(52, min_periods=26).mean()
        expanding_std = frame[f"{prefix}_net"].shift(1).rolling(52, min_periods=26).std()
        frame[f"{prefix}_z"] = (
            (frame[f"{prefix}_net"]-expanding_mean)/expanding_std.replace(0, np.nan))
    frame["spread_z"] = frame.am_z-frame.lev_z
    columns = ["available_time", "report_date", "am_z", "lev_z",
               "dealer_z", "spread_z"]
    return frame[columns].dropna().sort_values("available_time")


def attach(features: pd.DataFrame, cftc: pd.DataFrame) -> pd.DataFrame:
    left = features.sort_values("entry_time").copy()
    output = pd.merge_asof(
        left, cftc, left_on="entry_time", right_on="available_time",
        direction="backward", allow_exact_matches=True)
    return output.dropna(subset=["am_z", "lev_z", "dealer_z", "spread_z"])


def apply_rule(entries: pd.DataFrame, rule: dict) -> pd.DataFrame:
    found = entries.copy()
    direction = found.direction_n.astype(float)
    mask = pd.Series(True, index=found.index)
    if "am_min" in rule:
        mask &= direction*found.am_z >= float(rule["am_min"])
    if "lev_max" in rule:
        mask &= direction*found.lev_z <= float(rule["lev_max"])
    if "dealer_max" in rule:
        mask &= direction*found.dealer_z <= float(rule["dealer_max"])
    if "spread_min" in rule:
        mask &= direction*found.spread_z >= float(rule["spread_min"])
    output = found.loc[mask].copy()
    output["cftc_rule"] = rule["name"]
    # Preserve the price rank and use positioning only as a causal tie-breaker.
    output["score"] = output.score.astype(float) + 1e-6*(
        direction.loc[mask]*output.am_z-direction.loc[mask]*output.lev_z)
    return output


def accepted_rows(frame: pd.DataFrame) -> list[dict]:
    return base.portfolio_gate(
        frame.to_dict("records"), None,
        rank_key=lambda row: (-float(row["score"]), str(row["symbol"])))


def metrics(frame: pd.DataFrame, features: pd.DataFrame,
            years: tuple[int, ...]) -> dict:
    selected = frame[frame.year.isin(years)].copy()
    days = features[features.year.isin(years)].session_date.nunique()
    value = base.account_metrics(selected, days)
    rows = accepted_rows(selected)
    hits = sum(int(row["outcome_code"]) == 1 for row in rows)
    value["literal_target_hits"] = hits
    value["literal_target_hit_rate"] = round(100*hits/len(rows), 4) if rows else 0.0
    value["minimum_month_pct"] = round(
        min(value["monthly_return_pct"].values()), 4) if value["monthly_return_pct"] else 0.0
    value["economic_gate"] = economic_gate(
        [float(row["signed_return"]) for row in rows],
        float(value["trades_per_market_day"])) if rows else {
            "n": 0, "observed_trades_per_day": 0.0,
            "projected_monthly_return_pct": 0.0,
            "frequency_pass": False, "return_pass": False, "passes": False}
    hours: dict[int, int] = {}
    symbols: dict[str, int] = {}
    for row in rows:
        hour = int(float(row["entry_time"])//3600)
        hours[hour] = hours.get(hour, 0)+1
        symbol = str(row["symbol"])
        symbols[symbol] = symbols.get(symbol, 0)+1
    value["max_entries_same_utc_hour"] = max(hours.values(), default=0)
    value["largest_symbol_share_pct"] = round(
        100*max(symbols.values(), default=0)/len(rows), 4) if rows else 0.0
    return value


def annual(frame: pd.DataFrame, features: pd.DataFrame,
           years: tuple[int, ...]) -> dict:
    return {str(year): metrics(frame, features, (year,)) for year in years}


def development_precheck(candidate: dict, baseline: dict) -> bool:
    return bool(
        candidate["n"] >= 200
        and candidate["return_pct"] > 0
        and candidate["dd_pct"] <= 12
        and candidate["trades_per_market_day"] >= .25
        and candidate["economic_gate"]["projected_monthly_return_pct"]
            >= baseline["economic_gate"]["projected_monthly_return_pct"]
        and candidate["dd_pct"] <= baseline["dd_pct"]+1e-9
        and candidate["negative_months"] <= baseline["negative_months"])


def validation_precheck(candidate: dict, baseline: dict) -> bool:
    return bool(
        candidate["n"] >= 80
        and candidate["return_pct"] > 0
        and candidate["dd_pct"] <= 8
        and candidate["economic_gate"]["return_pass"]
        and candidate["economic_gate"]["projected_monthly_return_pct"]
            >= .9*baseline["economic_gate"]["projected_monthly_return_pct"]
        and candidate["dd_pct"] <= baseline["dd_pct"]+1e-9
        and candidate["negative_months"] <= baseline["negative_months"])


def holdout_ok(value: dict) -> bool:
    return bool(
        value["n"] >= 40 and value["return_pct"] > 0
        and value["dd_pct"] <= 5 and value["max_loss_streak"] <= 3
        and value["economic_gate"]["return_pass"])


def public(item: dict | None) -> dict | None:
    if item is None:
        return None
    return {key: value for key, value in item.items() if key != "_rows"}


def main() -> None:
    candles, price_sources = base.load_history()
    raw_features, metadata = base.build_features(candles)
    cftc = load_cftc()
    features = attach(raw_features, cftc)
    frontier: list[dict] = []
    survivors: list[dict] = []
    for spec, target, stop, sessions in itertools.product(
            base.SPECS, base.TARGETS, base.STOPS, base.HOLD_SESSIONS):
        entries = base.apply_spec(features, spec)
        if len(entries) < 300:
            continue
        resolved = base.add_outcomes(entries, candles, metadata, target, stop, sessions)
        baseline_dev = metrics(resolved, features, DEVELOPMENT)
        for rule in RULES:
            filtered_entries = apply_rule(entries, rule)
            if len(filtered_entries) < 100:
                continue
            allowed = set(zip(filtered_entries.symbol, filtered_entries.entry_time))
            candidate_rows = resolved[
                [pair in allowed for pair in zip(resolved.symbol, resolved.entry_time)]].copy()
            candidate_rows["cftc_rule"] = rule["name"]
            dev = metrics(candidate_rows, features, DEVELOPMENT)
            item = {
                "route_id": f"{spec['name']}__t{target}_s{stop}_h{sessions}__{rule['name']}",
                "entry": spec, "exit": {"target_pct": target,
                    "stop_pct": stop, "hold_sessions": sessions},
                "cftc_rule": rule,
                "development_2017_2021": dev,
                "development_baseline": baseline_dev,
            }
            frontier.append(item)
            if not development_precheck(dev, baseline_dev):
                continue
            dev_years = annual(candidate_rows, features, DEVELOPMENT)
            item["development_years"] = dev_years
            if sum(value["return_pct"] > 0 for value in dev_years.values()) < 4:
                continue
            baseline_validation = metrics(resolved, features, VALIDATION)
            validation = metrics(candidate_rows, features, VALIDATION)
            item["validation_2022_2023"] = validation
            item["validation_baseline"] = baseline_validation
            if not validation_precheck(validation, baseline_validation):
                continue
            validation_years = annual(candidate_rows, features, VALIDATION)
            item["validation_years"] = validation_years
            if not all(value["return_pct"] > 0 for value in validation_years.values()):
                continue
            item["score"] = round(
                validation["economic_gate"]["projected_monthly_return_pct"]
                -2*validation["dd_pct"]-validation["negative_months"], 8)
            item["_rows"] = candidate_rows
            survivors.append(item)
    survivors.sort(key=lambda item: item["score"], reverse=True)
    frontier.sort(key=lambda item: (
        item["development_2017_2021"]["economic_gate"]["projected_monthly_return_pct"],
        -item["development_2017_2021"]["dd_pct"]), reverse=True)
    frozen = survivors[0] if survivors else None
    results = checks = None
    if frozen is not None:
        rows = frozen["_rows"]
        rows.to_csv(ROWS_OUT, index=False)
        test_2024 = metrics(rows, features, (2024,))
        results = {"test_2024": test_2024, "test_2025": None, "test_2026": None}
        checks = {"test_2024": holdout_ok(test_2024),
                  "test_2025": None, "test_2026": None}
        if checks["test_2024"]:
            test_2025 = metrics(rows, features, (2025,))
            results["test_2025"] = test_2025
            checks["test_2025"] = holdout_ok(test_2025)
            if checks["test_2025"]:
                test_2026 = metrics(rows, features, (2026,))
                results["test_2026"] = test_2026
                checks["test_2026"] = holdout_ok(test_2026)
    passed = bool(checks and all(value is True for value in checks.values()))
    status = "RESEARCH_CANDIDATE_AWAITING_FORWARD" if passed else "REJECTED"
    report = {
        "status": status,
        "protocol": {
            "source": "official CFTC Traders in Financial Futures, futures only",
            "contract": "Nasdaq-100 Consolidated, code 20974+",
            "availability": "Tuesday report becomes usable seven full days later",
            "features": "52-report causal z-scores; rolling moments exclude current report",
            "fixed_rules": list(RULES),
            "selection": "2017-2021 development; 2022-2023 validation; freeze one route",
            "sealed_order": ["2024", "2025", "2026"],
            "money": "3% entry-equity margin and fixed 10x gross",
        },
        "feature_rows": len(features),
        "cftc_rows_after_warmup": len(cftc),
        "routes_evaluated": len(frontier),
        "survivors_before_hidden": len(survivors),
        "frozen_before_hidden": public(frozen),
        "top_survivors": [public(item) for item in survivors[:10]],
        "development_frontier": [public(item) for item in frontier[:15]],
        "results": results,
        "checks": checks,
        "sources": {**price_sources, str(CFTC): sha256(CFTC),
                    str(Path(__file__)): sha256(Path(__file__))},
        "limitations": [
            "CFTC positioning is weekly and cannot time intraday entries.",
            "The Nasdaq contract is a broad regime proxy, not stock-specific flow.",
            "Dukascopy cash prices establish history but are not X-Perp fills.",
            "Projected monthly return is a diagnostic and not a forecast.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": status, "feature_rows": len(features),
        "routes_evaluated": len(frontier),
        "survivors_before_hidden": len(survivors),
        "frozen": public(frozen), "results": results, "checks": checks,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
