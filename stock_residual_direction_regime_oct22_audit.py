"""Causal reversal/continuation router for stock residual extremes.

The same volatility-scaled extreme can mean exhaustion in a quiet tape or
momentum in a directional tape.  This audit learns one transparent direction
style per regime on 2017-2020 and freezes it before later eras.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

import stock_cash_residual_cross_section_oct21_audit as base
import stock_cash_xperp_leadlag_oct6_audit as resolver
import stock_vol_scaled_residual_cross_section_oct21_audit as source


OUT = base.ROOT / "stock_residual_direction_regime_oct22.json"
ROWS_OUT = base.ROOT / "stock_residual_direction_regime_oct22.csv"
DEVELOPMENT = (2017, 2018, 2019, 2020)
VALIDATION = (2021, 2022)
TEST = (2023, 2024)
CONFIRMATION = (2025, 2026)
RULE = {"lookback_bars": 4, "z_threshold": 2.0,
        "minimum_raw_residual_pct": .2, "target_pct": 2.0,
        "stop_pct": .5, "hold_bars": 32}
GROUPS = (
    ("market_strength", "residual_market_alignment"),
    ("market_strength", "residual_market_alignment", "direction_name"),
    ("market_strength", "residual_market_alignment", "volatility_regime"),
    ("market_strength", "volatility_regime"),
)
MARKET_THRESHOLDS = (.002, .004, .006)
MIN_COUNTS = (30, 75)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolved_styles(features: pd.DataFrame, candles: dict) -> pd.DataFrame:
    entries = source.select_entries(
        features, RULE["z_threshold"], RULE["minimum_raw_residual_pct"] / 100)
    pieces = []
    for style, multiplier in (("reversal", -1), ("continuation", 1)):
        part = entries.copy()
        part["direction_n"] = (
            multiplier * np.sign(part.scaled_residual)).astype(np.int8)
        found = resolver.add_outcomes(
            part, candles, RULE["target_pct"], RULE["stop_pct"],
            RULE["hold_bars"])
        found["style"] = style
        pieces.append(found)
    return pd.concat(pieces, ignore_index=True)


def enrich(frame: pd.DataFrame, market_threshold: float,
           volatility_cuts: tuple[float, float]) -> pd.DataFrame:
    result = frame.copy()
    result["market_strength"] = np.where(
        result.cross_mean.abs().ge(market_threshold), "directional", "quiet")
    aligned = np.sign(result.residual_return) == np.sign(result.cross_mean)
    result["residual_market_alignment"] = np.where(
        aligned, "aligned", "opposed")
    result["direction_name"] = np.where(
        result.residual_return.ge(0), "up_extreme", "down_extreme")
    result["volatility_regime"] = pd.cut(
        result.scaled_cross_std, [-np.inf, *volatility_cuts, np.inf],
        labels=["low", "normal", "high"]).astype(str)
    return result


def learn_policy(frame: pd.DataFrame, fields: tuple[str, ...],
                 minimum_count: int) -> pd.DataFrame:
    stats = frame[frame.year.isin(DEVELOPMENT)].groupby(
        ["year", *fields, "style"], dropna=False).signed_return.agg(
            count="size", mean="mean").reset_index()
    eligible = stats[stats["count"].ge(minimum_count)]
    summary = eligible.groupby([*fields, "style"], dropna=False).agg(
        years=("year", "nunique"), positive_years=("mean", lambda x: int((x > 0).sum())),
        worst_year_mean=("mean", "min"), median_year_mean=("mean", "median"),
        observations=("count", "sum")).reset_index()
    summary = summary[(summary.years.eq(len(DEVELOPMENT)))
                      & summary.positive_years.ge(3)
                      & summary.median_year_mean.gt(0)].copy()
    if summary.empty:
        return summary
    summary["robust_score"] = (
        summary.median_year_mean + summary.worst_year_mean.clip(upper=0))
    summary.sort_values([*fields, "robust_score"], ascending=True,
                        inplace=True, kind="mergesort")
    return summary.drop_duplicates(list(fields), keep="last")


def route(frame: pd.DataFrame, fields: tuple[str, ...], policy: pd.DataFrame,
          name: str) -> pd.DataFrame:
    if policy.empty:
        return frame.iloc[:0].copy()
    chosen = frame.merge(
        policy[[*fields, "style", "robust_score"]],
        on=[*fields, "style"], how="inner", validate="many_to_one")
    chosen["module"] = name
    chosen["module_priority"] = 0
    return chosen


def score(stage: dict) -> tuple:
    value = stage["margin_3pct"]
    floor = min(value.get("monthly_return_pct", {}).values(), default=-999.)
    return (value["trades_per_market_day"] >= 2.5,
            value["projected_22_market_day_pct"], -value["negative_months"],
            floor, value["return_pct"], -value["dd_pct"])


def pass_gate(stage: dict) -> bool:
    value = stage["margin_3pct"]
    annual = list(stage["annual_3pct"].values())
    return bool(value["n"] >= 400 and value["return_pct"] > 0
                and value["dd_pct"] <= 12
                and value["trades_per_market_day"] >= 2.5
                and all(item["return_pct"] > 0 for item in annual))


def public(item: dict) -> dict:
    return {key: value for key, value in item.items()
            if not key.startswith("_")}


def main() -> None:
    candles, sources = base.parent.load_history()
    features = source.feature_rows(candles, RULE["lookback_bars"])
    resolved = resolved_styles(features, candles)
    development = resolved[resolved.year.isin(DEVELOPMENT)]
    volatility_cuts = tuple(float(x) for x in
        development.scaled_cross_std.quantile([1/3, 2/3]))
    candidates = []
    for fields, market_threshold, minimum_count in itertools.product(
            GROUPS, MARKET_THRESHOLDS, MIN_COUNTS):
        enriched = enrich(resolved, market_threshold, volatility_cuts)
        policy = learn_policy(enriched, fields, minimum_count)
        name = ("residual_direction:" + "|".join(fields)
                + f":m{market_threshold}:n{minimum_count}")
        routed = route(enriched, fields, policy, name)
        stage, accepted = base.exact_bundle(routed, candles, DEVELOPMENT)
        candidates.append({
            "fields": list(fields), "market_threshold_pct": 100*market_threshold,
            "minimum_count_each_year": minimum_count,
            "policy_groups": len(policy), "development": stage,
            "_policy": policy, "_frame": enriched,
            "_accepted": accepted, "_name": name,
        })
    candidates.sort(key=lambda item: score(item["development"]), reverse=True)
    selected = candidates[0] if candidates and pass_gate(
        candidates[0]["development"]) else None
    validation = test = confirmation = None
    status = "REJECTED_DEVELOPMENT"
    accepted = selected["_accepted"] if selected else []
    if selected:
        fields = tuple(selected["fields"])
        routed = route(selected["_frame"], fields, selected["_policy"],
                       selected["_name"])
        validation, accepted = base.exact_bundle(routed, candles, VALIDATION)
        status = "REJECTED_VALIDATION"
        if pass_gate(validation):
            test, accepted = base.exact_bundle(routed, candles, TEST)
            status = "REJECTED_TEST"
            if pass_gate(test):
                confirmation, accepted = base.exact_bundle(
                    routed, candles, CONFIRMATION)
                status = ("DIRECTION_REGIME_CANDIDATE"
                          if pass_gate(confirmation) else "REJECTED_CONFIRMATION")
    pd.DataFrame(accepted).to_csv(ROWS_OUT, index=False)
    objective = bool(
        selected and validation and test and confirmation
        and all(base.objective_ok(stage) for stage in (
            selected["development"], validation, test, confirmation)))
    report = {
        "status": status, "ready_for_live_orders": False,
        "orders_possible": False,
        "protocol": {
            "hypothesis": "switch residual extremes between reversal and continuation by causal market regime",
            "development": list(DEVELOPMENT), "validation": list(VALIDATION),
            "test": list(TEST), "confirmation": list(CONFIRMATION),
            "entry": "next 15-minute market open", "orders": "market only",
            "money": "client-selected 3% current-equity margin; fixed 10x",
            "rule": RULE,
        },
        "grid": {"features": len(features), "resolved_both_styles": len(resolved),
                 "candidates": len(candidates)},
        "volatility_cuts_frozen_on_development": list(volatility_cuts),
        "selected_before_validation": public(selected) if selected else None,
        "validation_2021_2022": validation, "test_2023_2024": test,
        "confirmation_2025_2026": confirmation,
        "all_stages_meet_monthly_objective": objective,
        "top_development": [public(item) for item in candidates[:10]],
        "sources": {**sources,
                    str(Path(source.__file__).resolve()): digest(Path(source.__file__).resolve()),
                    str(Path(__file__).resolve()): digest(Path(__file__).resolve())},
        "limitations": [
            "All historical eras have been observed by prior research; any survivor requires new shadow-forward evidence.",
            "Cash-session history is a proxy for stock X-Perp execution.",
            "Gross results exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": status,
                      "selected": report["selected_before_validation"],
                      "validation": validation, "test": test,
                      "confirmation": confirmation,
                      "objective": objective}, indent=2), flush=True)


if __name__ == "__main__":
    main()
