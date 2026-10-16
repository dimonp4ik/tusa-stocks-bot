"""Cross-era stock audit of intraday breadth acceleration and reversal.

Unlike the earlier static-breadth routes, every rule uses the change in market
breadth between two completed regular-session checkpoints.  Entries occur at
the next market bar with fixed literal 0.5-2% targets.  Development is
2017-2021, validation is 2022-2023, and later periods remain sealed unless a
single frozen route passes both stages.
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
from stock_module_cluster_audit import portfolio_gate


OUT = base.ROOT / "stock_breadth_transition_oct16.json"
ROWS_OUT = base.ROOT / "stock_breadth_transition_oct16.csv"
DEVELOPMENT = (2017, 2018, 2019, 2020, 2021)
VALIDATION = (2022, 2023)
TEST = (2024,)
CONFIRMATION = (2025, 2026)
EXITS = (
    {"target_pct": .5, "stop_pct": .5, "hold_sessions": 1},
    {"target_pct": 1.0, "stop_pct": 1.0, "hold_sessions": 2},
    {"target_pct": 2.0, "stop_pct": 1.0, "hold_sessions": 3},
)
SPECS = (
    {"name": "breadth_acceleration_loose", "family": "acceleration",
     "qqq": .002, "breadth": .625, "delta": .125,
     "stock": 0.0, "last": 0.0},
    {"name": "breadth_acceleration_strict", "family": "acceleration",
     "qqq": .005, "breadth": .75, "delta": .125,
     "stock": .002, "last": .001},
    {"name": "breadth_majority_reclaim", "family": "reclaim",
     "breadth": .625, "delta": .125, "last": .001},
    {"name": "breadth_market_reversal", "family": "reversal",
     "qqq": .004, "delta": .125, "last": .001},
    {"name": "breadth_laggard_catchup", "family": "catchup",
     "qqq": .004, "lag": .005, "breadth": .625,
     "delta": 0.0, "last": .001},
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def transitions(frame: pd.DataFrame) -> pd.DataFrame:
    order = {"early": 0, "middle": 1, "late": 2}
    out = frame.copy()
    out["slot_order"] = out.slot.map(order).astype(int)
    out = out.sort_values(["symbol", "session_date", "slot_order"])
    grouped = out.groupby(["symbol", "session_date"], sort=False)
    out["prior_breadth"] = grouped.breadth_intra.shift(1)
    out["prior_qqq_intraday"] = grouped.qqq_intraday.shift(1)
    out["breadth_delta"] = out.breadth_intra - out.prior_breadth
    out["qqq_delta"] = out.qqq_intraday - out.prior_qqq_intraday
    return out.dropna(subset=["prior_breadth", "prior_qqq_intraday"]).copy()


def directed(frame: pd.DataFrame, direction: np.ndarray) -> pd.DataFrame:
    out = base.directed(frame, direction)
    sign = out.direction_n.to_numpy(float)
    out["breadth_delta_dir"] = out.breadth_delta * sign
    out["qqq_last_hour_dir"] = out.qqq_last_hour * sign
    out["prior_breadth_dir"] = np.where(
        sign > 0, out.prior_breadth, 1 - out.prior_breadth)
    out["qqq_lead_dir"] = (
        out.qqq_intraday - out.intraday) * sign
    return out


def apply_spec(frame: pd.DataFrame, spec: dict) -> pd.DataFrame:
    family = spec["family"]
    if family == "acceleration":
        found = directed(frame, np.sign(frame.qqq_intraday.to_numpy(float)))
        mask = (
            found.direction_n.ne(0)
            & found.qqq_intraday_dir.ge(spec["qqq"])
            & found.breadth_dir.ge(spec["breadth"])
            & found.breadth_delta_dir.ge(spec["delta"])
            & found.intraday_dir.ge(spec["stock"])
            & found.last_hour_dir.ge(spec["last"]))
    elif family == "reclaim":
        direction = np.sign(frame.breadth_intra.to_numpy(float) - .5)
        found = directed(frame, direction)
        mask = (
            found.direction_n.ne(0)
            & found.prior_breadth_dir.lt(.5)
            & found.breadth_dir.ge(spec["breadth"])
            & found.breadth_delta_dir.ge(spec["delta"])
            & found.qqq_last_hour_dir.ge(spec["last"])
            & found.last_hour_dir.ge(spec["last"]))
    elif family == "reversal":
        direction = -np.sign(frame.qqq_intraday.to_numpy(float))
        found = directed(frame, direction)
        mask = (
            found.direction_n.ne(0)
            & found.qqq_intraday_dir.le(-spec["qqq"])
            & found.breadth_delta_dir.ge(spec["delta"])
            & found.qqq_last_hour_dir.ge(spec["last"])
            & found.last_hour_dir.ge(0))
    elif family == "catchup":
        found = directed(frame, np.sign(frame.qqq_intraday.to_numpy(float)))
        mask = (
            found.direction_n.ne(0)
            & found.qqq_intraday_dir.ge(spec["qqq"])
            & found.qqq_lead_dir.ge(spec["lag"])
            & found.breadth_dir.ge(spec["breadth"])
            & found.breadth_delta_dir.ge(spec["delta"])
            & found.last_hour_dir.ge(spec["last"]))
    else:
        raise ValueError(family)
    selected = found.loc[mask].copy()
    selected["entry_spec"] = spec["name"]
    selected["score"] = (
        selected.breadth_delta_dir.abs()
        + selected.qqq_last_hour_dir.clip(lower=0)
        + selected.last_hour_dir.clip(lower=0))
    return selected


def accepted_rows(frame: pd.DataFrame, years: tuple[int, ...]) -> list[dict]:
    selected = frame[frame.year.isin(years)]
    return portfolio_gate(
        selected.to_dict("records"), None,
        rank_key=lambda row: (-float(row["score"]),))


def evaluate(frame: pd.DataFrame, features: pd.DataFrame,
             years: tuple[int, ...]) -> dict:
    value = base.evaluate(frame, features, years)
    rows = accepted_rows(frame, years)
    returns = [float(row["signed_return"]) for row in rows]
    value["literal_target_hits"] = sum(
        int(row["outcome_code"]) == 1 for row in rows)
    value["literal_target_hit_rate"] = round(
        100 * value["literal_target_hits"] / len(rows), 4) if rows else 0.0
    value["economic_gate"] = economic_gate(
        returns, float(value["trades_per_market_day"]), market_days=22)
    return value


def development_ok(value: dict, annual: dict[str, dict]) -> bool:
    return bool(
        value["n"] >= 150
        and value["return_pct"] > 0
        and value["dd_pct"] <= 10
        and value["trades_per_market_day"] >= .3
        and sum(item["return_pct"] > 0 for item in annual.values()) >= 4)


def validation_ok(value: dict, annual: dict[str, dict]) -> bool:
    return bool(
        value["n"] >= 60
        and value["return_pct"] > 0
        and value["dd_pct"] <= 7
        and value["trades_per_market_day"] >= .2
        and all(item["return_pct"] > 0 for item in annual.values()))


def objective_ok(value: dict) -> bool:
    return bool(
        75 <= value["literal_target_hit_rate"] <= 95
        and value["trades_per_market_day"] >= 3
        and value["negative_months"] == 0
        and value["dd_pct"] <= 5
        and value["economic_gate"]["passes"])


def public(item: dict | None) -> dict | None:
    return ({key: value for key, value in item.items() if key != "_rows"}
            if item else None)


def main() -> None:
    candles, sources = base.load_history()
    raw_features, metadata = base.build_features(candles)
    features = transitions(raw_features)
    frontier = []
    validated = []
    for spec, exit_spec in itertools.product(SPECS, EXITS):
        entries = apply_spec(features, spec)
        resolved = base.add_outcomes(
            entries, candles, metadata,
            exit_spec["target_pct"], exit_spec["stop_pct"],
            exit_spec["hold_sessions"])
        development = evaluate(resolved, features, DEVELOPMENT)
        development_annual = {
            str(year): evaluate(resolved, features, (year,))
            for year in DEVELOPMENT}
        item = {
            "entry": spec,
            "exit": exit_spec,
            "development": development,
            "development_years": development_annual,
            "development_pass": development_ok(
                development, development_annual),
            "_rows": resolved,
        }
        frontier.append(item)
        if not item["development_pass"]:
            continue
        validation = evaluate(resolved, features, VALIDATION)
        validation_annual = {
            str(year): evaluate(resolved, features, (year,))
            for year in VALIDATION}
        item["validation"] = validation
        item["validation_years"] = validation_annual
        item["validation_pass"] = validation_ok(validation, validation_annual)
        if item["validation_pass"]:
            item["score"] = round(
                min(development["return_pct"], validation["return_pct"])
                - 2 * max(development["dd_pct"], validation["dd_pct"])
                - validation["negative_months"], 8)
            validated.append(item)

    frontier.sort(key=lambda item: (
        item["development_pass"], item["development"]["return_pct"],
        item["development"]["literal_target_hit_rate"]), reverse=True)
    validated.sort(key=lambda item: item["score"], reverse=True)
    frozen = validated[0] if validated else None
    test = confirmation = None
    checks = None
    status = "REJECTED_BEFORE_TEST"
    if frozen:
        rows = frozen["_rows"]
        test = evaluate(rows, features, TEST)
        status = "REJECTED_TEST"
        if objective_ok(test):
            confirmation = {
                str(year): evaluate(rows, features, (year,))
                for year in CONFIRMATION}
            checks = {
                "test_2024": True,
                **{str(year): objective_ok(confirmation[str(year)])
                   for year in CONFIRMATION},
            }
            status = ("BREADTH_TRANSITION_RESEARCH_CANDIDATE"
                      if all(checks.values()) else "REJECTED_OBJECTIVE")
            if status == "BREADTH_TRANSITION_RESEARCH_CANDIDATE":
                rows.to_csv(ROWS_OUT, index=False)

    report = {
        "status": status,
        "ready_for_live_orders": False,
        "protocol": {
            "mechanism": "change in causal cross-sectional breadth between completed session checkpoints",
            "development": list(DEVELOPMENT),
            "validation": list(VALIDATION),
            "conditional_test": list(TEST),
            "confirmation_only": list(CONFIRMATION),
            "entry": "next regular-session market bar",
            "entry_specs": len(SPECS),
            "exit_specs": list(EXITS),
            "money": "3% current-equity margin, fixed 10x gross",
            "market_orders_only": True,
        },
        "feature_rows": len(features),
        "validated_before_test": len(validated),
        "frozen_before_test": public(frozen),
        "test_2024": test,
        "confirmation_2025_2026": confirmation,
        "objective_checks": checks,
        "development_frontier": [public(item) for item in frontier[:15]],
        "sources": {
            **sources,
            str(Path(base.__file__)): digest(Path(base.__file__)),
            str(Path(__file__)): digest(Path(__file__)),
        },
        "limitations": [
            "Dukascopy cash bars establish cross-era structure but are not X-Perp fills.",
            "Breadth covers seven liquid technology stocks available throughout the history.",
            "A historical pass would still require direct venue transfer and fresh shadow outcomes.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": status,
        "feature_rows": len(features),
        "validated_before_test": len(validated),
        "frozen_before_test": public(frozen),
        "test_2024": test,
        "confirmation": confirmation,
        "top_development": [public(item) for item in frontier[:5]],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
