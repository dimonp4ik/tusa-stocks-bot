"""Select material stock moves by predicted signed return, not hit rate.

The outcome model fits 2017-2021.  Target, holding period, and score cutoff are
selected jointly on 2022-2023, then frozen before 2024-2026 are evaluated.
This intentionally optimizes account return; win rate remains diagnostic only.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from filter_lab import gate
import stock_cash_literal_move_hgb_oct7_audit as base
import stock_profit_first_high_move_oct17_audit as high
import stock_profit_first_literal_runner_oct17_audit as parent
import stock_profit_first_multisession_oct17_audit as session_base
from stock_walk_forward_filter_lab import CATEGORICAL, NUMERIC


ROOT = Path("reports/audit_2026_09_08")
OUT = ROOT / "stock_profit_first_expected_return_oct18.json"
ROWS_OUT = ROOT / "stock_profit_first_expected_return_oct18.pkl"
TARGETS_PCT = (2.0, 2.5, 3.0, 4.0)
HOLD_SESSIONS = (1, 2, 3)
QUANTILES = (.55, .60, .65, .70, .75, .80, .85, .875, .90, .925, .95)
FIT_YEARS = parent.FIT_YEARS
DEVELOPMENT_YEARS = (2022, 2023)
TEST_YEARS = (2024, 2025, 2026)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_regressor() -> Pipeline:
    transform = ColumnTransformer([
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), list(NUMERIC)),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), list(CATEGORICAL)),
    ])
    estimator = HistGradientBoostingRegressor(
        loss="squared_error", learning_rate=.04, max_iter=180,
        max_leaf_nodes=15, min_samples_leaf=150, l2_regularization=4.0,
        early_stopping=True, validation_fraction=.15, random_state=20261018,
    )
    return Pipeline([("transform", transform), ("regressor", estimator)])


def evaluate(frame: pd.DataFrame, prediction: np.ndarray,
             year: int, threshold: float) -> dict:
    return parent.enrich(base.score(frame, prediction, (year,), threshold))


def development_ok(years: dict) -> bool:
    """Keep evidence and drawdown gates, without imposing a hit-rate gate."""
    return bool(all(value["n"] >= 200
                    and value["return_pct"] > 0
                    and value["dd_pct"] <= 18
                    and value["trades_per_calendar_day"] >= .6
                    for value in years.values()))


def development_score(years: dict) -> float:
    values = list(years.values())
    weakest_monthly = min(value["equivalent_monthly_return_pct"]
                          for value in values)
    drawdown = max(value["dd_pct"] for value in values)
    negative = sum(value["negative_months"] for value in values)
    activity = min(value["trades_per_calendar_day"] for value in values)
    return round(weakest_monthly - .08 * drawdown - .10 * negative
                 + .05 * min(activity, 3), 8)


def selected_rows(frame: pd.DataFrame, prediction: np.ndarray,
                  threshold: float) -> pd.DataFrame:
    mask = prediction >= threshold
    chosen = frame.loc[mask].copy()
    chosen["quality"] = prediction[mask]
    rows = gate(chosen.sort_values(
        ["entry_time", "quality", "symbol"], ascending=[True, False, True])
        .to_dict("records"), 4)
    return pd.DataFrame(rows)


def public(item: dict) -> dict:
    return {key: value for key, value in item.items()
            if key not in {"_frame", "_prediction"}}


def main() -> None:
    candles, sources = base.load()
    proposals = parent.build_target(candles, 2.0)
    metadata = session_base.regular_metadata(candles)
    features = list(NUMERIC + CATEGORICAL)
    candidates = []
    profiles = []
    for target, sessions in itertools.product(TARGETS_PCT, HOLD_SESSIONS):
        frame = high.add_outcomes(proposals, candles, metadata, target, sessions)
        fit = frame.year.isin(FIT_YEARS)
        estimator = make_regressor()
        estimator.fit(frame.loc[fit, features], frame.loc[fit, "signed_return"])
        prediction = estimator.predict(frame[features])
        selection = prediction[frame.year.eq(DEVELOPMENT_YEARS[0]).to_numpy()]
        qualified = 0
        for quantile in QUANTILES:
            threshold = float(np.quantile(selection, quantile))
            development = {
                str(year): evaluate(frame, prediction, year, threshold)
                for year in DEVELOPMENT_YEARS
            }
            if not development_ok(development):
                continue
            candidates.append({
                "target_pct": target, "hold_sessions": sessions,
                "selection_quantile": quantile, "threshold": threshold,
                "development": development,
                "score": development_score(development),
                "_frame": frame, "_prediction": prediction,
            })
            qualified += 1
        profiles.append({"target_pct": target, "hold_sessions": sessions,
                         "qualified": qualified})
        print(json.dumps(profiles[-1]), flush=True)

    candidates.sort(key=lambda item: item["score"], reverse=True)
    frozen = candidates[0] if candidates else None
    tests = checks = None
    status = "REJECTED_DEVELOPMENT"
    if frozen:
        tests = {
            str(year): evaluate(frozen["_frame"], frozen["_prediction"],
                                year, frozen["threshold"])
            for year in TEST_YEARS
        }
        all_years = {**frozen["development"], **tests}
        checks = {year: parent.objective_ok(value)
                  for year, value in all_years.items()}
        status = ("STOCK_EXPECTED_RETURN_RESEARCH_CANDIDATE"
                  if all(checks.values()) else "REJECTED_OBJECTIVE")
        selected_rows(frozen["_frame"], frozen["_prediction"],
                      frozen["threshold"]).to_pickle(ROWS_OUT)

    report = {
        "status": status, "ready_for_live_orders": False,
        "protocol": {
            "model_target": "signed trade return; hit rate is diagnostic only",
            "fit": list(FIT_YEARS), "development": list(DEVELOPMENT_YEARS),
            "frozen_test": list(TEST_YEARS),
            "targets_pct": list(TARGETS_PCT),
            "hold_sessions": list(HOLD_SESSIONS), "stop_pct": high.STOP_PCT,
            "entry": "next 15-minute market open", "orders": "market only",
            "money": "3% current-equity margin, fixed 10x gross",
            "objective": ">=30% account return in every month; >=3 trades/day; DD <=10%",
        },
        "profiles": profiles, "candidate_count": len(candidates),
        "development_top20": [public(item) for item in candidates[:20]],
        "frozen_before_2024": public(frozen) if frozen else None,
        "test_2024_2026": tests, "objective_checks": checks,
        "sources": {**sources, str(Path(base.__file__)): digest(Path(base.__file__)),
                    str(Path(high.__file__)): digest(Path(high.__file__)),
                    str(Path(__file__)): digest(Path(__file__))},
        "limitations": [
            "Cash-session data omit overnight X-Perp price discovery.",
            "The direct-return objective is noisier than target classification.",
            "Any survivor still requires direct X-Perp shadow transfer.",
            "Gross returns exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": status, "frozen": public(frozen) if frozen else None,
                      "tests": tests, "checks": checks}, indent=2), flush=True)


if __name__ == "__main__":
    main()
