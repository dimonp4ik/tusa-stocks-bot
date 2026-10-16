"""Profit-first stock classifier for literal one-to-two-percent moves.

Each model learns whether a proposal reaches its literal target before a fixed
one-percent stop or the cash-session close.  Models fit 2017-2021; target and
probability cutoff are chosen on 2022, frozen for 2023 validation and 2024
test, then checked on 2025-2026.  Win rate is diagnostic rather than a gate.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from filter_lab import gate
import stock_cash_literal_move_hgb_oct7_audit as base
from stock_walk_forward_filter_lab import CATEGORICAL, NUMERIC, make_model


ROOT = Path("reports/audit_2026_09_08")
OUT = ROOT / "stock_profit_first_literal_runner_oct17.json"
ROWS_OUT = ROOT / "stock_profit_first_literal_runner_oct17.pkl"
TARGETS_PCT = (1.0, 1.5, 2.0)
STOP_PCT = 1.0
QUANTILES = (.70, .75, .80, .85, .875, .90, .925, .95, .96, .97, .98)
FIT_YEARS = (2017, 2018, 2019, 2020, 2021)
SELECTION_YEAR = 2022
VALIDATION_YEAR = 2023
TEST_YEAR = 2024
CONFIRMATION_YEARS = (2025, 2026)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_target(candles: dict, target_pct: float) -> pd.DataFrame:
    old_target, old_stop = base.TARGET_PCT, base.STOP_PCT
    try:
        base.TARGET_PCT = target_pct
        base.STOP_PCT = STOP_PCT
        return base.build(candles)
    finally:
        base.TARGET_PCT, base.STOP_PCT = old_target, old_stop


def enrich(value: dict) -> dict:
    result = dict(value)
    months = result.get("months", 0)
    total = result.get("return_pct", 0.0)
    result["equivalent_monthly_return_pct"] = round(
        100 * ((1 + total / 100) ** (1 / months) - 1), 6
        if months and total > -100 else -100.0)
    result["profit_first_objective_pass"] = objective_ok(result)
    return result


def evaluate(frame: pd.DataFrame, probability: np.ndarray,
             years: tuple[int, ...], threshold: float) -> dict:
    return enrich(base.score(frame, probability, years, threshold))


def selection_ok(value: dict) -> bool:
    return bool(value["return_pct"] > 0 and value["dd_pct"] <= 15
                and value["trades_per_calendar_day"] >= 3)


def profile_score(value: dict) -> float:
    return round(value["equivalent_monthly_return_pct"]
                 - .10 * value["dd_pct"]
                 - .20 * value["negative_months"], 8)


def objective_ok(value: dict) -> bool:
    return bool(value.get("months", 0) > 0 and value["dd_pct"] <= 10
                and value["trades_per_calendar_day"] >= 3
                and value["months_ge_30pct"] == value["months"])


def public(item: dict) -> dict:
    return {key: value for key, value in item.items()
            if key not in {"_frame", "_probability"}}


def selected_rows(frame: pd.DataFrame, probability: np.ndarray,
                  threshold: float) -> pd.DataFrame:
    chosen = frame[probability >= threshold].copy()
    chosen["quality"] = probability[probability >= threshold]
    rows = gate(chosen.sort_values(
        ["entry_time", "quality", "symbol"], ascending=[True, False, True])
        .to_dict("records"), 4)
    return pd.DataFrame(rows)


def main() -> None:
    candles, sources = base.load()
    features = list(NUMERIC + CATEGORICAL)
    candidates = []
    for target in TARGETS_PCT:
        frame = build_target(candles, target)
        fit = frame.year.isin(FIT_YEARS)
        estimator = make_model()
        estimator.fit(frame.loc[fit, features], frame.loc[fit, "won"])
        probability = estimator.predict_proba(frame[features])[:, 1]
        selection_scores = probability[frame.year.eq(SELECTION_YEAR).to_numpy()]
        for quantile in QUANTILES:
            threshold = float(np.quantile(selection_scores, quantile))
            value = evaluate(frame, probability, (SELECTION_YEAR,), threshold)
            candidates.append({
                "target_pct": target,
                "selection_quantile": quantile,
                "threshold": threshold,
                "selection_2022": value,
                "selection_pass": selection_ok(value),
                "score": profile_score(value),
                "_frame": frame,
                "_probability": probability,
            })
    candidates.sort(key=lambda item: (
        item["selection_pass"], item["score"]), reverse=True)
    frozen = candidates[0] if candidates and candidates[0]["selection_pass"] else None
    validation = test = confirmation = checks = None
    status = "REJECTED_SELECTION"
    if frozen:
        frame = frozen["_frame"]
        probability = frozen["_probability"]
        threshold = frozen["threshold"]
        validation = evaluate(frame, probability, (VALIDATION_YEAR,), threshold)
        test = evaluate(frame, probability, (TEST_YEAR,), threshold)
        confirmation = {str(year): evaluate(frame, probability, (year,), threshold)
                        for year in CONFIRMATION_YEARS}
        yearly = {str(SELECTION_YEAR): frozen["selection_2022"],
                  str(VALIDATION_YEAR): validation, str(TEST_YEAR): test,
                  **confirmation}
        checks = {year: objective_ok(value) for year, value in yearly.items()}
        status = ("STOCK_LITERAL_RUNNER_RESEARCH_CANDIDATE"
                  if all(checks.values()) else "REJECTED_OBJECTIVE")
        selected_rows(frame, probability, threshold).to_pickle(ROWS_OUT)
    report = {
        "status": status,
        "ready_for_live_orders": False,
        "protocol": {
            "priority": "account return first; win rate diagnostic only",
            "venue": "Dukascopy US cash-session CFD proxy",
            "fit": list(FIT_YEARS), "profile_selection": SELECTION_YEAR,
            "frozen_validation": VALIDATION_YEAR,
            "untouched_test": TEST_YEAR,
            "confirmation_only": list(CONFIRMATION_YEARS),
            "targets_pct": list(TARGETS_PCT), "stop_pct": STOP_PCT,
            "hold": "same cash session", "entry": "next 15-minute market open",
            "orders": "market only",
            "money": "3% current-equity margin, fixed 10x gross",
            "objective": ">=30% account return in every month; >=3 trades/calendar day; DD <=10%",
        },
        "candidate_count": len(candidates),
        "top_selection": [public(item) for item in candidates[:15]],
        "frozen_before_2023": public(frozen) if frozen else None,
        "validation_2023": validation, "test_2024": test,
        "confirmation_2025_2026": confirmation,
        "objective_checks": checks,
        "sources": {**sources, str(Path(base.__file__)): digest(Path(base.__file__)),
                    str(Path(__file__)): digest(Path(__file__))},
        "limitations": [
            "Dukascopy cash history is structural evidence rather than X-Perp execution history.",
            "The same-session hold cannot capture an overnight two-percent move.",
            "Any historical survivor would still require direct X-Perp shadow transfer.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": status, "candidate_count": len(candidates),
        "frozen": public(frozen) if frozen else None,
        "validation_2023": validation, "test_2024": test,
        "confirmation_2025_2026": confirmation,
        "objective_checks": checks,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
