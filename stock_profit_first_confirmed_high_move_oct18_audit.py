"""Profit-first stock high-move audit confirmed across 2022 and 2023.

Models fit 2017-2021. Target, holding period and cutoff are selected using
both 2022 and 2023, then frozen before evaluation on 2024-2026. Win rate is a
diagnostic only; selection rewards the weaker of the two development years.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from filter_lab import gate
import stock_cash_literal_move_hgb_oct7_audit as base
import stock_profit_first_high_move_oct17_audit as high
import stock_profit_first_literal_runner_oct17_audit as parent
import stock_profit_first_multisession_oct17_audit as session_base
from stock_walk_forward_filter_lab import CATEGORICAL, NUMERIC, make_model


ROOT = Path("reports/audit_2026_09_08")
OUT = ROOT / "stock_profit_first_confirmed_high_move_oct18.json"
ROWS_OUT = ROOT / "stock_profit_first_confirmed_high_move_oct18.pkl"
TARGETS_PCT = (2.0, 2.5, 3.0, 4.0)
HOLD_SESSIONS = (1, 2, 3)
QUANTILES = (.60, .70, .75, .80, .85, .875, .90, .925, .95)
FIT_YEARS = parent.FIT_YEARS
DEVELOPMENT_YEARS = (2022, 2023)
TEST_YEARS = (2024, 2025, 2026)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def development_ok(years: dict) -> bool:
    return bool(all(value["n"] >= 100
                    and value["return_pct"] > 0
                    and value["dd_pct"] <= 15
                    and value["trades_per_calendar_day"] >= 1
                    for value in years.values()))


def development_score(years: dict) -> float:
    values = list(years.values())
    monthly = min(value["equivalent_monthly_return_pct"] for value in values)
    drawdown = max(value["dd_pct"] for value in values)
    negative = sum(value["negative_months"] for value in values)
    return round(monthly - .08 * drawdown - .10 * negative, 8)


def public(item: dict) -> dict:
    return {key: value for key, value in item.items()
            if key not in {"_frame", "_probability"}}


def selected_rows(frame: pd.DataFrame, probability: np.ndarray,
                  threshold: float) -> pd.DataFrame:
    mask = probability >= threshold
    chosen = frame.loc[mask].copy()
    chosen["quality"] = probability[mask]
    return pd.DataFrame(gate(chosen.sort_values(
        ["entry_time", "quality", "symbol"], ascending=[True, False, True])
        .to_dict("records"), 4))


def main() -> None:
    candles, sources = base.load()
    proposals = parent.build_target(candles, 2.0)
    metadata = session_base.regular_metadata(candles)
    features = list(NUMERIC + CATEGORICAL)
    candidates = []
    model_reports = []
    for target, sessions in itertools.product(TARGETS_PCT, HOLD_SESSIONS):
        frame = high.add_outcomes(proposals, candles, metadata, target, sessions)
        fit = frame.year.isin(FIT_YEARS)
        estimator = make_model()
        estimator.fit(frame.loc[fit, features], frame.loc[fit, "won"])
        probability = estimator.predict_proba(frame[features])[:, 1]
        selection_scores = probability[frame.year.eq(2022).to_numpy()]
        qualified = 0
        for quantile in QUANTILES:
            threshold = float(np.quantile(selection_scores, quantile))
            development = {
                str(year): high.evaluate(frame, probability, (year,), threshold)
                for year in DEVELOPMENT_YEARS
            }
            if not development_ok(development):
                continue
            candidates.append({
                "target_pct": target, "hold_sessions": sessions,
                "selection_quantile": quantile, "threshold": threshold,
                "development": development,
                "score": development_score(development),
                "_frame": frame, "_probability": probability,
            })
            qualified += 1
        model_reports.append({"target_pct": target, "hold_sessions": sessions,
                              "qualified": qualified})
        print(json.dumps(model_reports[-1]), flush=True)
    candidates.sort(key=lambda item: item["score"], reverse=True)
    frozen = candidates[0] if candidates else None
    tests = checks = None
    status = "REJECTED_DEVELOPMENT"
    if frozen:
        tests = {
            str(year): high.evaluate(frozen["_frame"], frozen["_probability"],
                                     (year,), frozen["threshold"])
            for year in TEST_YEARS
        }
        checks = {year: parent.objective_ok(value)
                  for year, value in {**frozen["development"], **tests}.items()}
        status = ("STOCK_CONFIRMED_HIGH_MOVE_RESEARCH_CANDIDATE"
                  if all(checks.values()) else "REJECTED_OBJECTIVE")
        selected_rows(frozen["_frame"], frozen["_probability"],
                      frozen["threshold"]).to_pickle(ROWS_OUT)
    report = {
        "status": status, "ready_for_live_orders": False,
        "protocol": {
            "priority": "account return first; win rate diagnostic only",
            "fit": list(FIT_YEARS), "development": list(DEVELOPMENT_YEARS),
            "frozen_test": list(TEST_YEARS),
            "targets_pct": list(TARGETS_PCT),
            "hold_sessions": list(HOLD_SESSIONS), "stop_pct": high.STOP_PCT,
            "entry": "next 15-minute market open", "orders": "market only",
            "money": "3% current-equity margin, fixed 10x gross",
            "role": "high-move module; combined bot must exceed three trades/day",
            "objective": ">=30% account return in every month; >=3 trades/day; DD <=10%",
        },
        "model_reports": model_reports, "candidate_count": len(candidates),
        "development_top20": [public(item) for item in candidates[:20]],
        "frozen_before_2024": public(frozen) if frozen else None,
        "test_2024_2026": tests, "objective_checks": checks,
        "sources": {**sources, str(Path(base.__file__)): digest(Path(base.__file__)),
                    str(Path(high.__file__)): digest(Path(high.__file__)),
                    str(Path(__file__)): digest(Path(__file__))},
        "limitations": [
            "Cash-session proxy omits overnight X-Perp price discovery.",
            "The module still needs direct X-Perp shadow transfer before activation.",
            "Gross returns exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": status,
                      "frozen": public(frozen) if frozen else None,
                      "test_2024_2026": tests,
                      "objective_checks": checks}, indent=2), flush=True)


if __name__ == "__main__":
    main()
