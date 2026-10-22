"""Annual causal adaptation audit for the liquidity-sweep stock family."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

import stock_liquidity_sweep_walkforward_oct22_audit as sweep


OUT = sweep.ROOT / "stock_liquidity_sweep_rolling_oct22.json"
DEV_YEARS = (2022, 2023)
ACCEPTANCE_YEAR = 2024
HIDDEN_YEARS = (2025, 2026)
QUANTILES = (.30, .40, .50, .60, .70, .80, .85, .90)


def fold(history, features, test_year: int, spec: dict, keep_model: bool = False) -> dict:
    validation_year = test_year-1
    train_years = tuple(range(max(2017, test_year-6), validation_year))
    train = history[history.year.isin(train_years)]
    validation = history[history.year.eq(validation_year)]
    model_cfg = spec["model"]
    model = HistGradientBoostingClassifier(
        max_iter=160, learning_rate=.06, max_leaf_nodes=model_cfg["leaves"],
        min_samples_leaf=model_cfg["leaf"], l2_regularization=model_cfg["l2"],
        random_state=20261022)
    model.fit(sweep.matrix(train), (train.outcome_code == 1).astype(int))
    validation = validation.copy()
    validation["quality"] = model.predict_proba(sweep.matrix(validation))[:, 1]
    choices = []
    for quantile in QUANTILES:
        threshold = float(validation.quality.quantile(quantile))
        value = sweep.evaluate(validation, features, (validation_year,), threshold)
        eligible = (value["n"] >= 100 and value["return_pct"] > 0
                    and value["dd_pct"] <= 12
                    and value["trades_per_market_day"] >= 1.0)
        score = (value["return_pct"] - 1.5*value["dd_pct"]
                 - .5*value["negative_months"]
                 + .02*value["literal_target_hit_rate"])
        choices.append({"quantile": quantile, "threshold": threshold,
                        "eligible": eligible, "score": score,
                        "validation": value})
    eligible = [item for item in choices if item["eligible"]]
    chosen = max(eligible, key=lambda item: item["score"], default=None)
    if chosen is None:
        chosen = max(choices, key=lambda item: item["score"])
    test = history[history.year.eq(test_year)].copy()
    test["quality"] = model.predict_proba(sweep.matrix(test))[:, 1]
    result = sweep.evaluate(test, features, (test_year,), chosen["threshold"])
    output = {
        "year": test_year, "train_years": list(train_years),
        "validation_year": validation_year,
        "threshold_selection_pass": bool(chosen["eligible"]),
        "chosen_quantile": chosen["quantile"],
        "threshold": chosen["threshold"],
        "validation": chosen["validation"], "test": result,
        "threshold_frontier": sorted(
            choices, key=lambda item: item["score"], reverse=True)[:4],
    }
    if keep_model:
        output["_model"] = model
        output["_test"] = test
    return output


def main() -> None:
    historical, _ = sweep.base.load_history()
    direct, direct_sources = sweep.load_direct_15m()
    features, meta = sweep.sweep_features(historical)
    direct_features, direct_meta = sweep.sweep_features(direct)
    exit_frames = {}
    for target in sweep.TARGETS:
        for stop in sweep.STOPS:
            for hold in sweep.HOLDS:
                exit_frames[(target, stop, hold)] = sweep.outcomes(
                    features, historical, meta, target, stop, hold)

    candidates = []
    for target in sweep.TARGETS:
        for stop in sweep.STOPS:
            for hold in sweep.HOLDS:
                history = exit_frames[(target, stop, hold)]
                for model in sweep.MODELS:
                    spec = {"target_pct": target, "stop_pct": stop,
                            "hold_sessions": hold, "model": model}
                    development = [fold(history, features, year, spec)
                                   for year in DEV_YEARS]
                    tests = [item["test"] for item in development]
                    if (all(item["threshold_selection_pass"] for item in development)
                            and all(value["n"] >= 100 and value["return_pct"] > 0
                                    and value["dd_pct"] <= 12 for value in tests)):
                        score = (sum(value["return_pct"] for value in tests)
                                 + 2*min(value["return_pct"] for value in tests)
                                 - sum(value["dd_pct"] for value in tests))
                        candidates.append({"spec": spec, "score": score,
                                           "development": development})
    candidates.sort(key=lambda item: item["score"], reverse=True)
    accepted = []
    for candidate in candidates:
        spec = candidate["spec"]
        history = exit_frames[(spec["target_pct"], spec["stop_pct"],
                               spec["hold_sessions"])]
        candidate["acceptance_2024"] = fold(
            history, features, ACCEPTANCE_YEAR, spec)
        value = candidate["acceptance_2024"]["test"]
        if (candidate["acceptance_2024"]["threshold_selection_pass"]
                and value["n"] >= 100 and value["return_pct"] > 0
                and value["dd_pct"] <= 8):
            accepted.append(candidate)
    # Development score fixes the order; 2024 is an acceptance gate only.
    frozen = accepted[0] if accepted else None
    acceptance = hidden = direct_result = None
    checks = None
    if frozen:
        spec = frozen["spec"]
        history = exit_frames[(spec["target_pct"], spec["stop_pct"], spec["hold_sessions"])]
        acceptance = frozen["acceptance_2024"]
        hidden = [fold(history, features, year, spec, keep_model=(year == 2026))
                  for year in HIDDEN_YEARS]
        final = hidden[-1]
        direct_exit = sweep.outcomes(
            direct_features, direct, direct_meta, spec["target_pct"],
            spec["stop_pct"], spec["hold_sessions"])
        direct_exit["quality"] = final["_model"].predict_proba(
            sweep.matrix(direct_exit))[:, 1]
        direct_result = sweep.evaluate(
            direct_exit, direct_features, (2026,), final["threshold"])
        final.pop("_model", None)
        final.pop("_test", None)
        checks = {
            "acceptance_2024_positive": acceptance["test"]["return_pct"] > 0,
            "test_2025_positive": hidden[0]["test"]["return_pct"] > 0,
            "observed_2026_positive": hidden[1]["test"]["return_pct"] > 0,
            "direct_xperp_positive": direct_result["return_pct"] > 0,
            "each_late_fold_dd_le_8": all(
                item["test"]["dd_pct"] <= 8 for item in [acceptance, *hidden]),
            "direct_n_ge_30": direct_result["n"] >= 30,
        }

    def public_candidate(item):
        return item if item is None else {
            "spec": item["spec"], "score": item["score"],
            "development": item["development"]}

    status = "SHADOW_CANDIDATE" if checks and all(checks.values()) else "REJECTED"
    report = {
        "status": status,
        "protocol": {
            "configuration_selection_folds": list(DEV_YEARS),
            "acceptance_year": ACCEPTANCE_YEAR,
            "hidden_years": list(HIDDEN_YEARS),
            "annual_adaptation": "fit up to five years ending at year-2; choose quality quantile on year-1",
            "entry": "next 15-minute market open after causal swing sweep",
            "orders": "market only",
            "money": "3% current-equity margin, fixed 10x gross",
        },
        "tested_configurations": len(sweep.TARGETS)*len(sweep.STOPS)*len(sweep.HOLDS)*len(sweep.MODELS),
        "development_qualified": len(candidates),
        "acceptance_qualified": len(accepted),
        "frozen_before_acceptance": public_candidate(frozen),
        "acceptance_2024": acceptance,
        "hidden_2025_2026": hidden,
        "direct_xperp": direct_result,
        "checks": checks,
        "top_development": [public_candidate(item) for item in candidates[:10]],
        "sources": direct_sources,
        "limitations": [
            "Cash candles are structural transfer evidence for stock perpetuals.",
            "Annual adaptation cannot react before the prior-year regime change is observed.",
            "The bounded configuration and threshold grid creates selection risk.",
            "Historical results cannot guarantee future profit or eliminate losing months.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": status, "qualified": len(candidates),
        "frozen": public_candidate(frozen), "acceptance": acceptance,
        "hidden": hidden, "direct": direct_result, "checks": checks,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
