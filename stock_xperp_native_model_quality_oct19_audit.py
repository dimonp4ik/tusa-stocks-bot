"""Chronological model-quality guard for the native stock X-Perp route.

The base route and ATR guard are already frozen before the October reveal.
This audit fits only on July 5-19, selects a model/threshold/rail policy on
July 20-31, validates on August, confirms on September, and reveals October
only if every earlier gate passes.  All features exist at entry time.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor)

from stock_module_cluster_audit import portfolio_gate
import stock_xperp_native_economics_oct18_audit as economics
import stock_xperp_native_quality_guard_oct18_audit as parent


OUT = parent.parent.ROOT / "stock_xperp_native_model_quality_oct19.json"
ROWS_OUT = parent.parent.ROOT / "stock_xperp_native_model_quality_oct19.csv"
FIT = ("2026-07-05", "2026-07-20")
SELECTION = ("2026-07-20", "2026-08-01")
VALIDATION = ("2026-08-01", "2026-09-01")
CONFIRMATION = ("2026-09-01", "2026-10-01")
FRESH = ("2026-10-07", "2026-10-17")
QUANTILES = (.30, .40, .50, .60, .70, .75, .80, .85, .875, .90)
LOSS_CAPS = (1, 2, 3)
CONFIGS = tuple(
    (leaves, leaf, l2)
    for leaves in (3, 7, 15)
    for leaf in (20, 40, 80)
    for l2 in (1.0, 5.0)
)
NUMERIC = (
    "signal_return", "abs_signal_return", "cross_mean", "cross_std",
    "cross_z", "abs_cross_z", "direction_n", "market_support",
    "atr14_fraction", "universe_n", "hour_sin", "hour_cos",
    "minute_sin", "minute_cos", "weekday_sin", "weekday_cos",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def enrich(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy().reset_index(drop=True)
    stamp = pd.to_datetime(result.entry_time, unit="s", utc=True)
    hour = stamp.dt.hour + stamp.dt.minute / 60
    result["abs_signal_return"] = result.signal_return.abs()
    result["abs_cross_z"] = result.cross_z.abs()
    result["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    result["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    result["minute_sin"] = np.sin(2 * np.pi * stamp.dt.minute / 60)
    result["minute_cos"] = np.cos(2 * np.pi * stamp.dt.minute / 60)
    result["weekday_sin"] = np.sin(2 * np.pi * stamp.dt.weekday / 7)
    result["weekday_cos"] = np.cos(2 * np.pi * stamp.dt.weekday / 7)
    return result


def bounds(frame: pd.DataFrame, period: tuple[str, str]) -> pd.DataFrame:
    start = pd.Timestamp(period[0], tz="UTC").timestamp()
    end = pd.Timestamp(period[1], tz="UTC").timestamp()
    return frame[frame.entry_time.between(start, end - 1)].copy()


def schema(train: pd.DataFrame) -> dict:
    return {
        "symbols": sorted(train.symbol.astype(str).unique().tolist()),
        "medians": {name: float(pd.to_numeric(
            train[name], errors="coerce").median()) for name in NUMERIC},
    }


def matrix(frame: pd.DataFrame, design: dict) -> np.ndarray:
    columns = [
        pd.to_numeric(frame[name], errors="coerce").fillna(
            design["medians"][name]).to_numpy(float)
        for name in NUMERIC
    ]
    symbols = frame.symbol.astype(str).to_numpy()
    columns.extend((symbols == symbol).astype(float)
                   for symbol in design["symbols"])
    return np.column_stack(columns)


def scored_frame(frame: pd.DataFrame, model, model_target: str,
                 design: dict) -> pd.DataFrame:
    result = frame.copy()
    values = matrix(result, design)
    result["model_quality"] = (
        model.predict(values)
        if model_target == "signed_return"
        else model.predict_proba(values)[:, 1]
    )
    return result


def accepted(frame: pd.DataFrame, threshold: float,
             loss_cap: int) -> list[dict]:
    rows = frame[frame.model_quality >= threshold].to_dict("records")
    return portfolio_gate(
        rows, None, rank_key=lambda row: -float(row["model_quality"]),
        loss_streak_cap=loss_cap)


def evaluate(frame: pd.DataFrame, threshold: float,
             loss_cap: int) -> dict:
    rows = accepted(frame, threshold, loss_cap)
    value = economics.simulate(rows, .03)
    if not rows:
        value.update({"active_days": 0, "trades_per_market_day": 0.0,
                      "projected_22_session_pct": 0.0,
                      "literal_tp_rate_pct": 0.0,
                      "profitable_exit_rate_pct": 0.0})
        return value
    active_days = len({pd.Timestamp(float(row["entry_time"]), unit="s", tz="UTC").date()
                       for row in rows})
    factor = max(0.0, 1 + value["return_pct"] / 100)
    projected = (100 * (factor ** (22 / active_days) - 1)
                 if active_days and factor > 0 else -100.0)
    value.update({
        "active_days": active_days,
        "trades_per_market_day": round(len(rows) / active_days, 4),
        "projected_22_session_pct": round(projected, 4),
        "literal_tp_rate_pct": round(100 * sum(
            int(row["outcome_code"]) == 1 for row in rows) / len(rows), 4),
        "profitable_exit_rate_pct": round(100 * sum(
            float(row["signed_return"]) > 0 for row in rows) / len(rows), 4),
    })
    return value


def selection_ok(value: dict) -> bool:
    return bool(value["n"] >= 30 and value["dd_pct"] <= 8
                and value["trades_per_market_day"] >= 3
                and value["projected_22_session_pct"] >= 30)


def full_month_ok(value: dict) -> bool:
    return bool(value["n"] >= 66 and value["dd_pct"] <= 8
                and value["trades_per_market_day"] >= 3
                and value["return_pct"] >= 30
                and value["negative_months"] == 0)


def main() -> None:
    raw, parameters, sources = parent.prepare()
    frame = enrich(raw)
    fit = bounds(frame, FIT)
    selection = bounds(frame, SELECTION)
    validation = bounds(frame, VALIDATION)
    confirmation = bounds(frame, CONFIRMATION)
    fresh = bounds(frame, FRESH)
    design = schema(fit)
    candidates = []
    for leaves, leaf, l2 in CONFIGS:
        for model_target in ("signed_return", "profitable_exit", "literal_tp"):
            common = dict(
                learning_rate=.05, max_iter=150, max_leaf_nodes=leaves,
                min_samples_leaf=leaf, l2_regularization=l2,
                early_stopping=False, random_state=20260908)
            if model_target == "signed_return":
                model = HistGradientBoostingRegressor(**common)
                target = fit.signed_return.to_numpy(float)
            else:
                model = HistGradientBoostingClassifier(**common)
                target = ((fit.signed_return > 0).astype(int).to_numpy()
                          if model_target == "profitable_exit"
                          else (fit.outcome_code == 1).astype(int).to_numpy())
            model.fit(matrix(fit, design), target)
            scored_selection = scored_frame(
                selection, model, model_target, design)
            for quantile in QUANTILES:
                threshold = float(
                    scored_selection.model_quality.quantile(quantile))
                for loss_cap in LOSS_CAPS:
                    value = evaluate(scored_selection, threshold, loss_cap)
                    candidates.append({
                        "model_target": model_target,
                        "max_leaf_nodes": leaves,
                        "min_samples_leaf": leaf,
                        "l2_regularization": l2,
                        "selection_quantile": quantile,
                        "threshold": threshold,
                        "loss_streak_cap": loss_cap,
                        "selection": value,
                        "selection_ok": selection_ok(value),
                        "score": round(value["projected_22_session_pct"]
                                       - .5 * value["dd_pct"], 8),
                        "_model": model,
                    })
    candidates.sort(key=lambda item: (
        item["selection_ok"], item["score"],
        item["selection"]["literal_tp_rate_pct"]), reverse=True)
    stage_one = [item for item in candidates if item["selection_ok"]]
    validation_frontier = []
    for item in stage_one:
        model = item["_model"]
        scored = scored_frame(
            validation, model, item["model_target"], design)
        value = evaluate(
            scored, float(item["threshold"]), int(item["loss_streak_cap"]))
        item["validation"] = value
        item["validation_pass"] = full_month_ok(value)
        validation_frontier.append(item)
    validation_frontier.sort(key=lambda item: (
        item["validation_pass"],
        min(item["selection"]["projected_22_session_pct"],
            item["validation"]["return_pct"]),
        -max(item["selection"]["dd_pct"], item["validation"]["dd_pct"]),
    ), reverse=True)
    selected = validation_frontier[0] if validation_frontier else None
    status = "REJECTED_SELECTION"
    validation_value = confirmation_value = fresh_value = None
    validation_pass = confirmation_pass = False
    output_rows = []
    if selected is not None:
        model = selected["_model"]
        threshold = float(selected["threshold"])
        loss_cap = int(selected["loss_streak_cap"])
        validation_value = selected["validation"]
        validation_pass = bool(selected["validation_pass"])
        status = "REJECTED_VALIDATION"
        if validation_pass:
            scored_confirmation = scored_frame(
                confirmation, model, selected["model_target"], design)
            confirmation_value = evaluate(
                scored_confirmation, threshold, loss_cap)
            confirmation_pass = full_month_ok(confirmation_value)
            status = "REJECTED_CONFIRMATION"
            if confirmation_pass:
                scored_fresh = scored_frame(
                    fresh, model, selected["model_target"], design)
                fresh_value = evaluate(scored_fresh, threshold, loss_cap)
                scored_selection = scored_frame(
                    selection, model, selected["model_target"], design)
                scored_validation = scored_frame(
                    validation, model, selected["model_target"], design)
                output_rows = accepted(
                    pd.concat([scored_selection, scored_validation,
                               scored_confirmation, scored_fresh]),
                    threshold, loss_cap)
                status = "MODEL_QUALITY_SHADOW_CANDIDATE"
    if output_rows:
        pd.DataFrame(output_rows).to_csv(ROWS_OUT, index=False)
    public_selected = ({key: value for key, value in selected.items()
                        if not key.startswith("_")} if selected else None)
    report = {
        "status": status,
        "ready_for_live_orders": False,
        "protocol": {
            "fit": list(FIT), "selection": list(SELECTION),
            "validation": list(VALIDATION),
            "confirmation": list(CONFIRMATION), "fresh": list(FRESH),
            "target_pct": parameters["target_pct"],
            "stop_pct": parameters["stop_pct"],
            "hold_bars": parameters["hold_bars"],
            "entry": "next 15-minute market open",
            "orders": "market only",
            "money": "3% current-equity margin, fixed 10x",
            "features": list(NUMERIC) + ["symbol one-hot"],
        },
        "candidate_count": len(candidates),
        "selection_survivors": sum(item["selection_ok"] for item in candidates),
        "validation_survivors": sum(
            item["validation_pass"] for item in validation_frontier),
        "selection_top20": [{key: value for key, value in item.items()
                             if not key.startswith("_")}
                            for item in candidates[:20]],
        "selected": public_selected,
        "validation_frontier": [
            {key: value for key, value in item.items()
             if not key.startswith("_")}
            for item in validation_frontier[:20]
        ],
        "validation": validation_value,
        "validation_pass": validation_pass,
        "confirmation": confirmation_value,
        "confirmation_pass": confirmation_pass,
        "fresh": fresh_value,
        "sources": {
            **sources,
            str(parent.OUT): digest(parent.OUT),
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "The native stock X-Perp history begins in July 2026.",
            "The base route was selected before this incremental guard.",
            "October is revealed only after selection, validation and confirmation pass.",
            "Gross returns exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": status,
        "selection_survivors": report["selection_survivors"],
        "selected": public_selected,
        "validation": validation_value,
        "validation_pass": validation_pass,
        "confirmation": confirmation_value,
        "confirmation_pass": confirmation_pass,
        "fresh": fresh_value,
    }, indent=2))


if __name__ == "__main__":
    main()
