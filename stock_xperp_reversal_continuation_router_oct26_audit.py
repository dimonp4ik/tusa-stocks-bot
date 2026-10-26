"""Chronological reversal-versus-continuation router for native stock X-Perps."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from stock_module_cluster_audit import portfolio_gate
import stock_cash_xperp_leadlag_oct6_audit as lead
import stock_xperp_native_economics_oct18_audit as economics
import stock_xperp_native_model_quality_oct19_audit as features
import stock_xperp_native_quality_guard_oct18_audit as parent


OUT = parent.parent.ROOT / "stock_xperp_reversal_continuation_router_oct26.json"
ROWS_OUT = parent.parent.ROOT / "stock_xperp_reversal_continuation_router_oct26.csv"
CONFIGS = tuple((leaves, leaf, l2) for leaves in (3, 7)
                for leaf in (40, 80) for l2 in (1.0, 5.0))
CONFIDENCE = (0.0, .025, .05, .075, .10, .15, .20)
LOSS_CAPS = (1, 2, 3)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired() -> tuple[pd.DataFrame, dict, dict]:
    reversal, parameters, sources = parent.prepare()
    candles, _ = parent.parent.load_sources()
    seed = reversal.copy()
    seed["direction_n"] = -seed.direction_n.astype(np.int8)
    continuation = lead.add_outcomes(
        seed, candles, float(parameters["target_pct"]),
        float(parameters["stop_pct"]), int(parameters["hold_bars"]))
    keys = ["symbol", "entry_index", "entry_time"]
    outcome = ["direction_n", "direction", "signed_return", "exit_time",
               "outcome_code", "outcome", "net_r"]
    extra = continuation[keys + outcome].rename(
        columns={name: f"continuation_{name}" for name in outcome})
    value = reversal.merge(extra, on=keys, how="inner", validate="one_to_one")
    value["continuation_better"] = (
        value.continuation_signed_return > value.signed_return).astype(int)
    return features.enrich(value), parameters, sources


def routed(frame: pd.DataFrame, probabilities: np.ndarray,
           minimum_confidence: float) -> pd.DataFrame:
    value = frame.copy()
    value["continuation_probability"] = probabilities
    value["router_confidence"] = np.abs(probabilities-.5)
    value = value[value.router_confidence.ge(minimum_confidence)].copy()
    choose = value.continuation_probability.ge(.5)
    for name in ("direction_n", "direction", "signed_return", "exit_time",
                 "outcome_code", "outcome", "net_r"):
        value[name] = np.where(
            choose, value[f"continuation_{name}"], value[name])
    value["module"] = "native_xperp_reversal_continuation_router"
    value["module_priority"] = 0
    return value


def accepted(frame: pd.DataFrame, loss_cap: int) -> list[dict]:
    return portfolio_gate(
        frame.to_dict("records"), None,
        rank_key=lambda row: -float(row["router_confidence"]),
        loss_streak_cap=loss_cap)


def evaluate(frame: pd.DataFrame, loss_cap: int) -> dict:
    rows = accepted(frame, loss_cap)
    result = economics.simulate(rows, .03)
    if not rows:
        result.update({"active_days": 0, "trades_per_day": 0.0,
                       "projected_22_session_pct": 0.0,
                       "continuation_share_pct": 0.0,
                       "literal_tp_rate_pct": 0.0})
        return result
    days = len({pd.Timestamp(float(row["entry_time"]), unit="s", tz="UTC").date()
                for row in rows})
    factor = max(0.0, 1+result["return_pct"]/100)
    result.update({
        "active_days": days,
        "trades_per_day": round(len(rows)/days, 4),
        "projected_22_session_pct": round(
            100*(factor**(22/days)-1) if factor > 0 else -100, 4),
        "continuation_share_pct": round(100*sum(
            float(row["continuation_probability"]) >= .5 for row in rows)/len(rows), 4),
        "literal_tp_rate_pct": round(100*sum(
            int(row["outcome_code"]) == 1 for row in rows)/len(rows), 4),
    })
    return result


def selection_ok(value: dict) -> bool:
    return bool(value["n"] >= 30 and value["dd_pct"] <= 8
                and value["trades_per_day"] >= 3
                and value["projected_22_session_pct"] >= 30)


def full_month_ok(value: dict) -> bool:
    return bool(value["n"] >= 66 and value["dd_pct"] <= 8
                and value["trades_per_day"] >= 3
                and value["return_pct"] >= 30
                and value["negative_months"] == 0)


def public(item: dict | None) -> dict | None:
    return ({key: value for key, value in item.items()
             if not key.startswith("_")} if item else None)


def main() -> None:
    frame, parameters, sources = paired()
    fit = features.bounds(frame, features.FIT)
    selection = features.bounds(frame, features.SELECTION)
    validation = features.bounds(frame, features.VALIDATION)
    confirmation = features.bounds(frame, features.CONFIRMATION)
    fresh = features.bounds(frame, features.FRESH)
    design = features.schema(fit)
    candidates = []
    for leaves, leaf, l2 in CONFIGS:
        model = HistGradientBoostingClassifier(
            learning_rate=.05, max_iter=150, max_leaf_nodes=leaves,
            min_samples_leaf=leaf, l2_regularization=l2,
            early_stopping=False, random_state=20260908)
        model.fit(features.matrix(fit, design), fit.continuation_better)
        probabilities = model.predict_proba(features.matrix(selection, design))[:, 1]
        for confidence in CONFIDENCE:
            routed_selection = routed(selection, probabilities, confidence)
            for loss_cap in LOSS_CAPS:
                value = evaluate(routed_selection, loss_cap)
                candidates.append({
                    "max_leaf_nodes": leaves, "min_samples_leaf": leaf,
                    "l2_regularization": l2,
                    "minimum_confidence": confidence,
                    "loss_streak_cap": loss_cap,
                    "selection": value, "selection_ok": selection_ok(value),
                    "_model": model,
                })
    stage_one = [item for item in candidates if item["selection_ok"]]
    for item in stage_one:
        model = item["_model"]
        probabilities = model.predict_proba(features.matrix(validation, design))[:, 1]
        value = evaluate(routed(
            validation, probabilities, float(item["minimum_confidence"])),
            int(item["loss_streak_cap"]))
        item["validation"] = value
        item["validation_pass"] = full_month_ok(value)
    stage_one.sort(key=lambda item: (
        item.get("validation_pass", False),
        min(item["selection"]["projected_22_session_pct"],
            item.get("validation", {}).get("return_pct", -100)),
        -max(item["selection"]["dd_pct"],
             item.get("validation", {}).get("dd_pct", 100))), reverse=True)
    selected = stage_one[0] if stage_one else None
    confirmation_value = fresh_value = None
    confirmation_pass = False
    status = "REJECTED_SELECTION"
    output_rows = []
    if selected:
        status = "REJECTED_VALIDATION"
        if selected.get("validation_pass"):
            model = selected["_model"]
            confidence = float(selected["minimum_confidence"])
            loss_cap = int(selected["loss_streak_cap"])
            probabilities = model.predict_proba(
                features.matrix(confirmation, design))[:, 1]
            routed_confirmation = routed(confirmation, probabilities, confidence)
            confirmation_value = evaluate(routed_confirmation, loss_cap)
            confirmation_pass = full_month_ok(confirmation_value)
            status = "REJECTED_CONFIRMATION"
            if confirmation_pass:
                probabilities = model.predict_proba(features.matrix(fresh, design))[:, 1]
                routed_fresh = routed(fresh, probabilities, confidence)
                fresh_value = evaluate(routed_fresh, loss_cap)
                pieces = []
                for part in (selection, validation, confirmation, fresh):
                    prob = model.predict_proba(features.matrix(part, design))[:, 1]
                    pieces.append(routed(part, prob, confidence))
                output_rows = accepted(pd.concat(pieces, ignore_index=True), loss_cap)
                status = "REVERSAL_CONTINUATION_SHADOW_CANDIDATE"
    if output_rows:
        pd.DataFrame(output_rows).to_csv(ROWS_OUT, index=False)
    report = {
        "status": status, "ready_for_live_orders": False,
        "protocol": {
            "fit": list(features.FIT), "selection": list(features.SELECTION),
            "validation": list(features.VALIDATION),
            "confirmation": list(features.CONFIRMATION),
            "conditional_fresh": list(features.FRESH),
            "decision": "choose reversal or continuation from entry-time features",
            "target_pct": parameters["target_pct"],
            "stop_pct": parameters["stop_pct"],
            "hold_bars": parameters["hold_bars"],
            "entry": "next 15-minute market open", "orders": "market only",
            "money": "3% current-equity margin, fixed 10x",
        },
        "paired_events": len(frame), "fit_continuation_better_pct": round(
            100*fit.continuation_better.mean(), 4),
        "candidate_count": len(candidates),
        "selection_survivors": len(stage_one),
        "validation_survivors": sum(bool(item.get("validation_pass"))
                                    for item in stage_one),
        "selected": public(selected),
        "confirmation": confirmation_value,
        "confirmation_pass": confirmation_pass,
        "fresh": fresh_value,
        "frontier": [public(item) for item in stage_one[:20]],
        "sources": {**sources,
                    str(Path(__file__).resolve()): digest(Path(__file__).resolve())},
        "limitations": [
            "Native stock X-Perp history begins in July 2026.",
            "The base event geometry was selected before this router.",
            "October remains sealed unless September confirms the objective.",
            "Gross returns exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": status, "paired_events": len(frame),
        "selection_survivors": report["selection_survivors"],
        "validation_survivors": report["validation_survivors"],
        "selected": public(selected), "confirmation": confirmation_value,
        "confirmation_pass": confirmation_pass, "fresh": fresh_value,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
