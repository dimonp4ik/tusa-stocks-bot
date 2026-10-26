"""Explicit basket-regime router between reversal and continuation entries.

The router uses only prior closed basket returns.  July and August select one
simple switching rule, September validates it, and October remains an already
observed diagnostic.  The frozen long-side basket guard and portfolio rails
are applied after the route chooses the final direction, matching runtime
order if the candidate is ever promoted.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import stock_cash_xperp_leadlag_oct6_audit as resolver
import stock_xperp_basket_trend_long_guard_oct26_audit as basket
import stock_xperp_native_growth_forward_monitor as monitor


ROOT = Path("reports/audit_2026_09_08")
OUT = ROOT / "stock_xperp_basket_direction_router_oct26.json"
ROWS_OUT = ROOT / "stock_xperp_basket_direction_router_oct26.csv"
HORIZONS = (16, 32, 64, 96)
QUANTILES = (.1, .2, .3, .4, .5, .6, .7, .8, .9)
RELATIONS = ("ge", "le")
SIGNAL_SIDES = ("ALL", "UP_MOVE", "DOWN_MOVE")
OUTCOME_COLUMNS = (
    "direction_n", "direction", "signed_return", "exit_time",
    "outcome_code", "outcome", "net_r", "risk_pct",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired() -> tuple[pd.DataFrame, dict[str, str]]:
    reversal, sources = basket.prepare()
    candles, candle_sources, _ = monitor.load_manifest(basket.MANIFEST)
    seed = reversal.drop(columns=[
        name for name in ("signed_return", "exit_time", "outcome_code",
                          "target_pct", "stop_pct", "hold_bars", "direction",
                          "outcome", "risk_pct", "net_r", "module",
                          "module_priority") if name in reversal.columns
    ]).copy()
    seed["direction_n"] = -seed.direction_n.astype(np.int8)
    continuation = resolver.add_outcomes(seed, candles, 2.0, 1.0, 64)
    keys = ["symbol", "entry_index", "entry_time"]
    extra = continuation[keys + list(OUTCOME_COLUMNS)].rename(
        columns={name: f"continuation_{name}" for name in OUTCOME_COLUMNS})
    frame = reversal.merge(extra, on=keys, how="inner", validate="one_to_one")
    if len(frame) != len(reversal):
        raise ValueError("continuation replay lost reversal events")
    return frame, {**sources, **candle_sources}


def routed(frame: pd.DataFrame, horizon: int, relation: str,
           signal_side: str, threshold: float) -> pd.DataFrame:
    value = frame.copy()
    signal_sign = -value.direction_n.astype(float)
    alignment = signal_sign * value[f"basket_trend_{horizon}"]
    relation_match = alignment.ge(threshold) if relation == "ge" else alignment.le(threshold)
    side_match = pd.Series(True, index=value.index)
    if signal_side == "UP_MOVE":
        side_match = signal_sign.gt(0)
    elif signal_side == "DOWN_MOVE":
        side_match = signal_sign.lt(0)
    choose = relation_match & side_match
    value["direction_router_alignment"] = alignment
    value["direction_router_continuation"] = choose.astype(np.int8)
    for name in OUTCOME_COLUMNS:
        value[name] = np.where(
            choose, value[f"continuation_{name}"], value[name])
    value["module"] = "native_xperp_basket_direction_router"
    value["module_priority"] = 0
    return value


def frozen_guard_and_gate(frame: pd.DataFrame) -> pd.DataFrame:
    guarded = basket.apply(
        frame, 96, "LONG", -0.005626823134606253)
    return basket.gated(guarded)


def score(values: dict) -> float:
    july = values["selection_july"]["margin_3pct"]
    august = values["selection_august"]["margin_3pct"]
    return round(min(july["return_pct"], august["return_pct"])
                 - .5 * max(july["dd_pct"], august["dd_pct"]), 8)


def eligible(values: dict) -> bool:
    return all(
        values[name]["margin_3pct"]["return_pct"] > 0
        and values[name]["margin_3pct"]["trades_per_active_day"] >= 3
        and values[name]["margin_3pct"]["dd_pct"] <= 6
        for name in ("selection_july", "selection_august"))


def compact(item: dict) -> dict:
    return {key: item[key] for key in (
        "n", "profitable_exit_rate_pct", "return_pct", "dd_pct",
        "negative_months", "trades_per_active_day", "mean_signed_move_pct")}


def period_summary(values: dict) -> dict:
    return {name: compact(item["margin_3pct"])
            for name, item in values.items()}


def main() -> None:
    frame, sources = paired()
    selection = pd.concat([
        basket.bounds(frame, basket.PERIODS["selection_july"]),
        basket.bounds(frame, basket.PERIODS["selection_august"]),
    ], ignore_index=True)
    baseline_rows = frozen_guard_and_gate(frame)
    baseline_values = basket.evaluate(baseline_rows)
    baseline_score = score(baseline_values)
    candidates = []
    for horizon, relation, signal_side in itertools.product(
            HORIZONS, RELATIONS, SIGNAL_SIDES):
        signal_sign = -selection.direction_n.astype(float)
        alignment = signal_sign * selection[f"basket_trend_{horizon}"]
        for quantile in QUANTILES:
            threshold = float(alignment.quantile(quantile))
            rows = frozen_guard_and_gate(routed(
                frame, horizon, relation, signal_side, threshold))
            values = basket.evaluate(rows)
            if eligible(values):
                candidates.append({
                    "horizon_bars": horizon, "relation": relation,
                    "signal_side": signal_side,
                    "selection_quantile": quantile,
                    "minimum_or_maximum_alignment": threshold,
                    "selection_score": score(values),
                    "continuation_share_pct": round(
                        100 * float(rows.direction_router_continuation.mean()), 4)
                        if len(rows) else 0.0,
                    "values": values, "_rows": rows,
                })
    candidates.sort(key=lambda item: item["selection_score"], reverse=True)
    selected = candidates[0]
    selected_values = selected["values"]
    validation = selected_values["validation_september"]["margin_3pct"]
    baseline_validation = baseline_values[
        "validation_september"]["margin_3pct"]
    october = selected_values["observed_oct17_oct26"]["margin_3pct"]
    baseline_october = baseline_values[
        "observed_oct17_oct26"]["margin_3pct"]
    robust = bool(
        selected["selection_score"] > baseline_score
        and validation["return_pct"] >= baseline_validation["return_pct"]
        and validation["dd_pct"] <= baseline_validation["dd_pct"] + .5
        and october["return_pct"] >= baseline_october["return_pct"]
        and october["trades_per_active_day"] >= 3)
    if robust:
        selected["_rows"].to_csv(ROWS_OUT, index=False)
    public = {key: value for key, value in selected.items()
              if not key.startswith("_") and key != "values"}
    report = {
        "status": ("SHADOW_CANDIDATE_REQUIRES_GENUINELY_NEW_FORWARD"
                   if robust else "REJECTED_NO_ROBUST_IMPROVEMENT"),
        "ready_for_live_orders": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "mechanism": "switch selected cross-sectional extremes from reversal to continuation using prior closed basket trend",
            "selection": ["2026-07", "2026-08"],
            "validation": "2026-09",
            "already_observed_diagnostic": "2026-10-01..26",
            "entry": "next 15-minute market open", "orders": "market only",
            "target_pct": 2.0, "stop_pct": 1.0, "hold_bars": 64,
            "retune_on_validation_or_october": False,
        },
        "grid": {"horizons": list(HORIZONS), "quantiles": list(QUANTILES),
                 "relations": list(RELATIONS), "signal_sides": list(SIGNAL_SIDES),
                 "tested": len(HORIZONS)*len(QUANTILES)*len(RELATIONS)*len(SIGNAL_SIDES),
                 "eligible": len(candidates)},
        "baseline": {"selection_score": baseline_score,
                     "periods": period_summary(baseline_values)},
        "selected_before_september": {
            **public, "periods": period_summary(selected_values)},
        "validation_improves_baseline":
            validation["return_pct"] >= baseline_validation["return_pct"],
        "october_diagnostic_improves_baseline":
            october["return_pct"] >= baseline_october["return_pct"],
        "robust_improvement": robust,
        "top20": [{
            **{key: value for key, value in item.items()
               if not key.startswith("_") and key != "values"},
            "periods": period_summary(item["values"]),
        } for item in candidates[:20]],
        "sources": {**sources, str(basket.MANIFEST): digest(basket.MANIFEST),
                    str(Path(__file__).resolve()): digest(Path(__file__).resolve())},
        "limitations": [
            "Native X-Perp history begins in 2026.",
            "October was already observed and cannot activate this candidate.",
            "Gross results exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "baseline": report["baseline"],
        "selected": report["selected_before_september"],
        "validation_improves_baseline": report["validation_improves_baseline"],
        "october_improves_baseline": report["october_diagnostic_improves_baseline"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
