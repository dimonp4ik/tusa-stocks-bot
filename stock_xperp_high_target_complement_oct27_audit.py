"""Audit a high-target multi-horizon complement to the frozen stock route.

The frozen two-bar 2% reversal always has priority.  Distinct one-, four- and
eight-bar extremes may fill otherwise unused portfolio capacity with a 2%-4%
target.  July and August choose one fixed rule; September validates it and the
already-observed October window is opened only after the rule is frozen.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import stock_cash_xperp_leadlag_oct6_audit as resolver
import stock_xperp_backing_level_guard_oct20_audit as backing
import stock_xperp_basket_trend_long_guard_oct26_audit as basket
import stock_xperp_current_barbell_oct27_audit as current
import stock_xperp_native_cross_section_oct18_audit as parent
import stock_xperp_native_growth_forward_monitor as monitor
import stock_xperp_native_volatility_guard_oct18_audit as volatility


ROOT = Path("reports/audit_2026_09_08")
OUT = ROOT / "stock_xperp_high_target_complement_oct27.json"
ROWS_OUT = ROOT / "stock_xperp_high_target_complement_oct27.csv"
LOOKBACKS = (1, 4, 8)
Z_THRESHOLDS = (1.0, 1.5, 2.0, 2.5)
MOVE_THRESHOLDS = (.002, .004, .006, .008)
TARGETS = (2.0, 3.0, 4.0)
STOPS = (.75, 1.0)
HOLDS = (32, 64, 96)
SIDES = ("BOTH", "LONG", "SHORT")
KEYS = ("symbol", "entry_time", "direction")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def basket_context(candles: dict) -> pd.DataFrame:
    features = parent.feature_rows(candles, 2)
    context = features.groupby("entry_time", as_index=False).cross_mean.first()
    context.sort_values("entry_time", inplace=True)
    context.rename(columns={"cross_mean": "basket_bar_return"}, inplace=True)
    for horizon in basket.HORIZONS:
        context[f"basket_trend_{horizon}"] = context.basket_bar_return.rolling(
            horizon, min_periods=horizon).sum()
    return context


def broad_entries(candles: dict, context: pd.DataFrame,
                  lookback: int) -> pd.DataFrame:
    features = parent.feature_rows(candles, lookback)
    entries = parent.select_entries(
        features, min(Z_THRESHOLDS), min(MOVE_THRESHOLDS))
    entries = volatility.add_atr14(entries, candles)
    entries = entries[entries.atr14_fraction.le(.01)]
    entries = backing.guarded(backing.add_backing_distance(
        entries, candles, 32))
    return entries.merge(context, on="entry_time", how="left",
                         validate="many_to_one")


def resolved_geometry(entries: pd.DataFrame, candles: dict, lookback: int,
                      target: float, stop: float, hold: int) -> pd.DataFrame:
    result = resolver.add_outcomes(entries, candles, target, stop, hold)
    result["module"] = (
        f"native_xperp_high_target_{lookback}bar_{target:g}pct")
    result["module_priority"] = 1
    result["near_activated"] = False
    return basket.apply(result, 96, "LONG", -0.005626823134606253)


def filter_rule(frame: pd.DataFrame, z: float, move: float,
                side: str) -> pd.DataFrame:
    found = frame[
        frame.cross_z.abs().ge(z)
        & frame.signal_return.abs().ge(move)
    ].copy()
    return found if side == "BOTH" else found[found.direction.eq(side)].copy()


def combine(base: pd.DataFrame, supplement: pd.DataFrame) -> pd.DataFrame:
    base_keys = set(zip(base.symbol.astype(str),
                        pd.to_numeric(base.entry_time).astype("int64"),
                        base.direction.astype(str)))
    keep = [
        (str(row.symbol), int(row.entry_time), str(row.direction)) not in base_keys
        for row in supplement.itertuples()
    ]
    extra = supplement.loc[keep].copy()
    columns = sorted(set(base.columns) | set(extra.columns))
    joined = pd.concat([
        base.reindex(columns=columns), extra.reindex(columns=columns)],
        ignore_index=True)
    return joined


def compact(value: dict) -> dict:
    return {key: value[key] for key in (
        "n", "profitable_exit_rate_pct", "literal_runner_rate_pct",
        "return_pct", "dd_pct", "negative_months",
        "trades_per_active_day", "mean_signed_move_pct")}


def summaries(values: dict) -> dict:
    return {name: compact(item["margin_3pct"])
            for name, item in values.items()}


def selection_values(frame: pd.DataFrame) -> dict:
    return {
        month: current.metrics(
            current.bounds(frame, current.PERIODS[month]), .03)
        for month in ("selection_july", "selection_august")
    }


def baseline_selection(values: dict) -> dict:
    return {month: values[month]["margin_3pct"]
            for month in ("selection_july", "selection_august")}


def selection_ok(values: dict, baseline: dict) -> bool:
    for month in ("selection_july", "selection_august"):
        value = values[month]
        base = baseline[month]
        if not (value["return_pct"] > base["return_pct"]
                and value["dd_pct"] <= base["dd_pct"] + .75
                and value["trades_per_active_day"] >= 3
                and value["negative_months"] == 0):
            return False
    return True


def selection_score(values: dict, baseline: dict) -> float:
    gains = []
    excess_dd = []
    for month in ("selection_july", "selection_august"):
        value = values[month]
        base = baseline[month]
        gains.append(value["return_pct"] - base["return_pct"])
        excess_dd.append(max(0.0, value["dd_pct"] - base["dd_pct"]))
    return round(min(gains) - .25 * max(excess_dd), 8)


def holdout_pass(values: dict, baseline: dict, month: str) -> bool:
    value = values[month]["margin_3pct"]
    base = baseline[month]["margin_3pct"]
    return bool(value["return_pct"] >= base["return_pct"]
                and value["dd_pct"] <= base["dd_pct"] + .5
                and value["negative_months"] <= base["negative_months"]
                and value["trades_per_active_day"] >= 3)


def public(item: dict) -> dict:
    return {key: value for key, value in item.items()
            if not key.startswith("_") and key != "values"}


def main() -> None:
    base_entries, candles, sources = current.prepare_entries()
    base = pd.DataFrame(monitor.resolve_entries(base_entries, candles, {
        "target_pct": 2.0, "stop_pct": 1.0, "hold_bars": 64}))
    base["module"] = "native_xperp_frozen"
    base["module_priority"] = 0
    base["near_activated"] = False
    baseline_rows = current.gated(base)
    baseline = current.evaluate(baseline_rows)
    selection_baseline = baseline_selection(baseline)
    context = basket_context(candles)

    candidates = []
    geometry_count = 0
    for lookback in LOOKBACKS:
        entries = broad_entries(candles, context, lookback)
        for target, stop, hold in itertools.product(TARGETS, STOPS, HOLDS):
            geometry_count += 1
            resolved = resolved_geometry(
                entries, candles, lookback, target, stop, hold)
            for z, move, side in itertools.product(
                    Z_THRESHOLDS, MOVE_THRESHOLDS, SIDES):
                supplement = filter_rule(resolved, z, move, side)
                combined = combine(base, supplement)
                rows = current.gated(current.bounds(
                    combined, ("2026-07-01", "2026-09-01")))
                values = selection_values(rows)
                if not selection_ok(values, selection_baseline):
                    continue
                candidates.append({
                    "lookback_bars": lookback, "z_threshold": z,
                    "minimum_move_pct": 100 * move,
                    "target_pct": target, "stop_pct": stop,
                    "hold_bars": hold, "side": side,
                    "selection_score": selection_score(
                        values, selection_baseline),
                    "supplement_rows_before_portfolio_gate": len(supplement),
                    "_selection": values, "_combined": combined,
                })
        print(json.dumps({"phase": "lookback", "lookback": lookback,
                          "selection_survivors": len(candidates)}), flush=True)

    candidates.sort(key=lambda item: (
        item["selection_score"],
        min(item["_selection"][month]["return_pct"]
            for month in ("selection_july", "selection_august")),
        -max(item["_selection"][month]["dd_pct"]
             for month in ("selection_july", "selection_august"))),
        reverse=True)
    selected = candidates[0] if candidates else None
    september_pass = october_pass = False
    if selected:
        selected["_rows"] = current.gated(selected["_combined"])
        selected["values"] = current.evaluate(selected["_rows"])
        september_pass = holdout_pass(
            selected["values"], baseline, "validation_september")
        october_pass = holdout_pass(
            selected["values"], baseline, "observed_october")
        selected["_rows"].to_csv(ROWS_OUT, index=False)
    accepted = bool(selected and september_pass and october_pass)

    report = {
        "status": ("SHADOW_CANDIDATE_REQUIRES_GENUINELY_NEW_FORWARD"
                   if accepted else "REJECTED_NO_ROBUST_IMPROVEMENT"),
        "ready_for_live_orders": False, "orders_possible": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "base": "frozen 2-bar 2% reversal route always first",
            "supplement": "one 1/4/8-bar reversal geometry with 2%-4% target",
            "selection": ["2026-07", "2026-08"],
            "validation": "2026-09",
            "already_observed_diagnostic": "2026-10 through snapshot end",
            "entry": "next 15-minute market open", "orders": "market only",
            "common_filters": "ATR<=1%, backing32 and frozen 96-bar basket long guard",
            "portfolio": "base priority, rolling-hour cap4, loss pause2, z-high rank",
            "money": "client-selected current-equity margin; audited at 3%, fixed 10x",
        },
        "grid": {
            "lookbacks": list(LOOKBACKS), "z_thresholds": list(Z_THRESHOLDS),
            "move_thresholds_pct": [100*x for x in MOVE_THRESHOLDS],
            "targets_pct": list(TARGETS), "stops_pct": list(STOPS),
            "hold_bars": list(HOLDS), "sides": list(SIDES),
            "resolved_geometries": geometry_count,
            "candidate_rules": (len(LOOKBACKS) * len(Z_THRESHOLDS)
                                * len(MOVE_THRESHOLDS) * len(TARGETS)
                                * len(STOPS) * len(HOLDS) * len(SIDES)),
            "selection_survivors": len(candidates),
        },
        "baseline": summaries(baseline),
        "selected_before_holdouts": (
            {**public(selected), "periods": summaries(selected["values"])}
            if selected else None),
        "checks": {"validation_september": september_pass,
                   "observed_october": october_pass,
                   "accepted": accepted},
        "top20_selection": [
            {**public(item), "selection_periods": {
                month: compact(item["_selection"][month])
                for month in ("selection_july", "selection_august")}}
            for item in candidates[:20]],
        "sources": {**sources, str(current.MANIFEST): digest(current.MANIFEST),
                    str(Path(__file__).resolve()): digest(Path(__file__).resolve())},
        "limitations": [
            "Native X-Perp history begins in July 2026.",
            "October was already observed and cannot activate a candidate.",
            "The grid creates multiple-testing risk, so both holdouts must dominate the frozen route.",
            "Public OHLC cannot reproduce live fill latency.",
            "Gross returns exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True),
                   encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "grid": report["grid"],
        "baseline": report["baseline"],
        "selected": report["selected_before_holdouts"],
        "checks": report["checks"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
