"""Add one causal reversal geometry to the frozen native stock route.

The base 2-bar route always remains first.  Candidate 1/4/8-bar reversal
events pass the same ATR, backing-level, basket-trend and portfolio rails.
July-August select a single complement; September and October only validate.
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
import stock_xperp_native_cross_section_oct18_audit as parent
import stock_xperp_native_growth_forward_monitor as monitor
import stock_xperp_native_quality_guard_oct18_audit as quality
import stock_xperp_native_volatility_guard_oct18_audit as volatility


ROOT = Path("reports/audit_2026_09_08")
OUT = ROOT / "stock_xperp_multi_horizon_complement_oct26.json"
ROWS_OUT = ROOT / "stock_xperp_multi_horizon_complement_oct26.csv"
LOOKBACKS = (1, 4, 8)
Z_THRESHOLDS = (1.5, 2.0, 2.5)
MOVE_THRESHOLDS = (.004, .006, .008)
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


def geometry(candles: dict, context: pd.DataFrame, features: pd.DataFrame,
             lookback: int,
             z_threshold: float, move_threshold: float) -> pd.DataFrame:
    entries = parent.select_entries(features, z_threshold, move_threshold)
    entries = volatility.add_atr14(entries, candles)
    entries = entries[entries.atr14_fraction.le(.01)]
    entries = backing.guarded(backing.add_backing_distance(
        entries, candles, 32))
    entries = entries.merge(context, on="entry_time", how="left",
                            validate="many_to_one")
    resolved = resolver.add_outcomes(entries, candles, 2.0, 1.0, 64)
    resolved["module"] = (
        f"native_xperp_complement_{lookback}bar_z{z_threshold:g}_m{move_threshold:g}")
    resolved["module_priority"] = 1
    return basket.apply(resolved, 96, "LONG", -0.005626823134606253)


def combine(base: pd.DataFrame, complement: pd.DataFrame,
            side: str) -> pd.DataFrame:
    extra = complement if side == "BOTH" else complement[
        complement.direction.eq(side)]
    base_keys = {(str(row["symbol"]), int(float(row["entry_time"])),
                  str(row["direction"])) for row in base.to_dict("records")}
    keep = [(str(row["symbol"]), int(float(row["entry_time"])),
             str(row["direction"])) not in base_keys
            for row in extra.to_dict("records")]
    extra = extra.loc[keep].copy()
    columns = sorted(set(base.columns) | set(extra.columns))
    joined = pd.concat([
        base.reindex(columns=columns), extra.reindex(columns=columns)],
        ignore_index=True)
    rows = basket.portfolio_gate(
        joined.to_dict("records"), None, rolling_hour_cap=4,
        rank_key=quality.ranker("z_high"), loss_streak_cap=2)
    return pd.DataFrame(rows, columns=columns)


def score(values: dict) -> float:
    july = values["selection_july"]["margin_3pct"]
    august = values["selection_august"]["margin_3pct"]
    return round(min(july["return_pct"], august["return_pct"])
                 - .5 * max(july["dd_pct"], august["dd_pct"]), 8)


def eligible(values: dict) -> bool:
    return all(
        values[name]["margin_3pct"]["return_pct"] > 0
        and values[name]["margin_3pct"]["trades_per_active_day"] >= 3
        and values[name]["margin_3pct"]["dd_pct"] <= 7
        for name in ("selection_july", "selection_august"))


def compact(item: dict) -> dict:
    return {key: item[key] for key in (
        "n", "profitable_exit_rate_pct", "return_pct", "dd_pct",
        "negative_months", "trades_per_active_day", "mean_signed_move_pct")}


def summaries(values: dict) -> dict:
    return {name: compact(value["margin_3pct"])
            for name, value in values.items()}


def main() -> None:
    raw_base, sources = basket.prepare()
    candles, candle_sources, _ = monitor.load_manifest(basket.MANIFEST)
    context = basket_context(candles)
    base = basket.apply(raw_base, 96, "LONG", -0.005626823134606253)
    baseline_rows = basket.gated(base)
    baseline_values = basket.evaluate(baseline_rows)
    baseline_score = score(baseline_values)
    candidates = []
    for lookback in LOOKBACKS:
        features = parent.feature_rows(candles, lookback)
        for z_threshold, move_threshold in itertools.product(
                Z_THRESHOLDS, MOVE_THRESHOLDS):
            extra = geometry(candles, context, features, lookback,
                             z_threshold, move_threshold)
            for side in SIDES:
                rows = combine(base, extra, side)
                values = basket.evaluate(rows)
                if eligible(values):
                    candidates.append({
                        "lookback_bars": lookback, "z_threshold": z_threshold,
                        "minimum_move_pct": 100*move_threshold,
                        "side": side, "selection_score": score(values),
                        "values": values, "_rows": rows,
                    })
    candidates.sort(key=lambda item: item["selection_score"], reverse=True)
    selected = candidates[0]
    values = selected["values"]
    september = values["validation_september"]["margin_3pct"]
    base_september = baseline_values["validation_september"]["margin_3pct"]
    fresh = values["observed_oct17_oct26"]["margin_3pct"]
    base_fresh = baseline_values["observed_oct17_oct26"]["margin_3pct"]
    robust = bool(
        selected["selection_score"] > baseline_score
        and september["return_pct"] >= base_september["return_pct"]
        and september["dd_pct"] <= base_september["dd_pct"] + .5
        and fresh["return_pct"] >= base_fresh["return_pct"]
        and fresh["trades_per_active_day"] >= 3)
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
            "base": "frozen 2-bar z>=2 move>=0.4% reversal route",
            "candidate": "one additional 1/4/8-bar reversal geometry",
            "selection": ["2026-07", "2026-08"],
            "validation": "2026-09",
            "already_observed_diagnostic": "2026-10-01..26",
            "entry": "next 15-minute market open", "orders": "market only",
            "target_pct": 2.0, "stop_pct": 1.0, "hold_bars": 64,
            "common_rails": "ATR<=1%, backing32, basket96 long guard, rolling-hour cap4, loss pause2",
            "retune_on_validation_or_october": False,
        },
        "grid": {"lookbacks": list(LOOKBACKS),
                 "z_thresholds": list(Z_THRESHOLDS),
                 "move_thresholds": [100*x for x in MOVE_THRESHOLDS],
                 "sides": list(SIDES),
                 "tested": len(LOOKBACKS)*len(Z_THRESHOLDS)*len(MOVE_THRESHOLDS)*len(SIDES),
                 "eligible": len(candidates)},
        "baseline": {"selection_score": baseline_score,
                     "periods": summaries(baseline_values)},
        "selected_before_september": {**public, "periods": summaries(values)},
        "validation_improves_baseline":
            september["return_pct"] >= base_september["return_pct"],
        "october_diagnostic_improves_baseline":
            fresh["return_pct"] >= base_fresh["return_pct"],
        "robust_improvement": robust,
        "top20": [{
            **{key: value for key, value in item.items()
               if not key.startswith("_") and key != "values"},
            "periods": summaries(item["values"]),
        } for item in candidates[:20]],
        "sources": {**sources, **candle_sources,
                    str(basket.MANIFEST): digest(basket.MANIFEST),
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
