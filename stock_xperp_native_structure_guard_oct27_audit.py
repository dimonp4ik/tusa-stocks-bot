"""Causal structure and stop-path audit for the frozen native X-Perp route.

All candidate fields use candles closed before the next-bar market entry.
July is development, August validation, September test, and October is opened
only for a rule that dominates the frozen route in all three earlier months.
Post-stop paths are diagnostic and never participate in rule selection.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.stock_xperp_native_shadow import AUDITED_SYMBOLS
from stock_module_cluster_audit import portfolio_gate
import stock_stop_cause_oct11_audit as structure
import stock_xperp_expanded_universe_oct27_audit as expanded
import stock_xperp_forward_loss_attribution_oct22_audit as path_audit
import stock_xperp_mature_supplement_oct27_audit as mature
import stock_xperp_native_cross_section_oct18_audit as parent
import stock_xperp_native_quality_guard_oct18_audit as quality


ROOT = Path("reports/audit_2026_09_08")
OUT = ROOT / "stock_xperp_native_structure_guard_oct27.json"
ROWS_OUT = ROOT / "stock_xperp_native_structure_guard_oct27.csv"
BASE_ROWS_OUT = ROOT / "stock_xperp_native_structure_baseline_oct27.csv"
MONTHS = ("2026-07", "2026-08", "2026-09", "2026-10")
RULES = (
    "require_supportive_sweep",
    "reject_breakout_hold",
    "reject_opposing_order_block",
    "reject_target_path_swing",
    "require_stop_zone_swing",
    "require_exhaustion_wick_25pct",
    "require_volume_wick_exhaustion",
    "reject_structure_conflict",
)
FIELDS = (
    "supportive_sweep", "breakout_hold", "opposing_order_block",
    "target_path_has_prior_swing", "stop_zone_has_prior_swing",
    "liquidity_closer_to_stop", "exhaustion_wick_25pct",
    "volume_wick_exhaustion", "structure_conflict",
)


def causal_features(row: dict, candle: dict) -> dict:
    """Attach only information available when the signal candle closes."""
    index = int(row["entry_index"])
    if index < 5 or int(candle["time"][index]) != int(row["entry_time"]):
        raise ValueError(f"entry alignment failed: {row['symbol']} {row['entry_time']}")
    entry = float(row["entry"])
    direction = 1 if str(row["direction"]) == "LONG" else -1
    risk = entry * float(row["stop_pct"]) / 100
    target_r = float(row["target_pct"]) / float(row["stop_pct"])
    target = entry + direction * risk * target_r
    highs, lows = structure.swing_levels(candle, index, 96)
    target_level = structure.nearest(
        highs if direction > 0 else lows, entry, direction > 0)
    stop_level = structure.nearest(
        lows if direction > 0 else highs, entry, direction < 0)
    target_level_r = (direction * (target_level-entry) / risk
                      if target_level is not None else None)
    stop_level_r = (-direction * (stop_level-entry) / risk
                    if stop_level is not None else None)

    signal = index - 1
    prior = slice(max(0, signal - 20), signal)
    prior_high = max(float(value) for value in candle["high"][prior])
    prior_low = min(float(value) for value in candle["low"][prior])
    opening = float(candle["open"][signal])
    high = float(candle["high"][signal])
    low = float(candle["low"][signal])
    close = float(candle["close"][signal])
    width = max(high-low, entry * 1e-12)
    supportive_sweep = (low < prior_low and close > prior_low
                        if direction > 0
                        else high > prior_high and close < prior_high)
    breakout_hold = (close < prior_low if direction > 0 else close > prior_high)
    rejection_wick = ((min(opening, close)-low) / width if direction > 0
                      else (high-max(opening, close)) / width)
    volumes = [float(value) for value in candle["volume"][prior]]
    median_volume = float(pd.Series(volumes).median()) if volumes else 0.0
    volume_ratio = (float(candle["volume"][signal]) / median_volume
                    if median_volume > 0 else math.nan)
    opposing = structure.opposing_order_block(
        candle, index, entry, target, direction)
    target_conflict = bool(target_level_r is not None
                           and target_level_r <= target_r)
    stop_support = bool(stop_level_r is not None and stop_level_r <= 1.0)
    liquidity_closer = bool(
        stop_level_r is not None and target_level_r is not None
        and stop_level_r < target_level_r)
    wick = rejection_wick >= .25
    volume_wick = wick and volume_ratio >= 1.5
    conflict = breakout_hold or opposing or target_conflict
    return {
        **row,
        "target_side_swing_r": target_level_r,
        "stop_side_swing_r": stop_level_r,
        "target_path_has_prior_swing": target_conflict,
        "stop_zone_has_prior_swing": stop_support,
        "liquidity_closer_to_stop": liquidity_closer,
        "opposing_order_block": bool(opposing),
        "supportive_sweep": bool(supportive_sweep),
        "breakout_hold": bool(breakout_hold),
        "rejection_wick_fraction": rejection_wick,
        "volume_ratio_20": volume_ratio,
        "exhaustion_wick_25pct": wick,
        "volume_wick_exhaustion": volume_wick,
        "structure_conflict": bool(conflict),
    }


def apply_rule(frame: pd.DataFrame, rule: str) -> pd.DataFrame:
    if rule == "require_supportive_sweep":
        mask = frame.supportive_sweep
    elif rule == "reject_breakout_hold":
        mask = ~frame.breakout_hold
    elif rule == "reject_opposing_order_block":
        mask = ~frame.opposing_order_block
    elif rule == "reject_target_path_swing":
        mask = ~frame.target_path_has_prior_swing
    elif rule == "require_stop_zone_swing":
        mask = frame.stop_zone_has_prior_swing
    elif rule == "require_exhaustion_wick_25pct":
        mask = frame.exhaustion_wick_25pct
    elif rule == "require_volume_wick_exhaustion":
        mask = frame.volume_wick_exhaustion
    elif rule == "reject_structure_conflict":
        mask = ~frame.structure_conflict
    else:
        raise KeyError(rule)
    return frame[mask].copy()


def accept(frame: pd.DataFrame) -> pd.DataFrame:
    rows = portfolio_gate(
        frame.to_dict("records"), None, rolling_hour_cap=4,
        rank_key=quality.ranker("z_high"), loss_streak_cap=2)
    return pd.DataFrame(rows, columns=frame.columns)


def month_rows(frame: pd.DataFrame, month: str) -> pd.DataFrame:
    values = pd.to_datetime(frame.entry_time, unit="s", utc=True).dt.strftime("%Y-%m")
    return frame[values.eq(month)]


def month_metrics(frame: pd.DataFrame) -> dict:
    return {month: mature.forward.metrics(month_rows(frame, month), .03)
            for month in MONTHS}


def dominates(value: dict, baseline: dict) -> bool:
    return bool(
        value["resolved"] >= .60 * baseline["resolved"]
        and value["return_pct"] >= baseline["return_pct"]
        and value["dd_pct"] <= baseline["dd_pct"]
        and value["max_loss_streak"] <= baseline["max_loss_streak"]
        and value["trades_per_active_day"] >= 3)


def factor_stats(frame: pd.DataFrame) -> dict:
    output = {}
    for month in MONTHS:
        part = month_rows(frame, month)
        base_stop_rate = float(part.signed_return.lt(0).mean()) if len(part) else 0
        output[month] = {}
        for field in FIELDS:
            exposed = part[part[field].astype(bool)]
            clear = part[~part[field].astype(bool)]
            rate = float(exposed.signed_return.lt(0).mean()) if len(exposed) else 0
            output[month][field] = {
                "exposed_n": len(exposed),
                "exposed_stop_rate_pct": round(100*rate, 4),
                "clear_stop_rate_pct": round(
                    100*float(clear.signed_return.lt(0).mean()), 4) if len(clear) else 0,
                "stop_rate_lift": round(rate/base_stop_rate, 4)
                                  if base_stop_rate else None,
            }
    return output


def stop_paths(frame: pd.DataFrame, candles: dict) -> dict:
    by_month: dict[str, list[dict]] = defaultdict(list)
    for row in frame[frame.outcome.eq("SL")].to_dict("records"):
        item = path_audit.attribute_stop(row, candles[str(row["symbol"])])
        month = pd.Timestamp(float(row["entry_time"]), unit="s", tz="UTC").strftime("%Y-%m")
        if month in MONTHS:
            by_month[month].append(item)
    result = {}
    for month, rows in sorted(by_month.items()):
        result[month] = {
            "stops": len(rows),
            "target_after_stop_pct": path_audit.rate(
                rows, "original_target_hit_after_stop"),
            "entry_reclaimed_after_stop_pct": path_audit.rate(
                rows, "entry_reclaimed_after_stop"),
            "another_1r_adverse_pct": path_audit.rate(
                rows, "another_1r_adverse_after_stop"),
            "backing_inside_stop_pct": path_audit.rate(
                rows, "backing_inside_stop_distance"),
        }
    return result


def main() -> None:
    candles, sources = expanded.load_manifest()
    expanded.same_snapshot_overlap(candles)
    base_candles = {symbol: candles[symbol] for symbol in AUDITED_SYMBOLS}
    features = parent.feature_rows(base_candles, 2)
    context = mature.stable_context(features)
    guarded = mature.guarded(features, context, base_candles,
                             "native_xperp_frozen")
    enriched = pd.DataFrame([
        causal_features(row, base_candles[str(row["symbol"])])
        for row in guarded.to_dict("records")])
    baseline_rows = accept(enriched)
    baseline_rows.to_csv(BASE_ROWS_OUT, index=False)
    baseline = month_metrics(baseline_rows)
    factors = factor_stats(baseline_rows)
    candidates = []
    selected_rows = None
    for rule in RULES:
        rows = accept(apply_rule(enriched, rule))
        values = month_metrics(rows)
        dev_pass = dominates(values["2026-07"], baseline["2026-07"])
        validation = (dominates(values["2026-08"], baseline["2026-08"])
                      if dev_pass else False)
        test = (dominates(values["2026-09"], baseline["2026-09"])
                if dev_pass and validation else False)
        confirmation = (dominates(values["2026-10"], baseline["2026-10"])
                        if dev_pass and validation and test else False)
        item = {
            "rule": rule,
            "development": values["2026-07"],
            "development_pass": dev_pass,
            "validation": values["2026-08"] if dev_pass else None,
            "validation_pass": validation,
            "test": values["2026-09"] if dev_pass and validation else None,
            "test_pass": test,
            "confirmation": (values["2026-10"]
                             if dev_pass and validation and test else None),
            "confirmation_pass": confirmation,
        }
        candidates.append(item)
        if confirmation and selected_rows is None:
            selected_rows = rows
    accepted_candidate = selected_rows is not None
    if selected_rows is not None:
        selected_rows.to_csv(ROWS_OUT, index=False)
    result = {
        "status": ("SHADOW_CANDIDATE_REQUIRES_GENUINELY_NEW_FORWARD"
                   if accepted_candidate else "REJECTED_NO_CROSS_MONTH_DOMINANCE"),
        "ready_for_live_orders": False,
        "orders_possible": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "entry_features": "only candles closed before next-15m market entry",
            "development": "2026-07", "validation": "2026-08",
            "test": "2026-09", "confirmation": "2026-10",
            "gate": "retain 60%; non-worse return, drawdown, loss streak; >=3/day",
            "candidate_rules": list(RULES),
            "post_stop_paths_used_for_selection": False,
        },
        "integrity": {"symbols": len(base_candles),
                      "guarded_candidates": len(guarded),
                      "accepted_baseline_rows": len(baseline_rows)},
        "baseline": baseline,
        "factor_stop_rate_by_month": factors,
        "post_stop_path_by_month": stop_paths(baseline_rows, base_candles),
        "candidates": candidates,
        "accepted": accepted_candidate,
        "sources": {**sources, str(Path(__file__).resolve()):
                    mature.digest(Path(__file__).resolve())},
        "limitations": [
            "The native X-Perp route has only four complete recent months.",
            "Swing and order-block fields are OHLC proxies, not order-book liquidity.",
            "Post-stop recovery is diagnostic and unavailable at entry time.",
            "Gross returns exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    compact = [{"rule": item["rule"], "dev": item["development_pass"],
                "val": item["validation_pass"], "test": item["test_pass"],
                "confirm": item["confirmation_pass"]} for item in candidates]
    print(json.dumps({"status": result["status"], "baseline": baseline,
                      "stop_paths": result["post_stop_path_by_month"],
                      "candidates": compact}, indent=2), flush=True)


if __name__ == "__main__":
    main()
