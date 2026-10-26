"""Chronological native X-Perp pullback-in-trend router audit.

The frozen native route fades a two-bar cross-sectional extreme.  This audit
tests a distinct causal mechanism: accept that reversal only when the extreme
is a pullback against a longer, efficient trend that points in the proposed
trade direction.  July selects; August validates; September confirms; the
Oct-17+ forward period is revealed only after all earlier stages stay positive.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from stock_module_cluster_audit import portfolio_gate
import stock_cash_xperp_leadlag_oct6_audit as lead
import stock_xperp_backing_level_guard_oct20_audit as backing
import stock_xperp_native_cross_section_oct18_audit as parent
import stock_xperp_native_economics_oct18_audit as economics
import stock_xperp_native_growth_forward_monitor as monitor
import stock_xperp_native_quality_guard_oct18_audit as quality
import stock_xperp_native_volatility_guard_oct18_audit as volatility


ROOT = parent.ROOT
MANIFEST = ROOT / "stock_xperp_forward_oct26" / "manifest.json"
OUT = ROOT / "stock_xperp_trend_pullback_router_oct26.json"
ROWS_OUT = ROOT / "stock_xperp_trend_pullback_router_oct26.csv"
PROFILE = ROOT / "stock_xperp_native_growth_profile_oct18.json"
QUALITY = ROOT / "stock_xperp_native_quality_guard_oct18.json"
VOLATILITY = ROOT / "stock_xperp_native_volatility_guard_oct18.json"
HOUR_CAP = ROOT / "stock_xperp_backing_hour_cap_oct20.json"

SELECTION = ("2026-07-05", "2026-08-01")
VALIDATION = ("2026-08-01", "2026-09-01")
CONFIRMATION = ("2026-09-01", "2026-10-01")
FRESH = ("2026-10-17", "2026-10-27")
HORIZONS = (8, 16, 32, 64, 96)
MIN_DIRECTED_PRIOR_RETURNS = (0.0, .0025, .005, .01)
MIN_PRIOR_EFFICIENCIES = (0.0, .2, .4)
TARGETS = (.5, 1.0, 1.5, 2.0)
STOPS = (.5, .75, 1.0)
HOLDS = (32, 64)
MARGINS = (.03, .05)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def add_path_features(frame: pd.DataFrame, candles: dict) -> pd.DataFrame:
    result = frame.copy()
    for horizon in HORIZONS:
        directed, efficiencies = [], []
        for row in result.itertuples():
            close = np.asarray(candles[row.symbol]["close"], dtype=float)
            signal_index = int(row.entry_index) - 1
            start = signal_index - horizon
            # Exclude the two-bar extreme itself.  Every value is therefore
            # fixed before the next-bar market entry.
            prior_end = signal_index - 2
            if start < 0 or prior_end <= start:
                directed.append(np.nan)
                efficiencies.append(np.nan)
                continue
            prior_return = close[prior_end] / close[start] - 1.0
            path = float(np.abs(np.diff(close[start:prior_end + 1])).sum()
                         / close[start])
            directed.append(float(row.direction_n) * prior_return)
            efficiencies.append(abs(prior_return) / path if path else np.nan)
        result[f"directed_prior_return_{horizon}"] = directed
        result[f"prior_efficiency_{horizon}"] = efficiencies
    return result


def rule_rows(frame: pd.DataFrame, horizon: int,
              minimum_return: float, minimum_efficiency: float) -> pd.DataFrame:
    return frame[
        frame[f"directed_prior_return_{horizon}"].ge(minimum_return)
        & frame[f"prior_efficiency_{horizon}"].ge(minimum_efficiency)
    ].copy()


def accepted(frame: pd.DataFrame, bounds: tuple[str, str], *,
             rolling_cap: int, rank_name: str, loss_cap: int) -> list[dict]:
    rows = lead.period_rows(frame, bounds).to_dict("records")
    return portfolio_gate(
        rows, None, rolling_hour_cap=rolling_cap,
        rank_key=quality.ranker(rank_name), loss_streak_cap=loss_cap)


def evaluate(rows: list[dict], margin: float) -> dict:
    value = economics.simulate(rows, margin)
    if not rows:
        value.update({"active_days": 0, "trades_per_market_day": 0.0,
                      "projected_22_session_pct": 0.0,
                      "literal_tp_rate_pct": 0.0,
                      "profitable_exit_rate_pct": 0.0,
                      "max_loss_streak": 0})
        return value
    days = len({pd.Timestamp(float(row["entry_time"]), unit="s", tz="UTC").date()
                for row in rows})
    factor = max(0.0, 1.0 + value["return_pct"] / 100.0)
    streak = maximum = 0
    for row in sorted(rows, key=lambda item: float(item["entry_time"])):
        if float(row["signed_return"]) <= 0:
            streak += 1
            maximum = max(maximum, streak)
        else:
            streak = 0
    value.update({
        "active_days": days,
        "trades_per_market_day": round(len(rows) / days, 4),
        "projected_22_session_pct": round(
            100 * (factor ** (22 / days) - 1) if factor > 0 else -100, 4),
        "literal_tp_rate_pct": round(100 * sum(
            int(row["outcome_code"]) == 1 for row in rows) / len(rows), 4),
        "profitable_exit_rate_pct": round(100 * sum(
            float(row["signed_return"]) > 0 for row in rows) / len(rows), 4),
        "max_loss_streak": maximum,
    })
    return value


def stage_ok(value: dict) -> bool:
    return bool(value["n"] >= 50 and value["return_pct"] > 0
                and value["negative_months"] == 0 and value["dd_pct"] <= 10
                and value["trades_per_market_day"] >= 3)


def objective_ok(value: dict) -> bool:
    return bool(stage_ok(value) and value["return_pct"] >= 30
                and value["dd_pct"] <= 8)


def public(candidate: dict | None) -> dict | None:
    return ({key: value for key, value in candidate.items()
             if not key.startswith("_")} if candidate else None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    csv_out = args.out.with_suffix(".csv")
    if args.out.exists() or csv_out.exists():
        raise FileExistsError("audit outputs are immutable")

    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    parameters = profile["selection"]["selected"]["parameters"]
    quality_report = json.loads(QUALITY.read_text(encoding="utf-8"))
    volatility_report = json.loads(VOLATILITY.read_text(encoding="utf-8"))
    cap_report = json.loads(HOUR_CAP.read_text(encoding="utf-8"))
    rank_name = str(quality_report["selection"]["selected_ranker"])
    loss_cap = int(quality_report["selection"]["selected_loss_streak_cap"])
    maximum_atr = float(volatility_report["selection"][
        "selected_maximum_atr14_fraction"])
    rolling_cap = int(cap_report["selected_rolling_hour_cap"])
    backing_lookback = int(cap_report["backing_level_lookback_bars"])

    candles, sources, overlap = monitor.load_manifest(args.manifest)
    features = parent.feature_rows(candles, int(parameters["lookback_bars"]))
    entries = parent.select_entries(
        features, float(parameters["z_threshold"]),
        float(parameters["minimum_move_pct"]) / 100.0)
    entries = volatility.add_atr14(entries, candles)
    entries = entries[entries.atr14_fraction.le(maximum_atr)]
    entries = backing.guarded(backing.add_backing_distance(
        entries, candles, backing_lookback))
    entries = add_path_features(entries, candles)

    candidates = []
    geometries = itertools.product(TARGETS, STOPS, HOLDS)
    rules = tuple(itertools.product(
        HORIZONS, MIN_DIRECTED_PRIOR_RETURNS, MIN_PRIOR_EFFICIENCIES))
    for target, stop, hold in geometries:
        resolved = lead.add_outcomes(entries, candles, target, stop, hold)
        resolved["module"] = "native_xperp_trend_pullback_router"
        resolved["module_priority"] = 0
        for horizon, minimum_return, minimum_efficiency in rules:
            filtered = rule_rows(
                resolved, horizon, minimum_return, minimum_efficiency)
            selected_rows = accepted(
                filtered, SELECTION, rolling_cap=rolling_cap,
                rank_name=rank_name, loss_cap=loss_cap)
            selection_5 = evaluate(selected_rows, .05)
            if not stage_ok(selection_5):
                continue
            candidates.append({
                "target_pct": target, "stop_pct": stop,
                "hold_bars": hold, "trend_horizon_bars": horizon,
                "minimum_directed_prior_return_pct": 100 * minimum_return,
                "minimum_prior_efficiency": minimum_efficiency,
                "selection_5pct": selection_5,
                "selection_3pct": evaluate(selected_rows, .03),
                "_rows": filtered,
            })

    for item in candidates:
        validation_rows = accepted(
            item["_rows"], VALIDATION, rolling_cap=rolling_cap,
            rank_name=rank_name, loss_cap=loss_cap)
        item["validation_5pct"] = evaluate(validation_rows, .05)
        item["validation_3pct"] = evaluate(validation_rows, .03)
        item["validation_pass"] = stage_ok(item["validation_5pct"])
        item["score"] = round(min(
            item["selection_5pct"]["return_pct"],
            item["validation_5pct"]["return_pct"])
            - .3 * max(item["selection_5pct"]["dd_pct"],
                       item["validation_5pct"]["dd_pct"]), 8)
    survivors = [item for item in candidates if item["validation_pass"]]
    survivors.sort(key=lambda item: (
        item["score"], item["validation_5pct"]["literal_tp_rate_pct"],
        -item["validation_5pct"]["dd_pct"]), reverse=True)
    selected = survivors[0] if survivors else None
    confirmation = fresh = None
    confirmation_pass = False
    objective_pass = False
    output_rows = []
    if selected:
        confirmation_rows = accepted(
            selected["_rows"], CONFIRMATION, rolling_cap=rolling_cap,
            rank_name=rank_name, loss_cap=loss_cap)
        confirmation = {f"margin_{int(100*m)}pct": evaluate(
            confirmation_rows, m) for m in MARGINS}
        confirmation_pass = stage_ok(confirmation["margin_5pct"])
        if confirmation_pass:
            fresh_rows = accepted(
                selected["_rows"], FRESH, rolling_cap=rolling_cap,
                rank_name=rank_name, loss_cap=loss_cap)
            fresh = {f"margin_{int(100*m)}pct": evaluate(
                fresh_rows, m) for m in MARGINS}
            output_rows = fresh_rows
            stages_3 = [selected["selection_3pct"],
                        selected["validation_3pct"],
                        confirmation["margin_3pct"], fresh["margin_3pct"]]
            objective_pass = all(objective_ok(value) for value in stages_3)
    if output_rows:
        pd.DataFrame(output_rows).to_csv(csv_out, index=False)

    status = ("OBJECTIVE_PASS_SHADOW_ONLY" if objective_pass else
              "REJECTED_CONFIRMATION" if selected and not confirmation_pass else
              "REJECTED_OBJECTIVE")
    report = {
        "status": status,
        "ready_for_live_orders": False,
        "orders_possible": False,
        "protocol": {
            "selection": list(SELECTION), "validation": list(VALIDATION),
            "confirmation": list(CONFIRMATION),
            "conditional_fresh": list(FRESH),
            "mechanism": (
                "fade the frozen two-bar cross-sectional extreme only when it "
                "is a pullback against a longer efficient trend pointing in "
                "the proposed trade direction"),
            "entry": "next 15-minute X-Perp market open",
            "orders": "market only",
            "money": "client-selected current-equity margin; fixed 10x",
            "targets_pct": list(TARGETS), "stops_pct": list(STOPS),
            "holds_bars": list(HOLDS), "trend_horizons": list(HORIZONS),
            "candidate_count": len(TARGETS) * len(STOPS) * len(HOLDS)
                               * len(rules),
        },
        "runtime_base": {
            "parameters": parameters, "maximum_atr14_fraction": maximum_atr,
            "backing_level_lookback_bars": backing_lookback,
            "ranker": rank_name, "loss_streak_cap": loss_cap,
            "rolling_hour_cap": rolling_cap,
        },
        "selection_survivors": len(candidates),
        "validation_survivors": len(survivors),
        "selected": public(selected),
        "confirmation": confirmation,
        "confirmation_pass": confirmation_pass,
        "fresh": fresh,
        "objective_pass": objective_pass,
        "frontier": [public(item) for item in survivors[:20]],
        "overlap_verification": {
            "symbols": len(overlap),
            "overlap_bars": sum(item["overlap_bars"] for item in overlap.values()),
            "new_bars_after_frozen_end": sum(
                item["new_bars_after_frozen_end"] for item in overlap.values()),
            "mismatches": 0,
        },
        "sources": {
            **sources, str(PROFILE): digest(PROFILE),
            str(QUALITY): digest(QUALITY), str(VOLATILITY): digest(VOLATILITY),
            str(HOUR_CAP): digest(HOUR_CAP),
            str(Path(__file__)): digest(Path(__file__)),
        },
        "limitations": [
            "Native stock X-Perp history begins in July 2026.",
            "The bounded path and bracket grid still creates selection risk.",
            "October is revealed only after positive July-August-September stages.",
            "Gross returns exclude fees at the user's request.",
        ],
    }
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True),
                        encoding="utf-8")
    print(json.dumps({
        "status": status, "selection_survivors": len(candidates),
        "validation_survivors": len(survivors),
        "selected": public(selected), "confirmation": confirmation,
        "confirmation_pass": confirmation_pass,
        "fresh": fresh, "objective_pass": objective_pass,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
