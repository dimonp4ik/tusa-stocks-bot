"""Test partial-profit runners on the frozen basket-trend stock route.

Exit choices are selected on July and August only.  September is a validation
month and October is an already-observed diagnostic.  Entries and every entry
filter remain identical to the current shadow runtime.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from stock_module_cluster_audit import portfolio_gate
import stock_xperp_backing_level_guard_oct20_audit as backing
import stock_xperp_basket_trend_long_guard_oct26_audit as frozen
import stock_xperp_native_cross_section_oct18_audit as parent
import stock_xperp_native_economics_oct18_audit as economics
import stock_xperp_native_growth_forward_monitor as monitor
import stock_xperp_native_quality_guard_oct18_audit as quality
import stock_xperp_native_volatility_guard_oct18_audit as volatility


ROOT = Path("reports/audit_2026_09_08")
MANIFEST = ROOT / "stock_xperp_forward_oct27" / "manifest.json"
OUT = ROOT / "stock_xperp_current_barbell_oct27.json"
ROWS_OUT = ROOT / "stock_xperp_current_barbell_oct27.csv"
TARGET_PCT = 2.0
STOP_PCT = 1.0
HOLD_BARS = 64
NEAR_TARGETS_PCT = (.5, 1.0, 1.5)
NEAR_FRACTIONS = (0.0, .05, .10, .20, .30, .50, .70)
PERIODS = {
    "selection_july": ("2026-07-01", "2026-08-01"),
    "selection_august": ("2026-08-01", "2026-09-01"),
    "validation_september": ("2026-09-01", "2026-10-01"),
    "observed_october": ("2026-10-01", "2026-11-01"),
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_entries() -> tuple[pd.DataFrame, dict, dict]:
    candles, sources, _ = monitor.load_manifest(MANIFEST)
    features = parent.feature_rows(candles, 2)
    entries = parent.select_entries(features, 2.0, .004)
    entries = volatility.add_atr14(entries, candles)
    entries = entries[entries.atr14_fraction.le(.01)]
    entries = backing.guarded(backing.add_backing_distance(entries, candles, 32))
    context = features.groupby("entry_time", as_index=False).cross_mean.first()
    context.sort_values("entry_time", inplace=True)
    context.rename(columns={"cross_mean": "basket_bar_return"}, inplace=True)
    for horizon in frozen.HORIZONS:
        context[f"basket_trend_{horizon}"] = context.basket_bar_return.rolling(
            horizon, min_periods=horizon).sum()
    entries = entries.merge(context, on="entry_time", how="left",
                            validate="many_to_one")
    entries["direction"] = np.where(entries.direction_n.gt(0), "LONG", "SHORT")
    entries = frozen.apply(entries, 96, "LONG", -0.005626823134606253)
    return entries, candles, sources


def resolve_row(row: dict, candle: dict, near_target_pct: float,
                near_fraction: float) -> dict:
    """Resolve one conservative partial/runner path from 15-minute OHLC.

    Stops win every same-bar ambiguity.  A runner touch is valid on the near
    target bar because the price must cross the near level first; break-even
    becomes active only on the following bar.
    """
    output = dict(row)
    open_, high, low, close = [np.asarray(candle[key], dtype=float)
                               for key in ("open", "high", "low", "close")]
    times = np.asarray(candle["time"], dtype=np.int64)
    entry_index = int(row["entry_index"])
    direction = int(row["direction_n"])
    entry = float(open_[entry_index])
    near = entry * (1 + direction * near_target_pct / 100)
    target = entry * (1 + direction * TARGET_PCT / 100)
    stop = entry * (1 - direction * STOP_PCT / 100)
    required_last = entry_index + HOLD_BARS - 1
    available_last = min(required_last, len(times) - 1)
    activated = False
    signed_return = None
    outcome_code = None
    outcome = None
    exit_index = available_last

    for index in range(entry_index, available_last + 1):
        if index > entry_index and times[index] - times[index - 1] != 900:
            exit_index = index - 1
            remainder = direction * (float(close[exit_index]) / entry - 1)
            signed_return = (near_fraction * near_target_pct / 100
                             + (1-near_fraction) * remainder
                             if activated else remainder)
            outcome_code, outcome = 0, "TIME"
            break
        opening = float(open_[index])
        stop_gap = opening <= stop if direction > 0 else opening >= stop
        stop_hit = float(low[index]) <= stop if direction > 0 else float(high[index]) >= stop
        near_gap = opening >= near if direction > 0 else opening <= near
        near_hit = float(high[index]) >= near if direction > 0 else float(low[index]) <= near
        runner_gap = opening >= target if direction > 0 else opening <= target
        runner_hit = float(high[index]) >= target if direction > 0 else float(low[index]) <= target

        if not activated:
            if stop_gap:
                signed_return = direction * (opening / entry - 1)
                exit_index, outcome_code, outcome = index, -1, "SL"
                break
            if stop_hit:
                signed_return = -STOP_PCT / 100
                exit_index, outcome_code, outcome = index, -1, "SL"
                break
            if runner_gap or runner_hit:
                signed_return = (near_fraction * near_target_pct / 100
                                 + (1-near_fraction) * TARGET_PCT / 100)
                exit_index, outcome_code, outcome = index, 1, "TP"
                break
            if near_gap or near_hit:
                activated = True
                continue
        else:
            break_even_gap = opening <= entry if direction > 0 else opening >= entry
            break_even_hit = (float(low[index]) <= entry if direction > 0
                              else float(high[index]) >= entry)
            if break_even_gap:
                remainder = direction * (opening / entry - 1)
                signed_return = (near_fraction * near_target_pct / 100
                                 + (1-near_fraction) * remainder)
                exit_index, outcome_code, outcome = index, 0, "BE"
                break
            if break_even_hit:
                signed_return = near_fraction * near_target_pct / 100
                exit_index, outcome_code, outcome = index, 0, "BE"
                break
            if runner_gap:
                remainder = direction * (opening / entry - 1)
                signed_return = (near_fraction * near_target_pct / 100
                                 + (1-near_fraction) * remainder)
                exit_index, outcome_code, outcome = index, 1, "TP"
                break
            if runner_hit:
                signed_return = (near_fraction * near_target_pct / 100
                                 + (1-near_fraction) * TARGET_PCT / 100)
                exit_index, outcome_code, outcome = index, 1, "TP"
                break

    pending = signed_return is None and available_last < required_last
    if signed_return is None and not pending:
        remainder = direction * (float(close[required_last]) / entry - 1)
        signed_return = (near_fraction * near_target_pct / 100
                         + (1-near_fraction) * remainder
                         if activated else remainder)
        exit_index, outcome_code, outcome = required_last, 0, "TIME"
    output.update({
        "entry": entry, "target_pct": TARGET_PCT, "stop_pct": STOP_PCT,
        "hold_bars": HOLD_BARS, "near_target_pct": near_target_pct,
        "near_fraction": near_fraction,
        "direction": "LONG" if direction > 0 else "SHORT",
        "signed_return": None if pending else float(signed_return),
        "outcome_code": None if pending else outcome_code,
        "outcome": "PENDING" if pending else outcome,
        "exit_time": (int(times[entry_index]) + HOLD_BARS * 900
                      if pending else int(times[exit_index]) + 900),
        "risk_pct": STOP_PCT / 100,
        "net_r": None if pending else float(signed_return / (STOP_PCT/100)),
        "module": "native_xperp_basket_trend_barbell", "module_priority": 0,
        "near_activated": activated,
    })
    return output


def resolve(entries: pd.DataFrame, candles: dict, near_target_pct: float,
            near_fraction: float) -> pd.DataFrame:
    return pd.DataFrame(resolve_row(row, candles[row["symbol"]],
                                    near_target_pct, near_fraction)
                        for row in entries.to_dict("records"))


def gated(frame: pd.DataFrame) -> pd.DataFrame:
    accepted = portfolio_gate(
        frame.to_dict("records"), None, rolling_hour_cap=4,
        rank_key=quality.ranker("z_high"), loss_streak_cap=2)
    return pd.DataFrame(accepted, columns=frame.columns)


def bounds(frame: pd.DataFrame, period: tuple[str, str]) -> pd.DataFrame:
    lo = pd.Timestamp(period[0], tz="UTC").timestamp()
    hi = pd.Timestamp(period[1], tz="UTC").timestamp()
    return frame[frame.entry_time.ge(lo) & frame.entry_time.lt(hi)]


def metrics(frame: pd.DataFrame, margin: float = .03) -> dict:
    resolved = frame[frame.outcome.ne("PENDING")].copy()
    value = economics.simulate(resolved.to_dict("records"), margin)
    days = max(1, pd.to_datetime(resolved.entry_time, unit="s", utc=True)
               .dt.date.nunique()) if len(resolved) else 1
    value.update({
        "resolved": len(resolved),
        "pending": int(frame.outcome.eq("PENDING").sum()),
        "trades_per_active_day": round(len(resolved)/days, 4),
        "profitable_exit_rate_pct": round(
            100*float(resolved.signed_return.gt(0).mean()), 4) if len(resolved) else 0.0,
        "literal_runner_rate_pct": round(
            100*float(resolved.outcome.eq("TP").mean()), 4) if len(resolved) else 0.0,
        "near_activation_rate_pct": round(
            100*float(resolved.near_activated.mean()), 4) if len(resolved) else 0.0,
        "mean_signed_move_pct": round(
            100*float(resolved.signed_return.mean()), 6) if len(resolved) else 0.0,
    })
    return value


def evaluate(frame: pd.DataFrame) -> dict:
    return {name: {"margin_3pct": metrics(bounds(frame, span), .03),
                   "margin_5pct": metrics(bounds(frame, span), .05)}
            for name, span in PERIODS.items()}


def main() -> None:
    entries, candles, sources = prepare_entries()
    base_raw = pd.DataFrame(monitor.resolve_entries(entries, candles, {
        "target_pct": TARGET_PCT, "stop_pct": STOP_PCT,
        "hold_bars": HOLD_BARS}))
    base_raw["near_activated"] = False
    baseline = gated(base_raw)
    baseline_values = evaluate(baseline)

    candidates = []
    for near_target, fraction in itertools.product(
            NEAR_TARGETS_PCT, NEAR_FRACTIONS):
        accepted = gated(resolve(entries, candles, near_target, fraction))
        values = evaluate(accepted)
        july = values["selection_july"]["margin_3pct"]
        august = values["selection_august"]["margin_3pct"]
        score = min(july["return_pct"], august["return_pct"]) - .5*max(
            july["dd_pct"], august["dd_pct"])
        candidates.append({
            "near_target_pct": near_target, "near_fraction": fraction,
            "selection_score": round(score, 8), "values": values,
            "_rows": accepted,
        })
    eligible = [item for item in candidates if all(
        item["values"][month]["margin_3pct"]["return_pct"] > 0
        and item["values"][month]["margin_3pct"]["trades_per_active_day"] >= 3
        for month in ("selection_july", "selection_august"))]
    eligible.sort(key=lambda item: item["selection_score"], reverse=True)
    selected = eligible[0]
    validation = selected["values"]["validation_september"]["margin_3pct"]
    base_validation = baseline_values["validation_september"]["margin_3pct"]
    october = selected["values"]["observed_october"]["margin_3pct"]
    base_october = baseline_values["observed_october"]["margin_3pct"]
    selection_improves = all(
        selected["values"][month]["margin_3pct"]["return_pct"]
        > baseline_values[month]["margin_3pct"]["return_pct"]
        for month in ("selection_july", "selection_august"))
    validation_improves = (
        validation["return_pct"] > base_validation["return_pct"]
        and validation["dd_pct"] <= 5
        and validation["trades_per_active_day"] >= 3)
    october_improves = (
        october["return_pct"] > base_october["return_pct"]
        and october["dd_pct"] <= 5
        and october["trades_per_active_day"] >= 3)
    accepted = bool(selection_improves and validation_improves
                    and october_improves)
    selected["_rows"].to_csv(ROWS_OUT, index=False)

    def public(item: dict) -> dict:
        return {key: value for key, value in item.items()
                if not key.startswith("_")}

    report = {
        "status": ("CANDIDATE_REQUIRES_NEW_FORWARD" if accepted
                   else "REJECTED_NO_ROBUST_IMPROVEMENT"),
        "ready_for_live_orders": False, "orders_possible": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "entry_and_filters": "frozen native reversal plus basket-trend guard",
            "selection": ["2026-07", "2026-08"],
            "validation": "2026-09",
            "already_observed_diagnostic": "2026-10 through snapshot end",
            "exit": "partial at near target; delayed break-even; 2% runner",
            "same_bar_policy": "stop first; runner may fill on near bar; BE starts next bar",
            "entry": "next 15-minute market open", "orders": "market only",
            "money": "client-selected 3% or 5% current-equity margin; fixed 10x",
        },
        "grid": {"near_targets_pct": list(NEAR_TARGETS_PCT),
                 "near_fractions": list(NEAR_FRACTIONS),
                 "candidates": len(candidates), "eligible": len(eligible)},
        "baseline": baseline_values,
        "selected": public(selected),
        "top10_selection": [public(item) for item in eligible[:10]],
        "acceptance": {
            "selection_returns_improve": selection_improves,
            "validation_return_improves_with_dd_at_most_5pct": validation_improves,
            "observed_october_return_improves_with_dd_at_most_5pct": october_improves,
            "accepted": accepted,
        },
        "sources": {**sources, str(MANIFEST): digest(MANIFEST),
                    str(Path(__file__).resolve()): digest(Path(__file__).resolve())},
        "limitations": [
            "October was already observed and is diagnostic, not fresh evidence.",
            "Native X-Perp history begins in July 2026.",
            "Public OHLC cannot prove the intrabar path or live fill latency.",
            "Gross returns exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": report["status"],
                      "selected": public(selected),
                      "baseline": baseline_values,
                      "acceptance": report["acceptance"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
