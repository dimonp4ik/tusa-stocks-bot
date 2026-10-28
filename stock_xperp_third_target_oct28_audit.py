"""Causal third-target audit for the frozen native stock X-Perp route.

The entry rule and portfolio rails stay frozen.  A candidate may bank part of
the position at the existing +2% target and protect the remainder at entry
from the next 15-minute bar while it attempts a farther target.  TP3 is never
awarded on the activation candle because OHLC cannot prove event order.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from stock_module_cluster_audit import portfolio_gate
import stock_xperp_backing_level_guard_oct20_audit as backing
import stock_xperp_basket_trend_forward_oct28_audit as forward
import stock_xperp_basket_trend_long_guard_oct26_audit as frozen
import stock_xperp_native_cross_section_oct18_audit as parent
import stock_xperp_native_growth_forward_monitor as monitor
import stock_xperp_native_quality_guard_oct18_audit as quality
import stock_xperp_native_volatility_guard_oct18_audit as volatility


ROOT = Path("reports/audit_2026_09_08")
MANIFEST = ROOT / "stock_xperp_forward_oct27_2156" / "manifest.json"
REFERENCE = ROOT / "stock_xperp_basket_trend_forward_oct28.csv"
OUT = ROOT / "stock_xperp_third_target_oct28.json"
ROWS_OUT = ROOT / "stock_xperp_third_target_oct28.csv"
TARGETS = (3.0, 4.0, 5.0)
CLOSE_FRACTIONS = (.50, .70, .80, .90, .95)
MONTHS = ("2026-07", "2026-08", "2026-09", "2026-10")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def raw_entries() -> tuple[pd.DataFrame, dict, dict]:
    candles, sources, _ = monitor.load_manifest(MANIFEST)
    features = parent.feature_rows(candles, 2)
    entries = parent.select_entries(features, 2.0, .004)
    entries = volatility.add_atr14(entries, candles)
    entries = entries[entries.atr14_fraction.le(.01)]
    entries = backing.guarded(backing.add_backing_distance(entries, candles, 32))
    context = features.groupby("entry_time", as_index=False).cross_mean.first()
    context.sort_values("entry_time", inplace=True)
    context.rename(columns={"cross_mean": "basket_bar_return"}, inplace=True)
    context["basket_trend_96"] = context.basket_bar_return.rolling(
        96, min_periods=96).sum()
    entries = entries.merge(context[["entry_time", "basket_trend_96"]],
                            on="entry_time", how="left", validate="many_to_one")
    return entries, candles, sources


def resolve(row: dict, candle: dict, close_fraction: float,
            third_target_pct: float) -> dict:
    output = dict(row)
    open_, high, low, close = [np.asarray(candle[key], dtype=float)
                               for key in ("open", "high", "low", "close")]
    times = np.asarray(candle["time"], dtype=np.int64)
    entry_index = int(row["entry_index"])
    direction = int(row["direction_n"])
    entry = float(open_[entry_index])
    stop = entry * (1 - direction * .01)
    first = entry * (1 + direction * .02)
    third = entry * (1 + direction * third_target_pct / 100)
    required_last = entry_index + 63
    available_last = min(required_last, len(times) - 1)
    activated = False
    activation_index = None
    first_leg_return = None
    runner_return = None
    exit_index = available_last
    outcome = None
    for index in range(entry_index, available_last + 1):
        if index > entry_index and times[index] - times[index - 1] != 900:
            exit_index = index - 1
            if activated:
                runner_return = max(0.0, direction * (close[exit_index] / entry - 1))
                outcome = "TP2_TIME"
            else:
                runner_return = direction * (close[exit_index] / entry - 1)
                outcome = "TIME"
            break
        opening = float(open_[index])
        if not activated:
            if direction > 0:
                if opening <= stop:
                    runner_return = opening / entry - 1
                    outcome, exit_index = "SL", index
                    break
                if opening >= first:
                    activated, activation_index = True, index
                    first_leg_return = opening / entry - 1
                elif low[index] <= stop:
                    runner_return, outcome, exit_index = -.01, "SL", index
                    break
                elif high[index] >= first:
                    activated, activation_index = True, index
                    first_leg_return = .02
            else:
                if opening >= stop:
                    runner_return = 1 - opening / entry
                    outcome, exit_index = "SL", index
                    break
                if opening <= first:
                    activated, activation_index = True, index
                    first_leg_return = 1 - opening / entry
                elif high[index] >= stop:
                    runner_return, outcome, exit_index = -.01, "SL", index
                    break
                elif low[index] <= first:
                    activated, activation_index = True, index
                    first_leg_return = .02
            if activated and close_fraction >= 1.0:
                runner_return, outcome, exit_index = first_leg_return, "TP2", index
                break
            # No TP3 or breakeven decision on the activation candle.
            continue
        if index <= int(activation_index):
            continue
        if direction > 0:
            if opening <= entry:
                runner_return, outcome, exit_index = opening / entry - 1, "TP2_BE", index
                break
            if opening >= third:
                runner_return, outcome, exit_index = opening / entry - 1, "TP3", index
                break
            if low[index] <= entry:
                runner_return, outcome, exit_index = 0.0, "TP2_BE", index
                break
            if high[index] >= third:
                runner_return, outcome, exit_index = third_target_pct / 100, "TP3", index
                break
        else:
            if opening >= entry:
                runner_return, outcome, exit_index = 1 - opening / entry, "TP2_BE", index
                break
            if opening <= third:
                runner_return, outcome, exit_index = 1 - opening / entry, "TP3", index
                break
            if high[index] >= entry:
                runner_return, outcome, exit_index = 0.0, "TP2_BE", index
                break
            if low[index] <= third:
                runner_return, outcome, exit_index = third_target_pct / 100, "TP3", index
                break
    if runner_return is None and available_last >= required_last:
        if activated:
            runner_return = max(0.0, direction * (close[required_last] / entry - 1))
            outcome = "TP2_TIME"
        else:
            runner_return = direction * (close[required_last] / entry - 1)
            outcome = "TIME"
        exit_index = required_last
    pending = runner_return is None
    if pending:
        signed_return = None
    elif outcome == "SL" or not activated:
        signed_return = float(runner_return)
    else:
        signed_return = (close_fraction * float(first_leg_return)
                         + (1-close_fraction) * float(runner_return))
    output.update({
        "entry": entry, "target_pct": 2.0, "third_target_pct": third_target_pct,
        "stop_pct": 1.0, "hold_bars": 64,
        "direction": "LONG" if direction > 0 else "SHORT",
        "signed_return": signed_return,
        "outcome_code": (None if pending else -1 if outcome == "SL" else
                         1 if outcome in ("TP2", "TP3") else 0),
        "outcome": "PENDING" if pending else outcome,
        "exit_time": (int(times[entry_index]) + 64*900 if pending
                      else int(times[exit_index]) + 900),
        "risk_pct": .01,
        "net_r": None if pending else signed_return / .01,
        "module": "native_xperp_third_target",
        "module_priority": 0,
        "tp2_close_fraction": close_fraction,
        "first_leg_return": first_leg_return,
        "runner_return": runner_return,
        "activation_bar_time": (None if activation_index is None
                                else int(times[activation_index])),
    })
    return output


def portfolio(entries: pd.DataFrame, candles: dict, close_fraction: float,
              third_target_pct: float) -> pd.DataFrame:
    resolved = pd.DataFrame(resolve(row, candles[row["symbol"]], close_fraction,
                                    third_target_pct)
                            for row in entries.to_dict("records"))
    guarded = frozen.apply(resolved, 96, "LONG", -0.005626823134606253)
    accepted = portfolio_gate(
        guarded.to_dict("records"), None, rolling_hour_cap=4,
        rank_key=quality.ranker("z_high"), loss_streak_cap=2)
    return pd.DataFrame(accepted, columns=guarded.columns)


def portfolio_fraction(resolved: pd.DataFrame, close_fraction: float) -> pd.DataFrame:
    """Reweight one resolved TP3 path; exit timing is fraction-independent."""
    rows = resolved.copy()
    active = rows.first_leg_return.notna() & rows.runner_return.notna()
    rows.loc[active, "signed_return"] = (
        close_fraction * rows.loc[active, "first_leg_return"].astype(float)
        + (1-close_fraction) * rows.loc[active, "runner_return"].astype(float))
    rows.loc[active, "net_r"] = rows.loc[active, "signed_return"] / .01
    rows["tp2_close_fraction"] = close_fraction
    guarded = frozen.apply(rows, 96, "LONG", -0.005626823134606253)
    accepted = portfolio_gate(
        guarded.to_dict("records"), None, rolling_hour_cap=4,
        rank_key=quality.ranker("z_high"), loss_streak_cap=2)
    return pd.DataFrame(accepted, columns=guarded.columns)


def period_metrics(frame: pd.DataFrame) -> dict:
    stamps = pd.to_datetime(frame.entry_time, unit="s", utc=True).dt.strftime("%Y-%m")
    return {month: forward.metrics(frame[stamps.eq(month)], .03) for month in MONTHS}


def compact(value: dict) -> dict:
    return {month: {key: item[key] for key in (
        "n", "return_pct", "dd_pct", "profitable_exit_rate_pct",
        "trades_per_active_day", "max_loss_streak")}
        for month, item in value.items()}


def main() -> None:
    entries, candles, sources = raw_entries()
    baseline = portfolio(entries, candles, 1.0, 3.0)
    published = pd.read_csv(REFERENCE)
    keys = ["symbol", "entry_time", "direction"]
    parity = baseline[keys].reset_index(drop=True).equals(
        published[keys].reset_index(drop=True)) and np.allclose(
            baseline.signed_return, published.signed_return,
            rtol=0, atol=1e-12, equal_nan=True)
    if not parity:
        raise ValueError("baseline replay does not match frozen forward rows")
    base_metrics = period_metrics(baseline)
    candidates = []
    for target in TARGETS:
        resolved = pd.DataFrame(
            resolve(row, candles[row["symbol"]], .5, target)
            for row in entries.to_dict("records"))
        for fraction in CLOSE_FRACTIONS:
            rows = portfolio_fraction(resolved, fraction)
            values = period_metrics(rows)
            development_pass = all(
                values[month]["return_pct"] > base_metrics[month]["return_pct"]
                and values[month]["dd_pct"] <= base_metrics[month]["dd_pct"] + .5
                and values[month]["trades_per_active_day"] >= 3
                for month in ("2026-07", "2026-08"))
            candidates.append({
                "third_target_pct": target, "tp2_close_fraction": fraction,
                "development_pass": development_pass,
                "development_min_return_gain": min(
                    values[m]["return_pct"]-base_metrics[m]["return_pct"]
                    for m in ("2026-07", "2026-08")),
                "periods": compact(values), "_rows": rows,
            })
    qualified = [item for item in candidates if item["development_pass"]]
    qualified.sort(key=lambda item: item["development_min_return_gain"], reverse=True)
    selected = qualified[0] if qualified else None
    holdout_pass = bool(selected and all(
        selected["periods"][month]["return_pct"] >=
        compact(base_metrics)[month]["return_pct"]
        and selected["periods"][month]["dd_pct"] <=
        compact(base_metrics)[month]["dd_pct"] + .5
        for month in ("2026-09", "2026-10")))
    if selected:
        selected["_rows"].to_csv(ROWS_OUT, index=False)
    report = {
        "status": ("RESEARCH_CANDIDATE_AWAIT_FRESH_FORWARD" if holdout_pass
                   else "REJECTED_HOLDOUT" if selected
                   else "REJECTED_DEVELOPMENT"),
        "ready_for_live_orders": False, "orders_possible": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "entry_and_filters": "unchanged frozen native route",
            "baseline": "close 100% at +2%; stop -1%; 64 bars",
            "candidate": "partial close at +2%, BE from next bar, TP3 at +3/+4/+5%",
            "same_activation_bar_tp3": False,
            "selection": ["2026-07", "2026-08"],
            "holdouts": ["2026-09", "2026-10"],
            "money": "3% current-equity margin; fixed 10x; market only",
        },
        "grid": {"candidates": len(candidates), "targets_pct": list(TARGETS),
                 "tp2_close_fractions": list(CLOSE_FRACTIONS)},
        "integrity": {"baseline_row_parity": parity,
                      "baseline_rows": len(baseline)},
        "baseline": compact(base_metrics),
        "selected_before_holdouts": (None if selected is None else
            {key: value for key, value in selected.items() if key != "_rows"}),
        "holdout_pass": holdout_pass,
        "top_development": [
            {key: value for key, value in item.items() if key != "_rows"}
            for item in sorted(candidates, key=lambda x: x["development_min_return_gain"],
                               reverse=True)[:10]],
        "sources": {str(MANIFEST): digest(MANIFEST), str(REFERENCE): digest(REFERENCE),
                    **sources, str(Path(__file__).resolve()): digest(Path(__file__).resolve())},
        "limitations": [
            "Native X-Perp history begins in July 2026.",
            "October was already observed and cannot activate the exit.",
            "Public OHLC cannot reproduce order-book slippage or latency.",
            "Gross results exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "status", "grid", "integrity", "baseline",
        "selected_before_holdouts", "holdout_pass")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
