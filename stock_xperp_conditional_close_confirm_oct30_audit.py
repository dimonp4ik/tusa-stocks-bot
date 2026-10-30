"""Causal audit of close-confirmed stops for the stock X-Perp route.

The candidate exits at the next 15-minute market open only after a 1% stop is
confirmed by a candle close, while retaining an intrabar 2% emergency stop.
The rule is applied before the causal portfolio gate. July selects one
pre-entry cohort, August validates it, September tests it and October is only a
diagnostic after a failed earlier gate.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path

import pandas as pd

import stock_xperp_basket_trend_forward_oct30_audit as forward
import stock_xperp_regime_target_router_oct30_audit as router
import stock_xperp_short_exhaustion_guard_oct30_audit as guard
import stock_xperp_short_stack_guard_oct30_forward as source


ROOT = Path("reports/audit_2026_09_08")
MANIFEST = ROOT / "stock_xperp_forward_oct30_1000" / "manifest.json"
OUT = ROOT / "stock_xperp_conditional_close_confirm_oct30.json"
MONTHS = ("2026-07", "2026-08", "2026-09", "2026-10")
FEATURES = {
    "atr14_fraction": (.002, .004, .006),
    "abs_z": (2.5, 3.0, 3.5, 4.0),
    "backing_level_distance": (.0025, .005, .01, .02),
    "directional_basket96": (-.02, -.01, 0.0, .01),
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(row: dict) -> tuple[str, int, str]:
    direction = row.get("direction")
    if direction is None:
        direction = "LONG" if int(row["direction_n"]) > 0 else "SHORT"
    return str(row["symbol"]), int(row["entry_time"]), str(direction)


def close_confirmed(row: dict, candle: dict, target_pct: float) -> dict:
    result = dict(row)
    first = int(row["entry_index"])
    direction = int(row["direction_n"])
    entry = float(candle["open"][first])
    target = entry * (1 + direction * target_pct / 100)
    soft_stop = entry * (1 - direction * .01)
    hard_stop = entry * (1 - direction * .02)
    required_last = first + 63
    last = min(required_last, len(candle["time"])-1)
    signed = outcome_code = None
    exit_index = last
    next_open_exit = False
    for index in range(first, last+1):
        if (index > first and int(candle["time"][index])
                - int(candle["time"][index-1]) != 900):
            exit_index = index-1
            signed = direction*(float(candle["close"][exit_index])/entry-1)
            outcome_code = 0
            break
        open_ = float(candle["open"][index])
        high = float(candle["high"][index])
        low = float(candle["low"][index])
        close = float(candle["close"][index])
        hard_open = open_ <= hard_stop if direction > 0 else open_ >= hard_stop
        target_open = open_ >= target if direction > 0 else open_ <= target
        hard_hit = low <= hard_stop if direction > 0 else high >= hard_stop
        target_hit = high >= target if direction > 0 else low <= target
        if hard_open:
            signed, outcome_code, exit_index = (
                direction*(open_/entry-1), -1, index)
            break
        if target_open:
            signed, outcome_code, exit_index = (
                direction*(open_/entry-1), 1, index)
            break
        if hard_hit:
            signed, outcome_code, exit_index = -.02, -1, index
            break
        if target_hit:
            signed, outcome_code, exit_index = target_pct/100, 1, index
            break
        stop_confirmed = close <= soft_stop if direction > 0 else close >= soft_stop
        if stop_confirmed:
            if index+1 < len(candle["time"]):
                exit_index = index+1
                signed = direction*(float(candle["open"][exit_index])/entry-1)
                outcome_code = -1
                next_open_exit = True
            break
    if signed is None and last >= required_last:
        exit_index = required_last
        signed = direction*(float(candle["close"][exit_index])/entry-1)
        outcome_code = 0
    pending = signed is None
    if pending:
        exit_time = int(candle["time"][first])+64*900
    elif next_open_exit:
        exit_time = int(candle["time"][exit_index])
    else:
        exit_time = int(candle["time"][exit_index])+900
    result.update({
        "entry": entry, "target_pct": target_pct, "stop_pct": 1.0,
        "hold_bars": 64, "direction": "LONG" if direction > 0 else "SHORT",
        "signed_return": None if pending else float(signed),
        "outcome_code": outcome_code,
        "outcome": ("PENDING" if pending else "TP" if outcome_code == 1
                    else "SL" if outcome_code == -1 else "TIME"),
        "exit_time": exit_time, "risk_pct": .01,
        "net_r": None if pending else float(signed/.01),
        "module": "native_xperp_conditional_close_confirm",
        "module_priority": 0, "audited_target_pct": target_pct,
    })
    return result


def feature_value(row: dict, feature: str) -> float:
    if feature == "abs_z":
        return abs(float(row["cross_z"]))
    if feature == "directional_basket96":
        return int(row["direction_n"])*float(row["basket_trend_96"])
    return float(row[feature])


def matches(row: dict, config: dict) -> bool:
    if config["direction"] != "ALL" and row["direction"] != config["direction"]:
        return False
    value = feature_value(row, config["feature"])
    threshold = float(config["threshold"])
    return value >= threshold if config["operator"] == "at_least" else value <= threshold


def compact(value: dict) -> dict:
    return {key: value[key] for key in (
        "return_pct", "dd_pct", "resolved", "pending",
        "trades_per_active_day", "literal_tp_rate_pct", "max_loss_streak")}


def evaluate(frame: pd.DataFrame) -> dict:
    labels = pd.to_datetime(frame.entry_time, unit="s", utc=True).dt.strftime("%Y-%m")
    return {month: compact(forward.metrics(frame[labels.eq(month)], .03))
            for month in MONTHS}


def stop_path(frame: pd.DataFrame, candles: dict) -> dict:
    rows = []
    for row in frame[frame.outcome.eq("SL")].to_dict("records"):
        candle = candles[row["symbol"]]
        lookup = {int(value): index for index, value in enumerate(candle["time"])}
        stop_index = lookup[int(row["exit_time"])-900]
        first = int(row["entry_index"])
        direction = 1 if row["direction"] == "LONG" else -1
        entry = float(row["entry"])
        stop = entry*(1-direction*.01)
        target = entry*(1+direction*float(row["audited_target_pct"])/100)
        last = min(first+63, len(candle["time"])-1)
        double_last = min(first+127, len(candle["time"])-1)
        close = float(candle["close"][stop_index])
        wick = close > stop if direction > 0 else close < stop

        def target_after(end: int) -> bool:
            return any((float(candle["high"][index]) >= target if direction > 0
                        else float(candle["low"][index]) <= target)
                       for index in range(stop_index+1, end+1))

        rows.append({
            "closed_back_inside": wick,
            "target_remaining_hold": target_after(last),
            "target_double_hold": target_after(double_last),
            "double_hold_complete": first+127 < len(candle["time"]),
        })
    path = pd.DataFrame(rows)

    def rate(field: str, subset: pd.DataFrame = path) -> float:
        return round(100*float(subset[field].mean()), 4) if len(subset) else 0.0

    complete = path[path.double_hold_complete]
    return {
        "stops": len(path),
        "closed_back_inside_pct": rate("closed_back_inside"),
        "target_remaining_hold_pct": rate("target_remaining_hold"),
        "double_hold_complete_stops": len(complete),
        "target_double_hold_pct_of_complete": rate("target_double_hold", complete),
    }


def main() -> None:
    raw, sources, candles = source.candidates(MANIFEST)
    entries = guard.apply_guard(raw, -.01)
    fixed = {target: router.resolve(entries, candles, target)
             for target in router.TARGETS}
    delayed = {
        target: pd.DataFrame(close_confirmed(row, candles[row["symbol"]], target)
                             for row in entries.to_dict("records"))
        for target in router.TARGETS
    }
    lookups = {
        "fixed": {target: {identity(row): row for row in frame.to_dict("records")}
                  for target, frame in fixed.items()},
        "delayed": {target: {identity(row): row for row in frame.to_dict("records")}
                    for target, frame in delayed.items()},
    }

    def route(config: dict | None = None, uniform: bool = False) -> pd.DataFrame:
        output = []
        for entry in entries.to_dict("records"):
            hour = int(float(entry["entry_time"])//3600) % 24
            target = (2.0 if float(entry["basket_trend_32"]) <= -.04 else
                      3.0 if 6 <= hour < 12 else 2.5)
            use_delayed = uniform or (config is not None and matches(entry, config))
            mode = "delayed" if use_delayed else "fixed"
            output.append(lookups[mode][target][identity(entry)])
        return guard.accept(pd.DataFrame(output))

    baseline_rows = route()
    baseline = evaluate(baseline_rows)
    uniform = evaluate(route(uniform=True))
    candidates = []
    for direction in ("ALL", "LONG", "SHORT"):
        for feature, thresholds in FEATURES.items():
            for threshold, operator in itertools.product(
                    thresholds, ("at_least", "at_most")):
                config = {"direction": direction, "feature": feature,
                          "threshold": threshold, "operator": operator}
                values = evaluate(route(config))
                candidates.append({
                    "config": config,
                    "july_return_gain_pp": round(
                        values["2026-07"]["return_pct"]
                        - baseline["2026-07"]["return_pct"], 4),
                    "metrics": values,
                })
    candidates.sort(key=lambda item: (
        -item["july_return_gain_pp"],
        json.dumps(item["config"], sort_keys=True)))
    selected = candidates[0]
    selection_pass = bool(
        selected["july_return_gain_pp"] > 0
        and selected["metrics"]["2026-07"]["dd_pct"]
        <= baseline["2026-07"]["dd_pct"]+.25)
    validation_pass = bool(selection_pass
        and selected["metrics"]["2026-08"]["return_pct"]
            > baseline["2026-08"]["return_pct"]
        and selected["metrics"]["2026-08"]["dd_pct"]
            <= baseline["2026-08"]["dd_pct"]+.25)
    test_pass = bool(validation_pass
        and selected["metrics"]["2026-09"]["return_pct"]
            >= baseline["2026-09"]["return_pct"]
        and selected["metrics"]["2026-09"]["dd_pct"]
            <= baseline["2026-09"]["dd_pct"]+.25)
    all_visible_qualified = [item for item in candidates if
        item["july_return_gain_pp"] > 0 and all(
            item["metrics"][month]["return_pct"] >= baseline[month]["return_pct"]
            and item["metrics"][month]["dd_pct"] <= baseline[month]["dd_pct"]+.10
            for month in MONTHS[1:])]
    report = {
        "status": "REJECTED_TEST_AND_UNIFORM_DAMAGE",
        "ready_for_live_orders": False, "orders_possible": False,
        "runtime_change_allowed": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "mechanism": ("confirm 1% stop by candle close; exit next 15m market "
                          "open; retain intrabar 2% emergency stop"),
            "selection": "2026-07", "validation": "2026-08",
            "test": "2026-09", "later_diagnostic": "2026-10",
            "candidate_count": len(candidates),
            "portfolio_gate_replayed": True,
            "position_fraction": "client setting", "orders": "market only",
        },
        "baseline": baseline, "uniform_close_confirmation": uniform,
        "baseline_stop_path": stop_path(baseline_rows, candles),
        "selected_on_july": selected,
        "selection_pass": selection_pass,
        "validation_pass": validation_pass, "test_pass": test_pass,
        "all_visible_qualified_candidates": len(all_visible_qualified),
        "top_july_candidates": candidates[:15],
        "sources": {**sources, str(MANIFEST): digest(MANIFEST),
                    str(Path(__file__).resolve()): digest(Path(__file__).resolve())},
        "limitations": [
            "Public OHLC cannot order hard-stop and target touches inside one candle; hard stop is evaluated first.",
            "The close-confirmed exit assumes execution at the next public candle open.",
            "Venue-native stock X-Perp history begins in late June 2026.",
            "Gross returns exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "baseline_stop_path": report["baseline_stop_path"],
        "uniform_close_confirmation": uniform,
        "selected_on_july": selected,
        "validation_pass": validation_pass, "test_pass": test_pass,
        "all_visible_qualified_candidates": len(all_visible_qualified),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
