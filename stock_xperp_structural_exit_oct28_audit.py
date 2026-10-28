"""Causal structural stop/target audit for frozen stock X-Perp entries.

Every level is built from candles closed before the next-bar market entry.
The audit selects on July, validates on August, tests on September and only
then reports October.  It cannot enable live orders; a winner would require a
new append-only shadow sample after this snapshot.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.stock_xperp_native_shadow import AUDITED_SYMBOLS
from stock_module_cluster_audit import portfolio_gate
import stock_stop_cause_oct11_audit as structure
import stock_xperp_basket_trend_forward_oct28_renewed_audit as forward
import stock_xperp_native_growth_forward_monitor as monitor
import stock_xperp_native_quality_guard_oct18_audit as quality
import stock_xperp_session_target_router_oct27_audit as base


ROOT = Path("reports/audit_2026_09_08")
MANIFEST = ROOT / "stock_xperp_forward_oct28_renewed" / "manifest.json"
OUT = ROOT / "stock_xperp_structural_exit_oct28.json"
ROWS_OUT = ROOT / "stock_xperp_structural_exit_oct28.csv"
MONTHS = ("2026-07", "2026-08", "2026-09", "2026-10")
BASE_TARGET = .02
BASE_STOP = .01
HOLD_BARS = 64


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(row: dict) -> tuple[str, int, str]:
    direction = row.get("direction")
    if direction is None:
        direction = "LONG" if int(row["direction_n"]) > 0 else "SHORT"
    return str(row["symbol"]), int(row["entry_time"]), str(direction)


def load_inputs() -> tuple[pd.DataFrame, dict, dict]:
    candles, sources, _ = monitor.load_manifest(MANIFEST)
    candles = {symbol: candles[symbol] for symbol in AUDITED_SYMBOLS}
    entries = base.prepare_entries(candles)
    entries.sort_values(["entry_time", "symbol", "direction_n"], inplace=True)
    entries.reset_index(drop=True, inplace=True)
    return entries, candles, sources


def structural_geometry(row: dict, candle: dict, config: dict,
                        cached_levels: tuple | None = None) -> dict | None:
    index = int(row["entry_index"])
    entry = float(candle["open"][index])
    direction = int(row["direction_n"])
    if cached_levels is None:
        highs, lows = structure.swing_levels(candle, index, int(config["lookback"]))
        adverse = structure.nearest(lows if direction > 0 else highs,
                                    entry, direction < 0)
        favorable = structure.nearest(highs if direction > 0 else lows,
                                      entry, direction > 0)
    else:
        adverse, favorable = cached_levels
    atr = entry * float(row["atr14_fraction"])
    buffer = float(config["buffer_atr"]) * atr
    stop = None if adverse is None else adverse - direction * buffer
    target = None if favorable is None else favorable - direction * buffer
    stop_fraction = (None if stop is None else
                     -direction * (stop / entry - 1))
    target_fraction = (None if target is None else
                       direction * (target / entry - 1))

    stop_valid = bool(
        stop_fraction is not None
        and float(config["stop_min"]) <= stop_fraction <= float(config["stop_max"]))
    target_valid = bool(
        target_fraction is not None
        and float(config["target_min"]) <= target_fraction <= float(config["target_max"]))
    mode = str(config["mode"])
    uses_stop = mode in ("both", "stop_only")
    uses_target = mode in ("both", "target_only")
    if uses_stop and not stop_valid:
        if config["fallback"] == "reject":
            return None
        stop, stop_fraction = entry * (1 - direction * BASE_STOP), BASE_STOP
    elif not uses_stop:
        stop, stop_fraction = entry * (1 - direction * BASE_STOP), BASE_STOP
    if uses_target and not target_valid:
        if config["fallback"] == "reject":
            return None
        target, target_fraction = entry * (1 + direction * BASE_TARGET), BASE_TARGET
    elif not uses_target:
        target, target_fraction = entry * (1 + direction * BASE_TARGET), BASE_TARGET
    if target_fraction / stop_fraction < float(config["minimum_rr"]):
        if config["fallback"] == "reject":
            return None
        stop = entry * (1 - direction * BASE_STOP)
        target = entry * (1 + direction * BASE_TARGET)
        stop_fraction, target_fraction = BASE_STOP, BASE_TARGET
    return {
        "entry": entry, "stop_price": float(stop), "target_price": float(target),
        "stop_fraction": float(stop_fraction),
        "target_fraction": float(target_fraction),
        "structural_stop_valid": stop_valid,
        "structural_target_valid": target_valid,
    }


def resolve_geometry(row: dict, candle: dict, geometry: dict) -> dict:
    result = dict(row)
    open_ = np.asarray(candle["open"], dtype=float)
    high = np.asarray(candle["high"], dtype=float)
    low = np.asarray(candle["low"], dtype=float)
    close = np.asarray(candle["close"], dtype=float)
    times = np.asarray(candle["time"], dtype=np.int64)
    index = int(row["entry_index"])
    direction = int(row["direction_n"])
    entry = float(geometry["entry"])
    stop, target = float(geometry["stop_price"]), float(geometry["target_price"])
    stop_fraction = float(geometry["stop_fraction"])
    target_fraction = float(geometry["target_fraction"])
    required_last = index + HOLD_BARS - 1
    available_last = min(required_last, len(times) - 1)
    signed_return = outcome_code = None
    exit_index = available_last
    for cursor in range(index, available_last + 1):
        if cursor > index and times[cursor] - times[cursor - 1] != 900:
            exit_index = cursor - 1
            signed_return = direction * (close[exit_index] / entry - 1)
            outcome_code = 0
            break
        opening = float(open_[cursor])
        if direction > 0:
            if opening <= stop:
                exit_index, signed_return, outcome_code = cursor, opening / entry - 1, -1
                break
            if opening >= target:
                exit_index, signed_return, outcome_code = cursor, opening / entry - 1, 1
                break
            if low[cursor] <= stop:  # conservative stop-first ambiguity
                exit_index, signed_return, outcome_code = cursor, -stop_fraction, -1
                break
            if high[cursor] >= target:
                exit_index, signed_return, outcome_code = cursor, target_fraction, 1
                break
        else:
            if opening >= stop:
                exit_index, signed_return, outcome_code = cursor, 1 - opening / entry, -1
                break
            if opening <= target:
                exit_index, signed_return, outcome_code = cursor, 1 - opening / entry, 1
                break
            if high[cursor] >= stop:
                exit_index, signed_return, outcome_code = cursor, -stop_fraction, -1
                break
            if low[cursor] <= target:
                exit_index, signed_return, outcome_code = cursor, target_fraction, 1
                break
    if signed_return is None and available_last >= required_last:
        exit_index = required_last
        signed_return = direction * (close[exit_index] / entry - 1)
        outcome_code = 0
    pending = signed_return is None
    result.update(geometry)
    result.update({
        "target_pct": 100 * target_fraction, "stop_pct": 100 * stop_fraction,
        "hold_bars": HOLD_BARS, "direction": "LONG" if direction > 0 else "SHORT",
        "signed_return": None if pending else float(signed_return),
        "outcome_code": outcome_code,
        "outcome": ("PENDING" if pending else "TP" if outcome_code == 1
                    else "SL" if outcome_code == -1 else "TIME"),
        "exit_time": (int(times[index]) + HOLD_BARS * 900 if pending
                      else int(times[exit_index]) + 900),
        "risk_pct": stop_fraction,
        "net_r": None if pending else float(signed_return / stop_fraction),
        "module": "native_xperp_structural_exit", "module_priority": 0,
    })
    return result


def resolve(entries: pd.DataFrame, candles: dict, config: dict,
            level_cache: dict | None = None) -> pd.DataFrame:
    rows = []
    for row in entries.to_dict("records"):
        cached = None if level_cache is None else level_cache[
            (identity(row), int(config["lookback"]))]
        geometry = structural_geometry(row, candles[row["symbol"]], config, cached)
        if geometry is not None:
            rows.append(resolve_geometry(row, candles[row["symbol"]], geometry))
    return pd.DataFrame(rows)


def gated(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    accepted = portfolio_gate(
        frame.to_dict("records"), None, rolling_hour_cap=4,
        rank_key=quality.ranker("z_high"), loss_streak_cap=2)
    return pd.DataFrame(accepted, columns=frame.columns)


def month_rows(frame: pd.DataFrame, month: str) -> pd.DataFrame:
    stamp = pd.to_datetime(frame.entry_time, unit="s", utc=True).dt.strftime("%Y-%m")
    return frame[stamp.eq(month)]


def evaluate(frame: pd.DataFrame) -> dict:
    return {month: {
        "margin_3pct": forward.metrics(month_rows(frame, month), .03),
        "margin_5pct": forward.metrics(month_rows(frame, month), .05),
    } for month in MONTHS}


def baseline(entries: pd.DataFrame, candles: dict) -> pd.DataFrame:
    rows = pd.DataFrame(monitor.resolve_entries(entries, candles, {
        "target_pct": 2.0, "stop_pct": 1.0, "hold_bars": HOLD_BARS}))
    rows["module"] = "native_xperp_fixed_exit"
    rows["module_priority"] = 0
    return gated(rows)


def eligible_july(values: dict, baseline_values: dict) -> bool:
    value = values["2026-07"]["margin_3pct"]
    reference = baseline_values["2026-07"]["margin_3pct"]
    return bool(value["return_pct"] >= reference["return_pct"]
                and value["dd_pct"] <= reference["dd_pct"] + .25
                and value["trades_per_active_day"] >= 3
                and value["negative_months"] == 0)


def score_july(values: dict) -> float:
    value = values["2026-07"]["margin_3pct"]
    return round(value["return_pct"] - .75 * value["dd_pct"], 8)


def dominates(month: str, values: dict, baseline_values: dict) -> bool:
    value = values[month]["margin_3pct"]
    reference = baseline_values[month]["margin_3pct"]
    return bool(value["return_pct"] >= reference["return_pct"]
                and value["dd_pct"] <= reference["dd_pct"] + .25
                and value["trades_per_active_day"] >= 3
                and value["negative_months"] == 0)


def main() -> None:
    entries, candles, sources = load_inputs()
    base_rows = baseline(entries, candles)
    base_values = evaluate(base_rows)
    level_cache = {}
    for row in entries.to_dict("records"):
        candle = candles[row["symbol"]]
        index = int(row["entry_index"])
        entry = float(candle["open"][index])
        direction = int(row["direction_n"])
        for lookback in (64, 96):
            highs, lows = structure.swing_levels(candle, index, lookback)
            level_cache[(identity(row), lookback)] = (
                structure.nearest(lows if direction > 0 else highs,
                                  entry, direction < 0),
                structure.nearest(highs if direction > 0 else lows,
                                  entry, direction > 0))
    print(json.dumps({"causal_level_cache": len(level_cache)}), flush=True)
    configs = []
    # Twelve mechanism-first routes keep the search small enough to avoid a
    # parameter-mining result: two causal horizons, two buffers, and either a
    # structural stop alone or structural stop plus structural target.
    iterator = itertools.product(
        (64, 96), (.1, .2), ((.005, .015),),
        ((.01, .03),), (1.5,), ("fixed",),
        ("both", "stop_only", "target_only"))
    for lookback, buffer_atr, stop_bounds, target_bounds, rr, fallback, mode in iterator:
        configs.append({
            "lookback": lookback, "buffer_atr": buffer_atr,
            "stop_min": stop_bounds[0], "stop_max": stop_bounds[1],
            "target_min": target_bounds[0], "target_max": target_bounds[1],
            "minimum_rr": rr, "fallback": fallback, "mode": mode,
        })
    candidates = []
    selected_rows = {}
    for number, config in enumerate(configs):
        rows = gated(resolve(entries, candles, config, level_cache))
        values = evaluate(rows)
        key = json.dumps(config, sort_keys=True)
        selected_rows[key] = rows
        candidates.append({
            "config": config, "july_eligible": eligible_july(values, base_values),
            "july_score": score_july(values), "values": values,
            "accepted_rows": len(rows),
        })
        print(json.dumps({"evaluated": number + 1, "total": len(configs)}), flush=True)
    eligible = [item for item in candidates if item["july_eligible"]]
    eligible.sort(key=lambda item: (-item["july_score"],
                                   json.dumps(item["config"], sort_keys=True)))
    selected = eligible[0] if eligible else max(candidates, key=lambda item: item["july_score"])
    selected_key = json.dumps(selected["config"], sort_keys=True)
    rows = selected_rows[selected_key]
    validation_pass = dominates("2026-08", selected["values"], base_values)
    test_pass = validation_pass and dominates("2026-09", selected["values"], base_values)
    confirmation_pass = test_pass and dominates("2026-10", selected["values"], base_values)
    if confirmation_pass:
        rows.to_csv(ROWS_OUT, index=False)
    public_top = [{key: value for key, value in item.items() if key != "values"}
                  | {"july": item["values"]["2026-07"],
                     "august": item["values"]["2026-08"]}
                  for item in eligible[:10]]
    result = {
        "status": ("SHADOW_CANDIDATE_REQUIRES_GENUINELY_NEW_FORWARD"
                   if confirmation_pass else "REJECTED_SEQUENTIAL_VALIDATION"),
        "ready_for_live_orders": False, "orders_possible": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "selection": "2026-07", "validation": "2026-08",
            "test": "2026-09", "confirmation": "2026-10",
            "entry": "unchanged next-15m market entry", "orders": "market only",
            "same_bar_rule": "stop first", "hold_bars": HOLD_BARS,
            "levels": "confirmed five-bar swings closed before entry",
            "candidate_count": len(configs),
        },
        "baseline": base_values,
        "selected": selected,
        "validation_pass": validation_pass, "test_pass": test_pass,
        "confirmation_pass": confirmation_pass,
        "selection_eligible": len(eligible), "selection_top10": public_top,
        "all_candidate_summaries": [{
            "config": item["config"], "july_eligible": item["july_eligible"],
            "july_score": item["july_score"], "accepted_rows": item["accepted_rows"],
            "margin_3pct": {month: item["values"][month]["margin_3pct"]
                            for month in MONTHS},
        } for item in candidates],
        "integrity": {
            "entries": len(entries), "symbols": len(candles),
            "future_bars_used_for_levels": 0,
        },
        "sources": {**sources, str(MANIFEST): digest(MANIFEST),
                    str(Path(__file__).resolve()): digest(Path(__file__).resolve())},
        "limitations": [
            "All four months were already observed; even a pass remains shadow-only.",
            "A five-bar swing is a causal price-structure proxy, not an order-book level.",
            "Gross returns exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": result["status"], "selected": selected,
        "validation_pass": validation_pass, "test_pass": test_pass,
        "confirmation_pass": confirmation_pass, "baseline": base_values,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
