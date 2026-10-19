"""Causal backing-level guard for the native stock X-Perp route.

A reversal entry is rejected when its next market open has already crossed
the prior backing level: below the prior low for a long or above the prior
high for a short.  July-August select one fixed lookback, September validates
it, and the partial October slice is revealed only after validation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import stock_xperp_native_cross_section_oct18_audit as parent
import stock_xperp_native_economics_oct18_audit as economics
import stock_xperp_native_growth_profile_oct18_audit as growth
import stock_xperp_native_quality_guard_oct18_audit as quality


OUT = parent.ROOT / "stock_xperp_backing_level_guard_oct20.json"
ROWS_OUT = parent.ROOT / "stock_xperp_backing_level_guard_oct20.csv"
LOOKBACKS = (16, 32, 64, 128)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def add_backing_distance(frame: pd.DataFrame, candles: dict,
                         lookback: int) -> pd.DataFrame:
    values = []
    for row in frame.itertuples(index=False):
        candle = candles[str(row.symbol)]
        index = int(row.entry_index)
        start = max(0, index - lookback)
        if start == index:
            values.append(np.nan)
            continue
        entry = float(candle["open"][index])
        prior_low = float(np.min(np.asarray(
            candle["low"][start:index], dtype=float)))
        prior_high = float(np.max(np.asarray(
            candle["high"][start:index], dtype=float)))
        distance = (entry / prior_low - 1
                    if int(row.direction_n) > 0
                    else prior_high / entry - 1)
        values.append(distance)
    result = frame.copy()
    result["backing_level_distance"] = values
    result["backing_level_lookback_bars"] = lookback
    return result


def guarded(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame.backing_level_distance >= 0].copy()


def evaluate(frame: pd.DataFrame, bounds: tuple[str, str], rank_name: str,
             loss_cap: int, margin: float) -> dict:
    rows = quality.gated(frame, bounds, rank_name, loss_cap)
    value = economics.simulate(rows, margin)
    if rows:
        days = len({pd.Timestamp(float(row["entry_time"]), unit="s",
                                 tz="UTC").date() for row in rows})
        value["active_market_days"] = days
        value["trades_per_market_day"] = round(len(rows) / days, 4)
        value["profitable_exit_rate_pct"] = round(
            100 * sum(float(row["signed_return"]) > 0 for row in rows)
            / len(rows), 4)
        value["literal_tp_rate_pct"] = round(
            100 * sum(int(row["outcome_code"]) == 1 for row in rows)
            / len(rows), 4)
        streak = maximum = 0
        for row in sorted(rows, key=lambda item: (
                float(item["exit_time"]), float(item["entry_time"]),
                str(item["symbol"]))):
            streak = streak + 1 if float(row["signed_return"]) <= 0 else 0
            maximum = max(maximum, streak)
        value["max_loss_streak"] = maximum
        value["concentration"] = economics.diagnostics(pd.DataFrame(rows))
    else:
        value.update({"active_market_days": 0,
                      "trades_per_market_day": 0.0,
                      "profitable_exit_rate_pct": 0.0,
                      "literal_tp_rate_pct": 0.0,
                      "max_loss_streak": 0,
                      "concentration": {}})
    return value


def period_ok(value: dict) -> bool:
    return bool(value["return_pct"] >= 30
                and value["negative_months"] == 0
                and value["dd_pct"] <= 8
                and value["trades_per_market_day"] >= 3)


def main() -> None:
    raw, parameters, sources = quality.prepare()
    candles, candle_sources = parent.load_sources()
    sources.update(candle_sources)
    frozen_quality = json.loads(quality.OUT.read_text(encoding="utf-8"))
    rank_name = str(frozen_quality["selection"]["selected_ranker"])
    loss_cap = int(frozen_quality["selection"]["selected_loss_streak_cap"])

    candidates = []
    for lookback in LOOKBACKS:
        found = guarded(add_backing_distance(raw, candles, lookback))
        july = evaluate(found, growth.COMPLETE_MONTHS["2026-07"],
                        rank_name, loss_cap, .05)
        august = evaluate(found, growth.COMPLETE_MONTHS["2026-08"],
                          rank_name, loss_cap, .05)
        candidates.append({
            "lookback_bars": lookback,
            "selection_months": {"2026-07": july, "2026-08": august},
            "selection_pass": period_ok(july) and period_ok(august),
            "selection_floor_pct": min(july["return_pct"],
                                         august["return_pct"]),
            "selection_max_dd_pct": max(july["dd_pct"], august["dd_pct"]),
            "_frame": found,
        })
    candidates.sort(key=lambda item: (
        item["selection_pass"], item["selection_floor_pct"],
        -item["selection_max_dd_pct"], -item["lookback_bars"]),
        reverse=True)
    selected = candidates[0]
    selected_frame = selected["_frame"]
    september = evaluate(
        selected_frame, growth.COMPLETE_MONTHS["2026-09"],
        rank_name, loss_cap, .05)
    validation_pass = bool(selected["selection_pass"]
                           and period_ok(september))

    periods = {**growth.COMPLETE_MONTHS,
               "fresh": parent.FRESH, "full": parent.FULL}
    scenarios = {
        f"margin_{round(100 * margin)}pct": {
            name: evaluate(selected_frame, bounds, rank_name,
                           loss_cap, margin)
            for name, bounds in periods.items()
        } for margin in (.03, .05)
    }
    full_rows = quality.gated(
        selected_frame, parent.FULL, rank_name, loss_cap)
    pd.DataFrame(full_rows).to_csv(ROWS_OUT, index=False)
    five = scenarios["margin_5pct"]
    objective_complete = bool(validation_pass and all(
        period_ok(five[month]) for month in growth.COMPLETE_MONTHS))
    report = {
        "status": ("FIVE_PCT_BACKING_LEVEL_CONFIRMED_FRESH_INCOMPLETE"
                   if objective_complete else "REJECTED_OBJECTIVE"),
        "ready_for_live_orders": False,
        "hypothesis": (
            "mean-reversion entries fail when the next market open has "
            "already crossed the prior backing level"),
        "protocol": {
            "candidate_lookbacks_bars": list(LOOKBACKS),
            "selection": ["2026-07", "2026-08"],
            "validation": "2026-09",
            "fresh_incomplete": list(parent.FRESH),
            "rule": (
                "long: entry open >= prior low; short: entry open <= prior high"),
            "feature_cutoff": "bars strictly before the entry bar",
            "base_ranker": rank_name,
            "base_daily_loss_streak_cap": loss_cap,
            "entry": "next 15-minute market open",
            "orders": "market only",
            "money": "client 3% or 5% current-equity margin; fixed 10x",
        },
        "base_parameters": parameters,
        "candidates": [{key: value for key, value in item.items()
                        if not key.startswith("_")} for item in candidates],
        "selected_lookback_bars": selected["lookback_bars"],
        "validation": september,
        "validation_pass": validation_pass,
        "margin_scenarios": scenarios,
        "objective_complete_months_pass": objective_complete,
        "sources": {
            **sources,
            str(quality.OUT): digest(quality.OUT),
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "The frozen base route parameters had already used September.",
            "October contains only ten calendar days and is diagnostic.",
            "Native stock X-Perp history begins in July 2026.",
            "Five-percent margin is a client setting and increases risk.",
            "Gross returns exclude fees at user request.",
            "No backtest guarantees a future monthly return.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True),
                   encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "selected_lookback_bars": report["selected_lookback_bars"],
        "selection": selected["selection_months"],
        "validation": september,
        "scenarios": scenarios,
        "objective_complete_months_pass": objective_complete,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
