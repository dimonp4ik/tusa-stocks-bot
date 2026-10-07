"""Chronological literal 0.5% cash-session model on deep stock history."""
from __future__ import annotations

import hashlib
import heapq
import json
import math
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from filter_lab import gate
from stock_session_family_lab import (
    attach_intraday_market_context, attach_market_context, candidate_records,
    market_context_by_day, market_intraday_context_by_time,
)
from stock_walk_forward_filter_lab import CATEGORICAL, NUMERIC, make_model


ROOT = Path("reports/audit_2026_09_08")
MANIFEST = ROOT / "stock_extended_history_manifest.json"
OUT = ROOT / "stock_cash_literal_move_hgb_oct7.json"
ROWS_OUT = ROOT / "stock_cash_literal_move_hgb_oct7.pkl"
TARGET_PCT = .5
STOP_PCT = 1.0
MARGIN_FRACTION = .03
LEVERAGE = 10.0
QUANTILES = (.70, .75, .80, .85, .90, .925, .95, .96, .97, .98, .985, .99)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load() -> tuple[dict, dict]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    candles, sources = {}, {str(MANIFEST): digest(MANIFEST)}
    for symbol, item in manifest["symbols"].items():
        path = Path(item["path"])
        raw = path.read_bytes()
        found = hashlib.sha256(raw).hexdigest()
        if found != item["sha256"]:
            raise ValueError(f"stock history hash mismatch: {symbol}")
        candle = pickle.loads(raw)
        if len(candle["time"]) != int(item["rows"]):
            raise ValueError(f"stock history row mismatch: {symbol}")
        candles[symbol] = candle
        sources[str(path)] = found
    return candles, sources


def literal_rows(candles: dict, symbol: str) -> list[dict]:
    output = []
    target_move = TARGET_PCT / 100.0
    stop_move = STOP_PCT / 100.0
    for item in candidate_records(candles):
        first, last = int(item["entry_bar"]), int(item["session_end"])
        entry = float(candles["open"][first])
        direction = int(item["direction_int"])
        target = entry * (1 + direction * target_move)
        stop = entry * (1 - direction * stop_move)
        value = 0.0
        reason = 2
        target_hit = False
        exit_bar = first
        for index in range(first, last + 1):
            exit_bar = index
            stop_hit = (float(candles["low"][index]) <= stop if direction > 0
                        else float(candles["high"][index]) >= stop)
            take_hit = (float(candles["high"][index]) >= target if direction > 0
                        else float(candles["low"][index]) <= target)
            if stop_hit:  # conservative same-bar ambiguity
                opening = float(candles["open"][index])
                fill = min(opening, stop) if direction > 0 else max(opening, stop)
                value = direction * (fill / entry - 1)
                reason = 0
                break
            if take_hit:
                value = target_move
                reason = 1
                target_hit = True
                break
        else:
            value = direction * (float(candles["close"][last]) / entry - 1)
            value = max(-stop_move, min(target_move, value))
        entry_time = int(candles["time"][first])
        stamp = datetime.fromtimestamp(entry_time, timezone.utc)
        output.append({
            "symbol": symbol, "family": item["family"],
            "direction": "LONG" if direction > 0 else "SHORT",
            "entry_time": entry_time,
            "exit_time": int(candles["time"][exit_bar]) + 900,
            "entry_bar": first, "session_end": last,
            "entry": entry, "signed_return": value,
            "target_hit": target_hit, "exit_reason": reason,
            "outcome": "SL" if value < 0 else "TP", "net_r": value / stop_move,
            "risk_pct": stop_move, "size_mult": 1.0,
            "year": stamp.year, "month": stamp.strftime("%Y-%m"),
            **{key: value for key, value in item.items()
               if key not in {"family", "direction_int", "signal_bar",
                              "entry_bar", "session_end"}},
        })
    return output


def build(candles: dict) -> pd.DataFrame:
    rows = []
    for symbol, candle in candles.items():
        rows.extend(literal_rows(candle, symbol))
    if "QQQUSDT" in candles:
        attach_market_context(rows, market_context_by_day(candles["QQQUSDT"]))
        attach_intraday_market_context(
            rows, market_intraday_context_by_time(candles["QQQUSDT"], "qqq"), "qqq")
    if "SPYUSDT" in candles:
        attach_intraday_market_context(
            rows, market_intraday_context_by_time(candles["SPYUSDT"], "spy"), "spy")
    frame = pd.DataFrame(rows)
    frame["won"] = frame.target_hit.astype(int)
    return frame


def account(rows: list[dict], *, already_gated: bool = False) -> dict:
    rows = list(rows)
    if any(float(right["entry_time"]) < float(left["entry_time"])
           for left, right in zip(rows, rows[1:])):
        raise ValueError("account rows must be ordered by entry time")
    if not already_gated:
        rows = gate(rows, 4)
    if not rows:
        return {"n": 0, "return_pct": 0.0, "dd_pct": 0.0,
                "literal_target_hit_rate": 0.0, "negative_months": 0,
                "trades_per_calendar_day": 0.0}
    equity = peak = 1.0
    maximum_drawdown = 0.0
    pending = []
    completed = []
    serial = 0

    def realize(until: float | None) -> None:
        nonlocal equity, peak, maximum_drawdown
        while pending and (until is None or pending[0][0] < until):
            exit_time, _, pnl, row = heapq.heappop(pending)
            equity += pnl
            peak = max(peak, equity)
            maximum_drawdown = max(maximum_drawdown, (peak - equity) / peak)
            completed.append((exit_time, pnl, row))

    for row in rows:
        realize(float(row["entry_time"]))
        pnl = equity * MARGIN_FRACTION * LEVERAGE * float(row["signed_return"])
        serial += 1
        heapq.heappush(pending, (float(row["exit_time"]), serial, pnl, row))
    realize(None)
    monthly_start, monthly_end = {}, {}
    ledger = 1.0
    streak = max_streak = 0
    for exit_time, pnl, row in sorted(
            completed, key=lambda item: (
                item[0], float(item[2]["entry_time"]), str(item[2]["symbol"]))):
        month = datetime.fromtimestamp(exit_time, timezone.utc).strftime("%Y-%m")
        monthly_start.setdefault(month, ledger)
        ledger += pnl
        monthly_end[month] = ledger
        streak = 0 if row["target_hit"] else streak + 1
        max_streak = max(max_streak, streak)
    monthly = {month: round(100 * (monthly_end[month] / monthly_start[month] - 1), 4)
               for month in sorted(monthly_start)}
    first = datetime.fromtimestamp(min(row["entry_time"] for row in rows), timezone.utc)
    last = datetime.fromtimestamp(max(row["entry_time"] for row in rows), timezone.utc)
    days = max(1, (last.date() - first.date()).days + 1)
    hits = sum(bool(row["target_hit"]) for row in rows)
    return {
        "n": len(rows), "return_pct": round(100 * (equity - 1), 4),
        "dd_pct": round(100 * maximum_drawdown, 4),
        "literal_target_hits": hits,
        "literal_target_hit_rate": round(100 * hits / len(rows), 4),
        "stops": sum(int(row["exit_reason"]) == 0 for row in rows),
        "timeouts": sum(int(row["exit_reason"]) == 2 for row in rows),
        "max_non_target_streak": max_streak,
        "negative_months": sum(value < 0 for value in monthly.values()),
        "months": len(monthly), "months_ge_30pct": sum(value >= 30 for value in monthly.values()),
        "minimum_month_pct": round(min(monthly.values()), 4),
        "median_month_pct": round(float(np.median(list(monthly.values()))), 4),
        "monthly_return_pct": monthly,
        "trades_per_calendar_day": round(len(rows) / days, 3),
        "trades_per_active_day": round(len(rows) / len({
            datetime.fromtimestamp(row["entry_time"], timezone.utc).date() for row in rows}), 3),
    }


def score(frame: pd.DataFrame, probability: np.ndarray,
          years: tuple[int, ...], threshold: float) -> dict:
    mask = frame.year.isin(years).to_numpy() & (probability >= threshold)
    selected = frame.loc[mask].copy()
    selected["quality"] = probability[mask]
    return account(selected.sort_values(
        ["entry_time", "quality", "symbol"], ascending=[True, False, True])
        .to_dict("records"))


def selection_ok(value: dict) -> bool:
    return bool(value["n"] >= 400 and value["return_pct"] > 0
                and value["literal_target_hit_rate"] >= 66
                and value["dd_pct"] <= 8
                and value["trades_per_calendar_day"] >= 1.5)


def validation_ok(value: dict) -> bool:
    return bool(value["n"] >= 300 and value["return_pct"] > 0
                and value["literal_target_hit_rate"] >= 64
                and value["dd_pct"] <= 8
                and value["trades_per_calendar_day"] >= 1.2)


def main() -> None:
    candles, sources = load()
    frame = build(candles)
    features = list(NUMERIC + CATEGORICAL)
    fit = frame.year.between(2017, 2021)
    model = make_model()
    model.fit(frame.loc[fit, features], frame.loc[fit, "won"])
    probability = model.predict_proba(frame[features])[:, 1]
    selection_values = probability[frame.year.eq(2022).to_numpy()]
    thresholds = sorted(set(
        [float(np.quantile(selection_values, quantile)) for quantile in QUANTILES]
        + list(np.linspace(.50, .90, 41))))
    frontier = []
    for threshold in thresholds:
        result = score(frame, probability, (2022,), threshold)
        frontier.append({"threshold": threshold, "selection_2022": result})
    qualified = [item for item in frontier if selection_ok(item["selection_2022"])]
    qualified.sort(key=lambda item: (
        item["selection_2022"]["return_pct"]
        - 2 * item["selection_2022"]["dd_pct"],
        item["selection_2022"]["literal_target_hit_rate"]), reverse=True)
    frozen = qualified[0] if qualified else None
    v23 = test24 = later = None
    status = "REJECTED_SELECTION"
    if frozen:
        v23 = score(frame, probability, (2023,), frozen["threshold"])
        if validation_ok(v23):
            test24 = score(frame, probability, (2024,), frozen["threshold"])
            if validation_ok(test24):
                later = {str(year): score(frame, probability, (year,), frozen["threshold"])
                         for year in (2025, 2026)}
                status = ("STOCK_CASH_LITERAL_CANDIDATE"
                          if all(validation_ok(value) for value in later.values())
                          else "REJECTED_LATER")
                if status == "STOCK_CASH_LITERAL_CANDIDATE":
                    selected = frame[probability >= frozen["threshold"]].copy()
                    selected["quality"] = probability[probability >= frozen["threshold"]]
                    selected.to_pickle(ROWS_OUT)
            else:
                status = "REJECTED_2024"
        else:
            status = "REJECTED_2023"
    report = {
        "status": status,
        "protocol": {"venue": "Dukascopy US cash-session CFD",
                     "fit": [2017, 2018, 2019, 2020, 2021], "selection": 2022,
                     "validation": 2023, "untouched_test": 2024,
                     "hidden_if_test_passes": [2025, 2026],
                     "entry": "next 15-minute market open", "target_pct": TARGET_PCT,
                     "stop_pct": STOP_PCT, "hold": "same cash session only",
                     "money": "3% current equity margin, fixed 10x gross"},
        "rows": len(frame), "fit_literal_prevalence_pct": round(
            100 * float(frame.loc[fit, "won"].mean()), 4),
        "selection_candidates": len(frontier), "qualified_selection": len(qualified),
        "frontier": frontier, "frozen_before_2023": frozen,
        "validation_2023": v23, "test_2024": test24, "later": later,
        "sources": {**sources, str(Path(__file__)): digest(Path(__file__))},
        "limitations": [
            "Dukascopy equity history contains the regular cash session, not genuine premarket.",
            "A passing deep-history result still requires direct OKX X-Perp transfer.",
        ],
    }
    if status != "STOCK_CASH_LITERAL_CANDIDATE" and ROWS_OUT.exists():
        ROWS_OUT.unlink()
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "status", "rows", "fit_literal_prevalence_pct", "qualified_selection",
        "frozen_before_2023", "validation_2023", "test_2024", "later")},
        indent=2), flush=True)


if __name__ == "__main__":
    main()
