"""Causal native stock X-Perp cross-sectional reversal audit.

The entry and exit grid is ranked on three periods ending before October 7.
October 7-16 is reported only after the single top route is frozen.  Signals
use closed 15-minute X-Perp candles and enter the next candle at market.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import stock_cash_xperp_leadlag_oct6_audit as lead


ROOT = Path("reports/audit_2026_09_08")
MANIFEST = ROOT / "stock_xperp_all_equities_oct17" / "manifest.json"
INSTRUMENT_SNAPSHOT = ROOT / "public_xperp_instruments.json"
OUT = ROOT / "stock_xperp_native_cross_section_oct18.json"
ROWS_OUT = ROOT / "stock_xperp_native_cross_section_oct18.csv"
DEVELOPMENT = ("2026-07-05", "2026-08-01")
VALIDATION = ("2026-08-01", "2026-09-01")
CONFIRMATION = ("2026-09-01", "2026-10-07")
FRESH = ("2026-10-07", "2026-10-17")
FULL = ("2026-07-05", "2026-10-17")
PERIOD_DAYS = {"development": 27, "validation": 31,
               "confirmation": 36}
LOOKBACKS = (1, 2, 4, 8)
Z_THRESHOLDS = (1.0, 1.5, 2.0)
MOVE_THRESHOLDS = (.002, .004, .006)
TARGETS = (1.5, 2.0, 3.0)
STOPS = (.75, 1.0)
HOLDS = (16, 32, 64)
EXCLUDED = {"QQQUSDT", "SPYUSDT", "TEST002USDT"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_sources() -> tuple[dict, dict]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    instrument_body = json.loads(
        INSTRUMENT_SNAPSHOT.read_text(encoding="utf-8"))
    expected = {
        item["instId"].split("-")[0] + "USDT"
        for item in instrument_body["data"]
        if item.get("ruleType") == "xperp"
        and item.get("state") == "live"
        and item.get("instCategory") == "3"
    }
    if set(manifest["symbols"]) != expected:
        raise ValueError("manifest is not the complete live equity X-Perp universe")
    candles, sources = {}, {str(MANIFEST): digest(MANIFEST),
                            str(INSTRUMENT_SNAPSHOT): digest(INSTRUMENT_SNAPSHOT)}
    for symbol, item in manifest["symbols"].items():
        if symbol in EXCLUDED or item["status"] not in ("OK", "SHORT_HISTORY"):
            continue
        source = Path(item["path"])
        raw = source.read_bytes()
        found = hashlib.sha256(raw).hexdigest()
        if found != item["sha256"]:
            raise ValueError(f"source hash mismatch: {symbol}")
        candles[symbol] = pickle.loads(raw)
        sources[str(source)] = found
    if len(candles) != 44:
        raise ValueError(f"expected 44 tradeable non-index instruments, got {len(candles)}")
    return candles, sources


def feature_rows(candles: dict, lookback: int) -> pd.DataFrame:
    rows = []
    for symbol, candle in candles.items():
        times = np.asarray(candle["time"], dtype=np.int64)
        close = np.asarray(candle["close"], dtype=float)
        for index in range(lookback, len(times) - 1):
            if times[index + 1] - times[index] != 900:
                continue
            move = close[index] / close[index - lookback] - 1
            rows.append((symbol, index + 1, int(times[index + 1]), move))
    frame = pd.DataFrame(rows, columns=(
        "symbol", "entry_index", "entry_time", "signal_return"))
    grouped = frame.groupby("entry_time", sort=False).signal_return
    frame["universe_n"] = grouped.transform("size")
    frame["cross_mean"] = grouped.transform("mean")
    frame["cross_std"] = grouped.transform("std")
    frame["cross_z"] = (
        (frame.signal_return - frame.cross_mean)
        / frame.cross_std.replace(0, np.nan))
    frame["direction_n"] = -np.sign(frame.signal_return).astype(np.int8)
    return frame.replace([np.inf, -np.inf], np.nan).dropna()


def select_entries(frame: pd.DataFrame, z_threshold: float,
                   move_threshold: float) -> pd.DataFrame:
    return frame[
        frame.cross_z.abs().ge(z_threshold)
        & frame.signal_return.abs().ge(move_threshold)
        & frame.universe_n.ge(10)
    ].copy()


def evaluate_periods(frame: pd.DataFrame) -> dict:
    return {
        "development": lead.evaluate(frame, DEVELOPMENT),
        "validation": lead.evaluate(frame, VALIDATION),
        "confirmation": lead.evaluate(frame, CONFIRMATION),
    }


def normalized_return(value: float, days: int) -> float:
    return (100 * ((1 + value / 100) ** (30 / days) - 1)
            if value > -100 else -100.0)


def candidate_ok(values: dict) -> bool:
    return all(value["n"] >= 100 and value["return_pct"] > 0
               and value["dd_pct"] <= 10
               and value["trades_per_calendar_day"] >= 3
               for value in values.values())


def candidate_score(values: dict) -> float:
    rate = min(normalized_return(values[name]["return_pct"], PERIOD_DAYS[name])
               for name in PERIOD_DAYS)
    drawdown = max(value["dd_pct"] for value in values.values())
    return round(rate - .3 * drawdown, 8)


def public(candidate: dict) -> dict:
    return {key: value for key, value in candidate.items() if key != "_rows"}


def main() -> None:
    candles, sources = load_sources()
    candidates = []
    for lookback in LOOKBACKS:
        features = feature_rows(candles, lookback)
        for z_threshold, move_threshold in itertools.product(
                Z_THRESHOLDS, MOVE_THRESHOLDS):
            entries = select_entries(features, z_threshold, move_threshold)
            for target, stop, hold in itertools.product(TARGETS, STOPS, HOLDS):
                resolved = lead.add_outcomes(entries, candles, target, stop, hold)
                resolved["module"] = "native_xperp_cross_section_reversal"
                values = evaluate_periods(resolved)
                if not candidate_ok(values):
                    continue
                candidates.append({
                    "lookback_bars": lookback,
                    "z_threshold": z_threshold,
                    "minimum_move_pct": 100 * move_threshold,
                    "target_pct": target, "stop_pct": stop,
                    "hold_bars": hold,
                    "score": candidate_score(values),
                    "periods": values, "_rows": resolved,
                })
        print(json.dumps({"lookback": lookback,
                          "qualified": len(candidates)}), flush=True)
    candidates.sort(key=lambda item: (
        -item["score"], item["lookback_bars"], item["z_threshold"],
        item["minimum_move_pct"], item["target_pct"],
        item["stop_pct"], item["hold_bars"]))
    if not candidates:
        raise ValueError("no native cross-sectional candidate survived")
    selected = candidates[0]
    fresh = lead.evaluate(selected["_rows"], FRESH)
    full = lead.evaluate(selected["_rows"], FULL)
    full_rows = lead.period_rows(selected["_rows"], FULL)
    accepted_rows = lead.accepted(full_rows.to_dict("records"))
    pd.DataFrame(accepted_rows).to_csv(ROWS_OUT, index=False)
    projected = (100 * ((1 + fresh["return_pct"] / 100)
                        ** (22 / max(fresh["active_days"], 1)) - 1)
                 if fresh["return_pct"] > -100 else -100)
    checks = {
        "selected_without_fresh": True,
        "fresh_positive": fresh["return_pct"] > 0,
        "fresh_no_negative_month": fresh["negative_months"] == 0,
        "fresh_at_least_3_trades_per_day":
            fresh["trades_per_calendar_day"] >= 3,
        "fresh_drawdown_at_most_5pct": fresh["dd_pct"] <= 5,
        "fresh_return_at_least_30pct": fresh["return_pct"] >= 30,
    }
    report = {
        "status": ("NATIVE_XPERP_OBJECTIVE_CANDIDATE"
                   if all(checks.values()) else "REJECTED_MONTHLY_OBJECTIVE"),
        "ready_for_live_orders": False,
        "protocol": {
            "universe": "OKX live instCategory=3 equity X-Perps",
            "tradeable_symbols": len(candles),
            "development": DEVELOPMENT, "validation": VALIDATION,
            "confirmation": CONFIRMATION,
            "untouched_fresh_reveal": FRESH,
            "entry": "next 15-minute X-Perp market open",
            "direction": "reversal against cross-sectional extreme",
            "orders": "market only",
            "money": "3% current-equity margin, fixed 10x gross",
        },
        "candidate_count": len(candidates),
        "selected": public(selected),
        "top20": [public(item) for item in candidates[:20]],
        "fresh_oct7_oct16": fresh,
        "full_jul5_oct16": full,
        "fresh_22_session_extrapolation_pct": round(projected, 4),
        "objective_checks": checks,
        "sources": {**sources, str(Path(__file__)): digest(Path(__file__))},
        "limitations": [
            "The X-Perps were listed in 2026, so native older-year history does not exist.",
            "The fresh reveal contains ten calendar days, not a complete month.",
            "The 22-session extrapolation is context and does not prove a monthly result.",
            "Gross returns exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "selected": public(selected),
        "fresh": fresh, "full": full,
        "projected_22_sessions": projected,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
