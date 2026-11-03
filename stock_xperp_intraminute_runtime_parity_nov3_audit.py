"""Audit native X-Perp publication delay and post-publication monitoring.

The strategy replay enters at the next 15-minute open, while the deployed
scanner previously waited about 60 seconds and then monitored the entire
15-minute entry candle.  This diagnostic uses public one-minute X-Perp bars
to measure the delayed price and resolve only bars observable after the paper
signal could exist.  It changes lifecycle accounting and scan timing; it does
not select a new trading rule or authorize real orders.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import pickle
from pathlib import Path

import pandas as pd

from src.backtest_integrity import simulate_exit


ROOT = Path("reports/audit_2026_09_08")
BASELINE = ROOT / "stock_xperp_premarket_liquidity_guard_nov3_baseline.csv"
MINUTE_ROOT = ROOT / "stock_xperp_forward_nov3_intraminute"
MINUTE_MANIFEST = MINUTE_ROOT / "manifest.json"
OUT = ROOT / "stock_xperp_intraminute_runtime_parity_nov3.json"
START = pd.Timestamp("2026-11-01", tz="UTC").timestamp()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(value: str) -> str:
    return {"TP2": "TP", "EXPIRED": "TIME"}.get(str(value), str(value))


def main() -> None:
    baseline = pd.read_csv(BASELINE)
    baseline = baseline[baseline.entry_time.ge(START)].copy()
    manifest = json.loads(MINUTE_MANIFEST.read_text(encoding="utf-8"))
    rows = []
    for row in baseline.to_dict("records"):
        path = Path(manifest["symbols"][row["symbol"]]["path"])
        candles = pickle.loads(path.read_bytes())
        lookup = {int(stamp): index
                  for index, stamp in enumerate(candles["time"])}
        entry_time = int(row["entry_time"])
        entry_index = lookup.get(entry_time)
        if entry_index is None:
            continue
        delayed_entry = float(candles["close"][entry_index])
        planned_entry = float(row["entry"])
        direction = str(row["direction"])
        sign = 1 if direction == "LONG" else -1
        target_fraction = float(row["audited_target_pct"]) / 100
        stop = delayed_entry * (1 - sign * .01)
        target = delayed_entry * (1 + sign * target_fraction)
        indices = [
            index for index, stamp in enumerate(candles["time"])
            if entry_time + 60 <= int(stamp) < entry_time + 64 * 900
        ]
        result = simulate_exit(
            candles, indices, direction=direction, entry=delayed_entry,
            sl=stop, tp1=target, tp2=target, atr=delayed_entry * .01,
            tp1_fraction=1.0, trail=False, trail_mult=0.0,
            stop_on_close=False, backstop_r=1.0,
            choose_trail=lambda _high, _low, _close: 0.0,
        )
        pre_high = float(candles["high"][entry_index])
        pre_low = float(candles["low"][entry_index])
        prepublication_touch = (
            pre_low <= stop or pre_high >= target
            if direction == "LONG"
            else pre_high >= stop or pre_low <= target
        )
        old_outcome = normalize(row["outcome"])
        delayed_outcome = normalize(result.outcome)
        rows.append({
            "symbol": row["symbol"], "direction": direction,
            "entry_utc": pd.Timestamp(
                entry_time, unit="s", tz="UTC").isoformat(),
            "planned_entry": planned_entry,
            "one_minute_entry_proxy": delayed_entry,
            "adverse_reversal_drift_pct": (
                sign * (delayed_entry / planned_entry - 1) * 100),
            "planned_outcome": old_outcome,
            "post_publication_minute_outcome": delayed_outcome,
            "outcome_changed": old_outcome != delayed_outcome,
            "prepublication_first_minute_touched_bracket": (
                prepublication_touch),
        })
    if len(rows) != len(baseline):
        raise ValueError(
            f"minute coverage mismatch: {len(rows)} of {len(baseline)}")
    changed = [row for row in rows if row["outcome_changed"]]
    drift = [float(row["adverse_reversal_drift_pct"]) for row in rows]
    report = {
        "status": "RUNTIME_PARITY_FIX",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "ready_for_live_orders": False,
        "orders_possible": False,
        "runtime_change_allowed": True,
        "scope": {
            "entries": len(rows), "start_utc": "2026-11-01T00:00:00+00:00",
            "entry_proxy": "close of first 1m candle after planned 15m open",
            "observable_path": "1m candles starting at planned entry + 60s",
            "hold": "64 planned 15m bars",
        },
        "result": {
            "outcome_changes": len(changed),
            "changed_rows": changed,
            "prepublication_bracket_touches": sum(
                bool(row["prepublication_first_minute_touched_bracket"])
                for row in rows),
            "adverse_reversal_drift_pct": {
                "mean": sum(drift) / len(drift),
                "minimum": min(drift), "maximum": max(drift),
            },
        },
        "deployed_contract": {
            "primary_scan": "+10 seconds after each 15m boundary",
            "retry_scan": "+70 seconds after each 15m boundary",
            "monitor": "1m X-Perp bars from the signal publication minute",
            "expiry": "original planned entry + 64x15m",
            "position_fraction": "unchanged client setting",
            "orders": "market only; shadow code still forbids real orders",
        },
        "live_boundary_probe": {
            "instrument": "AAPL X-Perp",
            "boundary_utc": "2026-11-03T19:15:00+00:00",
            "prior_15m_confirmed_at_plus_seconds": 2,
            "forming_candle_confirm_flag": 0,
            "prior_candle_confirm_flag": 1,
        },
        "sources": {
            str(BASELINE): digest(BASELINE),
            str(MINUTE_MANIFEST): digest(MINUTE_MANIFEST),
            **{str(Path(item["path"])): item["sha256"]
               for item in manifest["symbols"].values()},
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "The one-minute close is a proxy for the Telegram market price, not an exchange fill.",
            "One-minute OHLC cannot order a stop and target touched inside the same minute.",
            "The sample covers only the recent entries available in public one-minute history.",
            "This audit fixes measurement parity and does not prove a profitability increase.",
        ],
    }
    if not all(math.isfinite(value) for value in drift):
        raise ValueError("non-finite entry drift")
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "scope": report["scope"],
        "result": report["result"],
        "deployed_contract": report["deployed_contract"],
        "live_boundary_probe": report["live_boundary_probe"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
