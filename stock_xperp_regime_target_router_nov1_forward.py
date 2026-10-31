"""Append-only Nov-1 refresh of the frozen native stock X-Perp route."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pandas as pd

import stock_xperp_basket_trend_forward_oct30_audit as forward
import stock_xperp_regime_target_router_oct30_forward as replay_source


ROOT = Path("reports/audit_2026_09_08")
OLD_MANIFEST = ROOT / "stock_xperp_forward_oct30_2000" / "manifest.json"
NEW_MANIFEST = ROOT / "stock_xperp_forward_nov1" / "manifest.json"
OLD_ROWS = ROOT / "stock_xperp_regime_target_router_oct30_2000_forward.csv"
OUT = ROOT / "stock_xperp_regime_target_router_nov1_forward.json"
ROWS_OUT = ROOT / "stock_xperp_regime_target_router_nov1_forward.csv"
MONTHS = ("2026-07", "2026-08", "2026-09", "2026-10")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def month(frame: pd.DataFrame, value: str) -> pd.DataFrame:
    stamps = pd.to_datetime(frame.entry_time, unit="s", utc=True)
    return frame[stamps.dt.strftime("%Y-%m").eq(value)].copy()


def metrics(frame: pd.DataFrame) -> dict:
    return {
        "margin_3pct": forward.metrics(frame, .03),
        "margin_5pct": forward.metrics(frame, .05),
    }


def main() -> None:
    overlap = forward.candle_overlap(OLD_MANIFEST, NEW_MANIFEST)
    old_rows = pd.read_csv(OLD_ROWS)
    new_rows, sources, candles = replay_source.replay(NEW_MANIFEST)
    old_manifest = json.loads(OLD_MANIFEST.read_text(encoding="utf-8"))
    old_end = min(int(item["end"]) for item in old_manifest["symbols"].values()
                  if item.get("end") is not None)
    new_end = min(int(candle["time"][-1]) for candle in candles.values())
    stable_cutoff = old_end - 64 * 900
    parity = replay_source.parity(old_rows, new_rows, stable_cutoff)

    newly_resolved = new_rows[
        new_rows.outcome.ne("PENDING") & new_rows.exit_time.gt(old_end)].copy()
    new_entries = new_rows[new_rows.entry_time.gt(old_end)].copy()
    complete_months = {value: metrics(month(new_rows, value))
                       for value in MONTHS}
    pending = new_rows[new_rows.outcome.eq("PENDING")].copy()
    pending_latest = pending.sort_values(
        ["entry_time", "symbol"], ascending=[False, True]).head(20)
    pending_rows = [{
        "symbol": str(row.symbol),
        "direction": str(row.direction),
        "entry_time": int(row.entry_time),
        "target_pct": float(row.audited_target_pct),
    } for row in pending_latest.itertuples(index=False)]

    fresh = metrics(newly_resolved)
    new_entry_metrics = metrics(new_entries)
    objective = {}
    for margin_key in ("margin_3pct", "margin_5pct"):
        monthly = {
            value: complete_months[value][margin_key]["return_pct"] >= 30
            for value in MONTHS
        }
        monthly["all_complete_months_pass"] = all(monthly.values())
        objective[margin_key] = monthly
    gate_input = fresh["margin_3pct"]
    forward_gate = {
        "minimum_resolved_rows": 30,
        "observed_resolved_rows": int(len(newly_resolved)),
        "positive_return": gate_input["return_pct"] > 0,
        "drawdown_at_most_5pct": gate_input["dd_pct"] <= 5,
        "frequency_at_least_3": (
            new_entry_metrics["margin_3pct"]["trades_per_active_day"] >= 3),
    }
    forward_gate["passes"] = bool(
        forward_gate["observed_resolved_rows"]
        >= forward_gate["minimum_resolved_rows"]
        and forward_gate["positive_return"]
        and forward_gate["drawdown_at_most_5pct"]
        and forward_gate["frequency_at_least_3"])

    new_rows.to_csv(ROWS_OUT, index=False)
    report = {
        "status": ("FORWARD_GATE_PASSED" if forward_gate["passes"] else
                   "FORWARD_AWAIT_MORE_GENUINELY_NEW_OUTCOMES"),
        "ready_for_live_orders": False,
        "orders_possible": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "entry": "next 15-minute market open",
            "orders": "market only",
            "stop_pct": 1.0,
            "hold_bars": 64,
            "direction_cap": 4,
            "rolling_hour_cap": 4,
            "loss_pause": 2,
            "target_pct": {
                "basket_32_at_or_below_minus_4pct": 2.0,
                "utc_06_12_otherwise": 3.0,
                "default": 2.5,
            },
        },
        "integrity": {
            "raw_candle_overlap": overlap,
            "stable_accepted_row_parity": parity,
            "old_snapshot_end_utc": pd.Timestamp(
                old_end, unit="s", tz="UTC").isoformat(),
            "new_snapshot_end_utc": pd.Timestamp(
                new_end, unit="s", tz="UTC").isoformat(),
        },
        "complete_months": complete_months,
        "monthly_30pct_objective": objective,
        "newly_resolved_after_old_snapshot": fresh,
        "entries_after_old_snapshot": new_entry_metrics,
        "newly_resolved_rows": int(len(newly_resolved)),
        "new_entry_rows": int(len(new_entries)),
        "pending_rows": int(len(pending)),
        "pending_latest": pending_rows,
        "forward_gate": forward_gate,
        "sources": {
            **sources,
            str(OLD_MANIFEST): digest(OLD_MANIFEST),
            str(NEW_MANIFEST): digest(NEW_MANIFEST),
            str(OLD_ROWS): digest(OLD_ROWS),
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "Native stock X-Perp history begins in late June 2026.",
            "The append-only interval is shorter than one complete month.",
            "Public OHLC cannot reproduce latency or order-book slippage.",
            "Gross returns exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True),
                   encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "integrity": report["integrity"],
        "complete_months": complete_months,
        "monthly_30pct_objective": objective,
        "newly_resolved": fresh,
        "entries_after_old_snapshot": new_entry_metrics,
        "pending_latest": pending_rows,
        "forward_gate": forward_gate,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
