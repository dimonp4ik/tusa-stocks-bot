"""Append-only Oct-30 evening refresh of the frozen stock target router."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pandas as pd

import stock_xperp_basket_trend_forward_oct30_audit as forward
import stock_xperp_regime_target_router_oct30_forward as previous


ROOT = Path("reports/audit_2026_09_08")
OLD_MANIFEST = ROOT / "stock_xperp_forward_oct30_1000" / "manifest.json"
NEW_MANIFEST = ROOT / "stock_xperp_forward_oct30_2000" / "manifest.json"
OLD_ROWS = ROOT / "stock_xperp_regime_target_router_oct30_forward.csv"
OUT = ROOT / "stock_xperp_regime_target_router_oct30_2000_forward.json"
ROWS_OUT = ROOT / "stock_xperp_regime_target_router_oct30_2000_forward.csv"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    overlap = forward.candle_overlap(OLD_MANIFEST, NEW_MANIFEST)
    old_rows = pd.read_csv(OLD_ROWS)
    new_rows, sources, candles = previous.replay(NEW_MANIFEST)
    old_manifest = json.loads(OLD_MANIFEST.read_text(encoding="utf-8"))
    old_end = min(int(item["end"]) for item in old_manifest["symbols"].values()
                  if item.get("end") is not None)
    stable_cutoff = old_end - 64 * 900
    row_parity = previous.parity(old_rows, new_rows, stable_cutoff)
    newly_resolved = new_rows[
        new_rows.outcome.ne("PENDING") & new_rows.exit_time.gt(old_end)].copy()
    new_entries = new_rows[new_rows.entry_time.gt(old_end)].copy()
    october = new_rows[new_rows.entry_time.ge(
        pd.Timestamp("2026-10-01", tz="UTC").timestamp())].copy()
    new_rows.to_csv(ROWS_OUT, index=False)
    fresh = previous.metrics(newly_resolved)
    october_metrics = previous.metrics(october)
    gate_input = fresh["margin_3pct"]
    gate = {
        "minimum_resolved_rows": 30,
        "observed_resolved_rows": int(len(newly_resolved)),
        "positive_return": gate_input["return_pct"] > 0,
        "drawdown_at_most_5pct": gate_input["dd_pct"] <= 5,
        "overall_frequency_at_least_3": (
            october_metrics["margin_3pct"]["trades_per_active_day"] >= 3),
    }
    gate["passes"] = bool(
        gate["observed_resolved_rows"] >= gate["minimum_resolved_rows"]
        and gate["positive_return"] and gate["drawdown_at_most_5pct"]
        and gate["overall_frequency_at_least_3"])
    new_end = min(int(value["time"][-1]) for value in candles.values())
    report = {
        "status": ("FORWARD_GATE_PASSED" if gate["passes"] else
                   "FORWARD_AWAIT_MORE_GENUINELY_NEW_OUTCOMES"),
        "ready_for_live_orders": False, "orders_possible": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "target_pct": {
                "basket_32_at_or_below_minus_4pct": 2.0,
                "utc_06_12_otherwise": 3.0, "default": 2.5,
            },
            "stop_pct": 1.0, "hold_bars": 64,
            "entry": "next 15-minute market open", "orders": "market only",
            "position_fraction": "client setting",
        },
        "integrity": {
            "raw_candle_overlap": overlap,
            "stable_accepted_row_parity": row_parity,
            "old_snapshot_end_utc": pd.Timestamp(
                old_end, unit="s", tz="UTC").isoformat(),
            "new_snapshot_end_utc": pd.Timestamp(
                new_end, unit="s", tz="UTC").isoformat(),
        },
        "october_to_date": october_metrics,
        "newly_resolved_after_old_snapshot": fresh,
        "entries_after_old_snapshot": previous.metrics(new_entries),
        "newly_resolved_rows": int(len(newly_resolved)),
        "new_entry_rows": int(len(new_entries)),
        "forward_gate": gate,
        "sources": {**sources, str(OLD_MANIFEST): digest(OLD_MANIFEST),
                    str(NEW_MANIFEST): digest(NEW_MANIFEST),
                    str(OLD_ROWS): digest(OLD_ROWS),
                    str(Path(__file__).resolve()): digest(Path(__file__).resolve())},
        "limitations": [
            "The refresh adds less than one complete 64-bar holding window.",
            "Newly resolved rows can have entered before the old snapshot ended.",
            "The route remains paper-only until a materially larger forward sample exists.",
            "Gross returns exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "integrity": report["integrity"],
        "newly_resolved_rows": len(newly_resolved),
        "new_entry_rows": len(new_entries),
        "newly_resolved": fresh,
        "entries_after_old_snapshot": report["entries_after_old_snapshot"],
        "october_to_date": october_metrics,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
