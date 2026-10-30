"""Append-only forward refresh of the frozen stock regime target router."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import stock_xperp_basket_trend_forward_oct30_audit as forward
import stock_xperp_regime_target_router_oct30_audit as router
import stock_xperp_short_exhaustion_guard_oct30_audit as guard
import stock_xperp_short_stack_guard_oct30_forward as source


ROOT = Path("reports/audit_2026_09_08")
OLD_MANIFEST = ROOT / "stock_xperp_forward_oct30" / "manifest.json"
NEW_MANIFEST = ROOT / "stock_xperp_forward_oct30_1000" / "manifest.json"
OLD_ROWS = ROOT / "stock_xperp_regime_target_router_oct30.csv"
OUT = ROOT / "stock_xperp_regime_target_router_oct30_forward.json"
ROWS_OUT = ROOT / "stock_xperp_regime_target_router_oct30_forward.csv"
JULY_START = int(pd.Timestamp("2026-07-01", tz="UTC").timestamp())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replay(manifest: Path) -> tuple[pd.DataFrame, dict, dict]:
    raw, sources, candles = source.candidates(manifest)
    entries = guard.apply_guard(raw, -.01)
    resolved = {target: router.resolve(entries, candles, target)
                for target in router.TARGETS}
    rows = router.route(entries, resolved, -.04, (6, 12))
    return rows, sources, candles


def parity(old: pd.DataFrame, new: pd.DataFrame, cutoff: int) -> dict:
    keys = ["symbol", "entry_time", "direction"]
    fields = ["outcome", "signed_return", "exit_time", "audited_target_pct"]
    # A fixed 12,000-bar refresh drops a few earliest June bars for mature
    # instruments.  July is the first selected strategy month and is safely
    # beyond every feature warm-up, so parity begins there.
    a = old[old.entry_time.between(JULY_START, cutoff)].copy()
    b = new[new.entry_time.between(JULY_START, cutoff)].copy()
    joined = a[keys + fields].merge(
        b[keys + fields], on=keys, suffixes=("_old", "_new"),
        validate="one_to_one")
    mismatches = 0
    for field in fields:
        left, right = joined[f"{field}_old"], joined[f"{field}_new"]
        if pd.api.types.is_numeric_dtype(left):
            mismatches += int((~np.isclose(
                left, right, rtol=0, atol=1e-12, equal_nan=True)).sum())
        else:
            mismatches += int((left.fillna("") != right.fillna("")).sum())
    passed = bool(len(a) == len(b) == len(joined) and mismatches == 0)
    if not passed:
        raise ValueError({"old": len(a), "new": len(b),
                          "matched": len(joined), "mismatches": mismatches})
    return {
        "through_entry_utc": pd.Timestamp(cutoff, unit="s", tz="UTC").isoformat(),
        "old_rows": len(a), "new_rows": len(b), "matched_rows": len(joined),
        "value_mismatches": mismatches, "passes": passed,
    }


def metrics(frame: pd.DataFrame) -> dict:
    return {
        "margin_3pct": forward.metrics(frame, .03),
        "margin_5pct": forward.metrics(frame, .05),
    }


def main() -> None:
    candle_overlap = forward.candle_overlap(OLD_MANIFEST, NEW_MANIFEST)
    old_rows = pd.read_csv(OLD_ROWS)
    new_rows, sources, new_candles = replay(NEW_MANIFEST)
    old_manifest = json.loads(OLD_MANIFEST.read_text(encoding="utf-8"))
    old_end = min(int(item["end"]) for item in old_manifest["symbols"].values()
                  if item.get("end") is not None)
    stable_cutoff = old_end - 64 * 900
    row_parity = parity(old_rows, new_rows, stable_cutoff)
    newly_resolved = new_rows[
        new_rows.outcome.ne("PENDING") & new_rows.exit_time.gt(old_end)].copy()
    new_entries = new_rows[new_rows.entry_time.gt(old_end)].copy()
    october = new_rows[new_rows.entry_time.ge(
        pd.Timestamp("2026-10-01", tz="UTC").timestamp())].copy()
    target_counts = {
        str(key): int(value) for key, value in
        new_rows.audited_target_pct.value_counts().sort_index().items()}
    fresh_three = metrics(newly_resolved)["margin_3pct"]
    forward_gate = {
        "minimum_resolved_rows": 30,
        "observed_resolved_rows": int(len(newly_resolved)),
        "positive_return": fresh_three["return_pct"] > 0,
        "drawdown_at_most_5pct": fresh_three["dd_pct"] <= 5,
        "overall_frequency_at_least_3": forward.metrics(
            october, .03)["trades_per_active_day"] >= 3,
    }
    forward_gate["passes"] = bool(
        forward_gate["observed_resolved_rows"] >= forward_gate["minimum_resolved_rows"]
        and forward_gate["positive_return"]
        and forward_gate["drawdown_at_most_5pct"]
        and forward_gate["overall_frequency_at_least_3"])
    new_rows.to_csv(ROWS_OUT, index=False)
    report = {
        "status": ("FORWARD_GATE_PASSED" if forward_gate["passes"] else
                   "FORWARD_GATE_FAILED_SMALL_SAMPLE"),
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
            "raw_candle_overlap": candle_overlap,
            "stable_accepted_row_parity": row_parity,
            "old_snapshot_end_utc": pd.Timestamp(
                old_end, unit="s", tz="UTC").isoformat(),
            "new_snapshot_end_utc": pd.Timestamp(
                min(int(value["time"][-1]) for value in new_candles.values()),
                unit="s", tz="UTC").isoformat(),
        },
        "october_to_date": metrics(october),
        "newly_resolved_after_old_snapshot": metrics(newly_resolved),
        "entries_after_old_snapshot": metrics(new_entries),
        "newly_resolved_rows": int(len(newly_resolved)),
        "new_entry_rows": int(len(new_entries)),
        "forward_gate": forward_gate,
        "target_distribution_full": target_counts,
        "sources": {**sources, str(OLD_MANIFEST): digest(OLD_MANIFEST),
                    str(NEW_MANIFEST): digest(NEW_MANIFEST),
                    str(OLD_ROWS): digest(OLD_ROWS),
                    str(Path(__file__).resolve()): digest(Path(__file__).resolve())},
        "limitations": [
            "The new snapshot adds less than one full 64-bar holding window.",
            "A newly resolved row may have entered before the old snapshot ended.",
            "The route remains paper-only pending a materially larger forward sample.",
            "Gross returns exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "integrity": report["integrity"],
        "newly_resolved_rows": len(newly_resolved),
        "new_entry_rows": len(new_entries),
        "newly_resolved": report["newly_resolved_after_old_snapshot"],
        "entries_after_old_snapshot": report["entries_after_old_snapshot"],
        "october_to_date": report["october_to_date"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
