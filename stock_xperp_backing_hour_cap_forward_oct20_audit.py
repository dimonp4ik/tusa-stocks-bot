"""Untouched Oct-17+ replay of the frozen stock rolling-hour cap."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from stock_module_cluster_audit import portfolio_gate
import stock_xperp_backing_hour_cap_oct20_audit as hour_cap
import stock_xperp_backing_level_forward_oct20_audit as forward
import stock_xperp_backing_level_guard_oct20_audit as backing
import stock_xperp_native_cross_section_oct18_audit as parent
import stock_xperp_native_economics_oct18_audit as economics
import stock_xperp_native_growth_forward_monitor as monitor
import stock_xperp_native_growth_profile_oct18_audit as growth
import stock_xperp_native_quality_guard_oct18_audit as quality
import stock_xperp_native_volatility_guard_oct18_audit as volatility


ROOT = parent.ROOT
MANIFEST = ROOT / "stock_xperp_forward_oct20_sync" / "manifest.json"
OUT = ROOT / "stock_xperp_backing_hour_cap_forward_oct20.json"
ROWS_OUT = ROOT / "stock_xperp_backing_hour_cap_forward_oct20.csv"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    csv_out = args.out.with_suffix(".csv")
    if args.out != OUT and (args.out.exists() or csv_out.exists()):
        raise FileExistsError("forward output is immutable; choose a new path")
    profile = json.loads(growth.OUT.read_text(encoding="utf-8"))
    parameters = profile["selection"]["selected"]["parameters"]
    volatility_report = json.loads(volatility.OUT.read_text(encoding="utf-8"))
    maximum_atr = float(volatility_report["selection"][
        "selected_maximum_atr14_fraction"])
    quality_report = json.loads(quality.OUT.read_text(encoding="utf-8"))
    rank_name = str(quality_report["selection"]["selected_ranker"])
    loss_cap = int(quality_report["selection"]["selected_loss_streak_cap"])
    cap_report = json.loads(hour_cap.OUT.read_text(encoding="utf-8"))
    rolling_cap = int(cap_report["selected_rolling_hour_cap"])

    candles, sources, overlap = monitor.load_manifest(args.manifest)
    features = parent.feature_rows(candles, int(parameters["lookback_bars"]))
    entries = parent.select_entries(
        features, float(parameters["z_threshold"]),
        float(parameters["minimum_move_pct"]) / 100)
    entries = volatility.add_atr14(entries, candles)
    entries = entries[entries.atr14_fraction.le(maximum_atr)]
    entries = backing.guarded(backing.add_backing_distance(
        entries, candles, int(cap_report["backing_level_lookback_bars"])))
    cutoff = pd.Timestamp(parent.FRESH[1], tz="UTC").timestamp()
    fresh_raw = monitor.resolve_entries(
        entries[entries.entry_time.ge(cutoff)], candles, parameters)
    reference = pd.read_csv(backing.ROWS_OUT).to_dict("records")
    gated = portfolio_gate(
        reference + fresh_raw, None, rolling_hour_cap=rolling_cap,
        rank_key=quality.ranker(rank_name), loss_streak_cap=loss_cap)
    fresh = [row for row in gated if float(row["entry_time"]) >= cutoff]
    resolved = [row for row in fresh if row["outcome"] != "PENDING"]
    pending = [row for row in fresh if row["outcome"] == "PENDING"]
    result = {
        "status": "ROLLING_HOUR_CAP_FORWARD_SNAPSHOT",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "ready_for_live_orders": False,
        "orders_possible": False,
        "audit_cutoff": parent.FRESH[1],
        "rolling_hour_cap": rolling_cap,
        "backing_level_lookback_bars": int(
            cap_report["backing_level_lookback_bars"]),
        "ranker": rank_name,
        "loss_streak_cap": loss_cap,
        "fresh_guarded_raw_signals": len(fresh_raw),
        "fresh_portfolio_signals": len(fresh),
        "resolved_signals": len(resolved),
        "pending_signals": len(pending),
        "resolved_metrics_3pct": economics.simulate(resolved, 0.03),
        "resolved_metrics_5pct": economics.simulate(resolved, 0.05),
        "diagnostics_3pct": forward.diagnostics(resolved, 0.03),
        "diagnostics_5pct": forward.diagnostics(resolved, 0.05),
        "overlap_verification": {
            "symbols": len(overlap),
            "overlap_bars": sum(item["overlap_bars"] for item in overlap.values()),
            "new_bars_after_frozen_end": sum(
                item["new_bars_after_frozen_end"] for item in overlap.values()),
            "mismatches": 0,
        },
        "fresh": fresh,
        "sources": {
            **sources,
            str(hour_cap.OUT): digest(hour_cap.OUT),
            str(backing.ROWS_OUT): digest(backing.ROWS_OUT),
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "The fresh sample spans only about three resolved days.",
            "This is a public-data shadow snapshot, not account fills.",
            "Gross returns exclude fees at user request.",
        ],
    }
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    pd.DataFrame(fresh).to_csv(csv_out, index=False)
    print(json.dumps({key: result[key] for key in (
        "status", "rolling_hour_cap", "fresh_guarded_raw_signals",
        "fresh_portfolio_signals", "resolved_signals", "pending_signals",
        "resolved_metrics_3pct", "resolved_metrics_5pct",
        "diagnostics_3pct")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
