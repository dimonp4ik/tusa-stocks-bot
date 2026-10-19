"""Fresh immutable replay of the current backing-level stock shadow route."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from stock_module_cluster_audit import portfolio_gate
import stock_xperp_backing_level_guard_oct20_audit as backing
import stock_xperp_native_cross_section_oct18_audit as parent
import stock_xperp_native_economics_oct18_audit as economics
import stock_xperp_native_growth_forward_monitor as monitor
import stock_xperp_native_growth_profile_oct18_audit as growth
import stock_xperp_native_quality_guard_oct18_audit as quality
import stock_xperp_native_volatility_guard_oct18_audit as volatility


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def diagnostics(rows: list[dict], margin: float) -> dict:
    if not rows:
        return {"win_rate_pct": 0.0, "literal_tp_rate_pct": 0.0,
                "max_loss_streak": 0, "daily": {}}
    ordered = sorted(rows, key=lambda row: (
        float(row["exit_time"]), float(row["entry_time"]),
        str(row["symbol"])))
    streak = maximum = 0
    for row in ordered:
        streak = streak + 1 if float(row["signed_return"]) <= 0 else 0
        maximum = max(maximum, streak)
    frame = pd.DataFrame(rows)
    frame["day"] = pd.to_datetime(
        frame.entry_time, unit="s", utc=True).dt.strftime("%Y-%m-%d")
    daily = {}
    for day, part in frame.groupby("day", sort=True):
        day_rows = part.drop(columns="day").to_dict("records")
        value = economics.simulate(day_rows, margin)
        daily[str(day)] = {
            "n": len(day_rows),
            "win_rate_pct": round(100 * sum(
                float(row["signed_return"]) > 0 for row in day_rows)
                / len(day_rows), 4),
            "return_pct": value["return_pct"], "dd_pct": value["dd_pct"],
        }
    return {
        "win_rate_pct": round(100 * sum(
            float(row["signed_return"]) > 0 for row in rows) / len(rows), 4),
        "literal_tp_rate_pct": round(100 * sum(
            int(row["outcome_code"]) == 1 for row in rows) / len(rows), 4),
        "max_loss_streak": maximum, "daily": daily,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base-forward", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    csv_out = args.out.with_suffix(".csv")
    if args.out.exists() or csv_out.exists():
        raise FileExistsError("forward output is immutable; choose a new path")

    profile = json.loads(growth.OUT.read_text(encoding="utf-8"))
    parameters = profile["selection"]["selected"]["parameters"]
    volatility_report = json.loads(volatility.OUT.read_text(encoding="utf-8"))
    maximum_atr = float(
        volatility_report["selection"]["selected_maximum_atr14_fraction"])
    quality_report = json.loads(quality.OUT.read_text(encoding="utf-8"))
    rank_name = str(quality_report["selection"]["selected_ranker"])
    loss_cap = int(quality_report["selection"]["selected_loss_streak_cap"])
    backing_report = json.loads(backing.OUT.read_text(encoding="utf-8"))
    backing_lookback = int(backing_report["selected_lookback_bars"])

    candles, sources, overlap = monitor.load_manifest(args.manifest)
    features = parent.feature_rows(candles, int(parameters["lookback_bars"]))
    entries = parent.select_entries(
        features, float(parameters["z_threshold"]),
        float(parameters["minimum_move_pct"]) / 100)
    entries = volatility.add_atr14(entries, candles)
    entries = entries[entries.atr14_fraction.le(maximum_atr)]
    entries = backing.guarded(
        backing.add_backing_distance(entries, candles, backing_lookback))
    cutoff = pd.Timestamp(parent.FRESH[1], tz="UTC").timestamp()
    fresh_entries = entries[entries.entry_time.ge(cutoff)]
    fresh_raw = monitor.resolve_entries(fresh_entries, candles, parameters)
    reference = pd.read_csv(backing.ROWS_OUT).to_dict("records")
    gated = portfolio_gate(
        reference + fresh_raw, None, rank_key=quality.ranker(rank_name),
        loss_streak_cap=loss_cap)
    fresh = [row for row in gated if float(row["entry_time"]) >= cutoff]
    resolved = [row for row in fresh if row["outcome"] != "PENDING"]
    pending = [row for row in fresh if row["outcome"] == "PENDING"]
    base_forward = json.loads(args.base_forward.read_text(encoding="utf-8"))
    result = {
        "status": "CURRENT_BACKING_ROUTE_FORWARD_SNAPSHOT",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "ready_for_live_orders": False,
        "orders_possible": False,
        "audit_cutoff": parent.FRESH[1],
        "parameters": parameters,
        "maximum_atr14_fraction": maximum_atr,
        "backing_level_lookback_bars": backing_lookback,
        "ranker": rank_name, "loss_streak_cap": loss_cap,
        "fresh_guarded_raw_signals": len(fresh_raw),
        "fresh_portfolio_signals": len(fresh),
        "resolved_signals": len(resolved), "pending_signals": len(pending),
        "resolved_metrics_3pct": economics.simulate(resolved, .03),
        "resolved_metrics_5pct": economics.simulate(resolved, .05),
        "diagnostics_3pct": diagnostics(resolved, .03),
        "diagnostics_5pct": diagnostics(resolved, .05),
        "base_route_comparison": {
            "signals": base_forward["resolved_signals"],
            "metrics_3pct": base_forward["resolved_metrics_3pct"],
            "metrics_5pct": base_forward["resolved_metrics_5pct"],
        },
        "overlap_verification": {
            "symbols": len(overlap),
            "overlap_bars": sum(item["overlap_bars"] for item in overlap.values()),
            "new_bars_after_frozen_end": sum(
                item["new_bars_after_frozen_end"] for item in overlap.values()),
            "mismatches": 0, "by_symbol": overlap,
        },
        "fresh": fresh,
        "sources": {
            **sources, str(args.base_forward): digest(args.base_forward),
            str(backing.OUT): digest(backing.OUT),
            str(backing.ROWS_OUT): digest(backing.ROWS_OUT),
            str(quality.OUT): digest(quality.OUT),
            str(volatility.OUT): digest(volatility.OUT),
            str(growth.OUT): digest(growth.OUT),
            str(Path(__file__)): digest(Path(__file__)),
        },
        "invariant": (
            "frozen universe and parameters; closed 15-minute features; "
            "ATR and 32-bar backing guards; next-bar market entry; "
            "stop-first OHLC; frozen ranking; causal daily loss pause"),
        "limitations": [
            "This is a public-data shadow snapshot, not account fills.",
            "The fresh sample spans only about three days.",
            "Pending rows require a later immutable snapshot.",
            "Gross returns exclude fees at user request.",
        ],
    }
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True),
                        encoding="utf-8")
    pd.DataFrame(fresh).to_csv(csv_out, index=False)
    print(json.dumps({key: result[key] for key in (
        "status", "audit_cutoff", "backing_level_lookback_bars",
        "fresh_guarded_raw_signals", "fresh_portfolio_signals",
        "resolved_signals", "pending_signals", "resolved_metrics_3pct",
        "resolved_metrics_5pct", "diagnostics_3pct",
        "base_route_comparison")}, indent=2),
        flush=True)


if __name__ == "__main__":
    main()
