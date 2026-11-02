"""Append-only forward check of the frozen low-ATR SHORT 3% target.

The target override was selected and frozen on the Oct-30 snapshot.  This file
does not search thresholds.  It replays the exact incumbent and challenger on
the Nov-2 public snapshot, verifies stable historical parity, and evaluates
only entries strictly after the original freeze candle for activation.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pandas as pd

import stock_xperp_basket_trend_forward_oct30_audit as forward
import stock_xperp_conditional_weak_reversal_guard_nov2_audit as integrity
import stock_xperp_quality_target_challenger_oct30_audit as frozen
import stock_xperp_regime_target_router_oct30_audit as router
import stock_xperp_short_exhaustion_guard_oct30_audit as short_guard
import stock_xperp_short_stack_guard_oct30_forward as source


ROOT = Path("reports/audit_2026_09_08")
MANIFEST = ROOT / "stock_xperp_forward_nov2_stop_review" / "frozen_manifest.json"
OLD_ROWS = ROOT / "stock_xperp_regime_target_router_nov1_forward.csv"
FREEZE_REPORT = ROOT / "stock_xperp_quality_target_challenger_oct30.json"
OUT = ROOT / "stock_xperp_quality_target_challenger_nov2_forward.json"
BASELINE_ROWS_OUT = ROOT / "stock_xperp_quality_target_challenger_nov2_baseline.csv"
CHALLENGER_ROWS_OUT = ROOT / "stock_xperp_quality_target_challenger_nov2.csv"
FROZEN_CONFIG = {
    "direction": "SHORT",
    "feature": "atr14_fraction",
    "operator": "at_most",
    "target_pct": 3.0,
    "threshold": 0.002,
}
MINIMUM_NEW_RESOLVED = 30


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(row: dict) -> tuple[str, int, str]:
    return str(row["symbol"]), int(row["entry_time"]), str(row["direction"])


def incumbent_target(entry: dict) -> float:
    hour = int(float(entry["entry_time"]) // 3600) % 24
    if float(entry["basket_trend_32"]) <= -.04:
        return 2.0
    if 6 <= hour < 12:
        return 3.0
    return 2.5


def challenger_target(entry: dict) -> float:
    if (str(entry["direction"]) == "SHORT"
            and float(entry["atr14_fraction"]) <= .002):
        return 3.0
    return incumbent_target(entry)


def route(entries: pd.DataFrame, resolved: dict[float, pd.DataFrame],
          target_fn) -> pd.DataFrame:
    lookup = {
        target: {identity(row): row for row in frame.to_dict("records")}
        for target, frame in resolved.items()
    }
    selected = [
        lookup[target_fn(entry)][identity(entry)]
        for entry in entries.to_dict("records")
    ]
    return short_guard.accept(pd.DataFrame(selected))


def compact(frame: pd.DataFrame) -> dict:
    values = forward.metrics(frame, .03)
    return {key: values[key] for key in (
        "n", "resolved", "pending", "return_pct", "dd_pct",
        "profitable_exit_rate_pct", "literal_tp_rate_pct",
        "max_loss_streak", "trades_per_active_day",
    )}


def main() -> None:
    freeze = json.loads(FREEZE_REPORT.read_text(encoding="utf-8"))
    if freeze["frozen_challenger"]["config"] != FROZEN_CONFIG:
        raise ValueError("frozen challenger config drift")
    freeze_end = pd.Timestamp(
        freeze["forward_activation_gate"]["freeze_candle_end_utc"]
    ).timestamp()

    raw, sources, candles = source.candidates(MANIFEST)
    entries = short_guard.apply_guard(raw, -.01)
    resolved = {
        target: router.resolve(entries, candles, target)
        for target in router.TARGETS
    }
    baseline = route(entries, resolved, incumbent_target)
    challenger = route(entries, resolved, challenger_target)
    BASELINE_ROWS_OUT.write_text(
        baseline.to_csv(index=False, lineterminator="\n"), encoding="utf-8")
    CHALLENGER_ROWS_OUT.write_text(
        challenger.to_csv(index=False, lineterminator="\n"), encoding="utf-8")

    old_rows = pd.read_csv(OLD_ROWS)
    freeze_manifest = json.loads(
        integrity.OLD_MANIFEST.read_text(encoding="utf-8"))
    old_end = min(
        int(item["end"]) for item in freeze_manifest["symbols"].values()
        if item.get("end") is not None
    )
    parity = integrity.stable_parity(
        old_rows, baseline, old_end - 64 * 900)

    fresh_baseline = baseline[
        baseline.entry_time.gt(freeze_end)
        & baseline.outcome.ne("PENDING")].copy()
    fresh_challenger = challenger[
        challenger.entry_time.gt(freeze_end)
        & challenger.outcome.ne("PENDING")].copy()
    base_metrics = compact(fresh_baseline)
    challenger_metrics = compact(fresh_challenger)
    observed = min(len(fresh_baseline), len(fresh_challenger))
    enough = observed >= MINIMUM_NEW_RESOLVED
    improvement = bool(
        challenger_metrics["return_pct"] > base_metrics["return_pct"]
        and challenger_metrics["dd_pct"] <= base_metrics["dd_pct"]
        and challenger_metrics["trades_per_active_day"] >= 3
    )
    gate = enough and improvement

    monthly = {}
    stamps_base = pd.to_datetime(baseline.entry_time, unit="s", utc=True)
    stamps_challenger = pd.to_datetime(
        challenger.entry_time, unit="s", utc=True)
    for value in ("2026-07", "2026-08", "2026-09", "2026-10", "2026-11"):
        monthly[value] = {
            "baseline": compact(baseline[stamps_base.dt.strftime("%Y-%m").eq(value)]),
            "challenger": compact(challenger[
                stamps_challenger.dt.strftime("%Y-%m").eq(value)]),
        }

    report = {
        "status": (
            "FORWARD_GATE_PASSED_SHADOW_TARGET_OVERRIDE" if gate
            else "FORWARD_GATE_FAILED" if enough
            else "FORWARD_AWAIT_MORE_NEW_OUTCOMES"
        ),
        "ready_for_live_orders": False,
        "orders_possible": False,
        "runtime_change_allowed": gate,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "frozen_config": FROZEN_CONFIG,
        "protocol": {
            "freeze_candle_end_utc": pd.Timestamp(
                freeze_end, unit="s", tz="UTC").isoformat(),
            "minimum_new_resolved_trades": MINIMUM_NEW_RESOLVED,
            "selection_changes": "none",
            "entry_stop_hold_portfolio": "unchanged",
            "candidate_change": (
                "SHORT with ATR14 <= 0.2% uses 3% target; otherwise incumbent"),
            "activation_requirements": [
                "strictly higher fresh return",
                "fresh drawdown no higher",
                "at least 3 trades per active day",
            ],
        },
        "integrity": {
            "raw_candle_overlap": forward.candle_overlap(
                integrity.OLD_MANIFEST, MANIFEST),
            "baseline_route_parity": parity,
        },
        "new_forward": {
            "observed_resolved_trades": observed,
            "baseline": base_metrics,
            "challenger": challenger_metrics,
            "enough_observations": enough,
            "improves_all_requirements": improvement,
            "gate_passes": gate,
        },
        "monthly_diagnostic": monthly,
        "sources": {
            **sources,
            str(MANIFEST): digest(MANIFEST),
            str(OLD_ROWS): digest(OLD_ROWS),
            str(FREEZE_REPORT): digest(FREEZE_REPORT),
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "The override was discovered on July-October and only post-freeze rows activate it.",
            "Native X-Perp history begins in late June 2026.",
            "Public OHLC cannot reproduce live latency or order-book slippage.",
            "Gross returns exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "new_forward": report["new_forward"],
        "november": monthly["2026-11"],
        "integrity": report["integrity"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
