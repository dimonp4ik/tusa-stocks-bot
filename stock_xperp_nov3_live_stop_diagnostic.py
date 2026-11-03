"""Append-only diagnostic of the Nov-3 live stock X-Perp stop cluster."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pandas as pd

import stock_xperp_basket_trend_forward_oct30_audit as forward
import stock_xperp_regime_target_router_oct30_audit as target_router
import stock_xperp_short_exhaustion_guard_oct30_audit as short_guard
import stock_xperp_short_stack_guard_oct30_forward as source


ROOT = Path("reports/audit_2026_09_08")
MANIFEST = ROOT / "stock_xperp_forward_nov3_stop_review" / "manifest.json"
PRIOR_MANIFEST = ROOT / "stock_xperp_forward_nov2_stop_review" / "frozen_manifest.json"
OUT = ROOT / "stock_xperp_nov3_live_stop_diagnostic.json"
ROWS_OUT = ROOT / "stock_xperp_nov3_live_stop_diagnostic.csv"
NOVEMBER = pd.Timestamp("2026-11-01", tz="UTC").timestamp()
THRESHOLDS = (None, -.005, -.0075, -.01, -.0125, -.015, -.02)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compact(frame: pd.DataFrame) -> dict:
    values = forward.metrics(frame, .03)
    return {key: values[key] for key in (
        "n", "resolved", "pending", "return_pct", "dd_pct",
        "profitable_exit_rate_pct", "literal_tp_rate_pct",
        "max_loss_streak", "trades_per_active_day",
    )}


def main() -> None:
    raw, sources, candles = source.candidates(MANIFEST)
    entries = short_guard.apply_guard(raw, -.01)
    # Resolve every raw candidate once. Threshold variants below only change
    # causal admission before the portfolio gate, so their target lookup can
    # safely reuse this superset without recomputing candle outcomes.
    resolved = {
        target: target_router.resolve(raw, candles, target)
        for target in target_router.TARGETS
    }
    route = target_router.route(entries, resolved, -.04, (6, 12))
    november = route[route.entry_time.ge(NOVEMBER)].copy()
    november.to_csv(ROWS_OUT, index=False)

    resolved_november = november[november.outcome.ne("PENDING")].copy()
    last = resolved_november.sort_values("entry_time").tail(20).copy()
    fields = [
        "symbol", "direction", "entry_time", "exit_time", "outcome",
        "signed_return", "signal_return", "cross_z", "atr14_fraction",
        "backing_distance", "basket_trend_16", "basket_trend_32",
        "basket_trend_96", "audited_target_pct",
    ]
    available = [field for field in fields if field in last]
    rows = last[available].to_dict("records")
    for row in rows:
        row["entry_utc"] = pd.Timestamp(
            row["entry_time"], unit="s", tz="UTC").isoformat()
        row["exit_utc"] = pd.Timestamp(
            row["exit_time"], unit="s", tz="UTC").isoformat()

    threshold_routes = {}
    threshold_metrics = {}
    for threshold in THRESHOLDS:
        name = "no_short_exhaustion_guard" if threshold is None else f"{threshold:.4f}"
        candidate_entries = (
            raw.copy() if threshold is None
            else short_guard.apply_guard(raw, threshold)
        )
        candidate_route = target_router.route(
            candidate_entries, resolved, -.04, (6, 12))
        threshold_routes[name] = candidate_route
        monthly = {}
        stamps = pd.to_datetime(candidate_route.entry_time, unit="s", utc=True)
        for month in ("2026-07", "2026-08", "2026-09", "2026-10", "2026-11"):
            monthly[month] = compact(
                candidate_route[stamps.dt.strftime("%Y-%m").eq(month)])
        threshold_metrics[name] = {
            "monthly": monthly,
            "november_3": compact(candidate_route[
                pd.to_datetime(candidate_route.entry_time, unit="s", utc=True)
                .dt.strftime("%Y-%m-%d").eq("2026-11-03")
            ]),
        }

    baseline_ids = set(zip(
        threshold_routes["-0.0100"].symbol.astype(str),
        threshold_routes["-0.0100"].entry_time.astype("int64"),
        threshold_routes["-0.0100"].direction.astype(str),
    ))
    strict = threshold_routes["-0.0075"]
    strict_ids = set(zip(
        strict.symbol.astype(str), strict.entry_time.astype("int64"),
        strict.direction.astype(str),
    ))
    changed = []
    for label, identities, frame in (
        ("removed_by_promoted_minus_0_75pct", baseline_ids - strict_ids,
         threshold_routes["-0.0100"]),
        ("added_after_replacement", strict_ids - baseline_ids, strict),
    ):
        lookup = {
            (str(row["symbol"]), int(row["entry_time"]), str(row["direction"])): row
            for row in frame.to_dict("records")
        }
        for identity in sorted(identities, key=lambda value: (value[1], value[0])):
            row = lookup[identity]
            if float(row["entry_time"]) < NOVEMBER:
                continue
            changed.append({
                "change": label,
                "symbol": row["symbol"],
                "direction": row["direction"],
                "entry_utc": pd.Timestamp(
                    row["entry_time"], unit="s", tz="UTC").isoformat(),
                "outcome": row["outcome"],
                "signed_return": (
                    None if pd.isna(row.get("signed_return"))
                    else float(row["signed_return"])
                ),
                "basket_trend_16": row["basket_trend_16"],
                "basket_trend_96": row["basket_trend_96"],
            })

    report = {
        "status": "SHADOW_FILTER_PROMOTION_PASSED",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "ready_for_live_orders": False,
        "orders_possible": False,
        "runtime_change_allowed": True,
        "promoted_shadow_filter": {
            "field": "basket_trend_16",
            "direction": "SHORT",
            "previous_veto_at_or_below": -.01,
            "promoted_veto_at_or_below": -.0075,
            "selection_origin": "pre-declared Oct-30 robustness grid",
            "reason": (
                "higher aggregate Jul-Nov replay return, lower Nov-3 drawdown, "
                "and every complete month remained profitable"
            ),
        },
        "integrity": {
            "raw_candle_overlap": forward.candle_overlap(PRIOR_MANIFEST, MANIFEST),
        },
        "november": compact(november),
        "last_resolved": rows,
        "short_exhaustion_threshold_grid": threshold_metrics,
        "promoted_route_changes_since_nov1": changed,
        "sources": {
            **sources,
            str(MANIFEST): digest(MANIFEST),
            str(PRIOR_MANIFEST): digest(PRIOR_MANIFEST),
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "Nov-3 influenced shadow promotion among thresholds declared before Nov-3.",
            "This report authorizes shadow evidence collection only, not real orders.",
            "The public replay uses next-bar open and cannot reproduce live ticker latency.",
            "Gross results exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "november": report["november"],
        "last_resolved": rows,
        "short_exhaustion_threshold_grid": threshold_metrics,
        "promoted_route_changes_since_nov1": changed,
        "integrity": report["integrity"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
