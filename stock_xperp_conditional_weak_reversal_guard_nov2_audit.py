"""Audit a conditional weak-reversal guard on the frozen stock X-Perp route.

The guard rejects only a near-threshold reversal whose proposed direction is
opposed by the average 64-bar basket move.  It is applied before the portfolio
gate so a rejected row can be replaced by the next causally available setup.

This candidate was discovered after July-October were already observed.  The
report therefore cannot authorize a runtime filter; it records exact replay
evidence and freezes the rule for genuinely new forward tracking.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pandas as pd

import stock_xperp_basket_trend_forward_oct30_audit as forward
import stock_xperp_regime_target_router_oct30_audit as router
import stock_xperp_short_exhaustion_guard_oct30_audit as short_guard
import stock_xperp_short_stack_guard_oct30_forward as source


ROOT = Path("reports/audit_2026_09_08")
MANIFEST = ROOT / "stock_xperp_forward_nov2_stop_review" / "frozen_manifest.json"
OLD_MANIFEST = ROOT / "stock_xperp_forward_nov1" / "manifest.json"
OLD_ROWS = ROOT / "stock_xperp_regime_target_router_nov1_forward.csv"
OUT = ROOT / "stock_xperp_conditional_weak_reversal_guard_nov2.json"
ROWS_OUT = ROOT / "stock_xperp_conditional_weak_reversal_guard_nov2.csv"

# Frozen from the accepted July-August route before this exact full replay.
# They are the development 10th percentile of absolute two-bar move and the
# 30th percentile of direction-adjusted 64-bar basket average respectively.
MAX_WEAK_MOVE_FRACTION = 0.00487779385889752
MAX_OPPOSING_BASKET_AVG_64 = -0.00011090910245449376
MONTHS = ("2026-07", "2026-08", "2026-09", "2026-10")
FRESH_START = pd.Timestamp("2026-11-01", tz="UTC").timestamp()
PARITY_START = pd.Timestamp("2026-07-03", tz="UTC").timestamp()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def guard_mask(frame: pd.DataFrame) -> pd.Series:
    directional_average = (
        frame.direction_n.astype(float) * frame.basket_trend_64.astype(float) / 64
    )
    weak_opposed = (
        frame.signal_return.abs().le(MAX_WEAK_MOVE_FRACTION)
        & directional_average.le(MAX_OPPOSING_BASKET_AVG_64)
    )
    return ~weak_opposed


def apply_guard(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[guard_mask(frame)].copy()


def month(frame: pd.DataFrame, value: str) -> pd.DataFrame:
    stamps = pd.to_datetime(frame.entry_time, unit="s", utc=True)
    return frame[stamps.dt.strftime("%Y-%m").eq(value)].copy()


def metrics(frame: pd.DataFrame) -> dict:
    return {
        "margin_3pct": forward.metrics(frame, .03),
        "margin_5pct": forward.metrics(frame, .05),
    }


def compact(frame: pd.DataFrame) -> dict:
    values = metrics(frame)
    return {
        margin: {key: row[key] for key in (
            "n", "return_pct", "dd_pct", "profitable_exit_rate_pct",
            "max_loss_streak", "trades_per_active_day",
        )}
        for margin, row in values.items()
    }


def dominates(candidate: dict, baseline: dict) -> bool:
    left = candidate["margin_3pct"]
    right = baseline["margin_3pct"]
    return bool(
        left["return_pct"] >= right["return_pct"]
        and left["dd_pct"] <= right["dd_pct"] + .25
        and left["trades_per_active_day"] >= 3
    )


def stable_parity(old: pd.DataFrame, new: pd.DataFrame,
                  cutoff: int) -> dict:
    """Compare the route after the newest fixed-window cache is fully warm."""
    keys = ["symbol", "entry_time", "direction"]
    fields = ["outcome", "signed_return", "exit_time", "audited_target_pct"]
    left = old[old.entry_time.between(PARITY_START, cutoff)].copy()
    right = new[new.entry_time.between(PARITY_START, cutoff)].copy()
    joined = left[keys + fields].merge(
        right[keys + fields], on=keys, suffixes=("_old", "_new"),
        validate="one_to_one")
    mismatches = 0
    for field in fields:
        a, b = joined[f"{field}_old"], joined[f"{field}_new"]
        if pd.api.types.is_numeric_dtype(a):
            mismatches += int((a.fillna(0).astype(float)
                               - b.fillna(0).astype(float)).abs().gt(1e-12).sum())
        else:
            mismatches += int(a.fillna("").ne(b.fillna("")).sum())
    passed = bool(
        len(left) == len(right) == len(joined) and mismatches == 0)
    if not passed:
        raise ValueError({
            "old": len(left), "new": len(right), "matched": len(joined),
            "mismatches": mismatches,
        })
    return {
        "from_entry_utc": pd.Timestamp(
            PARITY_START, unit="s", tz="UTC").isoformat(),
        "through_entry_utc": pd.Timestamp(
            cutoff, unit="s", tz="UTC").isoformat(),
        "old_rows": len(left), "new_rows": len(right),
        "matched_rows": len(joined), "value_mismatches": mismatches,
        "passes": passed,
    }


def main() -> None:
    raw, sources, candles = source.candidates(MANIFEST)
    entries = short_guard.apply_guard(raw, -.01)

    baseline_resolved = {
        target: router.resolve(entries, candles, target)
        for target in router.TARGETS
    }
    baseline = router.route(entries, baseline_resolved, -.04, (6, 12))

    guarded_entries = apply_guard(entries)
    guarded_resolved = {
        target: router.resolve(guarded_entries, candles, target)
        for target in router.TARGETS
    }
    guarded = router.route(
        guarded_entries, guarded_resolved, -.04, (6, 12))

    old_manifest = json.loads(OLD_MANIFEST.read_text(encoding="utf-8"))
    old_end = min(
        int(item["end"]) for item in old_manifest["symbols"].values()
        if item.get("end") is not None
    )
    integrity = {
        "raw_candle_overlap": forward.candle_overlap(OLD_MANIFEST, MANIFEST),
        "baseline_route_parity": stable_parity(
            pd.read_csv(OLD_ROWS), baseline, old_end - 64 * 900),
    }

    periods = {}
    all_complete_dominate = True
    for value in MONTHS:
        base_rows = month(baseline, value)
        candidate_rows = month(guarded, value)
        base_metrics = metrics(base_rows)
        candidate_metrics = metrics(candidate_rows)
        pass_value = dominates(candidate_metrics, base_metrics)
        all_complete_dominate &= pass_value
        periods[value] = {
            "baseline": compact(base_rows),
            "candidate": compact(candidate_rows),
            "candidate_dominates": pass_value,
            "accepted_delta": len(candidate_rows) - len(base_rows),
        }

    fresh_base = baseline[baseline.entry_time.ge(FRESH_START)].copy()
    fresh_candidate = guarded[guarded.entry_time.ge(FRESH_START)].copy()
    fresh_values = {
        "baseline": compact(fresh_base),
        "candidate": compact(fresh_candidate),
        "candidate_dominates": dominates(
            metrics(fresh_candidate), metrics(fresh_base)),
        "accepted_delta": len(fresh_candidate) - len(fresh_base),
    }
    guarded.to_csv(ROWS_OUT, index=False)

    report = {
        "status": (
            "FROZEN_RESEARCH_CHALLENGER_AWAIT_NEW_FORWARD"
            if all_complete_dominate else "REJECTED_COMPLETE_MONTHS"
        ),
        "ready_for_live_orders": False,
        "runtime_change_allowed": False,
        "orders_possible": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rule": {
            "reject_when": (
                "abs(two_bar_move) <= 0.4877793859% and "
                "direction*basket_trend_64/64 <= -0.0110909102%"
            ),
            "move_threshold_fraction": MAX_WEAK_MOVE_FRACTION,
            "directional_basket_average_64_threshold": (
                MAX_OPPOSING_BASKET_AVG_64),
            "position_in_pipeline": "before exact portfolio gate",
        },
        "protocol": {
            "discovery": ["2026-07", "2026-08"],
            "already_observed_checks": ["2026-09", "2026-10"],
            "fresh_diagnostic": "2026-11-01 onward",
            "screened_rule_count": 23590,
            "entry": "next 15-minute market open",
            "stop": "unchanged 1% price move",
            "target_router": "unchanged frozen 2/2.5/3%",
            "money": "client-selected margin; fixed 10x; market only",
        },
        "integrity": integrity,
        "complete_months": periods,
        "all_complete_months_dominate": all_complete_dominate,
        "fresh_since_nov1": fresh_values,
        "sources": {
            **sources,
            str(MANIFEST): digest(MANIFEST),
            str(OLD_MANIFEST): digest(OLD_MANIFEST),
            str(OLD_ROWS): digest(OLD_ROWS),
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "All four complete native-history months were already observed.",
            "The candidate came from a 23,590-rule exploratory screen.",
            "The November diagnostic contains only two calendar days.",
            "This evidence freezes a challenger; it does not authorize filtering live orders.",
            "Public OHLC cannot reproduce latency or order-book slippage.",
            "Gross returns exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "all_complete_months_dominate": all_complete_dominate,
        "complete_months": periods,
        "fresh_since_nov1": fresh_values,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
