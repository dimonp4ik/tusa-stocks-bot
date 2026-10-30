"""Audit the shadow-only SHORT exhaustion guard on native stock X-Perps.

The guard withholds a reversal SHORT when the causal 16-bar basket move is
already at or below -1%.  It changes neither market entry execution, the
client's position fraction, target, stop nor maximum holding time.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pandas as pd

from stock_module_cluster_audit import portfolio_gate
import stock_xperp_basket_trend_forward_oct30_audit as forward
import stock_xperp_native_quality_guard_oct18_audit as quality
import stock_xperp_short_stack_guard_oct30_forward as source
from src.stock_xperp_native_shadow import (
    SHORT_EXHAUSTION_BARS, SHORT_MIN_BASKET_TREND,
)


ROOT = Path("reports/audit_2026_09_08")
MANIFEST = ROOT / "stock_xperp_forward_oct30" / "manifest.json"
FREEZE_MANIFEST = ROOT / "stock_xperp_forward_oct27_2156" / "manifest.json"
OUT = ROOT / "stock_xperp_short_exhaustion_guard_oct30.json"
ROWS_OUT = ROOT / "stock_xperp_short_exhaustion_guard_oct30.csv"
ROBUSTNESS_THRESHOLDS = (-.005, -.0075, -.01, -.0125, -.015, -.02)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def apply_guard(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    field = f"basket_trend_{SHORT_EXHAUSTION_BARS}"
    veto = frame.direction.eq("SHORT") & frame[field].le(float(threshold))
    return frame[~veto].copy()


def accept(frame: pd.DataFrame) -> pd.DataFrame:
    rows = portfolio_gate(
        frame.to_dict("records"), None, rolling_hour_cap=4,
        rank_key=quality.ranker("z_high"), loss_streak_cap=2)
    return pd.DataFrame(rows, columns=frame.columns)


def period(frame: pd.DataFrame, start: str, end: str | None = None,
           *, exit_after: int | None = None) -> pd.DataFrame:
    if exit_after is not None:
        return frame[frame.outcome.ne("PENDING")
                     & frame.exit_time.gt(exit_after)].copy()
    stamp = pd.to_datetime(frame.entry_time, unit="s", utc=True)
    mask = stamp.ge(pd.Timestamp(start, tz="UTC"))
    if end is not None:
        mask &= stamp.lt(pd.Timestamp(end, tz="UTC"))
    return frame[mask].copy()


def evaluate(frame: pd.DataFrame, freeze_end: int) -> dict:
    periods = {
        "selection_july_august": period(frame, "2026-07-01", "2026-09-01"),
        "test_september": period(frame, "2026-09-01", "2026-10-01"),
        "confirmation_october_to_freeze": frame[
            (frame.entry_time.ge(pd.Timestamp("2026-10-01", tz="UTC").timestamp()))
            & frame.entry_time.le(freeze_end)].copy(),
        "new_resolved_after_freeze": period(
            frame, "2026-10-01", exit_after=freeze_end),
        "october_to_date": period(frame, "2026-10-01", "2026-11-01"),
    }
    return {
        name: {
            "margin_3pct": forward.metrics(rows, .03),
            "margin_5pct": forward.metrics(rows, .05),
        }
        for name, rows in periods.items()
    }


def compact(values: dict) -> dict:
    return {
        name: {
            "return_pct": item["margin_3pct"]["return_pct"],
            "dd_pct": item["margin_3pct"]["dd_pct"],
            "resolved": item["margin_3pct"]["resolved"],
            "pending": item["margin_3pct"]["pending"],
        }
        for name, item in values.items()
    }


def main() -> None:
    candidates, sources, _ = source.candidates(MANIFEST)
    freeze = json.loads(FREEZE_MANIFEST.read_text(encoding="utf-8"))
    freeze_end = min(int(item["end"]) for item in freeze["symbols"].values()
                     if item.get("end") is not None)
    baseline_rows = accept(candidates)
    guarded_rows = accept(apply_guard(candidates, SHORT_MIN_BASKET_TREND))
    baseline = evaluate(baseline_rows, freeze_end)
    guarded = evaluate(guarded_rows, freeze_end)
    robustness = {}
    for threshold in ROBUSTNESS_THRESHOLDS:
        values = evaluate(accept(apply_guard(candidates, threshold)), freeze_end)
        robustness[f"{threshold:.4f}"] = compact(values)

    before_names = (
        "selection_july_august", "test_september",
        "confirmation_october_to_freeze")
    total_before = sum(
        guarded[name]["margin_3pct"]["return_pct"] for name in before_names)
    baseline_total_before = sum(
        baseline[name]["margin_3pct"]["return_pct"] for name in before_names)
    candidate_fresh = guarded["new_resolved_after_freeze"]["margin_3pct"]
    baseline_fresh = baseline["new_resolved_after_freeze"]["margin_3pct"]
    gate = {
        "all_observed_periods_positive": all(
            guarded[name]["margin_3pct"]["return_pct"] > 0
            for name in before_names),
        "aggregate_pre_freeze_return_not_lower": total_before >= baseline_total_before,
        "pre_freeze_max_dd_increase_at_most_0_25pp": max(
            guarded[name]["margin_3pct"]["dd_pct"] for name in before_names
        ) <= max(baseline[name]["margin_3pct"]["dd_pct"]
                 for name in before_names) + .25,
        "fresh_return_improved": (
            candidate_fresh["return_pct"] > baseline_fresh["return_pct"]),
        "fresh_drawdown_not_worse": (
            candidate_fresh["dd_pct"] <= baseline_fresh["dd_pct"]),
        "fresh_frequency_at_least_3": candidate_fresh["trades_per_active_day"] >= 3,
    }
    gate["passes"] = all(gate.values())
    guarded_rows.to_csv(ROWS_OUT, index=False)
    report = {
        "status": ("SHADOW_ONLY_FORWARD_EXPERIMENT"
                   if gate["passes"] else "REJECTED"),
        "ready_for_live_orders": False,
        "orders_possible": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "direction": "SHORT",
            "basket_bars": SHORT_EXHAUSTION_BARS,
            "veto_at_or_below": SHORT_MIN_BASKET_TREND,
            "entry": "unchanged next 15-minute market open",
            "target_pct": 2.0, "stop_pct": 1.0, "hold_bars": 64,
            "position_fraction": "unchanged client setting",
        },
        "protocol": {
            "selection_context": "July-August aggregate",
            "test": "September", "confirmation": "October through Oct-27 freeze",
            "forward_diagnostic": "resolved outcomes after Oct-27 freeze",
            "threshold_warning": (
                "The threshold was proposed after observing historical losses; "
                "deployment must remain shadow-only until more forward outcomes."),
        },
        "activation_gate": gate,
        "aggregate_pre_freeze_return_pct_at_3pct": {
            "baseline": round(baseline_total_before, 4),
            "guarded": round(total_before, 4),
        },
        "baseline": baseline,
        "guarded": guarded,
        "robustness": robustness,
        "sources": {
            **sources, str(MANIFEST): digest(MANIFEST),
            str(FREEZE_MANIFEST): digest(FREEZE_MANIFEST),
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "Native X-Perp history begins in late June 2026.",
            "The genuinely new post-freeze sample contains only two active days.",
            "OHLC replay cannot reproduce latency or intrabar order sequence.",
            "Gross returns exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "activation_gate": gate,
        "aggregate_pre_freeze": report["aggregate_pre_freeze_return_pct_at_3pct"],
        "baseline": compact(baseline), "guarded": compact(guarded),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
