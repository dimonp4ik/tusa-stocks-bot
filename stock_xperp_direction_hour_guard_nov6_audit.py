"""Audit a post-event cap for same-direction stock X-Perp entry clusters.

The Nov-6 forward loss cluster motivated the rule, so November cannot approve
it.  The frozen Jul-Oct pre-portfolio rows are replayed through every guard
that was already active before the event.  The candidate keeps the existing
four-entry total rolling-hour cap and admits no more than two entries in the
same direction during those four closed 15-minute bars.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import pickle
from pathlib import Path

import pandas as pd

from stock_module_cluster_audit import portfolio_gate
import stock_xperp_basket_trend_forward_oct30_audit as metric
import stock_xperp_long_stress_guard_nov5_audit as long_guard
import stock_xperp_native_quality_guard_oct18_audit as quality
import stock_xperp_premarket_liquidity_guard_nov3_audit as premarket


ROOT = Path("reports/audit_2026_09_08")
ROWS = long_guard.ROWS
OUT = ROOT / "stock_xperp_direction_hour_guard_nov6.json"
MONTHS = ("2026-07", "2026-08", "2026-09", "2026-10")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replay(direction_hour_cap: int | None) -> pd.DataFrame:
    raw = pickle.loads(ROWS.read_bytes())
    guarded = long_guard.apply_guard(premarket.apply_guard(raw))
    rows = portfolio_gate(
        guarded.to_dict("records"), None,
        rolling_hour_cap=4,
        rolling_direction_hour_cap=direction_hour_cap,
        rank_key=quality.ranker("z_high"),
        loss_streak_cap=2,
        direction_cap=4,
    )
    return pd.DataFrame(rows, columns=guarded.columns)


def monthly(frame: pd.DataFrame) -> dict:
    stamps = pd.to_datetime(frame.entry_time, unit="s", utc=True)
    output = {}
    for month in MONTHS:
        values = metric.metrics(
            frame[stamps.dt.strftime("%Y-%m").eq(month)], .03)
        output[month] = {key: values[key] for key in (
            "n", "return_pct", "dd_pct", "profitable_exit_rate_pct",
            "max_loss_streak", "trades_per_active_day",
        )}
    return output


def summary(values: dict) -> dict:
    return {
        "sum_monthly_return_pct": round(sum(
            float(item["return_pct"]) for item in values.values()), 4),
        "worst_month_return_pct": round(min(
            float(item["return_pct"]) for item in values.values()), 4),
        "max_monthly_drawdown_pct": round(max(
            float(item["dd_pct"]) for item in values.values()), 4),
        "max_loss_streak": max(
            int(item["max_loss_streak"]) for item in values.values()),
        "n": sum(int(item["n"]) for item in values.values()),
    }


def main() -> None:
    variants = {}
    for cap in (None, 1, 2, 3):
        name = "baseline" if cap is None else f"direction_cap_{cap}"
        values = monthly(replay(cap))
        variants[name] = {"monthly": values, "summary": summary(values)}

    baseline = variants["baseline"]
    selected = variants["direction_cap_2"]
    base_summary = baseline["summary"]
    selected_summary = selected["summary"]
    historical_pass = bool(
        selected_summary["sum_monthly_return_pct"]
            >= .95 * base_summary["sum_monthly_return_pct"]
        and selected_summary["worst_month_return_pct"]
            > base_summary["worst_month_return_pct"]
        and selected_summary["max_monthly_drawdown_pct"]
            <= base_summary["max_monthly_drawdown_pct"]
        and selected_summary["max_loss_streak"]
            <= base_summary["max_loss_streak"] + 1
        and all(item["return_pct"] > 0
                and item["trades_per_active_day"] >= 3
                for item in selected["monthly"].values())
    )
    report = {
        "status": ("PAPER_GUARD_ACCEPTED" if historical_pass
                   else "PAPER_GUARD_REJECTED"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "ready_for_live_orders": False,
        "runtime_change_allowed": historical_pass,
        "fresh_november_used_for_acceptance": False,
        "rule": {
            "total_entries_per_rolling_hour": 4,
            "same_direction_entries_per_rolling_hour": 2,
            "bar_seconds": 900,
            "interpretation": (
                "keep mixed LONG/SHORT capacity but cap one-way correlated "
                "exposure at two entries over four closed 15-minute bars"),
        },
        "protocol": {
            "motivation": "Nov-6 same-direction forward stop cluster",
            "verification_periods": list(MONTHS),
            "money": "3% current-equity margin per trade; fixed 10x",
            "fixed_ranker": "z_high",
            "fixed_loss_streak_cap": 2,
            "fixed_open_direction_cap": 4,
            "fixed_total_rolling_hour_cap": 4,
            "acceptance": (
                "retain at least 95% aggregate monthly return, improve the "
                "worst month, do not worsen maximum monthly drawdown, add at "
                "most one loss to the worst streak, remain profitable with "
                "at least three trades per active day in every complete month"),
        },
        "baseline": baseline,
        "selected": selected,
        "historical_pass": historical_pass,
        "variants": variants,
        "sources": {
            str(ROWS): digest(ROWS),
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "The direct native X-Perp history begins in July 2026.",
            "November motivated the rule and is excluded from acceptance.",
            "The stock route remains PAPER until its independent forward-health gate recovers.",
            "Returns are gross of fees at the user's instruction.",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "baseline": baseline["summary"],
        "selected": selected["summary"],
        "historical_pass": historical_pass,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
