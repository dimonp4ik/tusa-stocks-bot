"""Freeze a forward-only LONG guard for a sharply falling stock basket.

The rule was proposed after the Nov-5 LONG stop pair, so November is diagnostic
only.  July-October are replayed through the exact current portfolio gate.  A
rejected setup may therefore be replaced by the next causally available setup.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import pickle
from pathlib import Path

import pandas as pd

import stock_xperp_basket_trend_forward_oct30_audit as metric
import stock_xperp_premarket_liquidity_guard_nov3_audit as premarket
import stock_xperp_short_exhaustion_guard_oct30_audit as portfolio


ROOT = Path("reports/audit_2026_09_08")
ROWS = ROOT / "stock_xperp_nov5_routed_preportfolio.pkl"
MANIFEST = ROOT / "stock_xperp_forward_nov5" / "manifest.json"
OUT = ROOT / "stock_xperp_long_stress_guard_nov5.json"
RUNTIME = Path("src/stock_xperp_native_shadow.py")
THRESHOLDS = (-.02, -.03, -.04, -.05, -.06, -.08)
SELECTED = -.04
MONTHS = ("2026-07", "2026-08", "2026-09", "2026-10", "2026-11")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def apply_guard(frame: pd.DataFrame, threshold: float = SELECTED) -> pd.DataFrame:
    veto = frame.direction.eq("LONG") & frame.basket_trend_64.lt(threshold)
    return frame[~veto].copy()


def accepted(frame: pd.DataFrame) -> pd.DataFrame:
    return portfolio.accept(premarket.apply_guard(frame))


def monthly(frame: pd.DataFrame) -> dict:
    stamps = pd.to_datetime(frame.entry_time, unit="s", utc=True)
    output = {}
    for month in MONTHS:
        values = metric.metrics(frame[stamps.dt.strftime("%Y-%m").eq(month)], .03)
        output[month] = {key: values[key] for key in (
            "n", "return_pct", "dd_pct", "profitable_exit_rate_pct",
            "max_loss_streak", "trades_per_active_day",
        )}
    return output


def identities(frame: pd.DataFrame) -> set[tuple[str, int, str]]:
    return set(zip(frame.symbol.astype(str), frame.entry_time.astype("int64"),
                   frame.direction.astype(str)))


def main() -> None:
    raw = pickle.loads(ROWS.read_bytes())
    baseline_rows = accepted(raw)
    baseline = monthly(baseline_rows)
    variants = {}
    for threshold in THRESHOLDS:
        rows = accepted(apply_guard(raw, threshold))
        values = monthly(rows)
        historical_safe = all(
            values[month]["return_pct"] >= baseline[month]["return_pct"]
            and values[month]["dd_pct"] <= baseline[month]["dd_pct"]
            and values[month]["trades_per_active_day"] >= 3
            for month in MONTHS[:4]
        )
        variants[f"{threshold:.2f}"] = {
            "monthly": values,
            "historical_safe": historical_safe,
            "removed": len(identities(baseline_rows) - identities(rows)),
            "added_by_portfolio_replacement": len(
                identities(rows) - identities(baseline_rows)),
        }

    selected_rows = accepted(apply_guard(raw))
    base_ids = identities(baseline_rows)
    selected_ids = identities(selected_rows)
    lookup = {
        (str(row["symbol"]), int(row["entry_time"]), str(row["direction"])): row
        for row in baseline_rows.to_dict("records")
    }
    fresh_removed = []
    november_start = pd.Timestamp("2026-11-01", tz="UTC").timestamp()
    for identity in sorted(base_ids - selected_ids, key=lambda item: (item[1], item[0])):
        if identity[1] < november_start:
            continue
        row = lookup[identity]
        fresh_removed.append({
            "symbol": row["symbol"], "direction": row["direction"],
            "entry_utc": pd.Timestamp(
                row["entry_time"], unit="s", tz="UTC").isoformat(),
            "outcome": row["outcome"], "net_r": float(row["net_r"]),
            "basket_trend_64": float(row["basket_trend_64"]),
        })

    selected = variants[f"{SELECTED:.2f}"]
    neighbors_safe = all(
        variants[f"{threshold:.2f}"]["historical_safe"]
        for threshold in (-.03, -.04, -.05)
    )
    report = {
        "status": "FORWARD_ONLY_SHADOW_EXPERIMENT",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "ready_for_live_orders": False,
        "orders_possible": False,
        "runtime_change_allowed": bool(selected["historical_safe"] and neighbors_safe),
        "rule": {
            "direction": "LONG", "basket_horizon_closed_bars": 64,
            "reject_below_sum": SELECTED,
            "interpretation": "reject LONG while the broad 64-bar basket sum is below -4%",
        },
        "protocol": {
            "complete_month_safety": list(MONTHS[:4]),
            "fresh_diagnostic": "2026-11 through the Nov-5 snapshot",
            "position_fraction": "client setting; evaluated at 3%",
            "leverage": 10, "orders": "market only",
            "portfolio_replayed_after_each_filter": True,
        },
        "baseline": baseline,
        "selected": selected,
        "neighbor_thresholds_historically_safe": neighbors_safe,
        "variants": variants,
        "fresh_removed": fresh_removed,
        "sources": {
            str(ROWS): digest(ROWS), str(MANIFEST): digest(MANIFEST),
            str(RUNTIME): digest(RUNTIME),
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "The rule was formulated after observing the Nov-5 LONG stop pair.",
            "November cannot authorize live orders and is diagnostic only.",
            "Native stock X-Perp history begins in 2026, so this is a shadow candidate.",
            "Public OHLC replay cannot reproduce account fill latency or slippage.",
            "Gross results exclude fees at the account owner's request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "runtime_change_allowed": report["runtime_change_allowed"],
        "baseline": baseline, "selected": selected,
        "fresh_removed": fresh_removed,
    }, indent=2))


if __name__ == "__main__":
    main()
