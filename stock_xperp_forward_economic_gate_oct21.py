"""Economic objective gate for one immutable stock X-Perp forward snapshot."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_AT_THREE_PER_DAY_PCT = 1.327709


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def projected_month_pct(mean_signed: float, trades_per_day: float,
                        margin: float) -> float:
    trades = 22 * trades_per_day
    return 100 * ((1 + margin * 10 * mean_signed) ** trades - 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("economic-gate report is immutable")
    source = json.loads(args.input.read_text(encoding="utf-8"))
    rows = [row for row in source["fresh"] if row["outcome"] != "PENDING"]
    frame = pd.DataFrame(rows)
    frame["day"] = pd.to_datetime(
        frame.entry_time, unit="s", utc=True).dt.strftime("%Y-%m-%d")
    days = int(frame.day.nunique())
    trades_per_day = len(frame) / days if days else 0.0
    mean_signed = float(frame.signed_return.mean()) if len(frame) else 0.0
    times = sorted(frame.entry_time.astype(float))
    rolling_hour = max((sum(time - 3600 < other <= time for other in times)
                        for time in times), default=0)
    same_timestamp = int(frame.groupby("entry_time").size().max()) if len(frame) else 0
    clock_hour = int(frame.assign(
        clock_hour=(frame.entry_time // 3600).astype(int)
    ).groupby("clock_hour").size().max()) if len(frame) else 0
    symbol_counts = Counter(frame.symbol.astype(str))
    daily = source["diagnostics_3pct"]["daily"]
    projected = projected_month_pct(mean_signed, trades_per_day, .03)
    required_actual_frequency = (
        100 * ((1.30 ** (1 / (22 * trades_per_day)) - 1) / .30)
        if trades_per_day else None)
    metrics = source["resolved_metrics_3pct"]
    diagnostics = source["diagnostics_3pct"]
    passes = bool(
        len(rows) >= 30 and projected >= 30
        and metrics["negative_months"] == 0 and metrics["dd_pct"] <= 8
        and trades_per_day >= 3 and diagnostics["literal_tp_rate_pct"] >= 75)
    report = {
        "status": "FORWARD_ECONOMIC_GATE_PASS" if passes else "FORWARD_ECONOMIC_GATE_FAIL",
        "ready_for_live_orders": False,
        "orders_possible": False,
        "source_snapshot": str(args.input),
        "resolved_observations": len(rows),
        "active_market_days": days,
        "actual_return_3pct_margin_pct": metrics["return_pct"],
        "projected_22_market_day_pct": round(projected, 6),
        "mean_signed_price_return_pct": round(100 * mean_signed, 6),
        "required_constant_signed_return_pct_at_actual_frequency": (
            round(required_actual_frequency, 6)
            if required_actual_frequency is not None else None),
        "required_constant_signed_return_pct_at_3_per_day": REQUIRED_AT_THREE_PER_DAY_PCT,
        "negative_months": metrics["negative_months"],
        "negative_market_days": sum(
            float(value["return_pct"]) < 0 for value in daily.values()),
        "literal_tp_rate_pct": diagnostics["literal_tp_rate_pct"],
        "profitable_exit_rate_pct": diagnostics["win_rate_pct"],
        "drawdown_pct": metrics["dd_pct"],
        "max_loss_streak": diagnostics["max_loss_streak"],
        "trades_per_market_day": round(trades_per_day, 4),
        "cluster_concentration": {
            "maximum_entries_rolling_hour": rolling_hour,
            "maximum_entries_same_timestamp": same_timestamp,
            "maximum_entries_same_clock_hour": clock_hour,
            "top_symbol_counts": dict(symbol_counts.most_common(10)),
        },
        "daily": daily,
        "objective_pass": passes,
        "sources": {str(args.input): digest(args.input),
                    str(Path(__file__).resolve()): digest(Path(__file__).resolve())},
        "limitations": [
            "Only four fresh market days are observed.",
            "Projection is a constant-expectancy diagnostic, not a forecast.",
            "This is public candle replay, not account fills.",
            "Gross returns exclude fees at user request.",
        ],
    }
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
