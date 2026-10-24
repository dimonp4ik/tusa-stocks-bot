"""Single-feature alignment audit for index-led stock sweep entries.

Thresholds and the winning filter are chosen on August only.  September and
October are then evaluated chronologically without retuning.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

import stock_liquidity_sweep_cross_venue_oct24_audit as base


ROOT = base.ROOT
SOURCE = ROOT / "stock_liquidity_sweep_cross_venue_oct24.pkl"
SOURCE_REPORT = ROOT / "stock_liquidity_sweep_cross_venue_oct24.json"
OUT = ROOT / "stock_index_sweep_alignment_oct24.json"
ROWS_OUT = ROOT / "stock_index_sweep_alignment_oct24.pkl"
FEATURES = (
    "quality", "sweep_depth_atr", "rejection_atr", "body_dir_atr",
    "trend_dir_atr", "ret20_dir_atr", "qqq_trend_dir_atr",
    "qqq_ret20_dir_atr", "volume_ratio", "daily_ret5_dir",
    "daily_ret20_dir", "gap_dir", "intraday_dir",
    "qqq_daily_ret5_dir", "qqq_daily_ret20_dir", "qqq_gap_dir",
    "qqq_intraday_dir",
)
MARKET_DAYS = {"2026-08": 21, "2026-09": 22, "2026-10": 15}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compact(value: dict) -> dict:
    return {name: value[name] for name in (
        "n", "return_pct", "dd_pct", "wr", "max_loss_streak",
        "trades_per_market_day", "negative_months",
        "mean_signed_price_return_pct")}


def subset(frame: pd.DataFrame, feature: str, side: str,
           threshold: float) -> pd.DataFrame:
    if side == "gte":
        return frame[frame[feature] >= threshold].copy()
    return frame[frame[feature] <= threshold].copy()


def summary(frame: pd.DataFrame, month: str) -> dict:
    return base.economic_summary(frame, MARKET_DAYS[month])


def main() -> None:
    rows = pd.read_pickle(SOURCE).copy()
    rows["month"] = pd.to_datetime(
        rows.entry_time, unit="s", utc=True).dt.strftime("%Y-%m")
    august = rows[rows.month.eq("2026-08")].copy()
    september = rows[rows.month.eq("2026-09")].copy()
    october = rows[rows.month.eq("2026-10")].copy()
    baseline = {month: summary(part, month) for month, part in (
        ("2026-08", august), ("2026-09", september),
        ("2026-10", october))}

    frontier = []
    for feature in FEATURES:
        values = pd.to_numeric(august[feature], errors="coerce").dropna()
        for quantile in (.10, .20, .25, .30, .40):
            for side, q in (("gte", quantile), ("lte", 1-quantile)):
                threshold = float(values.quantile(q))
                chosen = subset(august, feature, side, threshold)
                value = summary(chosen, "2026-08")
                if value["trades_per_market_day"] < 2.5:
                    continue
                frontier.append({
                    "feature": feature, "side": side,
                    "threshold": threshold, "quantile": q,
                    "development": compact(value),
                    "score": round(value["return_pct"]-.25*value["dd_pct"], 8),
                })
    frontier.sort(key=lambda item: (
        item["score"], item["development"]["return_pct"],
        -item["development"]["dd_pct"]), reverse=True)
    selected = frontier[0]
    tests, test_rows = {}, []
    for month, frame in (("2026-09", september), ("2026-10", october)):
        chosen = subset(frame, selected["feature"], selected["side"],
                        float(selected["threshold"]))
        tests[month] = summary(chosen, month)
        test_rows.append(chosen)
    accepted = pd.concat(test_rows, ignore_index=True)
    accepted.to_pickle(ROWS_OUT)
    passed = bool(all(
        tests[month]["return_pct"] > 0
        and tests[month]["dd_pct"] <= 8
        and tests[month]["trades_per_market_day"] >= 3
        for month in tests)
        and tests["2026-10"]["economic_30pct_gate"])
    report = {
        "status": "ALIGNMENT_GATE_PASSED" if passed else "REJECTED",
        "ready_for_live_orders": False,
        "order_eligible": False,
        "protocol": {
            "development": "August 2026 only",
            "tests": ["September 2026", "October 2026 through Oct 22"],
            "filter_shape": "one entry-time feature and one threshold",
            "tested_configurations": len(frontier),
            "test_months_used_for_selection": False,
        },
        "baseline": {key: compact(value) for key, value in baseline.items()},
        "selected_on_august": selected,
        "chronological_tests": tests,
        "objective_pass": passed,
        "development_frontier": frontier[:20],
        "sources": {
            str(SOURCE): digest(SOURCE),
            str(SOURCE_REPORT): digest(SOURCE_REPORT),
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "The direct cross-venue sample starts at the end of July.",
            "Filtering already accepted rows is conservative about fallback entries.",
            "Gross results exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True),
                   encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "baseline": report["baseline"],
        "selected_on_august": selected,
        "chronological_tests": {
            key: compact(value) for key, value in tests.items()},
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
