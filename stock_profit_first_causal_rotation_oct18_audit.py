"""Causally rotate among frozen material-move stock modules.

The three source modules were fitted on 2017-2021 and selected on 2022.  This
audit selects a small trailing-performance rotation policy on 2023 only, then
freezes it before evaluating 2024-2026.  A month can use only trades that had
closed before that month began when ranking source modules.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from filter_lab import gate
import stock_cash_literal_move_hgb_oct7_audit as base
import stock_profit_first_literal_runner_oct17_audit as parent


ROOT = Path("reports/audit_2026_09_08")
OUT = ROOT / "stock_profit_first_causal_rotation_oct18.json"
ROWS_OUT = ROOT / "stock_profit_first_causal_rotation_oct18.pkl"
SOURCES = {
    "literal": ROOT / "stock_profit_first_literal_runner_oct17.pkl",
    "multisession": ROOT / "stock_profit_first_multisession_oct17.pkl",
    "high_move": ROOT / "stock_profit_first_high_move_oct17.pkl",
}
DEVELOPMENT_YEAR = 2023
TEST_YEARS = (2024, 2025, 2026)
LOOKBACK_MONTHS = (3, 6, 9, 12, 18, 24, 36)
TOP_K = (1, 2, 3)
SCORE_MODES = ("sum", "mean", "lcb", "monthly_lcb")
MIN_HISTORY_TRADES = 25
SIGNAL_KEY = ("symbol", "entry_time", "direction", "family")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_frames() -> dict[str, pd.DataFrame]:
    output = {}
    for source, path in SOURCES.items():
        frame = pd.read_pickle(path).copy()
        frame["source"] = source
        frame["entry_dt"] = pd.to_datetime(frame.entry_time, unit="s", utc=True)
        frame["exit_dt"] = pd.to_datetime(frame.exit_time, unit="s", utc=True)
        frame["entry_month"] = frame.entry_dt.dt.strftime("%Y-%m")
        output[source] = frame
    return output


def history_before(frame: pd.DataFrame, month: pd.Timestamp,
                   lookback: int) -> pd.DataFrame:
    lower = month - pd.DateOffset(months=lookback)
    return frame[(frame.entry_dt >= lower) & (frame.exit_dt < month)]


def rank_history(history: pd.DataFrame, mode: str) -> float:
    values = history.signed_return.to_numpy(float)
    if mode == "sum":
        return float(values.sum())
    if mode == "mean":
        return float(values.mean())
    if mode == "lcb":
        if len(values) < 2:
            return -1e9
        return float(values.mean() - values.std(ddof=1) / math.sqrt(len(values)))
    monthly = history.groupby("entry_month").signed_return.sum()
    if len(monthly) < 2:
        return float(monthly.mean()) if len(monthly) else -1e9
    return float(monthly.mean()
                 - monthly.std(ddof=1) / math.sqrt(len(monthly)))


def choose_sources(frames: dict[str, pd.DataFrame], month: pd.Timestamp,
                   policy: dict) -> tuple[str, ...]:
    ranked = []
    for source, frame in frames.items():
        history = history_before(frame, month, policy["lookback_months"])
        if len(history) < MIN_HISTORY_TRADES:
            continue
        ranked.append((rank_history(history, policy["score_mode"]), source))
    ranked.sort(reverse=True)
    return tuple(source for _, source in ranked[:policy["top_k"]])


def months_in_range(frames: dict[str, pd.DataFrame], start: str,
                    end: str) -> list[pd.Timestamp]:
    lower, upper = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    values = set()
    for frame in frames.values():
        part = frame[(frame.entry_dt >= lower) & (frame.entry_dt < upper)]
        values.update(pd.to_datetime(
            part.entry_dt.dt.strftime("%Y-%m-01"), utc=True).tolist())
    return sorted(values)


def make_plan(frames: dict[str, pd.DataFrame], start: str, end: str,
              policy: dict) -> tuple[tuple[str, tuple[str, ...]], ...]:
    return tuple((month.strftime("%Y-%m"), choose_sources(frames, month, policy))
                 for month in months_in_range(frames, start, end))


def execute(frames: dict[str, pd.DataFrame],
            plan: tuple[tuple[str, tuple[str, ...]], ...]) -> pd.DataFrame:
    parts = []
    for month, selected in plan:
        for priority, source in enumerate(selected):
            part = frames[source][frames[source].entry_month.eq(month)].copy()
            if part.empty:
                continue
            part["source_priority"] = priority
            parts.append(part)
    if not parts:
        return next(iter(frames.values())).iloc[0:0].copy()
    candidates = pd.concat(parts, ignore_index=True)
    candidates = candidates.sort_values(
        [*SIGNAL_KEY, "source_priority", "quality"],
        ascending=[True, True, True, True, True, False],
    ).drop_duplicates(list(SIGNAL_KEY), keep="first")
    rows = candidates.sort_values(
        ["entry_time", "source_priority", "quality", "symbol"],
        ascending=[True, True, False, True],
    ).to_dict("records")
    accepted = gate(rows, 4)
    return pd.DataFrame(accepted) if accepted else candidates.iloc[0:0].copy()


def evaluate(frame: pd.DataFrame) -> dict:
    rows = frame.sort_values(["entry_time", "symbol"]).to_dict("records")
    return parent.enrich(base.account(rows, already_gated=True))


def development_score(value: dict) -> float | None:
    if value["n"] < 500 or value["return_pct"] <= 0:
        return None
    return round(value["return_pct"] - 1.5 * value["dd_pct"]
                 - 2.0 * value["negative_months"], 8)


def policy_grid() -> list[dict]:
    return [{"lookback_months": lookback, "top_k": top_k,
             "score_mode": mode}
            for lookback, top_k, mode in itertools.product(
                LOOKBACK_MONTHS, TOP_K, SCORE_MODES)]


def main() -> None:
    frames = load_frames()
    candidates = []
    cache = {}
    for policy in policy_grid():
        plan = make_plan(frames, "2023-01-01", "2024-01-01", policy)
        if plan not in cache:
            selected = execute(frames, plan)
            cache[plan] = (selected, evaluate(selected))
        selected, value = cache[plan]
        score = development_score(value)
        if score is not None:
            candidates.append({"policy": policy, "score": score,
                               "development": value, "plan": plan})
    candidates.sort(key=lambda item: item["score"], reverse=True)
    frozen = candidates[0] if candidates else None
    tests = plans = checks = None
    combined = []
    status = "REJECTED_DEVELOPMENT"
    if frozen:
        tests, plans = {}, {}
        combined.append(execute(frames, frozen["plan"]))
        for year in TEST_YEARS:
            plan = make_plan(frames, f"{year}-01-01", f"{year + 1}-01-01",
                             frozen["policy"])
            selected = execute(frames, plan)
            tests[str(year)] = evaluate(selected)
            plans[str(year)] = plan
            combined.append(selected)
        all_years = {str(DEVELOPMENT_YEAR): frozen["development"], **tests}
        checks = {year: parent.objective_ok(value)
                  for year, value in all_years.items()}
        status = ("STOCK_CAUSAL_ROTATION_RESEARCH_CANDIDATE"
                  if all(checks.values()) else "REJECTED_OBJECTIVE")
        pd.concat(combined, ignore_index=True).sort_values(
            ["entry_time", "symbol"]).to_pickle(ROWS_OUT)

    public_frozen = None if not frozen else {
        "policy": frozen["policy"], "score": frozen["score"],
        "development": frozen["development"], "plan": frozen["plan"],
    }
    report = {
        "status": status, "ready_for_live_orders": False,
        "protocol": {
            "source_fit": "2017-2021", "source_profile_selection": 2022,
            "rotation_policy_selection": DEVELOPMENT_YEAR,
            "frozen_test": list(TEST_YEARS),
            "selection_uses_only_exits_before_month_start": True,
            "priority": "account return first; win rate diagnostic only",
            "entry": "market", "orders": "market only",
            "money": "3% current-equity margin, fixed 10x gross",
            "objective": ">=30% account return in every month; >=3 trades/day; DD <=10%",
        },
        "policy_count": len(policy_grid()),
        "unique_development_plans": len(cache),
        "eligible_policies": len(candidates),
        "development_top20": candidates[:20],
        "frozen_before_2024": public_frozen,
        "test_2024_2026": tests, "test_plans": plans,
        "objective_checks": checks,
        "sources": {str(path): digest(path) for path in SOURCES.values()},
        "limitations": [
            "Source pickles contain causally gated source selections, not every raw proposal.",
            "Cash-session data omit overnight X-Perp price discovery.",
            "Policy selection on one calendar year has selection risk.",
            "December entry cohorts may realize exits in the following January.",
            "Gross returns exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": status, "frozen": public_frozen,
                      "tests": tests, "checks": checks}, indent=2), flush=True)


if __name__ == "__main__":
    main()
