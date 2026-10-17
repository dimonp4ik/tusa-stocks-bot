"""Audit a causal per-module cap for simultaneous correlated stock signals."""
from __future__ import annotations

import argparse
import collections
import heapq
import json
import math
from pathlib import Path
from typing import Callable

import pandas as pd

from stock_forward_monitor import metrics


def portfolio_gate(rows: list[dict], module_cap: int | None,
                   hour_cap: int | None = None,
                   rolling_hour_cap: int | None = None,
                   weak_index_max_ratio: float | None = None,
                   rank_key: Callable[[dict], object] | None = None,
                   loss_streak_cap: int = 3) -> list[dict]:
    if loss_streak_cap < 1:
        raise ValueError("loss_streak_cap must be positive")
    pending: list[tuple[float, int, dict]] = []
    opened: dict[str, dict] = {}
    last: dict[tuple[str, str], float] = {}
    per_scan: collections.Counter[int] = collections.Counter()
    per_module: collections.Counter[tuple[int, str]] = collections.Counter()
    per_hour: collections.Counter[int] = collections.Counter()
    recent_entries: collections.deque[float] = collections.deque()
    output: list[dict] = []
    day = None
    streak = 0
    paused = False
    for row in sorted(rows, key=lambda item: (
            item["entry_time"],
            rank_key(item) if rank_key is not None else (),
            item["symbol"])):
        now = float(row["entry_time"])
        while recent_entries and recent_entries[0] <= now - 3600:
            recent_entries.popleft()
        new_day = int(now // 86400)
        if new_day != day:
            day, streak, paused = new_day, 0, False
        while pending and pending[0][0] < now:
            _, _, done = heapq.heappop(pending)
            opened.pop(done["symbol"], None)
            if int(float(done["exit_time"]) // 86400) == day:
                signed_return = done.get("signed_return")
                realized_loss = (
                    signed_return is not None
                    and not math.isnan(float(signed_return))
                    and float(signed_return) < 0)
                streak = streak + 1 if realized_loss else 0
                if streak >= loss_streak_cap:
                    paused = True
        if paused or row["symbol"] in opened:
            continue
        key = (row["symbol"], row["direction"])
        if now - last.get(key, -1e20) < 10800:
            continue
        scan = int(now // 900)
        hour = int(now // 3600)
        module_key = (scan, str(row["module"]))
        if per_scan[scan] >= 2:
            continue
        if hour_cap is not None and per_hour[hour] >= hour_cap:
            continue
        if rolling_hour_cap is not None and len(recent_entries) >= rolling_hour_cap:
            continue
        if module_cap is not None and per_module[module_key] >= module_cap:
            continue
        if weak_index_max_ratio is not None:
            qqq = abs(float(row.get("qqq_signal_range_atr") or 0))
            spy = abs(float(row.get("spy_signal_range_atr") or 0))
            inconsistent = (qqq <= 0 or spy <= 0 or
                            max(qqq / spy, spy / qqq) > weak_index_max_ratio)
            if inconsistent and per_module[module_key] >= 1:
                continue
        if sum(item["direction"] == row["direction"] for item in opened.values()) >= 4:
            continue
        opened[row["symbol"]] = row
        last[key] = now
        per_scan[scan] += 1
        per_hour[hour] += 1
        recent_entries.append(now)
        per_module[module_key] += 1
        output.append(row)
        heapq.heappush(pending, (float(row["exit_time"]), len(output), row))
    return output


def window(rows: list[dict], start: str, end: str | None = None) -> list[dict]:
    lower = pd.Timestamp(start, tz="UTC").timestamp()
    upper = pd.Timestamp(end, tz="UTC").timestamp() if end else float("inf")
    return [row for row in rows if lower <= float(row["entry_time"]) < upper]


def split(rows: list[dict]) -> dict:
    return {
        "calibration_july_august": metrics(window(rows, "2026-07-01", "2026-09-01")),
        "validation_sep_1_15": metrics(window(rows, "2026-09-01", "2026-09-16")),
        "all_reference": metrics(rows),
    }


def index_range_consistent(rows: list[dict], max_ratio: float) -> list[dict]:
    """Require QQQ and SPY closed bars to show comparable nonzero movement."""
    output = []
    for row in rows:
        qqq = abs(float(row.get("qqq_signal_range_atr") or 0))
        spy = abs(float(row.get("spy_signal_range_atr") or 0))
        if qqq <= 0 or spy <= 0:
            continue
        ratio = max(qqq / spy, spy / qqq)
        if ratio <= max_ratio:
            output.append(row)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--fresh-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    reference = pd.read_csv(args.reference).to_dict("records")
    fresh = json.loads(args.fresh_report.read_text(encoding="utf-8"))["fresh_raw"]
    policies = {
        "baseline": portfolio_gate(reference, None),
        "module_cap_1": portfolio_gate(reference, 1),
        "hour_cap_2": portfolio_gate(reference, None, 2),
        "hour_cap_3": portfolio_gate(reference, None, 3),
        "rolling_60m_cap_2": portfolio_gate(reference, None, None, 2),
        "rolling_60m_cap_3": portfolio_gate(reference, None, None, 3),
        "index_range_ratio_2": portfolio_gate(index_range_consistent(reference, 2), None),
        "index_range_ratio_3": portfolio_gate(index_range_consistent(reference, 3), None),
        "index_range_ratio_4": portfolio_gate(index_range_consistent(reference, 4), None),
        "weak_index_module_cap_ratio_2": portfolio_gate(reference, None, weak_index_max_ratio=2),
        "weak_index_module_cap_ratio_3": portfolio_gate(reference, None, weak_index_max_ratio=3),
        "weak_index_module_cap_ratio_4": portfolio_gate(reference, None, weak_index_max_ratio=4),
    }
    fresh_policies = {
        "baseline": portfolio_gate(fresh, None),
        "module_cap_1": portfolio_gate(fresh, 1),
        "hour_cap_2": portfolio_gate(fresh, None, 2),
        "hour_cap_3": portfolio_gate(fresh, None, 3),
        "rolling_60m_cap_2": portfolio_gate(fresh, None, None, 2),
        "rolling_60m_cap_3": portfolio_gate(fresh, None, None, 3),
        "index_range_ratio_2": portfolio_gate(index_range_consistent(fresh, 2), None),
        "index_range_ratio_3": portfolio_gate(index_range_consistent(fresh, 3), None),
        "index_range_ratio_4": portfolio_gate(index_range_consistent(fresh, 4), None),
        "weak_index_module_cap_ratio_2": portfolio_gate(fresh, None, weak_index_max_ratio=2),
        "weak_index_module_cap_ratio_3": portfolio_gate(fresh, None, weak_index_max_ratio=3),
        "weak_index_module_cap_ratio_4": portfolio_gate(fresh, None, weak_index_max_ratio=4),
    }
    result = {
        "hypothesis": (
            "signals from the same module and closed 15-minute bar are one "
            "correlated market bet; admit at most one while keeping total cap=2"
        ),
        "fresh_not_used_for_acceptance": True,
        **{name: {**split(rows), "fresh_sep_16_21": metrics(fresh_policies[name])}
           for name, rows in policies.items()},
    }
    base = result["baseline"]
    accepted = []
    for name in ("module_cap_1", "hour_cap_2", "hour_cap_3",
                 "rolling_60m_cap_2", "rolling_60m_cap_3",
                 "index_range_ratio_2", "index_range_ratio_3",
                 "index_range_ratio_4", "weak_index_module_cap_ratio_2",
                 "weak_index_module_cap_ratio_3", "weak_index_module_cap_ratio_4"):
        candidate = result[name]
        if all(candidate[period]["net_r"] >= .95 * base[period]["net_r"]
               and candidate[period]["dd_r"] <= base[period]["dd_r"]
               for period in ("calibration_july_august", "validation_sep_1_15")):
            accepted.append(name)
    result["acceptance_rule"] = (
        ">=95% of baseline net R and drawdown no higher in both July-August calibration "
        "and Sep 1-15 validation"
    )
    result["accepted_before_fresh"] = accepted
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
