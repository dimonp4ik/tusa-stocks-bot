"""Test feature-confirmed overflow slots above the frozen 4/4 capacity rails."""
from __future__ import annotations

import collections
import hashlib
import heapq
import json
import math
from pathlib import Path
from typing import Callable

import pandas as pd

import stock_xperp_profit_first_capacity_oct31_audit as capacity
import stock_xperp_native_economics_oct18_audit as economics
import stock_xperp_native_quality_guard_oct18_audit as quality


OUT = capacity.parent.ROOT / "stock_xperp_conditional_capacity_oct31.json"
Predicate = Callable[[dict], bool]
PERIODS = capacity.PERIODS


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def predicates() -> dict[str, Predicate]:
    return {
        "z_ge_3_0": lambda r: abs(float(r["cross_z"])) >= 3.0,
        "z_ge_3_5": lambda r: abs(float(r["cross_z"])) >= 3.5,
        "z_ge_4_0": lambda r: abs(float(r["cross_z"])) >= 4.0,
        "support_ge_0": lambda r: float(r["market_support"]) >= 0.0,
        "support_ge_0_001": lambda r: float(r["market_support"]) >= .001,
        "atr_le_0_0035": lambda r: float(r["atr14_fraction"]) <= .0035,
        "atr_le_0_005": lambda r: float(r["atr14_fraction"]) <= .005,
        "backing_le_0_005":
            lambda r: float(r["backing_level_distance"]) <= .005,
        "z3_support0": lambda r: (
            abs(float(r["cross_z"])) >= 3.0
            and float(r["market_support"]) >= 0.0),
        "z35_support0": lambda r: (
            abs(float(r["cross_z"])) >= 3.5
            and float(r["market_support"]) >= 0.0),
        "z3_atr005": lambda r: (
            abs(float(r["cross_z"])) >= 3.0
            and float(r["atr14_fraction"]) <= .005),
        "support0_atr005": lambda r: (
            float(r["market_support"]) >= 0.0
            and float(r["atr14_fraction"]) <= .005),
        "z3_backing01": lambda r: (
            abs(float(r["cross_z"])) >= 3.0
            and float(r["backing_level_distance"]) <= .01),
    }


def conditional_gate(rows: list[dict], overflow_ok: Predicate | None,
                     rank_name: str, loss_streak_cap: int,
                     base_direction_cap: int = 4, base_hour_cap: int = 4,
                     hard_direction_cap: int = 8,
                     hard_hour_cap: int = 8) -> list[dict]:
    """Allow slots above 4/4 only when their own closed features qualify."""
    pending: list[tuple[float, int, dict]] = []
    opened: dict[str, dict] = {}
    last: dict[tuple[str, str], float] = {}
    per_scan: collections.Counter[int] = collections.Counter()
    recent_entries: collections.deque[float] = collections.deque()
    output: list[dict] = []
    day = None
    streak = 0
    paused = False
    rank_key = quality.ranker(rank_name)
    for row in sorted(rows, key=lambda item: (
            item["entry_time"], rank_key(item), item["symbol"])):
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
                signed = done.get("signed_return")
                loss = (signed is not None and not math.isnan(float(signed))
                        and float(signed) < 0)
                streak = streak + 1 if loss else 0
                if streak >= loss_streak_cap:
                    paused = True
        if paused or row["symbol"] in opened:
            continue
        key = (row["symbol"], row["direction"])
        if now - last.get(key, -1e20) < 10800:
            continue
        scan = int(now // 900)
        if per_scan[scan] >= 2:
            continue
        same_direction = sum(
            item["direction"] == row["direction"] for item in opened.values())
        if same_direction >= hard_direction_cap or len(recent_entries) >= hard_hour_cap:
            continue
        overflow = (same_direction >= base_direction_cap
                    or len(recent_entries) >= base_hour_cap)
        if overflow and (overflow_ok is None or not overflow_ok(row)):
            continue
        opened[row["symbol"]] = row
        last[key] = now
        per_scan[scan] += 1
        recent_entries.append(now)
        output.append(row)
        heapq.heappush(pending, (float(row["exit_time"]), len(output), row))
    return output


def evaluate(raw: list[dict], predicate: Predicate | None,
             rank_name: str, loss_cap: int) -> dict:
    result = {}
    for name, bounds in PERIODS.items():
        selected = conditional_gate(
            capacity.window(raw, bounds), predicate, rank_name, loss_cap)
        result[name] = capacity.metrics(selected)
    return result


def main() -> None:
    raw, rank_name, loss_cap = capacity.raw_guarded_rows()
    candidates = [{
        "name": "baseline_4_4",
        "periods": evaluate(raw, None, rank_name, loss_cap),
    }]
    for name, predicate in predicates().items():
        candidates.append({
            "name": name,
            "periods": evaluate(raw, predicate, rank_name, loss_cap),
        })
    baseline = candidates[0]
    dev_key, val_key = "development_2026_07", "validation_2026_08"
    base_dev = baseline["periods"][dev_key]
    base_val = baseline["periods"][val_key]
    eligible = []
    for candidate in candidates[1:]:
        dev = candidate["periods"][dev_key]
        val = candidate["periods"][val_key]
        dev_ok = capacity.development_ok(dev, base_dev)
        val_ok = capacity.validation_ok(val, base_val)
        candidate["development_pass"] = dev_ok
        candidate["validation_pass"] = val_ok
        if dev_ok and val_ok:
            eligible.append(candidate)
    eligible.sort(key=lambda item: (
        min(item["periods"][dev_key]["return_pct"],
            item["periods"][val_key]["return_pct"]),
        -max(item["periods"][dev_key]["dd_pct"],
             item["periods"][val_key]["dd_pct"]),
        item["name"]), reverse=True)
    selected = eligible[0] if eligible else baseline
    hidden_keys = ("hidden_2026_09", "hidden_2026_10_01_17")
    hidden_pass = (selected is not baseline and all(
        capacity.hidden_ok(selected["periods"][name],
                           baseline["periods"][name])
        for name in hidden_keys))
    report = {
        "status": ("CONDITIONAL_CAPACITY_ACCEPTED" if hidden_pass
                   else "CONDITIONAL_CAPACITY_REJECTED"),
        "ready_for_live_orders": False,
        "protocol": {
            "base_caps": {"direction": 4, "rolling_hour": 4},
            "hard_caps": {"direction": 8, "rolling_hour": 8},
            "fixed_per_scan_cap": 2,
            "development": "2026-07",
            "validation": "2026-08",
            "hidden_reveal": ["2026-09", "2026-10-01..16"],
            "causality": "overflow decision uses only the candidate's closed-bar features",
            "money": "3% current-equity margin per trade; fixed 10x",
        },
        "baseline": baseline,
        "selected_before_hidden": selected,
        "hidden_pass": hidden_pass,
        "accepted": hidden_pass,
        "candidates": candidates,
        "sources": {
            str(capacity.OUT): digest(capacity.OUT),
            str(capacity.ROWS): digest(capacity.ROWS),
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "Direct native X-Perp history begins in July 2026.",
            "The base strategy parameters had already used September.",
            "October is an incomplete hidden period.",
            "Gross returns exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "baseline": baseline,
        "selected_before_hidden": selected,
        "hidden_pass": hidden_pass,
        "eligible": [item["name"] for item in eligible],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
