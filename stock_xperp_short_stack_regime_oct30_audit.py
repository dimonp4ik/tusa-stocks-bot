"""Sequential audit of causal same-scan SHORT concentration guards.

The mechanism addresses two distinct failure modes without changing entries,
position size, targets, stops or market-order execution: a second SHORT from
one scan is vetoed when the 96-bar basket is rising, when the preceding 16-bar
basket move is an exhausted fall, or in both regimes.  July selects one fixed
rule, August validates it, September tests it, and October through the Oct-27
freeze confirms it.  Later outcomes are reported only as forward diagnostics.
"""
from __future__ import annotations

import collections
from datetime import datetime, timezone
import hashlib
import heapq
import json
import math
from pathlib import Path

import pandas as pd

import stock_xperp_basket_trend_forward_oct30_audit as forward
import stock_xperp_native_quality_guard_oct18_audit as quality
import stock_xperp_short_stack_guard_oct30_forward as source


ROOT = Path("reports/audit_2026_09_08")
MANIFEST = ROOT / "stock_xperp_forward_oct30" / "manifest.json"
FREEZE_MANIFEST = ROOT / "stock_xperp_forward_oct27_2156" / "manifest.json"
OUT = ROOT / "stock_xperp_short_stack_regime_oct30.json"
ROWS_OUT = ROOT / "stock_xperp_short_stack_regime_oct30.csv"
POLICIES = (
    "baseline",
    "second_short_always",
    "second_short_rising96",
    "second_short_exhausted16_2pct",
    "second_short_unstable_2pct",
    "second_short_unstable_3pct",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def veto_second_short(row: dict, admitted: list[dict], policy: str) -> bool:
    if str(row["direction"]) != "SHORT":
        return False
    if not any(str(item["direction"]) == "SHORT" for item in admitted):
        return False
    rising = float(row.get("basket_trend_96") or 0) > 0
    exhausted_2 = float(row.get("basket_trend_16") or 0) <= -.02
    exhausted_3 = float(row.get("basket_trend_16") or 0) <= -.03
    return {
        "baseline": False,
        "second_short_always": True,
        "second_short_rising96": rising,
        "second_short_exhausted16_2pct": exhausted_2,
        "second_short_unstable_2pct": rising or exhausted_2,
        "second_short_unstable_3pct": rising or exhausted_3,
    }[policy]


def portfolio_gate(rows: list[dict], policy: str) -> list[dict]:
    if policy not in POLICIES:
        raise ValueError(f"unknown policy: {policy}")
    pending: list[tuple[float, int, dict]] = []
    opened: dict[str, dict] = {}
    last: dict[tuple[str, str], float] = {}
    per_scan: collections.Counter[int] = collections.Counter()
    recent_entries: collections.deque[float] = collections.deque()
    admitted_in_scan: dict[int, list[dict]] = collections.defaultdict(list)
    output: list[dict] = []
    day = None
    streak = 0
    paused = False
    rank_key = quality.ranker("z_high")
    for row in sorted(rows, key=lambda item: (
            item["entry_time"], rank_key(item), item["symbol"])):
        now = float(row["entry_time"])
        while recent_entries and recent_entries[0] <= now - 3600:
            recent_entries.popleft()
        current_day = int(now // 86400)
        if current_day != day:
            day, streak, paused = current_day, 0, False
        while pending and pending[0][0] < now:
            _, _, done = heapq.heappop(pending)
            opened.pop(done["symbol"], None)
            if int(float(done["exit_time"]) // 86400) == day:
                signed = done.get("signed_return")
                loss = (signed is not None and not math.isnan(float(signed))
                        and float(signed) < 0)
                streak = streak + 1 if loss else 0
                if streak >= 2:
                    paused = True
        if paused or row["symbol"] in opened:
            continue
        symbol_direction = (row["symbol"], row["direction"])
        if now - last.get(symbol_direction, -1e20) < 10800:
            continue
        scan = int(now // 900)
        if per_scan[scan] >= 2 or len(recent_entries) >= 4:
            continue
        if sum(item["direction"] == row["direction"]
               for item in opened.values()) >= 4:
            continue
        if veto_second_short(row, admitted_in_scan[scan], policy):
            continue
        opened[row["symbol"]] = row
        last[symbol_direction] = now
        per_scan[scan] += 1
        recent_entries.append(now)
        admitted_in_scan[scan].append(row)
        output.append(row)
        heapq.heappush(pending, (float(row["exit_time"]), len(output), row))
    return output


def accepted(candidates: pd.DataFrame, policy: str) -> pd.DataFrame:
    return pd.DataFrame(portfolio_gate(candidates.to_dict("records"), policy),
                        columns=candidates.columns)


def slice_entries(frame: pd.DataFrame, start: str,
                  end: str | None = None) -> pd.DataFrame:
    stamp = pd.to_datetime(frame.entry_time, unit="s", utc=True)
    mask = stamp.ge(pd.Timestamp(start, tz="UTC"))
    if end is not None:
        mask &= stamp.lt(pd.Timestamp(end, tz="UTC"))
    return frame[mask].copy()


def metrics(frame: pd.DataFrame) -> dict:
    return forward.metrics(frame, .03)


def evaluate(frame: pd.DataFrame, freeze_end: int) -> dict:
    freeze_iso = pd.Timestamp(freeze_end, unit="s", tz="UTC")
    before_freeze = frame[frame.entry_time.le(freeze_end)].copy()
    return {
        "selection_july": metrics(slice_entries(before_freeze, "2026-07-01", "2026-08-01")),
        "validation_august": metrics(slice_entries(before_freeze, "2026-08-01", "2026-09-01")),
        "test_september": metrics(slice_entries(before_freeze, "2026-09-01", "2026-10-01")),
        "confirmation_october_to_freeze": metrics(slice_entries(
            before_freeze, "2026-10-01",
            freeze_iso.strftime("%Y-%m-%dT%H:%M:%S"))),
        "new_outcomes_after_freeze": metrics(frame[
            frame.outcome.ne("PENDING") & frame.exit_time.gt(freeze_end)].copy()),
    }


def dominates(value: dict, baseline: dict, dd_tolerance: float = .25) -> bool:
    return bool(value["return_pct"] >= baseline["return_pct"]
                and value["dd_pct"] <= baseline["dd_pct"] + dd_tolerance
                and value["trades_per_active_day"] >= 3
                and value["negative_months"] == 0)


def main() -> None:
    candidates, sources, _ = source.candidates(MANIFEST)
    freeze = json.loads(FREEZE_MANIFEST.read_text(encoding="utf-8"))
    freeze_end = min(int(item["end"]) for item in freeze["symbols"].values()
                     if item.get("end") is not None)
    frames = {policy: accepted(candidates, policy) for policy in POLICIES}
    values = {policy: evaluate(frame, freeze_end)
              for policy, frame in frames.items()}
    baseline = values["baseline"]
    candidates_report = []
    for policy in POLICIES[1:]:
        july = values[policy]["selection_july"]
        july_base = baseline["selection_july"]
        eligible = dominates(july, july_base)
        candidates_report.append({
            "policy": policy, "selection_eligible": eligible,
            "selection_score": round(july["return_pct"]-.75*july["dd_pct"], 8),
            "periods": values[policy],
        })
    eligible = [item for item in candidates_report if item["selection_eligible"]]
    eligible.sort(key=lambda item: (-item["selection_score"], item["policy"]))
    selected = eligible[0] if eligible else None
    validation_pass = test_pass = confirmation_pass = False
    forward_better = False
    if selected:
        policy = selected["policy"]
        validation_pass = dominates(
            values[policy]["validation_august"], baseline["validation_august"])
        test_pass = validation_pass and dominates(
            values[policy]["test_september"], baseline["test_september"])
        confirmation_pass = test_pass and dominates(
            values[policy]["confirmation_october_to_freeze"],
            baseline["confirmation_october_to_freeze"])
        fresh, fresh_base = (values[policy]["new_outcomes_after_freeze"],
                             baseline["new_outcomes_after_freeze"])
        forward_better = bool(
            fresh["return_pct"] > fresh_base["return_pct"]
            and fresh["dd_pct"] <= fresh_base["dd_pct"])
        if confirmation_pass and forward_better:
            frames[policy].to_csv(ROWS_OUT, index=False)
    accepted_candidate = bool(confirmation_pass and forward_better)
    report = {
        "status": ("SHADOW_CANDIDATE_REQUIRES_MORE_FORWARD"
                   if accepted_candidate else "REJECTED_SEQUENTIAL_VALIDATION"),
        "ready_for_live_orders": False, "orders_possible": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "selection": "2026-07", "validation": "2026-08",
            "test": "2026-09", "confirmation": "2026-10 through frozen Oct-27 candle",
            "forward_diagnostic": "outcomes closing after frozen Oct-27 candle",
            "policies": list(POLICIES), "entry": "unchanged market order",
            "targets_stops_position_size": "unchanged",
        },
        "freeze_end_utc": pd.Timestamp(
            freeze_end, unit="s", tz="UTC").isoformat(),
        "baseline": baseline, "selected_on_july": selected,
        "validation_pass": validation_pass, "test_pass": test_pass,
        "confirmation_pass": confirmation_pass,
        "forward_better_return_no_worse_dd": forward_better,
        "candidate_count": len(candidates_report),
        "candidates": candidates_report,
        "sources": {**sources, str(MANIFEST): digest(MANIFEST),
                    str(FREEZE_MANIFEST): digest(FREEZE_MANIFEST),
                    str(Path(__file__).resolve()): digest(Path(__file__).resolve())},
        "limitations": [
            "All retrospective periods were already visible before this audit.",
            "The post-freeze diagnostic remains a small sample.",
            "Gross returns exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "selected_on_july": selected,
        "validation_pass": validation_pass, "test_pass": test_pass,
        "confirmation_pass": confirmation_pass,
        "forward_better": forward_better, "baseline": baseline,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
