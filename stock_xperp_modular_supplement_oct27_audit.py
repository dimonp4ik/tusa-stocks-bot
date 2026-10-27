"""Test module-local loss pauses for the mature X-Perp supplement.

The preceding audit showed a cascade: supplement outcomes changed the global
daily loss pause and replaced many otherwise accepted frozen-universe trades.
This diagnostic keeps every exposure/cooldown/hour rail global while tracking
the two-loss daily pause independently for the frozen and supplement modules.
September selects age and pause policy; October remains a retrospective test
because another supplement architecture has already exposed that month.
"""
from __future__ import annotations

import collections
import heapq
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.stock_xperp_native_shadow import AUDITED_SYMBOLS
import stock_xperp_expanded_universe_oct27_audit as expanded
import stock_xperp_mature_supplement_oct27_audit as mature
import stock_xperp_native_cross_section_oct18_audit as parent


OUT = mature.ROOT / "stock_xperp_modular_supplement_oct27.json"
ROWS_OUT = mature.ROOT / "stock_xperp_modular_supplement_oct27.csv"


def module_pause_gate(rows: list[dict], loss_streak_cap: int = 2) -> list[dict]:
    """Apply global exposure rails and a causal daily pause per module."""
    if loss_streak_cap < 1:
        raise ValueError("loss_streak_cap must be positive")
    old = set(AUDITED_SYMBOLS)
    rank = lambda row: (0 if row["symbol"] in old else 1,
                        -abs(float(row["cross_z"])), row["symbol"])
    pending: list[tuple[float, int, dict]] = []
    opened: dict[str, dict] = {}
    last: dict[tuple[str, str], float] = {}
    per_scan: collections.Counter[int] = collections.Counter()
    recent_entries: collections.deque[float] = collections.deque()
    output: list[dict] = []
    day = None
    streak: collections.Counter[str] = collections.Counter()
    paused: set[str] = set()
    for row in sorted(rows, key=lambda item: (item["entry_time"], rank(item))):
        now = float(row["entry_time"])
        while recent_entries and recent_entries[0] <= now - 3600:
            recent_entries.popleft()
        new_day = int(now // 86400)
        if new_day != day:
            day = new_day
            streak.clear()
            paused.clear()
        while pending and pending[0][0] < now:
            _, _, done = heapq.heappop(pending)
            opened.pop(done["symbol"], None)
            if int(float(done["exit_time"]) // 86400) == day:
                module = str(done["module"])
                signed = done.get("signed_return")
                loss = (signed is not None and not math.isnan(float(signed))
                        and float(signed) < 0)
                streak[module] = streak[module] + 1 if loss else 0
                if streak[module] >= loss_streak_cap:
                    paused.add(module)
        module = str(row["module"])
        if module in paused or row["symbol"] in opened:
            continue
        key = (row["symbol"], row["direction"])
        if now - last.get(key, -1e20) < 10800:
            continue
        scan = int(now // 900)
        if per_scan[scan] >= 2 or len(recent_entries) >= 4:
            continue
        if sum(item["direction"] == row["direction"]
               for item in opened.values()) >= 4:
            continue
        opened[row["symbol"]] = row
        last[key] = now
        per_scan[scan] += 1
        recent_entries.append(now)
        output.append(row)
        heapq.heappush(pending, (float(row["exit_time"]), len(output), row))
    return output


def accepted(base: pd.DataFrame, supplement: pd.DataFrame,
             mode: str) -> pd.DataFrame:
    combined = pd.concat([base, supplement], ignore_index=True)
    if mode == "global_pause":
        return mature.accepted(base, supplement)
    if mode != "module_pause":
        raise ValueError(f"unknown mode: {mode}")
    rows = module_pause_gate(combined.to_dict("records"), 2)
    return pd.DataFrame(rows, columns=combined.columns)


def main() -> None:
    candles, sources = expanded.load_manifest()
    expanded.same_snapshot_overlap(candles)
    base_candles = {symbol: candles[symbol] for symbol in AUDITED_SYMBOLS}
    new_candles = {symbol: values for symbol, values in candles.items()
                   if symbol not in AUDITED_SYMBOLS}
    base_features = parent.feature_rows(base_candles, 2)
    context = mature.stable_context(base_features)
    base_guarded = mature.guarded(base_features, context, base_candles,
                                  "native_xperp_frozen")
    new_features = mature.supplement_features(new_candles, context)
    new_guarded = mature.guarded(new_features, context, candles,
                                 "native_xperp_mature_supplement")
    base_rows = accepted(base_guarded, base_guarded.iloc[0:0], "module_pause")
    baseline_development = mature.metrics(base_rows, mature.DEVELOPMENT)
    candidates, rows_by_key = [], {}
    for mode in ("global_pause", "module_pause"):
        for age_days in mature.AGE_DAYS:
            supplement = mature.apply_age(new_guarded, new_candles, age_days)
            rows = accepted(base_guarded, supplement, mode)
            development = mature.metrics(rows, mature.DEVELOPMENT)
            passes = mature.development_pass(
                development, baseline_development)
            item = {"pause_mode": mode, "minimum_age_days": age_days,
                    "development": development,
                    "development_pass": passes}
            candidates.append(item)
            rows_by_key[(mode, age_days)] = rows
    survivors = [item for item in candidates if item["development_pass"]]
    survivors.sort(key=lambda item: (
        -item["development"]["margin_3pct"]["return_pct"],
        item["development"]["margin_3pct"]["dd_pct"],
        item["pause_mode"], -item["minimum_age_days"]))
    selected = survivors[0] if survivors else None
    test = None
    checks = {"development_candidate_exists": selected is not None}
    if selected:
        rows = rows_by_key[(selected["pause_mode"],
                            selected["minimum_age_days"])]
        rows.to_csv(ROWS_OUT, index=False)
        test = {"baseline": mature.metrics(base_rows, mature.TEST),
                "candidate": mature.metrics(rows, mature.TEST)}
        base = test["baseline"]["margin_3pct"]
        value = test["candidate"]["margin_3pct"]
        checks.update({
            "test_return_improves": value["return_pct"] > base["return_pct"],
            "test_drawdown_at_most_5pct": value["dd_pct"] <= 5,
            "test_frequency_at_least_3": value["trades_per_active_day"] >= 3,
            "test_no_negative_month": value["negative_months"] == 0,
        })
    passed = bool(checks and all(checks.values()))
    result = {
        "status": ("RETROSPECTIVE_SHADOW_CANDIDATE_NEEDS_NEW_FORWARD"
                   if passed else "REJECTED_NO_ROBUST_IMPROVEMENT"),
        "ready_for_live_orders": False,
        "orders_possible": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "mechanism": "global exposure rails with per-module daily loss pauses",
            "pause_modes": ["global_pause", "module_pause"],
            "minimum_age_days_grid": list(mature.AGE_DAYS),
            "selection": "September development; October retrospective test",
            "market_orders_only": True,
        },
        "baseline_development": baseline_development,
        "candidates": candidates,
        "selected": selected,
        "test": test,
        "checks": checks,
        "accepted": passed,
        "sources": {**sources, str(Path(__file__).resolve()):
                    mature.digest(Path(__file__).resolve())},
        "limitations": [
            "The cascade diagnosis and October month were known before this repair.",
            "A passing repair cannot be promoted without genuinely new forward data.",
            "Gross returns exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    compact = [{"mode": x["pause_mode"], "age": x["minimum_age_days"],
                "return": x["development"]["margin_3pct"]["return_pct"],
                "dd": x["development"]["margin_3pct"]["dd_pct"],
                "pass": x["development_pass"]} for x in candidates]
    print(json.dumps({"status": result["status"], "candidates": compact,
                      "selected": selected, "test": test,
                      "checks": checks}, indent=2), flush=True)


if __name__ == "__main__":
    main()
