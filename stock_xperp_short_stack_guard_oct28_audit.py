"""Audit a causal guard against stacking shorts into a rising stock basket.

The rule was frozen after inspecting the Oct-27 snapshot: when one SHORT has
already been admitted from a 15-minute scan, reject a second SHORT from that
same scan if the closed 96-bar basket trend is positive.  The Oct-28-renewed
snapshot is used only for append-only forward evidence; it is not used to
change the rule.
"""
from __future__ import annotations

import collections
import hashlib
import heapq
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import stock_xperp_backing_level_guard_oct20_audit as backing
import stock_xperp_basket_trend_long_guard_oct26_audit as frozen
import stock_xperp_basket_trend_forward_oct28_audit as forward
import stock_xperp_native_cross_section_oct18_audit as parent
import stock_xperp_native_growth_forward_monitor as monitor
import stock_xperp_native_quality_guard_oct18_audit as quality


ROOT = Path("reports/audit_2026_09_08")
OLD_MANIFEST = ROOT / "stock_xperp_forward_oct27_2156" / "manifest.json"
NEW_MANIFEST = ROOT / "stock_xperp_forward_oct28_renewed" / "manifest.json"
OLD_PUBLISHED_ROWS = ROOT / "stock_xperp_basket_trend_forward_oct28.csv"
OUT = ROOT / "stock_xperp_short_stack_guard_oct28.json"
ROWS_OUT = ROOT / "stock_xperp_short_stack_guard_oct28.csv"
PARAMETERS = {"target_pct": 2.0, "stop_pct": 1.0, "hold_bars": 64}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def candidates(manifest_path: Path) -> tuple[pd.DataFrame, dict, dict]:
    candles, sources, _ = monitor.load_manifest(manifest_path)
    features = parent.feature_rows(candles, 2)
    entries = parent.select_entries(features, 2.0, .004)
    entries = forward.volatility.add_atr14(entries, candles)
    entries = entries[entries.atr14_fraction.le(.01)]
    entries = backing.guarded(backing.add_backing_distance(entries, candles, 32))
    context = features.groupby("entry_time", as_index=False).cross_mean.first()
    context.sort_values("entry_time", inplace=True)
    context.rename(columns={"cross_mean": "basket_bar_return"}, inplace=True)
    for horizon in frozen.HORIZONS:
        context[f"basket_trend_{horizon}"] = context.basket_bar_return.rolling(
            horizon, min_periods=horizon).sum()
    entries = entries.merge(context, on="entry_time", how="left",
                            validate="many_to_one")
    outcomes = pd.DataFrame(monitor.resolve_entries(entries, candles, PARAMETERS))
    outcomes["module"] = "native_xperp_basket_trend_guard"
    outcomes["module_priority"] = 0
    guarded = frozen.apply(outcomes, 96, "LONG", -0.005626823134606253)
    return guarded, sources, candles


def portfolio_gate(rows: list[dict], short_stack_guard: bool) -> list[dict]:
    """Replay the production rails with the optional frozen second-short veto."""
    pending: list[tuple[float, int, dict]] = []
    opened: dict[str, dict] = {}
    last: dict[tuple[str, str], float] = {}
    per_scan: collections.Counter[int] = collections.Counter()
    per_hour: collections.Counter[int] = collections.Counter()
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
        new_day = int(now // 86400)
        if new_day != day:
            day, streak, paused = new_day, 0, False
        while pending and pending[0][0] < now:
            _, _, done = heapq.heappop(pending)
            opened.pop(done["symbol"], None)
            if int(float(done["exit_time"]) // 86400) == day:
                signed = done.get("signed_return")
                loss = signed is not None and not math.isnan(float(signed)) \
                    and float(signed) < 0
                streak = streak + 1 if loss else 0
                if streak >= 2:
                    paused = True
        if paused or row["symbol"] in opened:
            continue
        symbol_direction = (row["symbol"], row["direction"])
        if now - last.get(symbol_direction, -1e20) < 10800:
            continue
        scan = int(now // 900)
        hour = int(now // 3600)
        if per_scan[scan] >= 2 or len(recent_entries) >= 4:
            continue
        if sum(item["direction"] == row["direction"]
               for item in opened.values()) >= 4:
            continue
        if (short_stack_guard and str(row["direction"]) == "SHORT"
                and float(row.get("basket_trend_96") or 0) > 0
                and any(str(item["direction"]) == "SHORT"
                        for item in admitted_in_scan[scan])):
            continue
        opened[row["symbol"]] = row
        last[symbol_direction] = now
        per_scan[scan] += 1
        per_hour[hour] += 1
        recent_entries.append(now)
        admitted_in_scan[scan].append(row)
        output.append(row)
        heapq.heappush(pending, (float(row["exit_time"]), len(output), row))
    return output


def accepted(frame: pd.DataFrame, guard: bool) -> pd.DataFrame:
    return pd.DataFrame(portfolio_gate(frame.to_dict("records"), guard),
                        columns=frame.columns)


def metrics(frame: pd.DataFrame, margin: float = .03) -> dict:
    return forward.metrics(frame, margin)


def month(frame: pd.DataFrame, value: str) -> pd.DataFrame:
    stamp = pd.to_datetime(frame.entry_time, unit="s", utc=True)
    return frame[stamp.dt.strftime("%Y-%m").eq(value)].copy()


def compare(base: pd.DataFrame, guarded: pd.DataFrame) -> dict:
    return {
        "baseline": metrics(base),
        "guarded": metrics(guarded),
        "accepted_delta": len(guarded) - len(base),
        "return_delta_pp": round(
            metrics(guarded)["return_pct"] - metrics(base)["return_pct"], 4),
        "dd_delta_pp": round(
            metrics(guarded)["dd_pct"] - metrics(base)["dd_pct"], 4),
    }


def identity(frame: pd.DataFrame) -> set[tuple]:
    return set(zip(frame.symbol.astype(str),
                   pd.to_numeric(frame.entry_time).astype("int64"),
                   frame.direction.astype(str)))


def main() -> None:
    old_candidates, old_sources, old_candles = candidates(OLD_MANIFEST)
    new_candidates, new_sources, new_candles = candidates(NEW_MANIFEST)
    old_base = accepted(old_candidates, False)
    old_guard = accepted(old_candidates, True)
    new_base = accepted(new_candidates, False)
    new_guard = accepted(new_candidates, True)

    published = pd.read_csv(OLD_PUBLISHED_ROWS)
    parity = {
        "published_rows": len(published), "replayed_rows": len(old_base),
        "identity_match": identity(published) == identity(old_base),
    }
    parity["passes"] = bool(parity["identity_match"]
                            and parity["published_rows"] == parity["replayed_rows"])
    if not parity["passes"]:
        raise ValueError(f"production baseline parity failed: {parity}")

    old_end = min(int(candle["time"][-1]) for candle in old_candles.values())
    stable_cutoff = old_end - PARAMETERS["hold_bars"] * 900
    # A fixed 12,000-bar refresh necessarily drops the earliest 43 bars when
    # 43 new bars are appended.  Start parity in July, well beyond every
    # feature warm-up, so the comparison covers the strategy's selected and
    # forward periods without treating the shifted cache boundary as drift.
    parity_start = int(pd.Timestamp("2026-07-01", tz="UTC").timestamp())
    old_stable = old_base[old_base.entry_time.between(parity_start, stable_cutoff)]
    new_stable = new_base[new_base.entry_time.between(parity_start, stable_cutoff)]
    append_parity = {
        "through_entry_utc": pd.Timestamp(
            stable_cutoff, unit="s", tz="UTC").isoformat(),
        "old_rows": len(old_stable), "new_rows": len(new_stable),
        "identity_match": identity(old_stable) == identity(new_stable),
    }
    append_parity["passes"] = bool(
        append_parity["identity_match"]
        and append_parity["old_rows"] == append_parity["new_rows"])
    if not append_parity["passes"]:
        raise ValueError(f"append-only baseline parity failed: {append_parity}")

    # Outcomes closing after the freeze are genuinely new for this rule.  A
    # position that was already open at the freeze is valid forward evidence:
    # its future target/stop result had not yet been observed.
    def fresh(frame: pd.DataFrame) -> pd.DataFrame:
        return frame[frame.outcome.ne("PENDING") & frame.exit_time.gt(old_end)].copy()

    old_months = {
        value: compare(month(old_base, value), month(old_guard, value))
        for value in ("2026-07", "2026-08", "2026-09", "2026-10")
    }
    fresh_base, fresh_guard = fresh(new_base), fresh(new_guard)
    new_guard.to_csv(ROWS_OUT, index=False)
    forward_comparison = compare(fresh_base, fresh_guard)
    enough_forward = min(len(fresh_base), len(fresh_guard)) >= 30
    forward_better = bool(
        enough_forward
        and forward_comparison["guarded"]["return_pct"]
            > forward_comparison["baseline"]["return_pct"]
        and forward_comparison["guarded"]["dd_pct"]
            <= forward_comparison["baseline"]["dd_pct"])
    report = {
        "status": ("FORWARD_GUARD_CANDIDATE" if forward_better
                   else "AWAIT_MORE_FORWARD_OUTCOMES"),
        "ready_for_live_orders": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "policy_frozen_before_new_snapshot": {
            "rule": ("after one SHORT is admitted from a 15-minute scan, "
                     "reject another SHORT from that scan when the closed "
                     "96-bar basket trend is positive"),
            "orders": "market only", "changes_position_size": False,
            "freeze_candle_end_utc": pd.Timestamp(
                old_end, unit="s", tz="UTC").isoformat(),
        },
        "integrity": {
            "published_baseline": parity,
            "append_only_stable_prefix": append_parity,
        },
        "retrospective_visible_months": old_months,
        "new_outcomes_after_freeze": forward_comparison,
        "forward_gate": {
            "required_resolved_per_route": 30,
            "baseline_resolved": len(fresh_base),
            "guarded_resolved": len(fresh_guard),
            "enough_outcomes": enough_forward,
            "guard_better_return_and_no_worse_dd": forward_better,
        },
        "sources": {
            **old_sources, **new_sources,
            str(OLD_MANIFEST): digest(OLD_MANIFEST),
            str(NEW_MANIFEST): digest(NEW_MANIFEST),
            str(OLD_PUBLISHED_ROWS): digest(OLD_PUBLISHED_ROWS),
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "The rule was formulated after all retrospective months were visible.",
            "Only outcomes after the frozen Oct-27 candle are forward evidence.",
            "The venue-native history begins in 2026.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "integrity": report["integrity"],
        "retrospective_visible_months": old_months,
        "new_outcomes_after_freeze": forward_comparison,
        "forward_gate": report["forward_gate"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
