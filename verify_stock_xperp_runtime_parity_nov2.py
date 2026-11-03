"""Verify the deployed native setup function against the independent replay.

The comparison uses every aligned 15-minute decision from November 1 through
the frozen November 2 snapshot.  It checks candidate identity, quality order,
causal features, the backing-level market-open guard, and target routing.
"""
from __future__ import annotations

from bisect import bisect_right
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import stock_xperp_backing_level_guard_oct20_audit as backing
import stock_xperp_basket_trend_forward_oct28_audit as forward
import stock_xperp_basket_trend_long_guard_oct26_audit as long_guard
import stock_xperp_native_cross_section_oct18_audit as parent
import stock_xperp_native_growth_forward_monitor as monitor
import stock_xperp_short_exhaustion_guard_oct30_audit as short_guard
from src.stock_xperp_native_shadow import (
    AUDITED_SYMBOLS,
    BACKING_LEVEL_LOOKBACK_BARS,
    LONG_MIN_BASKET_TREND,
    REQUIRED_HISTORY_BARS,
    SHORT_MIN_BASKET_TREND,
    STOP_FRACTION,
    backing_level_allows_entry,
    native_shadow_setups,
    routed_target_fraction,
)


ROOT = Path("reports/audit_2026_09_08")
MANIFEST = ROOT / "stock_xperp_forward_nov2_stop_review" / "frozen_manifest.json"
OUT = ROOT / "verify_stock_xperp_runtime_parity_nov2.json"
START = pd.Timestamp("2026-11-01", tz="UTC").timestamp()
TOLERANCE = 1e-10


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def independent_candidates(candles: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build candidates without calling the production setup function."""
    features = parent.feature_rows(candles, 2)
    entries = parent.select_entries(features, 2.0, .004)
    entries["direction"] = np.where(
        entries.direction_n.gt(0), "LONG", "SHORT")
    entries = forward.volatility.add_atr14(entries, candles)
    entries = entries[entries.atr14_fraction.le(.01)].copy()
    entries = backing.guarded(backing.add_backing_distance(
        entries, candles, BACKING_LEVEL_LOOKBACK_BARS))
    context = features.groupby("entry_time", as_index=False).cross_mean.first()
    context.sort_values("entry_time", inplace=True)
    context.rename(columns={"cross_mean": "basket_bar_return"}, inplace=True)
    for horizon in (16, 32, 96):
        context[f"basket_trend_{horizon}"] = context.basket_bar_return.rolling(
            horizon, min_periods=horizon).sum()
    entries = entries.merge(context, on="entry_time", how="left",
                            validate="many_to_one")
    entries = long_guard.apply(
        entries, 96, "LONG", LONG_MIN_BASKET_TREND)
    entries = short_guard.apply_guard(entries, SHORT_MIN_BASKET_TREND)
    return entries, features


def history_at(candles: dict, signal_time: int) -> dict:
    output = {}
    for symbol in sorted(AUDITED_SYMBOLS & set(candles)):
        source = candles[symbol]
        times = [int(value) for value in source["time"]]
        right = bisect_right(times, int(signal_time))
        if not right or times[right - 1] != int(signal_time):
            continue
        left = max(0, right - REQUIRED_HISTORY_BARS)
        output[symbol] = {
            key: list(values[left:right]) for key, values in source.items()
        }
    return output


def next_open(candles: dict, symbol: str, entry_time: int) -> float | None:
    source = candles[symbol]
    times = [int(value) for value in source["time"]]
    index = bisect_right(times, int(entry_time)) - 1
    if index < 0 or times[index] != int(entry_time):
        return None
    return float(source["open"][index])


def expected_backing_level(row, candles: dict) -> float:
    source = candles[str(row.symbol)]
    index = int(row.entry_index)
    start = index - BACKING_LEVEL_LOOKBACK_BARS
    key = "low" if str(row.direction) == "LONG" else "high"
    values = np.asarray(source[key][start:index], dtype=float)
    return float(values.min() if str(row.direction) == "LONG" else values.max())


def identity(row) -> tuple[str, int, str]:
    return str(row["symbol"]), int(float(row["entry_time"])), str(row["direction"])


def main() -> None:
    candles, sources, _ = monitor.load_manifest(MANIFEST)
    expected, features = independent_candidates(candles)
    end = min(int(values["time"][-1]) for values in candles.values()) + 900
    expected = expected[
        expected.entry_time.ge(START) & expected.entry_time.le(end)].copy()
    aligned_times = sorted(
        int(value) for value in features.entry_time.unique()
        if START <= float(value) <= end)

    expected_by_time = {
        int(stamp): frame.copy()
        for stamp, frame in expected.groupby(expected.entry_time.astype("int64"))
    }
    missing = []
    unexpected = []
    feature_mismatches = []
    target_mismatches = []
    rank_mismatches = []
    observed_count = 0
    checked_pairs = 0
    max_abs_error = {name: 0.0 for name in (
        "cross_z", "signal_return", "atr14_fraction",
        "basket_trend_16", "basket_trend_32", "basket_trend_96",
        "backing_level",
    )}

    for entry_time in aligned_times:
        signal_time = entry_time - 900
        runtime_rows = native_shadow_setups(
            history_at(candles, signal_time), now=entry_time + 1)
        admitted = []
        for row in runtime_rows:
            market = next_open(candles, str(row["symbol"]), entry_time)
            if market is not None and backing_level_allows_entry(row, market):
                admitted.append(row)
        observed_count += len(admitted)

        exp_frame = expected_by_time.get(entry_time, expected.iloc[0:0]).copy()
        if len(exp_frame):
            exp_frame["_runtime_rank"] = -exp_frame.cross_z.abs()
            exp_frame.sort_values(
                ["_runtime_rank", "symbol"], inplace=True, kind="stable")
        exp_rows = exp_frame.to_dict("records")
        expected_ids = [identity(row) for row in exp_rows]
        observed_ids = [identity({**row, "entry_time": entry_time}) for row in admitted]
        missing.extend(sorted(set(expected_ids) - set(observed_ids)))
        unexpected.extend(sorted(set(observed_ids) - set(expected_ids)))
        if expected_ids != observed_ids:
            rank_mismatches.append({
                "entry_time": entry_time,
                "expected": expected_ids,
                "observed": observed_ids,
            })

        exp_lookup = {identity(row): row for row in exp_rows}
        for runtime in admitted:
            key = identity({**runtime, "entry_time": entry_time})
            if key not in exp_lookup:
                continue
            checked_pairs += 1
            audit = exp_lookup[key]
            comparisons = {
                "cross_z": (runtime["cross_z"], audit["cross_z"]),
                "signal_return": (runtime["signal_return"], audit["signal_return"]),
                "atr14_fraction": (runtime["atr14_fraction"], audit["atr14_fraction"]),
                "basket_trend_16": (runtime["basket_trend_16"], audit["basket_trend_16"]),
                "basket_trend_32": (runtime["basket_trend_32"], audit["basket_trend_32"]),
                "basket_trend_96": (runtime["basket_trend_96"], audit["basket_trend_96"]),
                "backing_level": (
                    runtime["backing_level"],
                    expected_backing_level(pd.Series(audit), candles)),
            }
            for name, (actual, wanted) in comparisons.items():
                error = abs(float(actual) - float(wanted))
                max_abs_error[name] = max(max_abs_error[name], error)
                if error > TOLERANCE:
                    feature_mismatches.append({
                        "identity": key, "feature": name,
                        "runtime": float(actual), "audit": float(wanted),
                        "abs_error": error,
                    })
            target = routed_target_fraction(
                float(audit["basket_trend_32"]), entry_time) / STOP_FRACTION
            if abs(float(runtime["target_r"]) - target) > TOLERANCE:
                target_mismatches.append({
                    "identity": key, "runtime": float(runtime["target_r"]),
                    "audit": target,
                })

    passed = not any((missing, unexpected, feature_mismatches,
                      target_mismatches, rank_mismatches))
    report = {
        "status": "RUNTIME_REPLAY_PARITY_PASSED" if passed else "RUNTIME_REPLAY_PARITY_FAILED",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "orders_possible": False,
        "ready_for_live_orders": False,
        "period": {
            "start_entry_utc": pd.Timestamp(START, unit="s", tz="UTC").isoformat(),
            "end_entry_utc": pd.Timestamp(end, unit="s", tz="UTC").isoformat(),
        },
        "checks": {
            "timestamps_checked": len(aligned_times),
            "expected_candidates": len(expected),
            "runtime_candidates": observed_count,
            "matched_feature_rows": checked_pairs,
            "missing_candidates": missing,
            "unexpected_candidates": unexpected,
            "rank_mismatches": rank_mismatches,
            "feature_mismatches": feature_mismatches,
            "target_mismatches": target_mismatches,
            "max_abs_feature_error": max_abs_error,
            "passes": passed,
        },
        "contract": {
            "signal_data": "confirmed candles through signal bar",
            "entry": "next 15-minute market open",
            "backing_level": f"prior {BACKING_LEVEL_LOOKBACK_BARS} closed bars",
            "candidate_order": "descending absolute cross-sectional z, then symbol",
            "target_router": "basket trend 32 plus UTC entry hour",
        },
        "sources": {
            **sources,
            str(MANIFEST): digest(MANIFEST),
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "Parity proves deterministic signal logic, not future profitability.",
            "Historical next-bar open substitutes for live market price.",
            "The comparison ends at the frozen November 2 candle snapshot.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "period": report["period"],
        "checks": report["checks"],
    }, indent=2), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
