"""Reproduce the frozen stock X-Perp route through the Nov-7 public snapshot.

This is a diagnostic, not a threshold search.  The Nov-6 loss cluster is kept
out of every acceptance decision.  Before describing fresh entries the audit
rebuilds the Nov-5 pre-portfolio rows and requires stable historical parity.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from stock_module_cluster_audit import portfolio_gate
import stock_xperp_basket_trend_forward_oct30_audit as metric
import stock_xperp_long_stress_guard_nov5_audit as long_guard
import stock_xperp_native_quality_guard_oct18_audit as quality
import stock_xperp_premarket_liquidity_guard_nov3_audit as premarket
import stock_xperp_regime_target_router_oct30_audit as router
import stock_xperp_short_exhaustion_guard_oct30_audit as short_guard
import stock_xperp_short_stack_guard_oct30_forward as source


ROOT = Path("reports/audit_2026_09_08")
OLD_MANIFEST = ROOT / "stock_xperp_forward_nov5" / "manifest.json"
NEW_MANIFEST = ROOT / "stock_xperp_forward_nov7_2" / "manifest_frozen.json"
OLD_PREPORTFOLIO = ROOT / "stock_xperp_nov5_routed_preportfolio.pkl"
OLD_POST_SHORT_CACHE = ROOT / "_tmp_nov5_post_short_guard.pkl"
OUT = ROOT / "stock_xperp_nov7_cluster_diagnostic.json"
ROWS_OUT = ROOT / "stock_xperp_nov7_current_route.csv"
TARGETS = (2.0, 2.5, 3.0)
SHORT_THRESHOLD = -.0075
KEYS = ("symbol", "entry_time", "direction")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(row: dict) -> tuple[str, int, str]:
    return str(row["symbol"]), int(row["entry_time"]), str(row["direction"])


def routed_preportfolio(manifest: Path) -> tuple[pd.DataFrame, dict, dict]:
    raw, sources, candles = source.candidates(manifest)
    entries = short_guard.apply_guard(raw, SHORT_THRESHOLD)
    resolved = {
        target: router.resolve(entries, candles, target)
        for target in TARGETS
    }
    lookup = {
        target: {identity(row): row for row in frame.to_dict("records")}
        for target, frame in resolved.items()
    }
    selected = []
    for entry in entries.to_dict("records"):
        hour = int(float(entry["entry_time"]) // 3600) % 24
        if float(entry["basket_trend_32"]) <= -.04:
            target = 2.0
        elif 6 <= hour < 12:
            target = 3.0
        else:
            target = 2.5
        selected.append(lookup[target][identity(entry)])
    frame = premarket.add_liquidity(pd.DataFrame(selected), candles)
    return frame, sources, candles


def accepted(frame: pd.DataFrame, direction_cap: int | None = 2) -> pd.DataFrame:
    guarded = long_guard.apply_guard(premarket.apply_guard(frame))
    rows = portfolio_gate(
        guarded.to_dict("records"), None,
        rolling_hour_cap=4,
        rolling_direction_hour_cap=direction_cap,
        rank_key=quality.ranker("z_high"),
        loss_streak_cap=2,
        direction_cap=4,
    )
    return pd.DataFrame(rows, columns=guarded.columns)


def parity(expected: pd.DataFrame, actual: pd.DataFrame, cutoff: int) -> dict:
    fields = (
        "signal_return", "cross_z", "atr14_fraction", "basket_trend_16",
        "basket_trend_32", "basket_trend_64", "basket_trend_96",
        "audited_target_pct", "signed_return", "net_r", "outcome",
        "exit_time", "liquid_universe_ratio_8x6",
    )
    # The frozen Nov-5 reference intentionally starts on July 4 even though
    # the refreshed 12,000-bar snapshot can warm up a few symbols earlier.
    # Compare the shared declared reference interval; pre-reference candidates
    # are neither expected rows nor strategy evidence.
    lower = int(expected.entry_time.min())
    left = expected[expected.entry_time.between(lower, cutoff)].copy()
    right = actual[actual.entry_time.between(lower, cutoff)].copy()
    joined = left[list(KEYS) + list(fields)].merge(
        right[list(KEYS) + list(fields)], on=list(KEYS), how="outer",
        suffixes=("_expected", "_actual"), indicator=True,
        validate="one_to_one")
    value_mismatches: dict[str, int] = {}
    matched = joined[joined._merge.eq("both")]
    for field in fields:
        a = matched[f"{field}_expected"]
        b = matched[f"{field}_actual"]
        if pd.api.types.is_numeric_dtype(a):
            value_mismatches[field] = int((~np.isclose(
                pd.to_numeric(a, errors="coerce"),
                pd.to_numeric(b, errors="coerce"),
                rtol=0, atol=1e-12, equal_nan=True)).sum())
        else:
            value_mismatches[field] = int((a.fillna("") != b.fillna("")).sum())
    only_expected = int(joined._merge.eq("left_only").sum())
    only_actual = int(joined._merge.eq("right_only").sum())
    passes = not only_expected and not only_actual and not sum(value_mismatches.values())
    return {
        "start_utc": pd.Timestamp(lower, unit="s", tz="UTC").isoformat(),
        "cutoff_utc": pd.Timestamp(cutoff, unit="s", tz="UTC").isoformat(),
        "expected_rows": int(len(left)), "actual_rows": int(len(right)),
        "matched_rows": int(len(matched)), "only_expected": only_expected,
        "only_actual": only_actual, "value_mismatches": value_mismatches,
        "passes": bool(passes),
    }


def compact_metrics(frame: pd.DataFrame) -> dict:
    values = metric.metrics(frame, .03)
    return {key: values[key] for key in (
        "n", "resolved", "pending", "return_pct", "dd_pct",
        "profitable_exit_rate_pct", "literal_tp_rate_pct",
        "max_loss_streak", "trades_per_active_day",
    )}


def row_summary(row: dict) -> dict:
    return {
        "symbol": str(row["symbol"]), "direction": str(row["direction"]),
        "entry_utc": pd.Timestamp(
            row["entry_time"], unit="s", tz="UTC").isoformat(),
        "exit_utc": (pd.Timestamp(row["exit_time"], unit="s", tz="UTC").isoformat()
                     if pd.notna(row.get("exit_time")) else None),
        "outcome": str(row["outcome"]), "net_r": float(row["net_r"]),
        "signal_return": float(row["signal_return"]),
        "cross_z": float(row["cross_z"]),
        "atr14_fraction": float(row["atr14_fraction"]),
        "basket_trend_16": float(row["basket_trend_16"]),
        "basket_trend_32": float(row["basket_trend_32"]),
        "basket_trend_64": float(row["basket_trend_64"]),
        "basket_trend_96": float(row["basket_trend_96"]),
        "liquid_universe_ratio_8x6": float(row["liquid_universe_ratio_8x6"]),
        "target_pct": float(row["audited_target_pct"]),
    }


def main() -> None:
    old_expected = pickle.loads(OLD_PREPORTFOLIO.read_bytes())
    old_post_minus_one = pickle.loads(OLD_POST_SHORT_CACHE.read_bytes())
    old_candidates = short_guard.apply_guard(
        old_post_minus_one, SHORT_THRESHOLD)
    expected_ids = {identity(row) for row in old_expected.to_dict("records")}
    candidate_ids = {identity(row) for row in old_candidates.to_dict("records")}
    reconstruction = {
        "expected_rows": int(len(old_expected)),
        "candidate_rows": int(len(old_candidates)),
        "only_expected": int(len(expected_ids - candidate_ids)),
        "only_candidates": int(len(candidate_ids - expected_ids)),
        "short_veto_at_or_below": SHORT_THRESHOLD,
    }
    reconstruction["passes"] = bool(
        not reconstruction["only_expected"]
        and not reconstruction["only_candidates"]
        and reconstruction["expected_rows"] == reconstruction["candidate_rows"])
    old_meta = json.loads(OLD_MANIFEST.read_text(encoding="utf-8"))
    old_end = min(int(item["end"]) for item in old_meta["symbols"].values()
                  if item.get("end") is not None)
    if not reconstruction["passes"]:
        raise ValueError({"old_reconstruction_parity": reconstruction})

    preportfolio, sources, candles = routed_preportfolio(NEW_MANIFEST)
    current = accepted(preportfolio, 2)
    without_direction_hour_cap = accepted(preportfolio, None)
    current.to_csv(ROWS_OUT, index=False)
    fresh = current[current.entry_time.gt(old_end)].copy()
    resolved_fresh = fresh[fresh.outcome.ne("PENDING")].copy()
    cluster_symbols = {"DELLUSDT", "ARMUSDT", "BEUSDT", "LITEUSDT"}
    cluster = fresh[fresh.symbol.isin(cluster_symbols)].sort_values(
        ["entry_time", "symbol"])

    current_ids = {identity(row) for row in current.to_dict("records")}
    uncapped_lookup = {
        identity(row): row
        for row in without_direction_hour_cap.to_dict("records")
    }
    withheld = [row_summary(uncapped_lookup[key]) for key in sorted(
        set(uncapped_lookup) - current_ids, key=lambda item: (item[1], item[0]))
        if key[1] > old_end]
    new_end = min(int(candle["time"][-1]) for candle in candles.values())
    report = {
        "status": "DIAGNOSTIC_ONLY",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "fresh_used_for_acceptance": False,
        "orders_possible": False,
        "integrity": {
            "old_reconstruction_parity": reconstruction,
            "raw_candle_overlap": metric.candle_overlap(
                OLD_MANIFEST, NEW_MANIFEST),
            "old_snapshot_end_utc": pd.Timestamp(
                old_end, unit="s", tz="UTC").isoformat(),
            "new_snapshot_end_utc": pd.Timestamp(
                new_end, unit="s", tz="UTC").isoformat(),
        },
        "current_route": {
            "rolling_total_entries_per_hour": 4,
            "rolling_same_direction_entries_per_hour": 2,
            "fresh": compact_metrics(fresh),
            "fresh_resolved": compact_metrics(resolved_fresh),
            "fresh_rows": [row_summary(row) for row in fresh.sort_values(
                ["entry_time", "symbol"]).to_dict("records")],
            "reported_cluster": [row_summary(row) for row in cluster.to_dict("records")],
        },
        "direction_hour_cap_impact": {
            "without_cap_fresh": compact_metrics(
                without_direction_hour_cap[
                    without_direction_hour_cap.entry_time.gt(old_end)]),
            "withheld_fresh": withheld,
        },
        "sources": {
            **sources, str(OLD_MANIFEST): digest(OLD_MANIFEST),
            str(NEW_MANIFEST): digest(NEW_MANIFEST),
            str(OLD_PREPORTFOLIO): digest(OLD_PREPORTFOLIO),
            str(OLD_POST_SHORT_CACHE): digest(OLD_POST_SHORT_CACHE),
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "The loss cluster is diagnostic and cannot authorize a new filter.",
            "Public OHLC cannot reproduce account fill latency or slippage.",
            "The current snapshot may leave recent 64-bar holds pending.",
            "Returns are gross of fees at the account owner's instruction.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "integrity": report["integrity"],
        "fresh": report["current_route"]["fresh"],
        "fresh_resolved": report["current_route"]["fresh_resolved"],
        "reported_cluster": report["current_route"]["reported_cluster"],
        "withheld_fresh": withheld,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
