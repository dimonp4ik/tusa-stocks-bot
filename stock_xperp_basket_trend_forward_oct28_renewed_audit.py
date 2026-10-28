"""Append-only renewed Oct-28 check of the frozen stock basket-trend shadow route.

Unlike the exploratory full-history resolver, this forward audit preserves
unresolved 64-bar positions as PENDING before portfolio gating.  Pending rows
occupy exposure but never contribute profit, loss or a loss-pause outcome.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import collections
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from stock_module_cluster_audit import portfolio_gate
import stock_xperp_backing_level_guard_oct20_audit as backing
import stock_xperp_basket_trend_long_guard_oct26_audit as frozen
import stock_xperp_native_cross_section_oct18_audit as parent
import stock_xperp_native_growth_forward_monitor as monitor
import stock_xperp_native_quality_guard_oct18_audit as quality
import stock_xperp_native_volatility_guard_oct18_audit as volatility


ROOT = Path("reports/audit_2026_09_08")
ORIGINAL_MANIFEST = ROOT / "stock_xperp_forward_oct26" / "manifest.json"
OLD_MANIFEST = ROOT / "stock_xperp_forward_oct27_2156" / "manifest.json"
NEW_MANIFEST = ROOT / "stock_xperp_forward_oct28_renewed" / "manifest.json"
OLD_EXPLORATORY_ROWS = ROOT / "stock_xperp_basket_trend_long_guard_oct26.csv"
OUT = ROOT / "stock_xperp_basket_trend_forward_oct28_renewed.json"
ROWS_OUT = ROOT / "stock_xperp_basket_trend_forward_oct28_renewed.csv"
PARAMETERS = {"target_pct": 2.0, "stop_pct": 1.0, "hold_bars": 64}
JULY_START = int(pd.Timestamp("2026-07-01", tz="UTC").timestamp())
OCTOBER_START = int(pd.Timestamp("2026-10-01", tz="UTC").timestamp())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(manifest_path: Path) -> tuple[pd.DataFrame, dict, dict]:
    candles, sources, _ = monitor.load_manifest(manifest_path)
    features = parent.feature_rows(candles, 2)
    entries = parent.select_entries(features, 2.0, .004)
    entries = volatility.add_atr14(entries, candles)
    entries = entries[entries.atr14_fraction.le(.01)]
    entries = backing.guarded(backing.add_backing_distance(
        entries, candles, 32))
    context = features.groupby("entry_time", as_index=False).cross_mean.first()
    context.sort_values("entry_time", inplace=True)
    context.rename(columns={"cross_mean": "basket_bar_return"}, inplace=True)
    for horizon in frozen.HORIZONS:
        context[f"basket_trend_{horizon}"] = context.basket_bar_return.rolling(
            horizon, min_periods=horizon).sum()
    entries = entries.merge(context, on="entry_time", how="left",
                            validate="many_to_one")
    outcomes = pd.DataFrame(monitor.resolve_entries(
        entries, candles, PARAMETERS))
    outcomes["module"] = "native_xperp_basket_trend_guard"
    outcomes["module_priority"] = 0
    guarded = frozen.apply(
        outcomes, 96, "LONG", -0.005626823134606253)
    accepted = portfolio_gate(
        guarded.to_dict("records"), None, rolling_hour_cap=4,
        rank_key=quality.ranker("z_high"), loss_streak_cap=2)
    return pd.DataFrame(accepted, columns=guarded.columns), sources, candles


def candle_overlap(old_manifest: Path, new_manifest: Path) -> dict:
    old = json.loads(old_manifest.read_text(encoding="utf-8"))
    new = json.loads(new_manifest.read_text(encoding="utf-8"))
    if set(old["symbols"]) != set(new["symbols"]):
        raise ValueError("snapshot universes differ")
    matched = mismatches = 0
    by_symbol = {}
    for symbol in old["symbols"]:
        old_item, new_item = old["symbols"][symbol], new["symbols"][symbol]
        if "path" not in old_item or "path" not in new_item:
            if old_item.get("status") != new_item.get("status"):
                raise ValueError(f"snapshot availability drift: {symbol}")
            continue
        old_path = Path(old_item["path"])
        new_path = Path(new_item["path"])
        a, b = pickle.loads(old_path.read_bytes()), pickle.loads(new_path.read_bytes())
        ia = {int(t): i for i, t in enumerate(a["time"])}
        ib = {int(t): i for i, t in enumerate(b["time"])}
        common = sorted(set(ia) & set(ib))
        local = 0
        for field in ("open", "high", "low", "close", "volume"):
            av = np.asarray([a[field][ia[t]] for t in common], dtype=float)
            bv = np.asarray([b[field][ib[t]] for t in common], dtype=float)
            local += int((~np.isclose(av, bv, rtol=0, atol=0,
                                     equal_nan=True)).sum())
        if local:
            raise ValueError(f"raw candle overlap drift: {symbol}={local}")
        matched += len(common)
        mismatches += local
        by_symbol[symbol] = {"overlap_bars": len(common),
                             "value_mismatches": local}
    return {"symbols": len(by_symbol), "overlap_bars": matched,
            "value_mismatches": mismatches, "passes": mismatches == 0,
            "by_symbol": by_symbol}


def parity(old: pd.DataFrame, new: pd.DataFrame, cutoff: int) -> dict:
    old_part = old[old.entry_time.between(JULY_START, cutoff)].copy()
    new_part = new[new.entry_time.between(JULY_START, cutoff)].copy()
    keys = ["symbol", "entry_time", "direction"]
    fields = ["signed_return", "exit_time", "outcome", "net_r"]
    joined = old_part[keys+fields].merge(
        new_part[keys+fields], on=keys, suffixes=("_old", "_new"),
        validate="one_to_one")
    mismatches = 0
    for field in fields:
        a, b = joined[f"{field}_old"], joined[f"{field}_new"]
        if pd.api.types.is_numeric_dtype(a):
            mismatches += int((~np.isclose(a, b, rtol=0, atol=1e-12,
                                           equal_nan=True)).sum())
        else:
            mismatches += int((a.fillna("") != b.fillna("")).sum())
    passed = bool(len(old_part) == len(new_part) == len(joined)
                  and mismatches == 0)
    if not passed:
        raise ValueError({"old": len(old_part), "new": len(new_part),
                          "matched": len(joined), "mismatches": mismatches})
    return {"through_entry_utc": pd.Timestamp(
                cutoff, unit="s", tz="UTC").isoformat(),
            "old_rows": len(old_part), "new_rows": len(new_part),
            "matched_rows": len(joined), "value_mismatches": mismatches,
            "passes": passed}


def metrics(frame: pd.DataFrame, margin: float) -> dict:
    resolved = frame[frame.outcome.ne("PENDING")].copy()
    value = frozen.metrics(resolved, margin)
    value["pending"] = int(frame.outcome.eq("PENDING").sum())
    value["resolved"] = len(resolved)
    if resolved.empty:
        value.update({
            "literal_tp_rate_pct": 0.0, "max_loss_streak": 0,
            "max_entries_same_timestamp": 0,
            "max_entries_same_clock_hour": 0,
            "largest_symbol_share_pct": 0.0,
            "observed_calendar_days": 0,
            "projected_22_active_day_pct": 0.0,
            "projected_30_calendar_day_pct": 0.0,
        })
    else:
        ordered = resolved.sort_values(["exit_time", "entry_time", "symbol"])
        streak = maximum_streak = 0
        for loss in ordered.signed_return.lt(0):
            streak = streak + 1 if loss else 0
            maximum_streak = max(maximum_streak, streak)
        stamps = pd.to_datetime(resolved.entry_time, unit="s", utc=True)
        calendar_days = (stamps.max().date()-stamps.min().date()).days+1
        active_days = max(1, stamps.dt.date.nunique())
        growth = max(0.0, 1+float(value["return_pct"])/100)
        by_stamp = collections.Counter(resolved.entry_time.astype("int64"))
        by_hour = collections.Counter((resolved.entry_time//3600).astype("int64"))
        by_symbol = resolved.symbol.astype(str).value_counts()
        value.update({
            "literal_tp_rate_pct": round(
                100*float(resolved.outcome_code.eq(1).mean()), 4),
            "max_loss_streak": maximum_streak,
            "max_entries_same_timestamp": max(by_stamp.values()),
            "max_entries_same_clock_hour": max(by_hour.values()),
            "largest_symbol_share_pct": round(
                100*float(by_symbol.iloc[0])/len(resolved), 4),
            "observed_calendar_days": calendar_days,
            "projected_22_active_day_pct": round(
                100*(growth**(22/active_days)-1), 4) if growth > 0 else -100.0,
            "projected_30_calendar_day_pct": round(
                100*(growth**(30/calendar_days)-1), 4) if growth > 0 else -100.0,
        })
    observed_mean = float(value.get("mean_signed_move_pct", 0.0))
    projected = float(value["projected_22_active_day_pct"])
    value["economic_gate"] = {
        "margin_fraction": margin, "leverage": 10.0,
        "required_trades_per_market_day": 3.0,
        "required_constant_signed_price_return_pct": 1.327709,
        "observed_mean_signed_price_return_pct": observed_mean,
        "frequency_pass": value["trades_per_active_day"] >= 3,
        "return_pass": projected >= 30,
        "passes": bool(value["trades_per_active_day"] >= 3
                       and projected >= 30
                       and value["negative_months"] == 0),
    }
    return value


def main() -> None:
    raw_overlap = candle_overlap(OLD_MANIFEST, NEW_MANIFEST)
    old, old_sources, old_candles = snapshot(OLD_MANIFEST)
    new, new_sources, _ = snapshot(NEW_MANIFEST)
    old_end = min(int(candle["time"][-1]) for candle in old_candles.values())
    stable_cutoff = old_end - PARAMETERS["hold_bars"]*900
    accepted_parity = parity(old, new, stable_cutoff)

    # Preserve the original Oct-26 selection cutoff across refreshes.  The
    # Oct-27 snapshot is used only for append-only parity, so the activation
    # sample grows instead of resetting on every monitor run.
    original_manifest = json.loads(
        ORIGINAL_MANIFEST.read_text(encoding="utf-8"))
    original_end = min(
        int(item["end"]) for item in original_manifest["symbols"].values()
        if item.get("end") is not None)
    incremental = new[new.entry_time.gt(original_end)].copy()
    newly_revealed = new[new.entry_time.gt(old_end)].copy()
    new.to_csv(ROWS_OUT, index=False)
    october = new[new.entry_time.ge(OCTOBER_START)].copy()
    results = {f"margin_{int(100*m)}pct": metrics(october, m)
               for m in (.03, .05)}
    incremental_results = {f"margin_{int(100*m)}pct": metrics(incremental, m)
                           for m in (.03, .05)}
    newly_revealed_results = {
        f"margin_{int(100*m)}pct": metrics(newly_revealed, m)
        for m in (.03, .05)}
    incremental_resolved = incremental[incremental.outcome.ne("PENDING")]
    observed_days = 0
    if len(incremental):
        stamp = pd.to_datetime(incremental.entry_time, unit="s", utc=True)
        observed_days = (stamp.max().date()-stamp.min().date()).days+1
    three = incremental_results["margin_3pct"]
    activation = bool(
        len(incremental_resolved) >= 60 and observed_days >= 14
        and three["return_pct"] > 0 and three["dd_pct"] <= 5
        and three["trades_per_active_day"] >= 3)
    exploratory = pd.read_csv(OLD_EXPLORATORY_ROWS)
    edge_pending = old[old.outcome.eq("PENDING")]
    report = {
        "status": ("ACTIVATION_GATE_PASSED" if activation
                   else "FORWARD_AWAIT_MORE_GENUINELY_NEW_OUTCOMES"),
        "ready_for_live_orders": False, "orders_possible": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "lookback_bars": 2, "z_threshold": 2.0,
            "minimum_move_pct": .4, "maximum_atr_pct": 1.0,
            "backing_level_bars": 32,
            "long_basket_trend_bars": 96,
            "long_minimum_basket_trend_pct": -0.5626823134606253,
            "target_pct": 2.0, "stop_pct": 1.0, "hold_bars": 64,
            "rolling_hour_cap": 4, "loss_pause": 2,
            "entry": "next 15-minute market open", "orders": "market only",
            "retuned_on_oct27": False, "retuned_on_oct28": False, "retuned_on_oct28_renewed": False,
        },
        "integrity": {
            "raw_candle_overlap": raw_overlap,
            "stable_accepted_prefix": accepted_parity,
            "original_selection_cutoff_utc": pd.Timestamp(
                original_end, unit="s", tz="UTC").isoformat(),
            "old_snapshot_rows_pending_aware": len(old),
            "old_exploratory_rows": len(exploratory),
            "old_edge_pending_positions": len(edge_pending),
            "pending_policy": "occupy portfolio exposure; excluded from PnL and loss-pause outcomes",
        },
        "october_to_date": results,
        "newly_revealed_after_oct26_snapshot": incremental_results,
        "newly_revealed_after_oct27_2156_snapshot": newly_revealed_results,
        "new_rows": len(incremental),
        "new_resolved_rows": len(incremental_resolved),
        "new_pending_rows": int(incremental.outcome.eq("PENDING").sum()),
        "activation_gate": {
            "required_resolved_trades": 60,
            "observed_resolved_trades": len(incremental_resolved),
            "required_calendar_days": 14,
            "observed_calendar_days": observed_days,
            "met": activation,
        },
        "sources": {**old_sources, **new_sources,
                    str(OLD_MANIFEST): digest(OLD_MANIFEST),
                    str(NEW_MANIFEST): digest(NEW_MANIFEST),
                    str(ORIGINAL_MANIFEST): digest(ORIGINAL_MANIFEST),
                    str(OLD_EXPLORATORY_ROWS): digest(OLD_EXPLORATORY_ROWS),
                    str(Path(__file__).resolve()): digest(Path(__file__).resolve())},
        "limitations": [
            "The basket guard was selected using data through Oct 26; only later rows are fresh.",
            "Public candles cannot reproduce latency or order-book slippage.",
            "Gross results exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "integrity": report["integrity"],
        "october_to_date": results,
        "newly_revealed_after_oct26_snapshot": incremental_results,
        "newly_revealed_after_oct27_2156_snapshot": newly_revealed_results,
        "activation_gate": report["activation_gate"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
