"""Outcome-blind expansion of the frozen native stock X-Perp route.

The candidate universe contains every live OKX equity X-Perp future returned
by the current instrument endpoint, except the QQQ/SPY index controls.  Entry,
exit, basket, backing, ATR and portfolio rules remain frozen.  A same-snapshot
replay of the original 44 symbols is the baseline, so any difference comes
from universe membership.  Newly listed symbols contribute only after their
own first available candle.
"""
from __future__ import annotations

import hashlib
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.stock_xperp_native_shadow import AUDITED_SYMBOLS
from stock_module_cluster_audit import portfolio_gate
import stock_xperp_backing_level_guard_oct20_audit as backing
import stock_xperp_basket_trend_forward_oct27_audit as forward
import stock_xperp_basket_trend_long_guard_oct26_audit as frozen
import stock_xperp_native_cross_section_oct18_audit as parent
import stock_xperp_native_growth_forward_monitor as monitor
import stock_xperp_native_quality_guard_oct18_audit as quality
import stock_xperp_native_volatility_guard_oct18_audit as volatility


ROOT = Path("reports/audit_2026_09_08")
MANIFEST = ROOT / "stock_xperp_all_live_equity_oct27" / "manifest.json"
OUT = ROOT / "stock_xperp_expanded_universe_oct27.json"
ROWS_OUT = ROOT / "stock_xperp_expanded_universe_oct27.csv"
BASE_ROWS_OUT = ROOT / "stock_xperp_expanded_universe_baseline_oct27.csv"
EXCLUDED = {"QQQUSDT", "SPYUSDT"}
PARAMETERS = {"target_pct": 2.0, "stop_pct": 1.0, "hold_bars": 64}
PERIODS = {
    "july": ("2026-07-01", "2026-08-01"),
    "august": ("2026-08-01", "2026-09-01"),
    "september": ("2026-09-01", "2026-10-01"),
    "october": ("2026-10-01", "2026-11-01"),
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest(path: Path = MANIFEST) -> tuple[dict, dict]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    candles, sources = {}, {str(path): digest(path)}
    for symbol, item in manifest["symbols"].items():
        if symbol in EXCLUDED or item.get("status") not in ("OK", "SHORT_HISTORY"):
            continue
        source = Path(item["path"])
        raw = source.read_bytes()
        found = hashlib.sha256(raw).hexdigest()
        if found != item["sha256"]:
            raise ValueError(f"source hash mismatch: {symbol}")
        candle = pickle.loads(raw)
        if not candle.get("time"):
            continue
        candles[symbol] = candle
        sources[str(source)] = found
    return candles, sources


def route(candles: dict) -> pd.DataFrame:
    features = parent.feature_rows(candles, 2)
    entries = parent.select_entries(features, 2.0, .004)
    entries = volatility.add_atr14(entries, candles)
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
    outcomes["module"] = "native_xperp_expanded_universe"
    outcomes["module_priority"] = 0
    guarded = frozen.apply(outcomes, 96, "LONG", -0.005626823134606253)
    accepted = portfolio_gate(
        guarded.to_dict("records"), None, rolling_hour_cap=4,
        rank_key=quality.ranker("z_high"), loss_streak_cap=2)
    return pd.DataFrame(accepted, columns=guarded.columns)


def bounds(frame: pd.DataFrame, span: tuple[str, str]) -> pd.DataFrame:
    lo = pd.Timestamp(span[0], tz="UTC").timestamp()
    hi = pd.Timestamp(span[1], tz="UTC").timestamp()
    return frame[frame.entry_time.ge(lo) & frame.entry_time.lt(hi)]


def evaluate(frame: pd.DataFrame) -> dict:
    return {
        name: {"margin_3pct": forward.metrics(bounds(frame, span), .03),
               "margin_5pct": forward.metrics(bounds(frame, span), .05)}
        for name, span in PERIODS.items()
    }


def same_snapshot_overlap(candles: dict) -> dict:
    old_manifest = json.loads(forward.NEW_MANIFEST.read_text(encoding="utf-8"))
    matched = mismatches = 0
    for symbol in sorted(AUDITED_SYMBOLS):
        old_item = old_manifest["symbols"].get(symbol)
        if not old_item or "path" not in old_item or symbol not in candles:
            raise ValueError(f"missing original-universe overlap: {symbol}")
        old = pickle.loads(Path(old_item["path"]).read_bytes())
        new = candles[symbol]
        old_index = {int(value): i for i, value in enumerate(old["time"])}
        new_index = {int(value): i for i, value in enumerate(new["time"])}
        common = sorted(set(old_index) & set(new_index))
        for field in ("open", "high", "low", "close", "volume"):
            left = np.asarray([old[field][old_index[t]] for t in common], float)
            right = np.asarray([new[field][new_index[t]] for t in common], float)
            mismatches += int((~np.isclose(left, right, rtol=0, atol=0,
                                           equal_nan=True)).sum())
        matched += len(common)
    if mismatches:
        raise ValueError(f"old-universe candle drift: {mismatches}")
    return {"symbols": len(AUDITED_SYMBOLS), "overlap_bars": matched,
            "value_mismatches": mismatches, "passes": mismatches == 0}


def main() -> None:
    candles, sources = load_manifest()
    missing = sorted(set(AUDITED_SYMBOLS) - set(candles))
    if missing:
        raise ValueError(f"expanded snapshot misses frozen symbols: {missing}")
    overlap = same_snapshot_overlap(candles)
    baseline_candles = {symbol: candles[symbol] for symbol in AUDITED_SYMBOLS}
    baseline_rows = route(baseline_candles)
    expanded_rows = route(candles)
    baseline = evaluate(baseline_rows)
    expanded = evaluate(expanded_rows)
    baseline_rows.to_csv(BASE_ROWS_OUT, index=False)
    expanded_rows.to_csv(ROWS_OUT, index=False)
    comparisons = {}
    all_improve = True
    for month in PERIODS:
        base = baseline[month]["margin_3pct"]
        value = expanded[month]["margin_3pct"]
        improves = bool(
            value["return_pct"] > base["return_pct"]
            and value["dd_pct"] <= 5
            and value["trades_per_active_day"] >= 3)
        comparisons[month] = {
            "return_delta_pct": round(value["return_pct"]-base["return_pct"], 4),
            "dd_delta_pct": round(value["dd_pct"]-base["dd_pct"], 4),
            "trade_delta": int(value["resolved"]-base["resolved"]),
            "improves": improves,
        }
        all_improve &= improves
    report = {
        "status": ("SHADOW_CANDIDATE_REQUIRES_GENUINELY_NEW_FORWARD"
                   if all_improve else "REJECTED_NO_ROBUST_IMPROVEMENT"),
        "ready_for_live_orders": False, "orders_possible": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "universe_rule": "all currently live OKX instCategory=3 X-Perp futures",
            "excluded_controls": sorted(EXCLUDED),
            "selection_uses_outcomes": False,
            "entry_and_exit": "frozen next-15m market entry; 2% target / 1% stop / 64 bars",
            "filters": "frozen ATR, backing level, long basket trend and portfolio rails",
            "money": "client-selected 3% or 5% current-equity margin; fixed 10x",
        },
        "integrity": {"same_snapshot_old_universe_overlap": overlap,
                      "baseline_symbols": len(baseline_candles),
                      "expanded_symbols": len(candles),
                      "new_symbols": len(candles)-len(baseline_candles)},
        "baseline_same_snapshot": baseline,
        "expanded_same_snapshot": expanded,
        "comparisons": comparisons,
        "accepted": all_improve,
        "sources": {**sources, str(forward.NEW_MANIFEST): digest(forward.NEW_MANIFEST),
                    str(Path(__file__).resolve()): digest(Path(__file__).resolve())},
        "limitations": [
            "The expanded universe is evaluated on already observed July-October history.",
            "A passing result would still require genuinely new forward outcomes.",
            "Public OHLC cannot reproduce latency or order-book slippage.",
            "Gross returns exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": report["status"], "integrity": report["integrity"],
                      "comparisons": comparisons,
                      "baseline": baseline, "expanded": expanded}, indent=2), flush=True)


if __name__ == "__main__":
    main()
