"""Independent trend-resumption complement for frozen stock X-Perp entries.

The module detects an efficient 32-96 bar trend, a controlled counter-trend
pullback and a closed resumption candle, then enters the next 15-minute market
open.  July-August select one fixed rule, September tests it and October is an
already-observed confirmation.  The frozen reversal route always ranks first.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.stock_xperp_native_shadow import AUDITED_SYMBOLS
from stock_module_cluster_audit import portfolio_gate
import stock_xperp_basket_trend_forward_oct28_audit as forward
import stock_xperp_mature_supplement_oct27_audit as mature
import stock_xperp_native_cross_section_oct18_audit as parent
import stock_xperp_native_growth_forward_monitor as monitor
import stock_xperp_native_volatility_guard_oct18_audit as volatility
import stock_xperp_session_target_router_oct27_audit as frozen


ROOT = Path("reports/audit_2026_09_08")
MANIFEST = ROOT / "stock_xperp_forward_oct27_2156" / "manifest.json"
OUT = ROOT / "stock_xperp_trend_resume_complement_oct28.json"
ROWS_OUT = ROOT / "stock_xperp_trend_resume_complement_oct28.csv"
HORIZONS = (32, 64, 96)
PULLBACKS = (2, 4)
TARGETS = (1.5, 2.0, 2.5)
STOPS = (.75, 1.0)
HOLDS = (32, 64)
TREND_MINIMUMS = (.005, .01)
EFFICIENCIES = (.15, .30)
PULLBACK_MINIMUMS = (.001, .002)
PULLBACK_RATIOS = (.50, .80)
CONFIRMATIONS = (.0005, .001)
BASKET_ALIGNMENTS = (-.0025, 0.0)
SIDES = ("BOTH", "LONG", "SHORT")
MONTHS = ("2026-07", "2026-08", "2026-09", "2026-10")
TOP_EXACT = 250


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def trend_rows(candles: dict, horizon: int, pullback: int) -> pd.DataFrame:
    rows = []
    for symbol, candle in candles.items():
        times = np.asarray(candle["time"], dtype=np.int64)
        close = np.asarray(candle["close"], dtype=float)
        for signal in range(horizon + pullback + 1, len(times)-1):
            trend_end = signal-pullback-1
            start = trend_end-horizon
            if (times[signal+1]-times[signal] != 900
                    or times[signal]-times[trend_end] != 900*(pullback+1)
                    or times[trend_end]-times[start] != 900*horizon):
                continue
            trend_return = close[trend_end]/close[start]-1
            direction = 1 if trend_return > 0 else -1 if trend_return < 0 else 0
            if not direction:
                continue
            path = float(np.abs(np.diff(close[start:trend_end+1])).sum()/close[start])
            efficiency = abs(trend_return)/path if path else 0.0
            pullback_return = close[signal-1]/close[trend_end]-1
            pullback_move = -direction*pullback_return
            confirmation = direction*(close[signal]/close[signal-1]-1)
            ratio = pullback_move/abs(trend_return) if trend_return else np.inf
            rows.append({
                "symbol": symbol, "entry_index": signal+1,
                "entry_time": int(times[signal+1]),
                "direction_n": direction,
                "direction": "LONG" if direction > 0 else "SHORT",
                "trend_return": trend_return,
                "trend_abs": abs(trend_return),
                "trend_efficiency": efficiency,
                "pullback_move": pullback_move,
                "pullback_ratio": ratio,
                "confirmation_move": confirmation,
                "trend_horizon_bars": horizon,
                "pullback_bars": pullback,
                "signal_return": direction*confirmation,
            })
    frame = pd.DataFrame(rows)
    grouped = frame.groupby("entry_time", sort=False).trend_return
    frame["universe_n"] = grouped.transform("size")
    frame["cross_mean"] = grouped.transform("mean")
    frame["cross_std"] = grouped.transform("std")
    frame["cross_z"] = ((frame.trend_return-frame.cross_mean)
                        / frame.cross_std.replace(0, np.nan))
    return frame.replace([np.inf, -np.inf], np.nan).dropna()


def basket_context(candles: dict) -> pd.DataFrame:
    features = parent.feature_rows(candles, 2)
    context = mature.stable_context(features)
    return context[["entry_time", "basket_bar_return", "basket_trend_96"]]


def prepare_geometry(candles: dict, context: pd.DataFrame,
                     horizon: int, pullback: int) -> pd.DataFrame:
    entries = trend_rows(candles, horizon, pullback)
    entries = volatility.add_atr14(entries, candles)
    entries = entries[entries.atr14_fraction.le(.01)]
    entries = entries.merge(context, on="entry_time", how="left",
                            validate="many_to_one")
    entries["basket_alignment"] = entries.direction_n*entries.basket_trend_96
    return entries[
        entries.trend_abs.ge(min(TREND_MINIMUMS))
        & entries.trend_efficiency.ge(min(EFFICIENCIES))
        & entries.pullback_move.ge(min(PULLBACK_MINIMUMS))
        & entries.pullback_ratio.le(max(PULLBACK_RATIOS))
        & entries.confirmation_move.ge(min(CONFIRMATIONS))
        & entries.basket_alignment.ge(min(BASKET_ALIGNMENTS))
    ].copy()


def filter_rule(frame: pd.DataFrame, rule: dict) -> pd.DataFrame:
    found = frame[
        frame.trend_abs.ge(rule["minimum_trend"])
        & frame.trend_efficiency.ge(rule["minimum_efficiency"])
        & frame.pullback_move.ge(rule["minimum_pullback"])
        & frame.pullback_ratio.le(rule["maximum_pullback_ratio"])
        & frame.confirmation_move.ge(rule["minimum_confirmation"])
        & frame.basket_alignment.ge(rule["minimum_basket_alignment"])
    ].copy()
    return found if rule["side"] == "BOTH" else found[found.direction.eq(rule["side"])].copy()


def month_rows(frame: pd.DataFrame, month: str) -> pd.DataFrame:
    values = pd.to_datetime(frame.entry_time, unit="s", utc=True).dt.strftime("%Y-%m")
    return frame[values.eq(month)]


def evaluate(frame: pd.DataFrame) -> dict:
    return {month: {
        "margin_3pct": forward.metrics(month_rows(frame, month), .03),
        "margin_5pct": forward.metrics(month_rows(frame, month), .05),
    } for month in MONTHS}


def gate(frame: pd.DataFrame) -> pd.DataFrame:
    rank = lambda row: (int(row.get("module_priority", 0)),
                        -abs(float(row.get("cross_z", 0))), str(row["symbol"]))
    rows = portfolio_gate(
        frame.sort_values(["entry_time", "module_priority", "symbol"])
        .to_dict("records"), None, rolling_hour_cap=4,
        rank_key=rank, loss_streak_cap=2)
    return pd.DataFrame(rows, columns=frame.columns)


def combine(base: pd.DataFrame, supplement: pd.DataFrame) -> pd.DataFrame:
    columns = sorted(set(base.columns) | set(supplement.columns))
    return pd.concat([base.reindex(columns=columns),
                      supplement.reindex(columns=columns)], ignore_index=True)


def exact_selection_ok(values: dict, baseline: dict) -> bool:
    for month in ("2026-07", "2026-08"):
        value = values[month]["margin_3pct"]
        base = baseline[month]["margin_3pct"]
        if not (value["return_pct"] > base["return_pct"]
                and value["dd_pct"] <= base["dd_pct"]+.75
                and value["trades_per_active_day"] >= 3
                and value["negative_months"] == 0):
            return False
    return True


def exact_score(values: dict, baseline: dict) -> float:
    gains = [values[month]["margin_3pct"]["return_pct"]
             - baseline[month]["margin_3pct"]["return_pct"]
             for month in ("2026-07", "2026-08")]
    dd = max(values[month]["margin_3pct"]["dd_pct"]
             for month in ("2026-07", "2026-08"))
    return round(min(gains)+.2*sum(gains)-.25*dd, 8)


def holdout_pass(values: dict, baseline: dict, month: str) -> bool:
    value, base = values[month]["margin_3pct"], baseline[month]["margin_3pct"]
    return bool(value["return_pct"] >= base["return_pct"]
                and value["dd_pct"] <= base["dd_pct"]+.5
                and value["negative_months"] == 0
                and value["trades_per_active_day"] >= 3)


def public(item: dict) -> dict:
    return {key: value for key, value in item.items() if not key.startswith("_")}


def main() -> None:
    candles, sources, _ = monitor.load_manifest(MANIFEST)
    candles = {symbol: candles[symbol] for symbol in AUDITED_SYMBOLS}
    context = basket_context(candles)
    base_entries = frozen.prepare_entries(candles)
    base = pd.DataFrame(monitor.resolve_entries(base_entries, candles, {
        "target_pct": 2.0, "stop_pct": 1.0, "hold_bars": 64}))
    base["module"] = "native_xperp_frozen"
    base["module_priority"] = 0
    baseline_rows = gate(base)
    baseline = evaluate(baseline_rows)

    raw_candidates = []
    resolved_geometries = 0
    for horizon, pullback in itertools.product(HORIZONS, PULLBACKS):
        entries = prepare_geometry(candles, context, horizon, pullback)
        for target, stop, hold in itertools.product(TARGETS, STOPS, HOLDS):
            resolved_geometries += 1
            resolved = pd.DataFrame(monitor.resolve_entries(entries, candles, {
                "target_pct": target, "stop_pct": stop, "hold_bars": hold}))
            resolved["module"] = "native_xperp_trend_resume_complement"
            resolved["module_priority"] = 1
            for values in itertools.product(
                    TREND_MINIMUMS, EFFICIENCIES, PULLBACK_MINIMUMS,
                    PULLBACK_RATIOS, CONFIRMATIONS, BASKET_ALIGNMENTS, SIDES):
                rule = dict(zip((
                    "minimum_trend", "minimum_efficiency", "minimum_pullback",
                    "maximum_pullback_ratio", "minimum_confirmation",
                    "minimum_basket_alignment", "side"), values))
                subset = filter_rule(resolved, rule)
                july, august = month_rows(subset, "2026-07"), month_rows(subset, "2026-08")
                if len(july) < 15 or len(august) < 15:
                    continue
                july_sum, august_sum = july.signed_return.sum(), august.signed_return.sum()
                if july_sum <= 0 or august_sum <= 0:
                    continue
                raw_candidates.append({
                    "trend_horizon_bars": horizon, "pullback_bars": pullback,
                    "target_pct": target, "stop_pct": stop, "hold_bars": hold,
                    **rule,
                    "raw_selection_score": float(
                        min(july_sum, august_sum)+.2*(july_sum+august_sum)),
                    "supplement_rows": len(subset), "_rows": subset,
                })
        print(json.dumps({"horizon": horizon, "pullback": pullback,
                          "raw_survivors": len(raw_candidates)}), flush=True)

    raw_candidates.sort(key=lambda item: item["raw_selection_score"], reverse=True)
    exact = []
    for item in raw_candidates[:TOP_EXACT]:
        rows = gate(combine(base, item["_rows"]))
        values = evaluate(rows)
        if exact_selection_ok(values, baseline):
            exact.append({**item, "selection_score": exact_score(values, baseline),
                          "_combined_rows": rows, "_values": values})
    exact.sort(key=lambda item: item["selection_score"], reverse=True)
    selected = exact[0] if exact else None
    test_pass = confirmation_pass = accepted = objective = False
    if selected:
        values = selected["_values"]
        test_pass = holdout_pass(values, baseline, "2026-09")
        confirmation_pass = bool(
            test_pass and holdout_pass(values, baseline, "2026-10"))
        accepted = bool(test_pass and confirmation_pass)
        objective = all(
            values[month]["margin_3pct"]["return_pct"] >= 30
            and values[month]["margin_3pct"]["negative_months"] == 0
            and values[month]["margin_3pct"]["trades_per_active_day"] >= 3
            for month in MONTHS)
        selected["_combined_rows"].to_csv(ROWS_OUT, index=False)

    report = {
        "status": ("SHADOW_CANDIDATE_REQUIRES_GENUINELY_NEW_FORWARD"
                   if accepted else "REJECTED_NO_ROBUST_IMPROVEMENT"),
        "ready_for_live_orders": False, "orders_possible": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "base": "frozen 2-bar cross-sectional reversal, priority zero",
            "supplement": "independent long-trend / pullback / resumption entry, priority one",
            "entry": "next 15-minute market open", "limit_orders": False,
            "selection": ["2026-07", "2026-08"],
            "test": "2026-09", "confirmation": "2026-10",
            "raw_prefilter_top_exact": TOP_EXACT,
        },
        "grid": {
            "candidate_rules": (len(HORIZONS)*len(PULLBACKS)*len(TARGETS)
                                * len(STOPS)*len(HOLDS)*len(TREND_MINIMUMS)
                                * len(EFFICIENCIES)*len(PULLBACK_MINIMUMS)
                                * len(PULLBACK_RATIOS)*len(CONFIRMATIONS)
                                * len(BASKET_ALIGNMENTS)*len(SIDES)),
            "resolved_geometries": resolved_geometries,
            "raw_survivors": len(raw_candidates), "exact_survivors": len(exact),
        },
        "baseline": baseline,
        "selected_before_holdouts": (
            {**public(selected), "all_months": selected["_values"]}
            if selected else None),
        "test_pass": test_pass, "confirmation_pass": confirmation_pass,
        "accepted": accepted, "objective_30pct_every_month_at_3pct": objective,
        "top20_selection": [public(item) for item in exact[:20]],
        "sources": {**sources, str(MANIFEST): digest(MANIFEST),
                    str(Path(__file__).resolve()): digest(Path(__file__).resolve())},
        "limitations": [
            "Native X-Perp history covers only late June through October 2026.",
            "October was already observed and cannot by itself activate the module.",
            "The raw prefilter reduces exact portfolio replays to the strongest 250 selection candidates.",
            "Public OHLC cannot reproduce latency or intrabar order sequence.",
            "Gross returns exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "grid": report["grid"],
        "baseline": baseline,
        "selected": report["selected_before_holdouts"],
        "test_pass": test_pass, "confirmation_pass": confirmation_pass,
        "objective": objective,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
