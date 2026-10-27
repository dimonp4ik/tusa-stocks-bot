"""Audit new X-Perps as a mature, lower-priority supplement.

The frozen 44-symbol route keeps its original cross-sectional statistics and
basket context.  New live contracts are normalized against that stable
context, enter only after a predeclared minimum observed age, and rank behind
frozen-universe candidates at the same scan.  September selects one age rule;
October is evaluated only if a September candidate passes the development
gate.  This prevents newly listed contracts from rewriting old signals merely
by joining the universe.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.stock_xperp_native_shadow import AUDITED_SYMBOLS
from stock_module_cluster_audit import portfolio_gate
import stock_xperp_backing_level_guard_oct20_audit as backing
import stock_xperp_basket_trend_forward_oct27_audit as forward
import stock_xperp_basket_trend_long_guard_oct26_audit as frozen
import stock_xperp_expanded_universe_oct27_audit as expanded
import stock_xperp_native_cross_section_oct18_audit as parent
import stock_xperp_native_growth_forward_monitor as monitor
import stock_xperp_native_volatility_guard_oct18_audit as volatility


ROOT = Path("reports/audit_2026_09_08")
OUT = ROOT / "stock_xperp_mature_supplement_oct27.json"
ROWS_OUT = ROOT / "stock_xperp_mature_supplement_oct27.csv"
BASE_ROWS_OUT = ROOT / "stock_xperp_mature_supplement_baseline_oct27.csv"
AGE_DAYS = (0, 7, 14, 30)
PARAMETERS = expanded.PARAMETERS
DEVELOPMENT = ("2026-09-01", "2026-10-01")
TEST = ("2026-10-01", "2026-11-01")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_context(base_features: pd.DataFrame) -> pd.DataFrame:
    columns = ["entry_time", "universe_n", "cross_mean", "cross_std"]
    context = base_features.groupby("entry_time", as_index=False)[
        ["universe_n", "cross_mean", "cross_std"]].first()
    context["basket_bar_return"] = context.cross_mean
    context.sort_values("entry_time", inplace=True)
    for horizon in frozen.HORIZONS:
        context[f"basket_trend_{horizon}"] = context.basket_bar_return.rolling(
            horizon, min_periods=horizon).sum()
    return context[columns + ["basket_bar_return"] + [
        f"basket_trend_{horizon}" for horizon in frozen.HORIZONS]]


def guarded(features: pd.DataFrame, context: pd.DataFrame,
            candles: dict, module: str) -> pd.DataFrame:
    entries = parent.select_entries(features, 2.0, .004)
    entries = volatility.add_atr14(entries, candles)
    entries = entries[entries.atr14_fraction.le(.01)]
    entries = backing.guarded(backing.add_backing_distance(entries, candles, 32))
    trend_columns = ["entry_time", "basket_bar_return"] + [
        f"basket_trend_{horizon}" for horizon in frozen.HORIZONS]
    entries = entries.merge(context[trend_columns], on="entry_time", how="left",
                            validate="many_to_one")
    outcomes = pd.DataFrame(monitor.resolve_entries(entries, candles, PARAMETERS))
    outcomes["module"] = module
    outcomes["module_priority"] = 0 if module == "native_xperp_frozen" else 1
    return frozen.apply(outcomes, 96, "LONG", -0.005626823134606253)


def supplement_features(new_candles: dict, context: pd.DataFrame) -> pd.DataFrame:
    raw = parent.feature_rows(new_candles, 2)
    raw.drop(columns=["universe_n", "cross_mean", "cross_std", "cross_z"],
             inplace=True)
    joined = raw.merge(context[["entry_time", "universe_n", "cross_mean",
                                "cross_std"]], on="entry_time", how="inner",
                       validate="many_to_one")
    joined["cross_z"] = ((joined.signal_return - joined.cross_mean)
                         / joined.cross_std.replace(0, np.nan))
    return joined.replace([np.inf, -np.inf], np.nan).dropna()


def apply_age(frame: pd.DataFrame, candles: dict, days: int) -> pd.DataFrame:
    first = {symbol: int(values["time"][0]) for symbol, values in candles.items()}
    age_seconds = frame.entry_time - frame.symbol.map(first)
    return frame[age_seconds.ge(days * 86400)].copy()


def accepted(base: pd.DataFrame, supplement: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([base, supplement], ignore_index=True)
    old = set(AUDITED_SYMBOLS)
    rank = lambda row: (0 if row["symbol"] in old else 1,
                        -abs(float(row["cross_z"])))
    rows = portfolio_gate(combined.to_dict("records"), None,
                          rolling_hour_cap=4, rank_key=rank,
                          loss_streak_cap=2)
    return pd.DataFrame(rows, columns=combined.columns)


def period(frame: pd.DataFrame, span: tuple[str, str]) -> pd.DataFrame:
    lo = pd.Timestamp(span[0], tz="UTC").timestamp()
    hi = pd.Timestamp(span[1], tz="UTC").timestamp()
    return frame[frame.entry_time.ge(lo) & frame.entry_time.lt(hi)]


def metrics(frame: pd.DataFrame, span: tuple[str, str]) -> dict:
    part = period(frame, span)
    return {"margin_3pct": forward.metrics(part, .03),
            "margin_5pct": forward.metrics(part, .05)}


def development_pass(candidate: dict, baseline: dict) -> bool:
    value = candidate["margin_3pct"]
    base = baseline["margin_3pct"]
    return bool(
        value["return_pct"] >= base["return_pct"] + .5
        and value["dd_pct"] <= base["dd_pct"] + .5
        and value["max_loss_streak"] <= base["max_loss_streak"]
        and value["trades_per_active_day"] >= 3)


def main() -> None:
    candles, sources = expanded.load_manifest()
    expanded.same_snapshot_overlap(candles)
    base_candles = {symbol: candles[symbol] for symbol in AUDITED_SYMBOLS}
    new_candles = {symbol: value for symbol, value in candles.items()
                   if symbol not in AUDITED_SYMBOLS}
    base_features = parent.feature_rows(base_candles, 2)
    context = stable_context(base_features)
    base_guarded = guarded(base_features, context, base_candles,
                           "native_xperp_frozen")
    base_rows = accepted(base_guarded, base_guarded.iloc[0:0])
    base_rows.to_csv(BASE_ROWS_OUT, index=False)
    base_development = metrics(base_rows, DEVELOPMENT)

    new_features = supplement_features(new_candles, context)
    new_guarded = guarded(new_features, context, candles,
                          "native_xperp_mature_supplement")
    candidates = []
    candidate_rows = {}
    for age_days in AGE_DAYS:
        aged = apply_age(new_guarded, new_candles, age_days)
        rows = accepted(base_guarded, aged)
        values = metrics(rows, DEVELOPMENT)
        passes = development_pass(values, base_development)
        candidates.append({"minimum_age_days": age_days,
                           "development": values,
                           "development_pass": passes,
                           "supplement_guarded_rows": len(aged)})
        candidate_rows[age_days] = rows

    survivors = [item for item in candidates if item["development_pass"]]
    survivors.sort(key=lambda item: (
        -item["development"]["margin_3pct"]["return_pct"],
        item["development"]["margin_3pct"]["dd_pct"],
        -item["minimum_age_days"]))
    selected = survivors[0] if survivors else None
    test = None
    checks = {"development_candidate_exists": selected is not None}
    if selected is not None:
        rows = candidate_rows[selected["minimum_age_days"]]
        rows.to_csv(ROWS_OUT, index=False)
        base_test = metrics(base_rows, TEST)
        test = {"baseline": base_test, "candidate": metrics(rows, TEST)}
        candidate_test = test["candidate"]["margin_3pct"]
        baseline_test = test["baseline"]["margin_3pct"]
        checks.update({
            "test_return_improves": candidate_test["return_pct"]
                                    > baseline_test["return_pct"],
            "test_drawdown_at_most_5pct": candidate_test["dd_pct"] <= 5,
            "test_frequency_at_least_3":
                candidate_test["trades_per_active_day"] >= 3,
            "test_no_negative_month": candidate_test["negative_months"] == 0,
        })
    accepted_candidate = bool(checks and all(checks.values()))
    report = {
        "status": ("SHADOW_CANDIDATE_REQUIRES_GENUINELY_NEW_FORWARD"
                   if accepted_candidate else "REJECTED_NO_ROBUST_IMPROVEMENT"),
        "ready_for_live_orders": False,
        "orders_possible": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "base": "frozen 44-symbol context and candidates",
            "supplement": "new X-Perps normalized to frozen context and ranked second",
            "minimum_age_days_grid": list(AGE_DAYS),
            "selection": "September only; October revealed only after a pass",
            "entry_exit": "next-15m market entry; 2% target / 1% stop / 64 bars",
            "money": "client-selected 3% or 5% current-equity margin; fixed 10x",
        },
        "integrity": {"base_symbols": len(base_candles),
                      "new_symbols": len(new_candles)},
        "baseline_development": base_development,
        "candidates": candidates,
        "selected": selected,
        "test": test,
        "checks": checks,
        "accepted": accepted_candidate,
        "sources": {**sources, str(Path(__file__).resolve()):
                    digest(Path(__file__).resolve())},
        "limitations": [
            "October had already been seen for a different all-universe architecture.",
            "Any passing supplement still requires genuinely new forward outcomes.",
            "Public OHLC cannot reproduce latency or order-book slippage.",
            "Gross returns exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": report["status"],
                      "baseline_development": base_development,
                      "candidates": candidates, "selected": selected,
                      "test": test, "checks": checks}, indent=2), flush=True)


if __name__ == "__main__":
    main()
