"""Chronological session-specific target audit for native stock X-Perps.

The entry filters, one-percent stop and 64-bar horizon remain frozen.  Only
the literal market-exit target may vary across four fixed UTC sessions.  All
81 target maps are selected on July-August, tested in September, and confirmed
in October.  The search never changes position sizing or uses limit entries.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.stock_xperp_native_shadow import AUDITED_SYMBOLS
from stock_module_cluster_audit import portfolio_gate
import stock_cash_xperp_leadlag_oct6_audit as resolver
import stock_xperp_backing_level_guard_oct20_audit as backing
import stock_xperp_basket_trend_long_guard_oct26_audit as frozen
import stock_xperp_expanded_universe_oct27_audit as expanded
import stock_xperp_mature_supplement_oct27_audit as mature
import stock_xperp_native_cross_section_oct18_audit as parent
import stock_xperp_native_quality_guard_oct18_audit as quality
import stock_xperp_native_volatility_guard_oct18_audit as volatility


ROOT = Path("reports/audit_2026_09_08")
OUT = ROOT / "stock_xperp_session_target_router_oct27.json"
ROWS_OUT = ROOT / "stock_xperp_session_target_router_oct27.csv"
TARGETS = (1.0, 1.5, 2.0)
SESSIONS = ("utc_00_02", "utc_03_07", "utc_08_15", "utc_16_23")
MONTHS = ("2026-07", "2026-08", "2026-09", "2026-10")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def session_name(entry_time: float) -> str:
    hour = pd.Timestamp(entry_time, unit="s", tz="UTC").hour
    if hour <= 2:
        return "utc_00_02"
    if hour <= 7:
        return "utc_03_07"
    if hour <= 15:
        return "utc_08_15"
    return "utc_16_23"


def prepare_entries(candles: dict) -> pd.DataFrame:
    features = parent.feature_rows(candles, 2)
    entries = parent.select_entries(features, 2.0, .004)
    entries = volatility.add_atr14(entries, candles)
    entries = entries[entries.atr14_fraction.le(.01)]
    entries = backing.guarded(backing.add_backing_distance(entries, candles, 32))
    context = mature.stable_context(features)
    trend_columns = ["entry_time", "basket_bar_return"] + [
        f"basket_trend_{horizon}" for horizon in frozen.HORIZONS]
    entries = entries.merge(context[trend_columns], on="entry_time", how="left",
                            validate="many_to_one")
    alignment = entries.direction_n * entries.basket_trend_96
    entries = entries[(entries.direction_n.ne(1))
                      | alignment.ge(-0.005626823134606253)].copy()
    entries["session"] = entries.entry_time.map(session_name)
    return entries


def resolve_targets(entries: pd.DataFrame, candles: dict) -> dict[float, pd.DataFrame]:
    output = {}
    for target in TARGETS:
        resolved = resolver.add_outcomes(entries, candles, target, 1.0, 64)
        resolved["module"] = "native_xperp_session_target"
        resolved["module_priority"] = 0
        output[target] = resolved
    return output


def route(resolved: dict[float, pd.DataFrame], mapping: dict[str, float]) -> pd.DataFrame:
    pieces = [frame[frame.session.eq(session)]
              for session, target in mapping.items()
              for frame in [resolved[target]]]
    combined = pd.concat(pieces, ignore_index=True)
    rows = portfolio_gate(
        combined.to_dict("records"), None, rolling_hour_cap=4,
        rank_key=quality.ranker("z_high"), loss_streak_cap=2)
    return pd.DataFrame(rows, columns=combined.columns)


def month_rows(frame: pd.DataFrame, month: str) -> pd.DataFrame:
    values = pd.to_datetime(frame.entry_time, unit="s", utc=True).dt.strftime("%Y-%m")
    return frame[values.eq(month)]


def evaluate(frame: pd.DataFrame) -> dict:
    return {month: {
        "margin_3pct": mature.forward.metrics(month_rows(frame, month), .03),
        "margin_5pct": mature.forward.metrics(month_rows(frame, month), .05),
    } for month in MONTHS}


def selection_ok(values: dict) -> bool:
    return all(
        values[month]["margin_3pct"]["return_pct"] > 0
        and values[month]["margin_3pct"]["dd_pct"] <= 5
        and values[month]["margin_3pct"]["trades_per_active_day"] >= 3
        and values[month]["margin_3pct"]["negative_months"] == 0
        for month in ("2026-07", "2026-08"))


def score(values: dict) -> float:
    return round(
        min(values[month]["margin_3pct"]["return_pct"]
            for month in ("2026-07", "2026-08"))
        - .5 * max(values[month]["margin_3pct"]["dd_pct"]
                   for month in ("2026-07", "2026-08")), 8)


def main() -> None:
    candles, sources = expanded.load_manifest()
    expanded.same_snapshot_overlap(candles)
    candles = {symbol: candles[symbol] for symbol in AUDITED_SYMBOLS}
    entries = prepare_entries(candles)
    resolved = resolve_targets(entries, candles)
    candidates = []
    rows_by_map = {}
    for targets in itertools.product(TARGETS, repeat=len(SESSIONS)):
        mapping = dict(zip(SESSIONS, targets))
        rows = route(resolved, mapping)
        values = evaluate(rows)
        item = {"targets_pct": mapping, "selection_ok": selection_ok(values),
                "selection_score": score(values),
                "selection": {month: values[month]
                              for month in ("2026-07", "2026-08")}}
        candidates.append(item)
        rows_by_map[targets] = (rows, values)
    eligible = [item for item in candidates if item["selection_ok"]]
    eligible.sort(key=lambda item: (
        -item["selection_score"],
        tuple(-item["targets_pct"][session] for session in SESSIONS)))
    selected = eligible[0]
    selected_key = tuple(selected["targets_pct"][session] for session in SESSIONS)
    rows, values = rows_by_map[selected_key]
    test = values["2026-09"]
    confirmation = values["2026-10"]
    baseline_key = tuple(2.0 for _ in SESSIONS)
    baseline_rows, baseline_values = rows_by_map[baseline_key]
    test_pass = bool(
        test["margin_3pct"]["return_pct"] >=
        baseline_values["2026-09"]["margin_3pct"]["return_pct"]
        and test["margin_3pct"]["dd_pct"] <=
        baseline_values["2026-09"]["margin_3pct"]["dd_pct"]
        and test["margin_3pct"]["trades_per_active_day"] >= 3)
    confirmation_pass = bool(
        test_pass
        and confirmation["margin_3pct"]["return_pct"] >=
            baseline_values["2026-10"]["margin_3pct"]["return_pct"]
        and confirmation["margin_3pct"]["dd_pct"] <=
            baseline_values["2026-10"]["margin_3pct"]["dd_pct"]
        and confirmation["margin_3pct"]["trades_per_active_day"] >= 3)
    if confirmation_pass:
        rows.to_csv(ROWS_OUT, index=False)
    public = sorted(candidates, key=lambda item: item["selection_score"], reverse=True)[:10]
    result = {
        "status": ("SHADOW_CANDIDATE_REQUIRES_GENUINELY_NEW_FORWARD"
                   if confirmation_pass else "REJECTED_OUT_OF_SAMPLE"),
        "ready_for_live_orders": False,
        "orders_possible": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "sessions_utc": list(SESSIONS), "targets_pct": list(TARGETS),
            "stop_pct": 1.0, "hold_bars": 64,
            "selection": ["2026-07", "2026-08"],
            "test": "2026-09", "confirmation": "2026-10",
            "entry": "frozen next-15m market entry", "limit_orders": False,
            "candidate_maps": len(candidates),
        },
        "integrity": {"symbols": len(candles), "eligible_maps": len(eligible)},
        "baseline_all_2pct": baseline_values,
        "selected": selected,
        "test": test,
        "test_pass": test_pass,
        "confirmation": confirmation,
        "confirmation_pass": confirmation_pass,
        "selection_top10": public,
        "sources": {**sources, str(Path(__file__).resolve()): digest(Path(__file__).resolve())},
        "limitations": [
            "Session boundaries are fixed UTC buckets rather than exchange sessions.",
            "All four months were already observed by earlier route research.",
            "Any passing map would still require genuinely new forward data.",
            "Gross returns exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": result["status"],
                      "selected_targets": selected["targets_pct"],
                      "selection": selected["selection"],
                      "test": test, "test_pass": test_pass,
                      "confirmation": confirmation,
                      "confirmation_pass": confirmation_pass,
                      "baseline_test": baseline_values["2026-09"],
                      "baseline_confirmation": baseline_values["2026-10"]},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
