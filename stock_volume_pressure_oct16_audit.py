"""Cross-era audit of causal market volume pressure before stock entries.

The signal uses volume and candle location only through the completed signal
bar.  Development is 2017-2021 and validation is 2022-2023.  Later years stay
sealed unless one fixed rule survives both stages.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

import stock_15m_multisession_regime_oct6_audit as base
from src.research_profit_gate import economic_gate
from stock_5m_family_oct3_audit import regular_groups
from stock_module_cluster_audit import portfolio_gate


OUT = base.ROOT / "stock_volume_pressure_oct16.json"
ROWS_OUT = base.ROOT / "stock_volume_pressure_oct16.csv"
DEVELOPMENT = (2017, 2018, 2019, 2020, 2021)
VALIDATION = (2022, 2023)
TEST = (2024,)
CONFIRMATION = (2025, 2026)
EXITS = (
    {"target_pct": .5, "stop_pct": .5, "hold_sessions": 1},
    {"target_pct": 1.0, "stop_pct": 1.0, "hold_sessions": 2},
    {"target_pct": 2.0, "stop_pct": 1.0, "hold_sessions": 3},
)
SPECS = (
    {"name": "flow_continuation_loose", "family": "continuation",
     "qqq_pressure": .10, "stock_pressure": .05,
     "breadth": .625, "qqq_rvol": .70},
    {"name": "flow_continuation_strict", "family": "continuation",
     "qqq_pressure": .20, "stock_pressure": .15,
     "breadth": .75, "qqq_rvol": 1.0},
    {"name": "flow_last_hour_confirmation", "family": "last_hour",
     "qqq_pressure": .15, "stock_pressure": .10,
     "breadth": .625, "stock_rvol": .8},
    {"name": "flow_reversal", "family": "reversal",
     "qqq_pressure": .20, "stock_pressure": .10,
     "breadth": .625, "stock_rvol": .8},
    {"name": "flow_breadth_acceleration", "family": "acceleration",
     "qqq_pressure": .10, "stock_pressure": .05,
     "breadth": .625, "delta": .125},
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def weighted_pressure(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                      volume: np.ndarray, indices: np.ndarray) -> float:
    """Volume weighted close location in [-1, 1] for completed bars."""
    if len(indices) == 0:
        return 0.0
    width = high[indices] - low[indices]
    location = np.divide(
        2 * close[indices] - high[indices] - low[indices], width,
        out=np.zeros(len(indices), dtype=float), where=width > 0)
    weights = np.maximum(volume[indices], 0)
    total = float(weights.sum())
    return float(np.dot(location, weights) / total) if total > 0 else 0.0


def checkpoint_volume(symbol: str, candle: dict) -> pd.DataFrame:
    frame, _, _ = base.session_table(symbol, candle)
    groups = regular_groups(candle)
    high, low, close, volume = (
        np.asarray(candle[key], dtype=float)
        for key in ("high", "low", "close", "volume"))
    session_pressure = []
    last_hour_pressure = []
    signal_volume = []
    for row in frame.itertuples(index=False):
        bars = np.asarray(groups[row.session_date], dtype=np.int64)
        signal_index = int(row.entry_index) - 1
        completed = bars[bars <= signal_index]
        session_pressure.append(weighted_pressure(
            high, low, close, volume, completed))
        last_hour_pressure.append(weighted_pressure(
            high, low, close, volume, completed[-4:]))
        signal_volume.append(float(volume[signal_index]))
    out = frame.copy()
    out["session_pressure"] = session_pressure
    out["last_hour_pressure"] = last_hour_pressure
    out["signal_volume"] = signal_volume
    prior_median = out.groupby("slot", sort=False).signal_volume.transform(
        lambda value: value.shift(1).rolling(20, min_periods=10).median())
    out["rvol"] = out.signal_volume / prior_median.replace(0, np.nan)
    return out


def build_features(candles: dict[str, dict]) -> pd.DataFrame:
    raw = pd.concat(
        [checkpoint_volume(symbol, candles[symbol]) for symbol in base.ALL_SYMBOLS],
        ignore_index=True)
    qqq = raw[raw.symbol.eq("QQQUSDT")][[
        "session_date", "slot", "session_pressure", "last_hour_pressure", "rvol",
    ]].rename(columns={
        "session_pressure": "qqq_session_pressure",
        "last_hour_pressure": "qqq_last_hour_pressure",
        "rvol": "qqq_rvol",
    })
    stocks = raw[raw.symbol.ne("QQQUSDT")].copy()
    group = stocks.groupby(["session_date", "slot"], sort=False)
    stocks["flow_breadth"] = group.session_pressure.transform(
        lambda value: (value > 0).mean())
    stocks["last_flow_breadth"] = group.last_hour_pressure.transform(
        lambda value: (value > 0).mean())
    stocks = stocks.merge(
        qqq, on=["session_date", "slot"], how="inner", validate="many_to_one")
    order = {"early": 0, "middle": 1, "late": 2}
    stocks["slot_order"] = stocks.slot.map(order).astype(int)
    stocks = stocks.sort_values(["symbol", "session_date", "slot_order"])
    grouped = stocks.groupby(["symbol", "session_date"], sort=False)
    stocks["prior_flow_breadth"] = grouped.flow_breadth.shift(1)
    stocks["flow_breadth_delta"] = (
        stocks.flow_breadth - stocks.prior_flow_breadth)
    return stocks.replace([np.inf, -np.inf], np.nan).dropna().copy()


def directed(frame: pd.DataFrame, direction: np.ndarray) -> pd.DataFrame:
    out = frame.copy()
    out["direction_n"] = direction.astype(np.int8)
    sign = out.direction_n.to_numpy(float)
    for column in ("intraday", "last_hour", "session_pressure",
                   "last_hour_pressure", "qqq_intraday",
                   "qqq_session_pressure", "qqq_last_hour_pressure"):
        out[f"{column}_dir"] = out[column] * sign
    out["flow_breadth_dir"] = np.where(
        sign > 0, out.flow_breadth, 1 - out.flow_breadth)
    out["last_flow_breadth_dir"] = np.where(
        sign > 0, out.last_flow_breadth, 1 - out.last_flow_breadth)
    out["flow_breadth_delta_dir"] = out.flow_breadth_delta * sign
    return out


def apply_spec(frame: pd.DataFrame, spec: dict) -> pd.DataFrame:
    family = spec["family"]
    if family == "continuation":
        found = directed(frame, np.sign(frame.qqq_session_pressure.to_numpy(float)))
        mask = (
            found.direction_n.ne(0)
            & found.qqq_session_pressure_dir.ge(spec["qqq_pressure"])
            & found.session_pressure_dir.ge(spec["stock_pressure"])
            & found.flow_breadth_dir.ge(spec["breadth"])
            & found.qqq_rvol.ge(spec["qqq_rvol"]))
    elif family == "last_hour":
        found = directed(frame, np.sign(frame.qqq_last_hour_pressure.to_numpy(float)))
        mask = (
            found.direction_n.ne(0)
            & found.qqq_last_hour_pressure_dir.ge(spec["qqq_pressure"])
            & found.last_hour_pressure_dir.ge(spec["stock_pressure"])
            & found.last_flow_breadth_dir.ge(spec["breadth"])
            & found.rvol.ge(spec["stock_rvol"]))
    elif family == "reversal":
        direction = np.sign(frame.qqq_last_hour_pressure.to_numpy(float))
        found = directed(frame, direction)
        mask = (
            found.direction_n.ne(0)
            & found.qqq_last_hour_pressure_dir.ge(spec["qqq_pressure"])
            & found.qqq_session_pressure_dir.lt(0)
            & found.last_hour_pressure_dir.ge(spec["stock_pressure"])
            & found.last_flow_breadth_dir.ge(spec["breadth"])
            & found.rvol.ge(spec["stock_rvol"]))
    elif family == "acceleration":
        direction = np.sign(frame.qqq_last_hour_pressure.to_numpy(float))
        found = directed(frame, direction)
        mask = (
            found.direction_n.ne(0)
            & found.qqq_last_hour_pressure_dir.ge(spec["qqq_pressure"])
            & found.last_hour_pressure_dir.ge(spec["stock_pressure"])
            & found.last_flow_breadth_dir.ge(spec["breadth"])
            & found.flow_breadth_delta_dir.ge(spec["delta"]))
    else:
        raise ValueError(family)
    selected = found.loc[mask].copy()
    selected["entry_spec"] = spec["name"]
    selected["score"] = (
        selected.qqq_last_hour_pressure_dir.clip(lower=0)
        + selected.last_hour_pressure_dir.clip(lower=0)
        + selected.flow_breadth_dir)
    return selected


def accepted_rows(frame: pd.DataFrame, years: tuple[int, ...]) -> list[dict]:
    selected = frame[frame.year.isin(years)]
    return portfolio_gate(
        selected.to_dict("records"), None,
        rank_key=lambda row: (-float(row["score"]),))


def evaluate(frame: pd.DataFrame, features: pd.DataFrame,
             years: tuple[int, ...]) -> dict:
    value = base.evaluate(frame, features, years)
    rows = accepted_rows(frame, years)
    returns = [float(row["signed_return"]) for row in rows]
    value["literal_target_hits"] = sum(
        int(row["outcome_code"]) == 1 for row in rows)
    value["literal_target_hit_rate"] = round(
        100 * value["literal_target_hits"] / len(rows), 4) if rows else 0.0
    value["economic_gate"] = economic_gate(
        returns, float(value["trades_per_market_day"]), market_days=22)
    return value


def development_ok(value: dict, annual: dict[str, dict]) -> bool:
    return bool(
        value["n"] >= 150
        and value["return_pct"] > 0
        and value["dd_pct"] <= 10
        and value["trades_per_market_day"] >= .3
        and sum(item["return_pct"] > 0 for item in annual.values()) >= 4)


def validation_ok(value: dict, annual: dict[str, dict]) -> bool:
    return bool(
        value["n"] >= 60
        and value["return_pct"] > 0
        and value["dd_pct"] <= 7
        and value["trades_per_market_day"] >= .2
        and all(item["return_pct"] > 0 for item in annual.values()))


def objective_ok(value: dict) -> bool:
    return bool(
        75 <= value["literal_target_hit_rate"] <= 95
        and value["trades_per_market_day"] >= 3
        and value["negative_months"] == 0
        and value["dd_pct"] <= 5
        and value["economic_gate"]["passes"])


def public(item: dict | None) -> dict | None:
    return ({key: value for key, value in item.items() if key != "_rows"}
            if item else None)


def main() -> None:
    candles, sources = base.load_history()
    base_features, metadata = base.build_features(candles)
    volume_features = build_features(candles)
    feature_columns = [
        "symbol", "session_date", "slot", "session_pressure",
        "last_hour_pressure", "rvol", "flow_breadth", "last_flow_breadth",
        "qqq_session_pressure", "qqq_last_hour_pressure", "qqq_rvol",
        "prior_flow_breadth", "flow_breadth_delta",
    ]
    features = base_features.merge(
        volume_features[feature_columns],
        on=["symbol", "session_date", "slot"], how="inner",
        validate="one_to_one")
    frontier = []
    validated = []
    for spec, exit_spec in itertools.product(SPECS, EXITS):
        entries = apply_spec(features, spec)
        resolved = base.add_outcomes(
            entries, candles, metadata,
            exit_spec["target_pct"], exit_spec["stop_pct"],
            exit_spec["hold_sessions"])
        development = evaluate(resolved, features, DEVELOPMENT)
        annual = {
            str(year): evaluate(resolved, features, (year,))
            for year in DEVELOPMENT}
        item = {
            "entry": spec, "exit": exit_spec,
            "development": development, "development_years": annual,
            "development_pass": development_ok(development, annual),
            "_rows": resolved,
        }
        frontier.append(item)
        if not item["development_pass"]:
            continue
        validation = evaluate(resolved, features, VALIDATION)
        validation_annual = {
            str(year): evaluate(resolved, features, (year,))
            for year in VALIDATION}
        item["validation"] = validation
        item["validation_years"] = validation_annual
        item["validation_pass"] = validation_ok(validation, validation_annual)
        if item["validation_pass"]:
            item["score"] = round(
                min(development["return_pct"], validation["return_pct"])
                - 2 * max(development["dd_pct"], validation["dd_pct"])
                - validation["negative_months"], 8)
            validated.append(item)

    frontier.sort(key=lambda item: (
        item["development_pass"], item["development"]["return_pct"],
        item["development"]["literal_target_hit_rate"]), reverse=True)
    validated.sort(key=lambda item: item["score"], reverse=True)
    frozen = validated[0] if validated else None
    test = confirmation = checks = None
    status = "REJECTED_BEFORE_TEST"
    if frozen:
        rows = frozen["_rows"]
        test = evaluate(rows, features, TEST)
        status = "REJECTED_TEST"
        if objective_ok(test):
            confirmation = {
                str(year): evaluate(rows, features, (year,))
                for year in CONFIRMATION}
            checks = {
                "test_2024": True,
                **{str(year): objective_ok(confirmation[str(year)])
                   for year in CONFIRMATION},
            }
            status = ("VOLUME_PRESSURE_RESEARCH_CANDIDATE"
                      if all(checks.values()) else "REJECTED_OBJECTIVE")
            if status == "VOLUME_PRESSURE_RESEARCH_CANDIDATE":
                rows.to_csv(ROWS_OUT, index=False)

    report = {
        "status": status,
        "ready_for_live_orders": False,
        "protocol": {
            "mechanism": "causal volume-weighted close-location pressure",
            "development": list(DEVELOPMENT), "validation": list(VALIDATION),
            "conditional_test": list(TEST),
            "confirmation_only": list(CONFIRMATION),
            "entry": "next regular-session market bar",
            "entry_specs": len(SPECS), "exit_specs": list(EXITS),
            "money": "3% current-equity margin, fixed 10x gross",
            "market_orders_only": True,
        },
        "feature_rows": len(features),
        "validated_before_test": len(validated),
        "frozen_before_test": public(frozen),
        "test_2024": test,
        "confirmation_2025_2026": confirmation,
        "objective_checks": checks,
        "development_frontier": [public(item) for item in frontier[:15]],
        "sources": {
            **sources,
            str(Path(base.__file__)): digest(Path(base.__file__)),
            str(Path(__file__)): digest(Path(__file__)),
        },
        "limitations": [
            "Dukascopy volume is relative CFD activity, not consolidated US share volume.",
            "Cross-era evidence must transfer to direct X-Perp volume before runtime use.",
            "A historical pass would still require fresh shadow outcomes.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": status, "feature_rows": len(features),
        "validated_before_test": len(validated),
        "frozen_before_test": public(frozen), "test_2024": test,
        "confirmation": confirmation,
        "top_development": [public(item) for item in frontier[:5]],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
