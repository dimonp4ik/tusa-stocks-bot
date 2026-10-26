"""Cross-era audit of stock shocks after removing a causal index beta.

The signal is deliberately different from the native cross-sectional route.
For every stock, a rolling beta is estimated only from bars completed before
the signal bar.  The current multi-bar stock return minus beta times the QQQ
or SPY return is standardized against that stock's own earlier residuals.
Both reversion and continuation are evaluated, with next-bar market entries.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import stock_cash_residual_cross_section_oct21_audit as base
import stock_cash_xperp_leadlag_oct6_audit as resolver
import stock_xperp_current_route_cash_transfer_oct21_audit as parent


ROOT = Path("reports/audit_2026_09_08")
MANIFEST = ROOT / "stock_extended_history_manifest.json"
OUT = ROOT / "stock_factor_residual_shock_oct26.json"
ROWS_OUT = ROOT / "stock_factor_residual_shock_oct26.csv"
DEVELOPMENT = (2017, 2018, 2019, 2020, 2021)
VALIDATION = (2022, 2023)
TEST = (2024,)
CONFIRMATION = (2025, 2026)
FACTORS = ("QQQUSDT", "SPYUSDT")
LOOKBACKS = (1, 2, 4, 8)
BETA_WINDOWS = (26, 78, 260)
Z_THRESHOLDS = (2.0, 2.5, 3.0)
RAW_THRESHOLDS = (.002, .004, .006)
STYLES = ("reversion", "continuation")
TARGETS = (1.0, 2.0)
STOPS = (.75, 1.0)
HOLDS = (16, 32, 64)
TOP_EXACT = 100


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load() -> tuple[dict[str, dict], dict[str, dict], dict[str, str]]:
    stocks, sources = parent.load_history()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    factors = {}
    for symbol in FACTORS:
        path = Path(manifest["symbols"][symbol]["path"])
        raw = path.read_bytes()
        if digest(path) != manifest["symbols"][symbol]["sha256"]:
            raise ValueError(f"source hash mismatch: {symbol}")
        factors[symbol] = pickle.loads(raw)
        sources[str(path)] = digest(path)
    return stocks, factors, sources


def _series(candle: dict, prefix: str) -> pd.DataFrame:
    return pd.DataFrame({
        "entry_index": np.arange(len(candle["time"]), dtype=np.int64),
        "entry_time": np.asarray(candle["time"], dtype=np.int64),
        f"{prefix}_close": np.asarray(candle["close"], dtype=float),
    })


def feature_rows(stocks: dict[str, dict], factor: dict,
                 lookback: int, beta_window: int) -> pd.DataFrame:
    benchmark = _series(factor, "factor").drop(columns="entry_index")
    pieces = []
    for symbol, candle in stocks.items():
        frame = _series(candle, "stock").merge(
            benchmark, on="entry_time", how="inner", validate="one_to_one")
        contiguous = frame.entry_time.diff().eq(900)
        stock_log = np.log(frame.stock_close)
        factor_log = np.log(frame.factor_close)
        stock_one = stock_log.diff().where(contiguous)
        factor_one = factor_log.diff().where(contiguous)
        # Session gaps are NaN by design.  Requiring every observation would
        # make a 26-bar regular-session window impossible, so require 75% of
        # the declared trailing rows while still excluding gap returns.
        minimum = max(12, int(np.ceil(.75 * beta_window)))
        covariance = stock_one.rolling(
            beta_window, min_periods=minimum).cov(factor_one)
        variance = factor_one.rolling(
            beta_window, min_periods=minimum).var(ddof=1)
        # shift(1) prevents the signal bar from influencing its own beta.
        beta = (covariance / variance.replace(0, np.nan)).shift(1)
        window_ok = frame.entry_time.diff(lookback).eq(900 * lookback)
        stock_move = stock_log.diff(lookback).where(window_ok)
        factor_move = factor_log.diff(lookback).where(window_ok)
        residual = stock_move - beta * factor_move
        residual_scale = residual.rolling(
            beta_window, min_periods=minimum).std(ddof=1).shift(1)
        frame["symbol"] = symbol
        frame["signal_return"] = np.expm1(stock_move)
        frame["factor_return"] = np.expm1(factor_move)
        frame["beta"] = beta
        frame["factor_residual"] = np.expm1(residual)
        frame["residual_z"] = residual / residual_scale.replace(0, np.nan)
        frame["entry_index"] += 1
        frame["entry_time"] += 900
        next_exists = frame.entry_index.lt(len(candle["time"]))
        next_contiguous = frame.entry_time.eq(
            pd.Series(np.asarray(candle["time"], dtype=np.int64)).reindex(
                frame.entry_index).to_numpy())
        pieces.append(frame.loc[next_exists & next_contiguous, [
            "symbol", "entry_index", "entry_time", "signal_return",
            "factor_return", "beta", "factor_residual", "residual_z"]])
    result = pd.concat(pieces, ignore_index=True)
    result["year"] = pd.to_datetime(
        result.entry_time, unit="s", utc=True).dt.year.astype(np.int16)
    return result.replace([np.inf, -np.inf], np.nan).dropna()


def select(frame: pd.DataFrame, z: float, raw: float,
           style: str) -> pd.DataFrame:
    found = frame[
        frame.residual_z.abs().ge(z)
        & frame.factor_residual.abs().ge(raw)
    ].copy()
    sign = np.sign(found.factor_residual.to_numpy(float)).astype(np.int8)
    found["direction_n"] = -sign if style == "reversion" else sign
    found["style"] = style
    found["cross_z"] = found.residual_z
    found["residual_return"] = found.factor_residual
    return found


def parameters(factor, lookback, beta_window, z, raw, style,
               target, stop, hold) -> dict:
    return {
        "factor": factor, "lookback_bars": lookback,
        "beta_window_bars": beta_window, "z_threshold": z,
        "minimum_residual_pct": 100 * raw, "style": style,
        "target_pct": target, "stop_pct": stop, "hold_bars": hold,
    }


def resolve(rule: dict, features: pd.DataFrame,
            candles: dict[str, dict], years: tuple[int, ...]) -> pd.DataFrame:
    entries = select(features, float(rule["z_threshold"]),
                     float(rule["minimum_residual_pct"]) / 100,
                     str(rule["style"]))
    entries = entries[entries.year.isin(years)]
    found = resolver.add_outcomes(
        entries, candles, float(rule["target_pct"]),
        float(rule["stop_pct"]), int(rule["hold_bars"]))
    found["module"] = "factor_residual_shock_" + str(rule["style"])
    found["module_priority"] = 0
    return found


def public(item: dict) -> dict:
    return {key: value for key, value in item.items()
            if key not in ("_accepted", "_rows")}


def main() -> None:
    stocks, factors, sources = load()
    features: dict[tuple[str, int, int], pd.DataFrame] = {}
    raw_candidates = []
    searched = 0
    for factor_name, lookback, beta_window in itertools.product(
            FACTORS, LOOKBACKS, BETA_WINDOWS):
        frame = feature_rows(stocks, factors[factor_name], lookback, beta_window)
        features[(factor_name, lookback, beta_window)] = frame
        broad = frame[
            frame.year.isin(DEVELOPMENT)
            & frame.residual_z.abs().ge(min(Z_THRESHOLDS))
            & frame.factor_residual.abs().ge(min(RAW_THRESHOLDS))]
        for style, target, stop, hold in itertools.product(
                STYLES, TARGETS, STOPS, HOLDS):
            signed = broad.copy()
            sign = np.sign(signed.factor_residual.to_numpy(float)).astype(np.int8)
            signed["direction_n"] = -sign if style == "reversion" else sign
            resolved = resolver.add_outcomes(signed, stocks, target, stop, hold)
            for z, raw in itertools.product(Z_THRESHOLDS, RAW_THRESHOLDS):
                searched += 1
                subset = resolved[
                    resolved.residual_z.abs().ge(z)
                    & resolved.factor_residual.abs().ge(raw)]
                if len(subset) < 300:
                    continue
                bundle = base.raw_bundle(subset, stocks, DEVELOPMENT)
                combined = bundle["combined"]
                annual = list(bundle["annual"].values())
                if (combined["return_pct"] <= 0 or combined["dd_pct"] > 20
                        or combined["trades_per_market_day"] < 2.5
                        or sum(x["return_pct"] > 0 for x in annual) < 3):
                    continue
                score = (min(x["return_pct"] for x in annual)
                         + .2 * combined["return_pct"] - combined["dd_pct"])
                raw_candidates.append({
                    "parameters": parameters(
                        factor_name, lookback, beta_window, z, raw,
                        style, target, stop, hold),
                    "development_raw": bundle,
                    "development_raw_score": score,
                })
        print(json.dumps({"factor": factor_name, "lookback": lookback,
                          "beta_window": beta_window,
                          "raw_passes": len(raw_candidates)}), flush=True)

    raw_candidates.sort(key=lambda x: x["development_raw_score"], reverse=True)
    exact = []
    for item in raw_candidates[:TOP_EXACT]:
        rule = item["parameters"]
        frame = features[(rule["factor"], rule["lookback_bars"],
                          rule["beta_window_bars"])]
        resolved = resolve(rule, frame, stocks, DEVELOPMENT)
        stage, accepted = base.exact_bundle(resolved, stocks, DEVELOPMENT)
        if base.development_ok(stage):
            item["development"] = stage
            item["development_score"] = (
                min(x["return_pct"] for x in stage["annual_3pct"].values())
                + .2 * stage["margin_3pct"]["return_pct"]
                - stage["margin_3pct"]["dd_pct"])
            item["_accepted"] = accepted
            exact.append(item)
    exact.sort(key=lambda x: x["development_score"], reverse=True)

    validated = []
    for item in exact:
        rule = item["parameters"]
        frame = features[(rule["factor"], rule["lookback_bars"],
                          rule["beta_window_bars"])]
        stage, _ = base.exact_bundle(
            resolve(rule, frame, stocks, VALIDATION), stocks, VALIDATION)
        item["validation"] = stage
        if base.later_ok(stage):
            validated.append(item)

    frozen = validated[0] if validated else None
    test = confirmation = None
    status = "REJECTED_DEVELOPMENT" if not exact else "REJECTED_VALIDATION"
    if frozen:
        rule = frozen["parameters"]
        frame = features[(rule["factor"], rule["lookback_bars"],
                          rule["beta_window_bars"])]
        test, _ = base.exact_bundle(
            resolve(rule, frame, stocks, TEST), stocks, TEST)
        status = "REJECTED_TEST"
        if base.later_ok(test):
            confirmation, accepted = base.exact_bundle(
                resolve(rule, frame, stocks, CONFIRMATION),
                stocks, CONFIRMATION)
            frozen["_accepted"] = accepted
            status = ("CROSS_ERA_CANDIDATE" if base.later_ok(confirmation)
                      else "REJECTED_CONFIRMATION")

    chosen = frozen or (exact[0] if exact else None)
    pd.DataFrame(chosen.get("_accepted", []) if chosen else []).to_csv(
        ROWS_OUT, index=False)
    report = {
        "status": status, "ready_for_live_orders": False,
        "protocol": {
            "hypothesis": "stock-specific shock after removing lagged rolling beta to QQQ or SPY",
            "development": list(DEVELOPMENT), "validation": list(VALIDATION),
            "test": list(TEST), "confirmation": list(CONFIRMATION),
            "beta": "estimated from prior completed 15m returns; signal bar excluded",
            "standardization": "each stock's prior residual distribution",
            "entry": "next 15m market open", "orders": "market only",
            "money": "client 3% or 5% current-equity margin; fixed 10x",
        },
        "grid": {"searched": searched, "raw_passes": len(raw_candidates),
                 "exact_development_passes": len(exact),
                 "validation_passes": len(validated)},
        "selected": public(chosen) if chosen else None,
        "test": test, "confirmation": confirmation,
        "sources": {**sources, str(Path(__file__).resolve()): digest(Path(__file__))},
        "limitations": [
            "Cash-proxy history tests structure, not historical X-Perp fills.",
            "Only ten equities have long aligned history.",
            "Gross returns exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: report[k] for k in
                      ("status", "grid", "selected", "test", "confirmation")},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
