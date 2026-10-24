"""Causal macro-regime enrichment of the stock liquidity-sweep audit."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import stock_liquidity_sweep_walkforward_oct22_audit as sweep


OUT = sweep.ROOT / "stock_liquidity_sweep_macro_rolling_oct22.json"
DEV_YEARS = (2022, 2023)
ACCEPTANCE_YEAR = 2024
HIDDEN_YEARS = (2025, 2026)
QUANTILES = (.30, .40, .50, .60, .70, .80, .85, .90)
MACRO_FEATURES = (
    "daily_ret5_dir", "daily_ret20_dir", "daily_vol20",
    "gap_dir", "intraday_dir", "qqq_daily_ret5_dir",
    "qqq_daily_ret20_dir", "qqq_daily_vol20", "qqq_gap_dir",
    "qqq_intraday_dir",
)


def daily_context(candle: dict, symbol: str) -> pd.DataFrame:
    groups = sweep.base.regular_groups(candle)
    sessions = [(day, bars) for day, bars in sorted(groups.items()) if len(bars) >= 20]
    closes = []
    rows = []
    close = np.asarray(candle["close"], dtype=float)
    open_ = np.asarray(candle["open"], dtype=float)
    for day, bars in sessions:
        if len(closes) >= 21:
            previous = closes[-1]
            returns = np.diff(np.log(np.asarray(closes[-21:], dtype=float)))
            rows.append({
                "symbol": symbol, "session_date": day,
                "daily_ret5": previous/closes[-6]-1,
                "daily_ret20": previous/closes[-21]-1,
                "daily_vol20": float(np.std(returns)),
                "session_open": float(open_[bars[0]]),
                "previous_close": previous,
            })
        closes.append(float(close[bars[-1]]))
    return pd.DataFrame(rows)


def enrich(features: pd.DataFrame, candles: dict[str, dict]) -> pd.DataFrame:
    contexts = pd.concat(
        [daily_context(candles[symbol], symbol) for symbol in (*sweep.SYMBOLS, "QQQUSDT")],
        ignore_index=True)
    stocks = contexts[contexts.symbol.ne("QQQUSDT")]
    qqq = contexts[contexts.symbol.eq("QQQUSDT")].drop(columns="symbol").rename(
        columns={column: f"qqq_daily_{column}" for column in
                 ("daily_ret5", "daily_ret20", "daily_vol20",
                  "session_open", "previous_close")})
    value = features.merge(stocks, on=["symbol", "session_date"], how="left",
                           validate="many_to_one")
    value = value.merge(qqq, on="session_date", how="left", validate="many_to_one")
    close_maps = {
        symbol: {int(timestamp): float(close) for timestamp, close in
                 zip(candles[symbol]["time"], candles[symbol]["close"], strict=True)}
        for symbol in (*sweep.SYMBOLS, "QQQUSDT")
    }
    signal_times = value.entry_time.astype(int)-900
    value["signal_close"] = [close_maps[symbol].get(int(timestamp), np.nan)
                             for symbol, timestamp in zip(value.symbol, signal_times, strict=True)]
    value["qqq_signal_close"] = [close_maps["QQQUSDT"].get(int(timestamp), np.nan)
                                 for timestamp in signal_times]
    direction = value.direction_n.astype(float)
    value["daily_ret5_dir"] = direction*value.daily_ret5
    value["daily_ret20_dir"] = direction*value.daily_ret20
    value["gap_dir"] = direction*(value.session_open/value.previous_close-1)
    value["intraday_dir"] = direction*(value.signal_close/value.session_open-1)
    value["qqq_daily_ret5_dir"] = direction*value.qqq_daily_daily_ret5
    value["qqq_daily_ret20_dir"] = direction*value.qqq_daily_daily_ret20
    value["qqq_daily_vol20"] = value.qqq_daily_daily_vol20
    value["qqq_gap_dir"] = direction*(
        value.qqq_daily_session_open/value.qqq_daily_previous_close-1)
    value["qqq_intraday_dir"] = direction*(
        value.qqq_signal_close/value.qqq_daily_session_open-1)
    return value.replace([np.inf, -np.inf], np.nan).dropna(subset=list(MACRO_FEATURES))


def matrix(frame: pd.DataFrame) -> np.ndarray:
    macro = frame[list(MACRO_FEATURES)].fillna(0).to_numpy(np.float32)
    return np.column_stack([sweep.matrix(frame), macro])


def choose(frame: pd.DataFrame, features: pd.DataFrame, threshold: float) -> dict:
    chosen = frame[frame.quality >= threshold].copy()
    chosen["score"] = chosen.quality
    rows = sweep.portfolio_gate(chosen.to_dict("records"), None,
                                rank_key=lambda row: (-float(row["quality"]),))
    accepted = pd.DataFrame(rows) if rows else chosen.iloc[:0].copy()
    return sweep.base.account_metrics(
        accepted, features[features.year.isin(frame.year.unique())].session_date.nunique())


def fold(history: pd.DataFrame, features: pd.DataFrame,
         test_year: int, spec: dict, keep_model: bool = False) -> dict:
    validation_year = test_year-1
    train_years = tuple(range(max(2017, test_year-6), validation_year))
    train = history[history.year.isin(train_years)]
    validation = history[history.year.eq(validation_year)].copy()
    cfg = spec["model"]
    model = HistGradientBoostingClassifier(
        max_iter=180, learning_rate=.05, max_leaf_nodes=cfg["leaves"],
        min_samples_leaf=cfg["leaf"], l2_regularization=cfg["l2"],
        random_state=20261022)
    model.fit(matrix(train), (train.outcome_code == 1).astype(int))
    validation["quality"] = model.predict_proba(matrix(validation))[:, 1]
    choices = []
    for quantile in QUANTILES:
        threshold = float(validation.quality.quantile(quantile))
        value = choose(validation, features, threshold)
        eligible = (value["n"] >= 100 and value["return_pct"] > 0
                    and value["dd_pct"] <= 12
                    and value["trades_per_market_day"] >= 1)
        score = (value["return_pct"]-1.5*value["dd_pct"]
                 -.5*value["negative_months"]+.01*value["wr"])
        choices.append({"quantile": quantile, "threshold": threshold,
                        "eligible": eligible, "score": score, "validation": value})
    eligible = [item for item in choices if item["eligible"]]
    selected = max(eligible, key=lambda item: item["score"], default=None)
    if selected is None:
        selected = max(choices, key=lambda item: item["score"])
    test = history[history.year.eq(test_year)].copy()
    test["quality"] = model.predict_proba(matrix(test))[:, 1]
    output = {
        "year": test_year, "train_years": list(train_years),
        "validation_year": validation_year,
        "threshold_selection_pass": bool(selected["eligible"]),
        "chosen_quantile": selected["quantile"], "threshold": selected["threshold"],
        "validation": selected["validation"],
        "test": choose(test, features, selected["threshold"]),
        "threshold_frontier": sorted(choices, key=lambda x: x["score"], reverse=True)[:4],
    }
    if keep_model:
        output["_model"] = model
    return output


def main() -> None:
    historical, _ = sweep.base.load_history()
    direct, direct_sources = sweep.load_direct_15m()
    history_raw, history_meta = sweep.sweep_features(historical)
    direct_raw, direct_meta = sweep.sweep_features(direct)
    features = enrich(history_raw, historical)
    direct_features = enrich(direct_raw, direct)
    exits = {(target, stop, hold): sweep.outcomes(
        features, historical, history_meta, target, stop, hold)
             for target in sweep.TARGETS for stop in sweep.STOPS for hold in sweep.HOLDS}
    candidates = []
    for (target, stop, hold), history in exits.items():
        for model in sweep.MODELS:
            spec = {"target_pct": target, "stop_pct": stop,
                    "hold_sessions": hold, "model": model}
            development = [fold(history, features, year, spec) for year in DEV_YEARS]
            values = [item["test"] for item in development]
            if (all(item["threshold_selection_pass"] for item in development)
                    and all(value["n"] >= 100 and value["return_pct"] > 0
                            and value["dd_pct"] <= 12 for value in values)):
                score = (sum(value["return_pct"] for value in values)
                         + 2*min(value["return_pct"] for value in values)
                         - sum(value["dd_pct"] for value in values))
                candidates.append({"spec": spec, "score": score,
                                   "development": development})
    candidates.sort(key=lambda item: item["score"], reverse=True)
    accepted = []
    for candidate in candidates:
        spec = candidate["spec"]
        history = exits[(spec["target_pct"], spec["stop_pct"], spec["hold_sessions"])]
        candidate["acceptance_2024"] = fold(history, features, ACCEPTANCE_YEAR, spec)
        value = candidate["acceptance_2024"]["test"]
        if (candidate["acceptance_2024"]["threshold_selection_pass"]
                and value["n"] >= 100 and value["return_pct"] > 0
                and value["dd_pct"] <= 8):
            accepted.append(candidate)
    frozen = accepted[0] if accepted else None
    hidden = direct_result = checks = None
    if frozen:
        spec = frozen["spec"]
        history = exits[(spec["target_pct"], spec["stop_pct"], spec["hold_sessions"])]
        hidden = [fold(history, features, year, spec, keep_model=(year == 2026))
                  for year in HIDDEN_YEARS]
        final = hidden[-1]
        direct_exit = sweep.outcomes(
            direct_features, direct, direct_meta, spec["target_pct"],
            spec["stop_pct"], spec["hold_sessions"])
        direct_exit["quality"] = final["_model"].predict_proba(matrix(direct_exit))[:, 1]
        direct_result = choose(direct_exit, direct_features, final["threshold"])
        final.pop("_model", None)
        checks = {
            "acceptance_positive": frozen["acceptance_2024"]["test"]["return_pct"] > 0,
            "test_2025_positive": hidden[0]["test"]["return_pct"] > 0,
            "observed_2026_positive": hidden[1]["test"]["return_pct"] > 0,
            "direct_positive": direct_result["return_pct"] > 0,
            "late_dd_le_8": all(item["test"]["dd_pct"] <= 8
                                for item in [frozen["acceptance_2024"], *hidden]),
            "direct_n_ge_30": direct_result["n"] >= 30,
        }

    def public(item):
        return item if item is None else {key: value for key, value in item.items()}

    status = "SHADOW_CANDIDATE" if checks and all(checks.values()) else "REJECTED"
    report = {
        "status": status,
        "protocol": {
            "setup": "causal confirmed-swing liquidity sweep with prior-day and intraday macro context",
            "configuration_selection_folds": list(DEV_YEARS),
            "acceptance_year": ACCEPTANCE_YEAR, "hidden_years": list(HIDDEN_YEARS),
            "annual_adaptation": "fit through year-2 and select threshold on year-1",
            "features": list((*sweep.FEATURES, *MACRO_FEATURES)),
            "entry": "next 15-minute market open", "orders": "market only",
            "money": "3% current-equity margin, fixed 10x gross",
        },
        "rows": {"history": len(features), "direct": len(direct_features)},
        "tested_configurations": len(exits)*len(sweep.MODELS),
        "development_qualified": len(candidates),
        "acceptance_qualified": len(accepted),
        "frozen_before_hidden": public(frozen),
        "hidden_2025_2026": hidden, "direct_xperp": direct_result,
        "checks": checks,
        "top_development": [public(item) for item in candidates[:10]],
        "sources": direct_sources,
        "limitations": [
            "Cash candles are structural transfer evidence for stock perpetuals.",
            "The bounded model and threshold grid creates selection risk.",
            "Annual adaptation cannot react before a regime change appears in completed outcomes.",
            "Historical results cannot guarantee future profit or eliminate losing months.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": status, "rows": report["rows"],
        "development_qualified": len(candidates), "acceptance_qualified": len(accepted),
        "frozen": frozen, "hidden": hidden, "direct": direct_result,
        "checks": checks,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
