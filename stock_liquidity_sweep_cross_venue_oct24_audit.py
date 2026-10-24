"""Audit cash/index liquidity-sweep signals with X-Perp execution.

The model and threshold reproduce the macro sweep policy frozen before its
direct venue check.  Signals are rebuilt from timestamp-matched OKX index
candles; entries, stops, targets, and exits use only X-Perp candles.
"""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import stock_liquidity_sweep_macro_rolling_oct22_audit as macro
import stock_liquidity_sweep_walkforward_oct22_audit as sweep
from stock_module_cluster_audit import portfolio_gate


ROOT = sweep.ROOT
FROZEN_REPORT = ROOT / "stock_liquidity_sweep_macro_rolling_oct22.json"
INDEX_DIR = ROOT / "stock_index_forward_oct22"
EXECUTION_DIR = ROOT / "stock_xperp_forward_oct22"
OUT = ROOT / "stock_liquidity_sweep_cross_venue_oct24.json"
ROWS_OUT = ROOT / "stock_liquidity_sweep_cross_venue_oct24.pkl"
REQUIRED_MEAN_PRICE_RETURN_PCT = 1.327709


def load_folder(root: Path) -> tuple[dict[str, dict], dict[str, str]]:
    candles, sources = {}, {}
    for symbol in (*sweep.SYMBOLS, "QQQUSDT"):
        path = root / f"{symbol}_15min_current.pkl"
        raw = path.read_bytes()
        candles[symbol] = pickle.loads(raw)
        sources[str(path)] = hashlib.sha256(raw).hexdigest()
    return candles, sources


def execution_outcomes(features: pd.DataFrame, execution: dict[str, dict],
                       target: float, stop: float, hold: int) -> pd.DataFrame:
    remapped = []
    metadata = {}
    for symbol, part in features.groupby("symbol", sort=False):
        candle = execution[symbol]
        regular, _, bars_per_session = sweep.regular_meta(candle)
        timestamp_to_index = {
            int(candle["time"][int(index)]): int(index) for index in regular
        }
        chosen = part[part.entry_time.astype(int).isin(timestamp_to_index)].copy()
        chosen["entry_index"] = chosen.entry_time.astype(int).map(timestamp_to_index)
        remapped.append(chosen)
        metadata[symbol] = {"regular": regular,
                            "bars_per_session": bars_per_session}
    frame = pd.concat(remapped, ignore_index=True) if remapped else features.iloc[:0]
    return sweep.outcomes(frame, execution, metadata, target, stop, hold)


def selected_rows(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    candidates = frame[frame.quality >= threshold].copy()
    candidates["score"] = candidates.quality
    rows = portfolio_gate(candidates.to_dict("records"), None,
                          rank_key=lambda row: (-float(row["quality"]),))
    return pd.DataFrame(rows) if rows else candidates.iloc[:0].copy()


def economic_summary(rows: pd.DataFrame, market_days: int) -> dict:
    value = sweep.base.account_metrics(rows, market_days)
    if rows.empty:
        mean_price_return = 0.0
        literal_rate = 0.0
        max_cluster = 0
        clustered = 0
    else:
        mean_price_return = 100*float(rows.signed_return.mean())
        literal_rate = 100*float((rows.outcome_code == 1).mean())
        clusters = rows.groupby("entry_time").size()
        max_cluster = int(clusters.max())
        clustered = int((clusters > 1).sum())
    factor = 1 + .03*10*(mean_price_return/100)
    projected = (factor**(3*22)-1)*100 if factor > 0 else -100.0
    observed_projected = (
        (factor**(value["trades_per_market_day"]*22)-1)*100
        if factor > 0 else -100.0)
    value.update({
        "literal_tp_rate_pct": round(literal_rate, 4),
        "mean_signed_price_return_pct": round(mean_price_return, 6),
        "projected_22_market_day_pct_at_3_trades_day": round(projected, 4),
        "projected_22_market_day_pct_at_observed_frequency": round(
            observed_projected, 4),
        "required_mean_price_return_pct_at_3_trades_day":
            REQUIRED_MEAN_PRICE_RETURN_PCT,
        "economic_30pct_gate": bool(
            mean_price_return >= REQUIRED_MEAN_PRICE_RETURN_PCT
            and value["trades_per_market_day"] >= 3
            and projected >= 30),
        "max_entries_same_15m_open": max_cluster,
        "clustered_15m_opens": clustered,
    })
    return value


def main() -> None:
    frozen_raw = FROZEN_REPORT.read_bytes()
    frozen = json.loads(frozen_raw)
    candidate = frozen["frozen_before_hidden"]
    spec = candidate["spec"]
    threshold = float(frozen["hidden_2025_2026"][-1]["threshold"])

    historical, historical_sources = sweep.base.load_history()
    history_raw, history_meta = sweep.sweep_features(historical)
    history_features = macro.enrich(history_raw, historical)
    history = sweep.outcomes(
        history_features, historical, history_meta,
        float(spec["target_pct"]), float(spec["stop_pct"]),
        int(spec["hold_sessions"]))
    training = history[history.year.between(2020, 2024)]
    cfg = spec["model"]
    model = HistGradientBoostingClassifier(
        max_iter=180, learning_rate=.05, max_leaf_nodes=int(cfg["leaves"]),
        min_samples_leaf=int(cfg["leaf"]), l2_regularization=float(cfg["l2"]),
        random_state=20261022)
    model.fit(macro.matrix(training),
              (training.outcome_code == 1).astype(int))

    index, index_sources = load_folder(INDEX_DIR)
    execution, execution_sources = load_folder(EXECUTION_DIR)
    signal_raw, _ = sweep.sweep_features(index)
    signal_features = macro.enrich(signal_raw, index)
    direct = execution_outcomes(
        signal_features, execution, float(spec["target_pct"]),
        float(spec["stop_pct"]), int(spec["hold_sessions"]))
    direct["quality"] = model.predict_proba(macro.matrix(direct))[:, 1]
    accepted = selected_rows(direct, threshold)
    accepted.to_pickle(ROWS_OUT)
    market_days = signal_features.session_date.nunique()
    result = economic_summary(accepted, market_days)

    by_month = {}
    if not accepted.empty:
        months = pd.to_datetime(
            accepted.entry_time, unit="s", utc=True).dt.strftime("%Y-%m")
        for month, part in accepted.groupby(months):
            days = signal_features[signal_features.entry_time.map(
                lambda value: pd.Timestamp(value, unit="s", tz="UTC").strftime("%Y-%m")
                == month)].session_date.nunique()
            by_month[str(month)] = economic_summary(part, int(days))

    baseline = frozen["direct_xperp"]
    checks = {
        "all_index_candles_timestamp_matched": all(
            json.loads((INDEX_DIR / "manifest.json").read_text())["symbols"][symbol]
            ["matched_fraction"] == 1.0 for symbol in (*sweep.SYMBOLS, "QQQUSDT")),
        "n_ge_30": result["n"] >= 30,
        "positive_return": result["return_pct"] > 0,
        "dd_le_8": result["dd_pct"] <= 8,
        "at_least_3_trades_per_market_day": result["trades_per_market_day"] >= 3,
        "no_negative_months": result["negative_months"] == 0,
        "economic_30pct_gate": result["economic_30pct_gate"],
    }
    report = {
        "status": "SHADOW_CANDIDATE" if all(checks.values()) else "REJECTED",
        "protocol": {
            "model": "exact frozen macro-sweep 2026 model fit on 2020-2024 cash proxy",
            "threshold": threshold,
            "threshold_source": "2025 cash validation, frozen before direct evidence",
            "signal": "OKX underlying-index 15m candles",
            "execution": "timestamp-matched OKX X-Perp 15m candles",
            "entry": "next 15-minute X-Perp market open",
            "target_pct": spec["target_pct"], "stop_pct": spec["stop_pct"],
            "hold_sessions": spec["hold_sessions"],
            "money": "3% current-equity margin, fixed 10x gross",
            "orders": "research only; no order placement",
        },
        "rows": {"index_sweep_features": len(signal_features),
                 "timestamp_matched_outcomes": len(direct),
                 "accepted": len(accepted)},
        "result": result,
        "by_month": by_month,
        "comparison": {
            "xperp_signal_xperp_execution": baseline,
            "index_signal_xperp_execution": result,
            "return_delta_pct": round(result["return_pct"]-baseline["return_pct"], 4),
        },
        "checks": checks,
        "sources": {
            str(FROZEN_REPORT): hashlib.sha256(frozen_raw).hexdigest(),
            **historical_sources, **index_sources, **execution_sources,
            str(Path(__file__).resolve()): hashlib.sha256(
                Path(__file__).read_bytes()).hexdigest(),
        },
        "limitations": [
            "The direct sample begins in July 2026 and remains short.",
            "OKX index candles are an underlying-price reference, not exchange stock prints.",
            "Gross outcomes exclude fees at the user's request.",
            "Historical results cannot guarantee future profit or eliminate losing months.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "rows": report["rows"],
                      "result": result, "comparison": report["comparison"],
                      "checks": checks}, indent=2), flush=True)


if __name__ == "__main__":
    main()
