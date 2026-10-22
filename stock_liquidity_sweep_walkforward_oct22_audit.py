"""Causal long-history audit of stock liquidity-sweep reversals."""
from __future__ import annotations

import hashlib
import json
import math
import pickle
from pathlib import Path
from statistics import median

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import stock_15m_multisession_regime_oct6_audit as base
from stock_module_cluster_audit import portfolio_gate


ROOT = Path("reports/audit_2026_09_08")
OUT = ROOT / "stock_liquidity_sweep_walkforward_oct22.json"
ROWS_OUT = ROOT / "stock_liquidity_sweep_walkforward_oct22.csv"
SYMBOLS = base.SYMBOLS
TARGETS = (.5, 1.0, 2.0)
STOPS = (.5, 1.0)
HOLDS = (1, 2)
SWING_LOOKBACK = 3
MAX_LEVEL_AGE = 80
FEATURES = (
    "sweep_depth_atr", "rejection_atr", "body_dir_atr",
    "trend_dir_atr", "ret20_dir_atr", "qqq_trend_dir_atr",
    "qqq_ret20_dir_atr", "volume_ratio", "level_age",
    "signal_minutes",
)
MODELS = (
    {"leaves": 7, "leaf": 80, "l2": 2.0},
    {"leaves": 15, "leaf": 120, "l2": 3.0},
    {"leaves": 15, "leaf": 240, "l2": 5.0},
)
QUANTILES = (.30, .40, .50, .60, .70, .80, .85, .90)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resample_15m(candle: dict) -> dict:
    frame = pd.DataFrame({key: candle[key] for key in
                          ("time", "open", "high", "low", "close", "volume")})
    frame.index = pd.to_datetime(frame.pop("time"), unit="s", utc=True)
    value = frame.resample("15min", label="left", closed="left").agg({
        "open": "first", "high": "max", "low": "min", "close": "last",
        "volume": "sum",
    }).dropna()
    timestamps = [int(stamp.timestamp()) for stamp in value.index]
    return {"time": timestamps,
            **{column: value[column].astype(float).tolist() for column in value.columns}}


def load_direct_15m() -> tuple[dict[str, dict], dict[str, str]]:
    candles, sources = {}, {}
    for symbol in (*SYMBOLS, "QQQUSDT"):
        path = base.DIRECT_CACHE / f"{symbol}_5min_current.pkl"
        raw = path.read_bytes()
        candles[symbol] = resample_15m(pickle.loads(raw))
        sources[str(path)] = hashlib.sha256(raw).hexdigest()
    return candles, sources


def regular_meta(candle: dict) -> tuple[np.ndarray, np.ndarray, int]:
    groups = base.regular_groups(candle)
    valid = [(day, bars) for day, bars in sorted(groups.items()) if len(bars) >= 20]
    regular = np.asarray([index for _, bars in valid for index in bars], dtype=np.int64)
    dates = np.asarray([day for day, bars in valid for _ in bars], dtype=object)
    bars_per_session = int(round(float(np.median([len(bars) for _, bars in valid]))))
    return regular, dates, bars_per_session


def rolling_context(candle: dict, regular: np.ndarray) -> dict[str, np.ndarray]:
    high = np.asarray(candle["high"], dtype=float)[regular]
    low = np.asarray(candle["low"], dtype=float)[regular]
    close = np.asarray(candle["close"], dtype=float)[regular]
    open_ = np.asarray(candle["open"], dtype=float)[regular]
    volume = np.asarray(candle["volume"], dtype=float)[regular]
    previous = np.r_[close[0], close[:-1]]
    tr = np.maximum(high-low, np.maximum(np.abs(high-previous), np.abs(low-previous)))
    atr = pd.Series(tr).rolling(14, min_periods=14).mean().to_numpy()
    ma20 = pd.Series(close).rolling(20, min_periods=20).mean().to_numpy()
    ma96 = pd.Series(close).rolling(96, min_periods=96).mean().to_numpy()
    volume_median = pd.Series(volume).shift(1).rolling(20, min_periods=20).median().to_numpy()
    return {"high": high, "low": low, "close": close, "open": open_,
            "volume": volume, "atr": atr, "ma20": ma20, "ma96": ma96,
            "volume_median": volume_median}


def qqq_map(candle: dict) -> dict[int, dict]:
    regular, _, _ = regular_meta(candle)
    values = rolling_context(candle, regular)
    output = {}
    for pos in range(96, len(regular)):
        unit = values["atr"][pos]
        if not math.isfinite(unit) or unit <= 0:
            continue
        timestamp = int(candle["time"][int(regular[pos])])
        output[timestamp] = {
            "trend": (values["ma20"][pos]-values["ma96"][pos])/unit,
            "ret20": (values["close"][pos]-values["close"][pos-20])/unit,
        }
    return output


def sweep_features(candles: dict[str, dict]) -> tuple[pd.DataFrame, dict]:
    market = qqq_map(candles["QQQUSDT"])
    rows, metadata = [], {}
    for symbol in SYMBOLS:
        candle = candles[symbol]
        regular, dates, bars_per_session = regular_meta(candle)
        values = rolling_context(candle, regular)
        metadata[symbol] = {"regular": regular, "bars_per_session": bars_per_session}
        swing_highs: list[tuple[int, float]] = []
        swing_lows: list[tuple[int, float]] = []
        for pos in range(96, len(regular)-1):
            pivot = pos-SWING_LOOKBACK
            if pivot >= SWING_LOOKBACK:
                left, right = pivot-SWING_LOOKBACK, pivot+SWING_LOOKBACK+1
                if values["high"][pivot] >= np.max(values["high"][left:right]):
                    swing_highs.append((pivot, float(values["high"][pivot])))
                if values["low"][pivot] <= np.min(values["low"][left:right]):
                    swing_lows.append((pivot, float(values["low"][pivot])))
            if dates[pos+1] != dates[pos]:
                continue
            timestamp = int(candle["time"][int(regular[pos])])
            qqq = market.get(timestamp)
            unit = float(values["atr"][pos])
            if qqq is None or not math.isfinite(unit) or unit <= 0:
                continue
            local = pd.Timestamp(timestamp, unit="s", tz="UTC").tz_convert("America/New_York")
            signal_minutes = local.hour*60+local.minute-570
            candidates = []
            if swing_lows and pos-swing_lows[-1][0] <= MAX_LEVEL_AGE:
                level_pos, level = swing_lows[-1]
                if values["low"][pos] < level and values["close"][pos] > level:
                    candidates.append((1, level_pos, level-values["low"][pos],
                                       values["close"][pos]-values["low"][pos]))
            if swing_highs and pos-swing_highs[-1][0] <= MAX_LEVEL_AGE:
                level_pos, level = swing_highs[-1]
                if values["high"][pos] > level and values["close"][pos] < level:
                    candidates.append((-1, level_pos, values["high"][pos]-level,
                                       values["high"][pos]-values["close"][pos]))
            for direction, level_pos, depth, rejection in candidates:
                volume_base = float(values["volume_median"][pos])
                rows.append({
                    "symbol": symbol, "family": "liquidity_sweep",
                    "direction_n": direction,
                    "direction": "LONG" if direction > 0 else "SHORT",
                    "entry_index": int(regular[pos+1]),
                    "entry_time": int(candle["time"][int(regular[pos+1])]),
                    "year": int(dates[pos][:4]), "session_date": dates[pos],
                    "signal_minutes": signal_minutes,
                    "sweep_depth_atr": depth/unit,
                    "rejection_atr": rejection/unit,
                    "body_dir_atr": direction*(values["close"][pos]-values["open"][pos])/unit,
                    "trend_dir_atr": direction*(values["ma20"][pos]-values["ma96"][pos])/unit,
                    "ret20_dir_atr": direction*(values["close"][pos]-values["close"][pos-20])/unit,
                    "qqq_trend_dir_atr": direction*qqq["trend"],
                    "qqq_ret20_dir_atr": direction*qqq["ret20"],
                    "volume_ratio": values["volume"][pos]/volume_base if volume_base > 0 else 1.0,
                    "level_age": pos-level_pos,
                    "module": "liquidity_sweep",
                })
    return pd.DataFrame(rows), metadata


def outcomes(features: pd.DataFrame, candles: dict[str, dict], metadata: dict,
             target: float, stop: float, hold: int) -> pd.DataFrame:
    parts = []
    for symbol, part in features.groupby("symbol", sort=False):
        candle = candles[symbol]
        regular = metadata[symbol]["regular"]
        lookup = np.full(len(candle["time"]), -1, dtype=np.int64)
        lookup[regular] = np.arange(len(regular), dtype=np.int64)
        valid = part[part.entry_index.map(lambda index: lookup[int(index)] >= 0)].copy()
        arrays = [np.asarray(candle[key], dtype=float) for key in ("open", "high", "low", "close")]
        values, exits, codes = base.resolve(
            *arrays, np.asarray(candle["time"], dtype=np.int64), regular, lookup,
            valid.entry_index.to_numpy(np.int64), valid.direction_n.to_numpy(np.int8),
            target, stop, hold*metadata[symbol]["bars_per_session"])
        valid["signed_return"] = values
        valid["exit_time"] = exits
        valid["outcome_code"] = codes
        valid["outcome"] = np.where(codes == 1, "TP", np.where(codes == -1, "SL", "TIMEOUT"))
        valid["target_pct"] = target
        valid["stop_pct"] = stop
        valid["hold_sessions"] = hold
        valid["net_r"] = values/(stop/100)
        valid["risk_pct"] = stop/100
        parts.append(valid)
    return pd.concat(parts, ignore_index=True) if parts else features.iloc[:0].copy()


def matrix(frame: pd.DataFrame) -> np.ndarray:
    numeric = frame[list(FEATURES)].replace([np.inf, -np.inf], np.nan).fillna(0)
    categories = pd.concat([
        pd.get_dummies(frame.symbol, prefix="symbol"),
        pd.get_dummies(pd.cut(frame.signal_minutes, (-1, 60, 150, 240, 390)), prefix="slot"),
    ], axis=1)
    expected_symbols = [f"symbol_{symbol}" for symbol in SYMBOLS]
    expected_slots = [f"slot_{label}" for label in pd.IntervalIndex.from_breaks((-1, 60, 150, 240, 390))]
    categories = categories.reindex(columns=[*expected_symbols, *expected_slots], fill_value=False)
    return np.column_stack([numeric.to_numpy(np.float32), categories.to_numpy(np.float32)])


def selected(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    chosen = frame[frame.quality >= threshold].copy()
    chosen["score"] = chosen.quality
    rows = portfolio_gate(chosen.to_dict("records"), None,
                          rank_key=lambda row: (-float(row["quality"]),))
    return pd.DataFrame(rows) if rows else chosen.iloc[:0].copy()


def evaluate(frame: pd.DataFrame, features: pd.DataFrame,
             years: tuple[int, ...], threshold: float) -> dict:
    chosen = selected(frame[frame.year.isin(years)], threshold)
    market_days = features[features.year.isin(years)].session_date.nunique()
    value = base.account_metrics(chosen, market_days)
    value["literal_target_hits"] = int((chosen.outcome_code == 1).sum()) if len(chosen) else 0
    value["literal_target_hit_rate"] = round(
        100*value["literal_target_hits"]/len(chosen), 4) if len(chosen) else 0.0
    return value


def main() -> None:
    historical, history_sources = base.load_history()
    direct, direct_sources = load_direct_15m()
    history_features, history_meta = sweep_features(historical)
    direct_features, direct_meta = sweep_features(direct)
    frontier, qualified = [], []
    for target in TARGETS:
        for stop in STOPS:
            for hold in HOLDS:
                history = outcomes(history_features, historical, history_meta, target, stop, hold)
                fit = history[history.year.between(2017, 2021)]
                selection_rows = history[history.year.between(2022, 2023)]
                for config in MODELS:
                    model = HistGradientBoostingClassifier(
                        max_iter=160, learning_rate=.06, max_leaf_nodes=config["leaves"],
                        min_samples_leaf=config["leaf"], l2_regularization=config["l2"],
                        random_state=20261022)
                    model.fit(matrix(fit), (fit.outcome_code == 1).astype(int))
                    scored = history.copy()
                    scored["quality"] = model.predict_proba(matrix(scored))[:, 1]
                    selection_quality = model.predict_proba(matrix(selection_rows))[:, 1]
                    for quantile in QUANTILES:
                        threshold = float(np.quantile(selection_quality, quantile))
                        selection = evaluate(scored, history_features, (2022, 2023), threshold)
                        item = {"target_pct": target, "stop_pct": stop,
                                "hold_sessions": hold, "model": config,
                                "quantile": quantile, "threshold": threshold,
                                "selection_2022_2023": selection}
                        frontier.append(item)
                        if not (selection["n"] >= 250 and selection["return_pct"] > 0
                                and selection["dd_pct"] <= 10
                                and selection["trades_per_market_day"] >= 1.0):
                            continue
                        validation = evaluate(scored, history_features, (2024,), threshold)
                        item["validation_2024"] = validation
                        if not (validation["n"] >= 80 and validation["return_pct"] > 0
                                and validation["dd_pct"] <= 8
                                and validation["trades_per_market_day"] >= 1.0):
                            continue
                        item["score"] = (selection["return_pct"] + 2*validation["return_pct"]
                                         - selection["dd_pct"] - 2*validation["dd_pct"])
                        item["_model"] = model
                        item["_history"] = scored
                        qualified.append(item)
        print(json.dumps({"phase": "target", "target": target,
                          "qualified": len(qualified)}), flush=True)
    qualified.sort(key=lambda item: item["score"], reverse=True)
    frozen = qualified[0] if qualified else None
    results = checks = None
    if frozen:
        model = frozen["_model"]
        direct_exit = outcomes(direct_features, direct, direct_meta,
                               frozen["target_pct"], frozen["stop_pct"],
                               frozen["hold_sessions"])
        direct_exit["quality"] = model.predict_proba(matrix(direct_exit))[:, 1]
        threshold = frozen["threshold"]
        results = {
            "selection_2022_2023": frozen["selection_2022_2023"],
            "validation_2024": frozen["validation_2024"],
            "test_2025": evaluate(frozen["_history"], history_features, (2025,), threshold),
            "observed_2026": evaluate(frozen["_history"], history_features, (2026,), threshold),
            "direct_xperp": evaluate(direct_exit, direct_features, (2026,), threshold),
        }
        chosen = selected(frozen["_history"], threshold)
        chosen.to_csv(ROWS_OUT, index=False)
        checks = {name: bool(value["return_pct"] > 0 and value["dd_pct"] <= 8)
                  for name, value in results.items()}

    def public(item: dict) -> dict:
        return {key: value for key, value in item.items()
                if key not in {"_model", "_history"}}

    frontier.sort(key=lambda item: (
        item["selection_2022_2023"]["return_pct"]
        - item["selection_2022_2023"]["dd_pct"],
        item["selection_2022_2023"]["literal_target_hit_rate"]), reverse=True)
    objective = bool(checks and all(checks.values())
                     and results["test_2025"]["wr"] >= 75
                     and results["observed_2026"]["wr"] >= 75
                     and results["direct_xperp"]["n"] >= 30
                     and results["direct_xperp"]["return_pct"] > 0)
    report = {
        "status": "SHADOW_CANDIDATE" if objective else "REJECTED",
        "protocol": {
            "setup": "wick beyond last causally confirmed 15m swing and close back inside",
            "swing_lookback": SWING_LOOKBACK, "max_level_age_bars": MAX_LEVEL_AGE,
            "entry": "next 15-minute market open", "orders": "market only",
            "fit": "2017-2021 cash proxy", "selection": "2022-2023 cash proxy",
            "validation": 2024, "test": 2025, "observed": 2026,
            "money": "3% current-equity margin, fixed 10x gross",
        },
        "rows": {"history_features": len(history_features),
                 "direct_features": len(direct_features)},
        "tested": len(frontier), "qualified_before_test": len(qualified),
        "frozen_before_test": public(frozen) if frozen else None,
        "top_selection": [public(item) for item in frontier[:20]],
        "results": results, "checks": checks,
        "sources": {**history_sources, **direct_sources,
                    str(Path(__file__).resolve()): digest(Path(__file__).resolve())},
        "limitations": [
            "Cash candles are structural transfer evidence for stock perpetuals.",
            "Dukascopy volume is tick volume; only within-symbol ratios are used.",
            "The bounded model and threshold grid creates selection risk.",
            "Historical results cannot guarantee future profit or eliminate losing months.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "rows": report["rows"],
                      "qualified": len(qualified),
                      "frozen": public(frozen) if frozen else None,
                      "results": results, "checks": checks}, indent=2), flush=True)


if __name__ == "__main__":
    main()
