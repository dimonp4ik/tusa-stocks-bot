"""Causal direct-X-Perp intraday state experiment with a sealed month holdout.

Every candidate uses a closed cash-session bar and enters at the next bar open.
July trains the probability model, August selects a target/model/threshold, and
September is reported once as the untouched chronological check.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from src.stock_venue_router import atr_at, core_groups


NY = ZoneInfo("America/New_York")
TARGETS = (.25, .5, .75, 1.0)


def _load(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    return pickle.loads(raw), hashlib.sha256(raw).hexdigest()


def _aligned_features(candles: dict, bar: int, direction: int,
                      groups: dict[str, list[int]], day: str) -> dict | None:
    risk = atr_at(candles, bar + 1)
    if not risk or bar < 20:
        return None
    bars = groups[day]
    offset = bars.index(bar)
    close = float(candles["close"][bar])
    open_ = float(candles["open"][bar])
    high = float(candles["high"][bar])
    low = float(candles["low"][bar])
    prior = [float(value) for value in candles["close"]]
    volumes = [float(value) for value in candles["volume"]]
    volume_base = np.median(volumes[max(0, bar - 20):bar])
    rolling_high = max(float(value) for value in candles["high"][bar - 7:bar + 1])
    rolling_low = min(float(value) for value in candles["low"][bar - 7:bar + 1])
    local = datetime.fromtimestamp(candles["time"][bar], NY)
    return {
        "signal_minutes": local.hour * 60 + local.minute - 570,
        "session_bucket": ("00_60" if offset <= 4 else "75_150"
                           if offset <= 10 else "165_plus"),
        "ret1_dir_atr": direction * (close - prior[bar - 1]) / risk,
        "ret2_dir_atr": direction * (close - prior[bar - 2]) / risk,
        "ret4_dir_atr": direction * (close - prior[bar - 4]) / risk,
        "ret8_dir_atr": direction * (close - prior[bar - 8]) / risk,
        "trend5_20_dir_atr": direction * (
            np.mean(prior[bar - 4:bar + 1]) - np.mean(prior[bar - 19:bar + 1])) / risk,
        "day_move_dir_atr": direction * (
            close - float(candles["open"][bars[0]])) / risk,
        "body_dir_atr": direction * (close - open_) / risk,
        "range_atr": (high - low) / risk,
        "close_location_dir": direction * (
            (close - rolling_low) / max(rolling_high - rolling_low, 1e-12) - .5),
        "volume_ratio": volumes[bar] / volume_base if volume_base > 0 else 1.0,
        "atr_pct": risk / close if close > 0 else None,
    }


def _index_features(candles: dict) -> dict[int, dict]:
    groups = core_groups(candles)
    result = {}
    for day, bars in groups.items():
        for bar in bars:
            risk = atr_at(candles, bar + 1)
            if not risk or bar < 20:
                continue
            close = float(candles["close"][bar])
            closes = candles["close"]
            result[int(candles["time"][bar])] = {
                "ret1": (close - float(closes[bar - 1])) / risk,
                "ret4": (close - float(closes[bar - 4])) / risk,
                "ret8": (close - float(closes[bar - 8])) / risk,
                "trend5_20": (np.mean(closes[bar - 4:bar + 1])
                               - np.mean(closes[bar - 19:bar + 1])) / risk,
                "day_move": (close - float(candles["open"][bars[0]])) / risk,
            }
    return result


def _outcome(candles: dict, entry_bar: int, end_bar: int, direction: int,
             risk: float, target_r: float) -> tuple[float, int]:
    entry = float(candles["open"][entry_bar])
    stop = entry - direction * risk
    target = entry + direction * risk * target_r
    for bar in range(entry_bar, end_bar + 1):
        high, low = float(candles["high"][bar]), float(candles["low"][bar])
        stopped = low <= stop if direction == 1 else high >= stop
        reached = high >= target if direction == 1 else low <= target
        if stopped:  # conservative ordering when both levels occur in one bar
            return -1.0, int(candles["time"][bar]) + 900
        if reached:
            return target_r, int(candles["time"][bar]) + 900
    result = direction * (float(candles["close"][end_bar]) - entry) / risk
    return float(max(-1.0, min(target_r, result))), int(candles["time"][end_bar]) + 900


def build(cache_dir: Path, symbols: list[str]) -> tuple[pd.DataFrame, dict]:
    caches, hashes = {}, {}
    for symbol in sorted(set(symbols) | {"QQQUSDT", "SPYUSDT"}):
        path = cache_dir / f"{symbol}_15min_current.pkl"
        caches[symbol], hashes[str(path)] = _load(path)
    qqq = _index_features(caches["QQQUSDT"])
    spy = _index_features(caches["SPYUSDT"])
    rows = []
    for symbol in symbols:
        candles = caches[symbol]
        groups = core_groups(candles)
        for day, bars in groups.items():
            if len(bars) < 20:
                continue
            for offset in range(1, len(bars) - 1):
                signal_bar, entry_bar = bars[offset], bars[offset + 1]
                timestamp = int(candles["time"][signal_bar])
                if timestamp not in qqq or timestamp not in spy:
                    continue
                for direction in (-1, 1):
                    features = _aligned_features(candles, signal_bar, direction, groups, day)
                    if not features:
                        continue
                    risk = atr_at(candles, entry_bar)
                    for target_r in TARGETS:
                        net_r, exit_time = _outcome(
                            candles, entry_bar, bars[-1], direction, risk, target_r)
                        row = {
                            "symbol": symbol,
                            "direction": "LONG" if direction == 1 else "SHORT",
                            "target_r": target_r,
                            "entry_time": int(candles["time"][entry_bar]),
                            "exit_time": exit_time, "net_r": net_r,
                            **features,
                        }
                        for prefix, context in (("qqq", qqq[timestamp]),
                                                ("spy", spy[timestamp])):
                            for key, value in context.items():
                                row[f"{prefix}_{key}_dir"] = direction * value
                        rows.append(row)
    frame = pd.DataFrame(rows)
    frame["month"] = pd.to_datetime(frame["entry_time"], unit="s", utc=True).dt.strftime("%Y-%m")
    return frame, hashes


def metrics(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {"n": 0, "wr": 0.0, "net_r": 0.0, "pf": 0.0, "dd_r": 0.0}
    values = frame.sort_values(["exit_time", "entry_time"])["net_r"].to_numpy(float)
    curve = np.cumsum(values)
    drawdown = np.maximum.accumulate(np.r_[0.0, curve])[1:] - curve
    wins, losses = values[values > 0].sum(), -values[values < 0].sum()
    return {
        "n": int(len(values)), "wr": float(100 * np.mean(values > 0)),
        "net_r": float(values.sum()), "mean_r": float(values.mean()),
        "pf": float(wins / losses) if losses else math.inf,
        "dd_r": float(drawdown.max(initial=0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--symbols", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    symbols = [item.strip() for item in args.symbols.split(",") if item.strip()]
    frame, hashes = build(args.cache_dir, symbols)
    exclude = {"net_r", "exit_time", "entry_time", "month", "target_r"}
    features = [column for column in frame if column not in exclude]
    categorical = [column for column in features if not is_numeric_dtype(frame[column])]
    numeric = [column for column in features if column not in categorical]
    preprocess = ColumnTransformer([
        ("numeric", make_pipeline(SimpleImputer(strategy="median")), numeric),
        ("categorical", make_pipeline(SimpleImputer(strategy="most_frequent"),
         OneHotEncoder(handle_unknown="ignore", sparse_output=False)), categorical),
    ])
    models = {
        "extra_leaf10": ExtraTreesClassifier(
            n_estimators=100, min_samples_leaf=10, max_features=.7,
            class_weight="balanced", random_state=17, n_jobs=-1),
        "extra_leaf25": ExtraTreesClassifier(
            n_estimators=100, min_samples_leaf=25, max_features=.7,
            class_weight="balanced", random_state=17, n_jobs=-1),
        "rf_leaf15": RandomForestClassifier(
            n_estimators=100, min_samples_leaf=15, max_features=.7,
            class_weight="balanced", random_state=17, n_jobs=-1),
    }
    trials = []
    for target_r in TARGETS:
        target = frame[frame["target_r"] == target_r].copy()
        train = target[target["month"] == "2026-07"]
        validation = target[target["month"] == "2026-08"]
        holdout = target[target["month"] == "2026-09"]
        for name, estimator in models.items():
            model = make_pipeline(preprocess, estimator)
            model.fit(train[features], train["net_r"].gt(0))
            candidates = {}
            for split, source in (("validation", validation), ("holdout", holdout)):
                scored = source.copy()
                scored["probability"] = model.predict_proba(source[features])[:, 1]
                # Only one side may be entered for a symbol at one timestamp.
                scored = scored.sort_values("probability", ascending=False).drop_duplicates(
                    ["symbol", "entry_time"], keep="first")
                candidates[split] = scored
            valid = candidates["validation"]
            # August has 21 US trading days. 63 trades is the requested average
            # of three per trading day; choose the highest validation WR among
            # thresholds that preserve at least that count.
            ranked = valid.sort_values("probability", ascending=False).copy()
            grouped = ranked.groupby("probability", sort=False).agg(
                n=("net_r", "size"), wins=("net_r", lambda values: int((values > 0).sum())),
                net_r=("net_r", "sum"),
            ).reset_index()
            grouped[["n", "wins", "net_r"]] = grouped[["n", "wins", "net_r"]].cumsum()
            choices = []
            for item in grouped.itertuples(index=False):
                if item.n < 63 or item.net_r <= 0:
                    continue
                wr = 100 * item.wins / item.n
                choices.append((wr, item.net_r, item.probability))
            if not choices:
                continue
            _, _, threshold = max(choices)
            validation_stat = metrics(valid[valid["probability"] >= threshold])
            held = candidates["holdout"]
            held = held[held["probability"] >= threshold]
            trials.append({
                "target_r": target_r, "model": name,
                "threshold": float(threshold),
                "validation": validation_stat, "holdout": metrics(held),
                "holdout_rows": held,
            })
    chosen = max(trials, key=lambda item: (
        item["validation"]["wr"], item["validation"]["net_r"]))
    report_trials = [{key: value for key, value in item.items() if key != "holdout_rows"}
                     for item in trials]
    selected_rows = chosen["holdout_rows"].sort_values("entry_time")
    report = {
        "status": "REJECT" if (chosen["holdout"]["wr"] < 75
                                  or chosen["holdout"]["net_r"] <= 0) else "PAPER_CANDIDATE",
        "selection_rule": "max August WR, then August net R, with at least 63 trades",
        "chosen": {key: value for key, value in chosen.items() if key != "holdout_rows"},
        "trials": report_trials, "rows": len(frame), "sources": hashes,
        "limitations": [
            "Direct stock X-Perp history begins in June 2026.",
            "July trains, August selects, September is the chronological holdout.",
            "Gross R excludes costs and client position sizing by user request.",
        ],
    }
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    frame.to_csv(args.out.with_name(args.out.stem + "_dataset.csv"), index=False)
    selected_rows.to_csv(args.out.with_name(args.out.stem + "_holdout.csv"), index=False)
    print(json.dumps({"status": report["status"], "chosen": report["chosen"]}, indent=2))


if __name__ == "__main__":
    main()
