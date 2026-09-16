"""Walk-forward regime classifier for prespecified stock session families.

Each test year is predicted by a model trained on older years.  The immediately
preceding year is used only to choose a probability threshold.  This tests a
conditional family selector without allowing the tested year's outcomes into
either training or threshold selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from filter_lab import gate, metrics
from stock_session_family_lab import (
    attach_intraday_market_context, attach_market_context,
    market_context_by_day, market_intraday_context_by_time,
    replay,
)


NUMERIC = (
    "signal_minutes", "opening_range_atr", "extension_atr", "gap_dir_atr",
    "prior_day_dir_atr", "prior_close_sma20_dir_atr", "trend_5_20_dir_atr",
    "prior_range_ratio", "signal_body_dir_atr", "signal_range_atr",
    "opening_move_dir_atr", "atr_pct", "qqq_trend_5_20_atr",
    "qqq_distance_sma20_atr", "qqq_prior_day_atr", "qqq_gap_atr",
    "qqq_vol_ratio", "qqq_trend_dir_atr", "qqq_distance_dir_atr",
    "qqq_prior_day_dir_atr", "qqq_gap_dir_atr",
    "qqq_intraday_move_atr", "qqq_signal_body_atr", "qqq_signal_range_atr",
    "qqq_opening_range_atr", "qqq_opening_position_atr",
    "qqq_intraday_move_dir_atr", "qqq_signal_body_dir_atr",
    "qqq_opening_position_dir_atr",
    "spy_intraday_move_atr", "spy_signal_body_atr", "spy_signal_range_atr",
    "spy_opening_range_atr", "spy_opening_position_atr",
    "spy_intraday_move_dir_atr", "spy_signal_body_dir_atr",
    "spy_opening_position_dir_atr",
)
CATEGORICAL = ("family", "direction", "qqq_regime")


def make_model(extra_numeric: tuple[str, ...] = ()) -> Pipeline:
    transform = ColumnTransformer([
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), list(NUMERIC + extra_numeric)),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), list(CATEGORICAL)),
    ])
    classifier = HistGradientBoostingClassifier(
        learning_rate=.045, max_iter=140, max_leaf_nodes=15,
        min_samples_leaf=150, l2_regularization=3.0,
        early_stopping=True, validation_fraction=.15, random_state=20260914,
    )
    return Pipeline([("transform", transform), ("classifier", classifier)])


def rows_at_probability(frame: pd.DataFrame, probability: np.ndarray,
                        threshold: float) -> list[dict]:
    chosen = frame.loc[probability >= threshold].copy()
    chosen["model_probability"] = probability[probability >= threshold]
    return gate(chosen.sort_values(["entry_time", "symbol", "family"])
                .to_dict("records"), 4)


def choose_threshold(frame: pd.DataFrame, probability: np.ndarray,
                     minimum_threshold: float = .50) -> dict | None:
    candidates = set(np.linspace(max(.50, minimum_threshold), .90, 41))
    candidates.update(float(np.quantile(probability, q))
                      for q in (.70, .75, .80, .85, .875, .90, .925, .95, .975))
    candidates = {value for value in candidates if value >= minimum_threshold}
    tested = []
    for threshold in sorted(candidates):
        selected = rows_at_probability(frame, probability, threshold)
        score = metrics(selected)
        if score["n"] < 35:
            continue
        tested.append({"threshold": float(threshold), "metrics": score})
    viable = [item for item in tested
              if item["metrics"]["wr"] >= 82
              and item["metrics"]["net_r"] > 0
              and (item["metrics"]["pf"] or 0) >= 1.15]
    if not viable:
        return None
    return max(viable, key=lambda item: (
        item["metrics"]["net_r"] / max(item["metrics"]["dd_r"], .25),
        item["metrics"]["wr_lower_95"], item["metrics"]["net_r"],
    ))


def compact(value: dict) -> dict:
    return {key: value[key] for key in
            ("n", "wr", "wr_lower_95", "net_r", "mean_r", "pf", "dd_r")}


def load_external(index_dir: Path, execution_dir: Path,
                  symbols: tuple[str, ...]) -> tuple[pd.DataFrame, dict]:
    rows, signals, sources = [], {}, {}
    for symbol in symbols:
        signal_path = index_dir / f"{symbol}_15min_18000.pkl"
        execution_path = execution_dir / f"{symbol}_15min_18000.pkl"
        if not signal_path.exists() or not execution_path.exists():
            continue
        signal_raw, execution_raw = signal_path.read_bytes(), execution_path.read_bytes()
        sources[str(signal_path)] = hashlib.sha256(signal_raw).hexdigest()
        sources[str(execution_path)] = hashlib.sha256(execution_raw).hexdigest()
        signals[symbol] = pickle.loads(signal_raw)
        rows.extend(replay(signals[symbol], symbol, targets=(.25,),
                           execution=pickle.loads(execution_raw)))
    if "QQQUSDT" in signals:
        attach_market_context(rows, market_context_by_day(signals["QQQUSDT"]))
        attach_intraday_market_context(
            rows, market_intraday_context_by_time(signals["QQQUSDT"], "qqq"), "qqq")
    if "SPYUSDT" in signals:
        attach_intraday_market_context(
            rows, market_intraday_context_by_time(signals["SPYUSDT"], "spy"), "spy")
    return pd.DataFrame(rows), sources


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("deep_csv", type=Path)
    parser.add_argument("--external-index-dir", type=Path, required=True)
    parser.add_argument("--external-execution-dir", type=Path, required=True)
    parser.add_argument("--symbols", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--start-year", type=int, default=2020)
    parser.add_argument("--first-test-year", type=int, default=2022)
    parser.add_argument("--minimum-threshold", type=float, default=.50)
    args = parser.parse_args()

    frame = pd.read_csv(args.deep_csv)
    frame = frame[(frame["target_r"] == .25)].copy()
    frame["year"] = pd.to_datetime(frame["entry_time"], unit="s", utc=True).dt.year
    frame = frame[(frame["year"] >= args.start_year) & (frame["year"] <= 2026)]
    frame["won"] = (frame["net_r"] > 0).astype(int)
    folds, out_of_sample = [], []
    fixed_values = (.82, .83, .84, .85, .86)
    fixed_out_of_sample = {str(value): [] for value in fixed_values}
    for test_year in range(args.first_test_year, 2027):
        validation_year = test_year - 1
        train = frame[frame["year"] < validation_year]
        validation = frame[frame["year"] == validation_year]
        test = frame[frame["year"] == test_year]
        if train.empty or validation.empty or test.empty:
            continue
        model = make_model()
        model.fit(train[list(NUMERIC + CATEGORICAL)], train["won"])
        validation_probability = model.predict_proba(
            validation[list(NUMERIC + CATEGORICAL)])[:, 1]
        threshold = choose_threshold(validation, validation_probability,
                                     args.minimum_threshold)
        probability = model.predict_proba(test[list(NUMERIC + CATEGORICAL)])[:, 1]
        fixed_test = {
            str(value): rows_at_probability(test, probability, value)
            for value in fixed_values
        }
        for key, selected_rows in fixed_test.items():
            fixed_out_of_sample[key].extend(selected_rows)
        if threshold is None:
            folds.append({
                "test_year": test_year, "status": "NO_THRESHOLD",
                "fixed_threshold_test": {
                    key: compact(metrics(value)) for key, value in fixed_test.items()
                },
            })
            continue
        selected = rows_at_probability(test, probability, threshold["threshold"])
        out_of_sample.extend(selected)
        folds.append({
            "test_year": test_year,
            "train_years": [int(train["year"].min()), int(train["year"].max())],
            "validation_year": validation_year,
            "threshold": threshold["threshold"],
            "validation": compact(threshold["metrics"]),
            "test": compact(metrics(selected)),
            "fixed_threshold_test": {
                key: compact(metrics(value)) for key, value in fixed_test.items()
            },
        })

    # Frozen current model: train through 2024, calibrate on 2025, then apply to
    # both 2026 deep data and timestamp-matched recent X-Perp execution.
    final_train = frame[frame["year"] <= 2024]
    final_validation = frame[frame["year"] == 2025]
    final_model = make_model()
    final_model.fit(final_train[list(NUMERIC + CATEGORICAL)], final_train["won"])
    validation_probability = final_model.predict_proba(
        final_validation[list(NUMERIC + CATEGORICAL)])[:, 1]
    final_threshold = choose_threshold(final_validation, validation_probability,
                                       args.minimum_threshold)

    symbols = tuple(value.strip() for value in args.symbols.split(",") if value.strip())
    external, external_sources = load_external(
        args.external_index_dir, args.external_execution_dir, symbols)
    if final_threshold is not None and not external.empty:
        external_probability = final_model.predict_proba(
            external[list(NUMERIC + CATEGORICAL)])[:, 1]
        external_selected = rows_at_probability(
            external, external_probability, final_threshold["threshold"])
    else:
        external_probability = np.asarray([])
        external_selected = []

    monthly = {}
    for row in out_of_sample:
        month = datetime.fromtimestamp(row["entry_time"], timezone.utc).strftime("%Y-%m")
        monthly.setdefault(month, []).append(row)
    model_path = args.out.with_suffix(".joblib")
    if final_threshold is not None:
        joblib.dump({"model": final_model, "threshold": final_threshold["threshold"],
                     "numeric": NUMERIC, "categorical": CATEGORICAL}, model_path)
    report = {
        "status": "RESEARCH_ONLY",
        "target_r": .25, "stop_atr": 1,
        "start_year": args.start_year,
        "first_test_year": args.first_test_year,
        "minimum_threshold": args.minimum_threshold,
        "walk_forward": folds,
        "walk_forward_all": compact(metrics(out_of_sample)),
        "fixed_threshold_walk_forward": {
            key: compact(metrics(value)) for key, value in fixed_out_of_sample.items()
        },
        "walk_forward_months": {key: compact(metrics(value))
                                for key, value in sorted(monthly.items())},
        "final_threshold": final_threshold["threshold"] if final_threshold else None,
        "final_validation_2025": (compact(final_threshold["metrics"])
                                  if final_threshold else None),
        "external_xperp_2026": compact(metrics(external_selected)),
        "external_threshold_sweep": {
            str(value): compact(metrics(rows_at_probability(
                external, external_probability, value)))
            for value in (.82, .83, .84, .85, .86)
        } if len(external_probability) else {},
        "external_selected": external_selected,
        "model_path": str(model_path) if final_threshold else None,
        "sources": {
            "deep": {"path": str(args.deep_csv),
                     "sha256": hashlib.sha256(args.deep_csv.read_bytes()).hexdigest()},
            **external_sources,
        },
        "limitations": [
            "Every test year is unseen, but adjacent years can still share market structure.",
            "The probability model is a conditional filter, not a guarantee of future wins.",
            "Recent X-Perp history is short; paper-forward validation remains required.",
            "Gross R excludes fees and client-selected position sizing.",
        ],
    }
    args.out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({
        "folds": folds,
        "walk_forward_all": report["walk_forward_all"],
        "final_threshold": report["final_threshold"],
        "external_xperp_2026": report["external_xperp_2026"],
    }, indent=2))


if __name__ == "__main__":
    main()
