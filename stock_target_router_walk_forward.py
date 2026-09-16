"""Causal walk-forward selector that routes each stock setup to one target.

The model estimates target-hit probability for all prespecified target sizes.
Only the highest expected-R target for a setup can be traded.  The score cutoff
is selected on the immediately preceding year, never on the tested year.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from filter_lab import gate, metrics
from stock_session_family_lab import (
    TARGETS, attach_intraday_market_context, attach_market_context,
    market_context_by_day, market_intraday_context_by_time, replay,
)
from stock_walk_forward_filter_lab import CATEGORICAL, NUMERIC, make_model


FEATURES = list(NUMERIC + ("target_r",) + CATEGORICAL)
SIGNAL_KEY = ["entry_time", "symbol", "family", "direction"]


def score_targets(frame: pd.DataFrame, probability: np.ndarray) -> pd.DataFrame:
    scored = frame.copy()
    scored["model_probability"] = probability
    # A full loss is -1R and a target hit pays target_r.  EOD partial exits are
    # retained in realized metrics, while this causal score is only a ranking.
    scored["expected_r_score"] = (
        probability * (scored["target_r"].to_numpy(dtype=float) + 1.0) - 1.0)
    scored = scored.sort_values(
        SIGNAL_KEY + ["expected_r_score", "target_r"],
        ascending=[True, True, True, True, False, False],
    )
    return scored.drop_duplicates(SIGNAL_KEY, keep="first")


def selected_rows(scored: pd.DataFrame, cutoff: float) -> list[dict]:
    chosen = scored.loc[scored["expected_r_score"] >= cutoff].copy()
    chosen = chosen.sort_values(
        ["entry_time", "expected_r_score", "symbol", "family"],
        ascending=[True, False, True, True],
    )
    return gate(chosen.to_dict("records"), 4)


def compact(value: dict) -> dict:
    return {key: value[key] for key in
            ("n", "wr", "wr_lower_95", "net_r", "mean_r", "pf", "dd_r")}


def choose_cutoff(scored: pd.DataFrame, minimum_trades: int) -> dict | None:
    values = scored["expected_r_score"].to_numpy(dtype=float)
    candidates = set(np.linspace(-.02, .20, 45))
    candidates.update(float(np.quantile(values, q))
                      for q in (.50, .60, .70, .75, .80, .85, .90, .925, .95, .975))
    tested = []
    for cutoff in sorted(candidates):
        rows = selected_rows(scored, cutoff)
        score = metrics(rows)
        if score["n"] < minimum_trades:
            continue
        tested.append({"cutoff": float(cutoff), "metrics": score})
    viable = [item for item in tested
              if item["metrics"]["net_r"] > 0
              and (item["metrics"]["pf"] or 0) >= 1.15
              and item["metrics"]["wr"] >= 58]
    if not viable:
        return None
    return max(viable, key=lambda item: (
        item["metrics"]["net_r"] / max(item["metrics"]["dd_r"], .25),
        item["metrics"]["mean_r"], item["metrics"]["net_r"],
    ))


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
        rows.extend(replay(signals[symbol], symbol, targets=TARGETS,
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
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--external-index-dir", type=Path, required=True)
    parser.add_argument("--external-execution-dir", type=Path, required=True)
    parser.add_argument("--symbols", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lookback-years", type=int, default=0,
                        help="0 uses all prior years; otherwise use this rolling window")
    parser.add_argument("--minimum-validation-trades", type=int, default=100)
    args = parser.parse_args()

    frame = pd.read_csv(args.dataset)
    frame["year"] = pd.to_datetime(frame["entry_time"], unit="s", utc=True).dt.year
    frame["won"] = (frame["net_r"] > 0).astype(int)
    folds, all_rows = [], []
    for test_year in range(2019, 2027):
        validation_year = test_year - 1
        train = frame[frame["year"] < validation_year]
        if args.lookback_years:
            train = train[train["year"] >= validation_year - args.lookback_years]
        validation = frame[frame["year"] == validation_year]
        test = frame[frame["year"] == test_year]
        if train.empty or validation.empty or test.empty:
            continue
        model = make_model(("target_r",))
        model.fit(train[FEATURES], train["won"])
        validation_scored = score_targets(
            validation, model.predict_proba(validation[FEATURES])[:, 1])
        cutoff = choose_cutoff(validation_scored, args.minimum_validation_trades)
        if cutoff is None:
            folds.append({"test_year": test_year, "status": "NO_CUTOFF"})
            continue
        test_scored = score_targets(test, model.predict_proba(test[FEATURES])[:, 1])
        rows = selected_rows(test_scored, cutoff["cutoff"])
        all_rows.extend(rows)
        folds.append({
            "test_year": test_year,
            "train_years": [int(train["year"].min()), int(train["year"].max())],
            "validation_year": validation_year,
            "cutoff": cutoff["cutoff"],
            "validation": compact(cutoff["metrics"]),
            "test": compact(metrics(rows)),
            "test_targets": {str(target): compact(metrics(
                [row for row in rows if float(row["target_r"]) == target]))
                for target in TARGETS},
        })

    final_train = frame[frame["year"] <= 2024]
    if args.lookback_years:
        final_train = final_train[final_train["year"] >= 2025 - args.lookback_years]
    final_validation = frame[frame["year"] == 2025]
    final_model = make_model(("target_r",))
    final_model.fit(final_train[FEATURES], final_train["won"])
    final_validation_scored = score_targets(
        final_validation, final_model.predict_proba(final_validation[FEATURES])[:, 1])
    final_cutoff = choose_cutoff(
        final_validation_scored, args.minimum_validation_trades)
    symbols = tuple(value.strip() for value in args.symbols.split(",") if value.strip())
    external, external_sources = load_external(
        args.external_index_dir, args.external_execution_dir, symbols)
    external_rows = []
    if final_cutoff and not external.empty:
        external_scored = score_targets(
            external, final_model.predict_proba(external[FEATURES])[:, 1])
        external_rows = selected_rows(external_scored, final_cutoff["cutoff"])

    monthly = {}
    for row in all_rows:
        month = datetime.fromtimestamp(row["entry_time"], timezone.utc).strftime("%Y-%m")
        monthly.setdefault(month, []).append(row)
    report = {
        "status": "RESEARCH_ONLY",
        "method": "one causal target per setup, expected-R rank, prior-year cutoff",
        "lookback_years": args.lookback_years,
        "minimum_validation_trades": args.minimum_validation_trades,
        "folds": folds,
        "walk_forward_all": compact(metrics(all_rows)),
        "walk_forward_months": {key: compact(metrics(rows))
                                for key, rows in sorted(monthly.items())},
        "final_cutoff": final_cutoff,
        "external_xperp_2026": compact(metrics(external_rows)),
        "external_selected": external_rows,
        "sources": {
            "dataset": {"path": str(args.dataset),
                        "sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest()},
            **external_sources,
        },
        "limitations": [
            "Repeated research on the same history creates selection risk.",
            "X-Perp transfer history is short and is not live proof.",
            "Gross R excludes fees and client-selected position sizing.",
        ],
    }
    args.out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({
        "folds": folds,
        "walk_forward_all": report["walk_forward_all"],
        "final_cutoff": final_cutoff,
        "external_xperp_2026": report["external_xperp_2026"],
    }, indent=2))


if __name__ == "__main__":
    main()
