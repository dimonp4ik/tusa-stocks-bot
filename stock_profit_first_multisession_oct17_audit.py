"""Profit-first two-percent stock runner held for one to three cash sessions.

The target and entry family come from the preceding literal-runner audit.  This
stage compares only the maximum hold using 2022, then freezes it for 2023-2026.
Targets and stops are checked on regular-session 15-minute bars with pessimistic
stop-first handling and opening-gap fills.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from filter_lab import gate
from stock_5m_family_oct3_audit import regular_groups
import stock_15m_multisession_regime_oct6_audit as multi
import stock_cash_literal_move_hgb_oct7_audit as base
import stock_profit_first_literal_runner_oct17_audit as parent
from stock_walk_forward_filter_lab import CATEGORICAL, NUMERIC, make_model


ROOT = Path("reports/audit_2026_09_08")
OUT = ROOT / "stock_profit_first_multisession_oct17.json"
ROWS_OUT = ROOT / "stock_profit_first_multisession_oct17.pkl"
TARGET_PCT = 2.0
STOP_PCT = 1.0
HOLD_SESSIONS = (1, 2, 3)
QUANTILES = (.70, .75, .80, .85, .90, .925, .95)
FIT_YEARS = parent.FIT_YEARS
SELECTION_YEAR = 2022
VALIDATION_YEAR = 2023
TEST_YEAR = 2024
CONFIRMATION_YEARS = (2025, 2026)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def regular_metadata(candles: dict) -> dict:
    output = {}
    for symbol, candle in candles.items():
        groups = regular_groups(candle)
        lengths = [len(indices) for indices in groups.values() if indices]
        expected = int(round(float(np.median(lengths))))
        complete = [indices for _, indices in sorted(groups.items())
                    if len(indices) >= int(.9 * expected)]
        regular = np.asarray([index for indices in complete for index in indices],
                             dtype=np.int64)
        output[symbol] = {"bars_per_session": expected, "regular": regular}
    return output


def add_outcomes(frame: pd.DataFrame, candles: dict, metadata: dict,
                 sessions: int) -> pd.DataFrame:
    pieces = []
    for symbol, part in frame.groupby("symbol", sort=False):
        candle = candles[symbol]
        times = np.asarray(candle["time"], dtype=np.int64)
        regular = metadata[symbol]["regular"]
        lookup = np.full(len(times), -1, dtype=np.int64)
        lookup[regular] = np.arange(len(regular), dtype=np.int64)
        valid = part[part.entry_bar.map(
            lambda value: 0 <= int(value) < len(lookup)
            and lookup[int(value)] >= 0)].copy()
        directions = valid.direction.map({"LONG": 1, "SHORT": -1}).to_numpy(np.int8)
        arrays = [np.asarray(candle[key], dtype=float)
                  for key in ("open", "high", "low", "close")]
        values, exits, outcomes = multi.resolve(
            *arrays, times, regular, lookup,
            valid.entry_bar.to_numpy(np.int64), directions,
            TARGET_PCT, STOP_PCT,
            sessions * int(metadata[symbol]["bars_per_session"]))
        valid["signed_return"] = values
        valid["exit_time"] = exits
        valid["target_hit"] = outcomes == 1
        valid["exit_reason"] = np.where(outcomes == 1, 1,
                                         np.where(outcomes == -1, 0, 2))
        valid["won"] = valid.target_hit.astype(int)
        valid["net_r"] = values / (STOP_PCT / 100)
        valid["risk_pct"] = STOP_PCT / 100
        valid["hold_sessions"] = sessions
        pieces.append(valid)
    return pd.concat(pieces, ignore_index=True)


def evaluate(frame: pd.DataFrame, probability: np.ndarray,
             years: tuple[int, ...], threshold: float) -> dict:
    return parent.enrich(base.score(frame, probability, years, threshold))


def selection_ok(value: dict) -> bool:
    return bool(value["return_pct"] > 0 and value["dd_pct"] <= 15
                and value["trades_per_calendar_day"] >= 3)


def objective_ok(value: dict) -> bool:
    return parent.objective_ok(value)


def selected_rows(frame: pd.DataFrame, probability: np.ndarray,
                  threshold: float) -> pd.DataFrame:
    chosen = frame[probability >= threshold].copy()
    chosen["quality"] = probability[probability >= threshold]
    return pd.DataFrame(gate(chosen.sort_values(
        ["entry_time", "quality", "symbol"], ascending=[True, False, True])
        .to_dict("records"), 4))


def public(item: dict) -> dict:
    return {key: value for key, value in item.items()
            if key not in {"_frame", "_probability"}}


def main() -> None:
    candles, sources = base.load()
    original = parent.build_target(candles, TARGET_PCT)
    metadata = regular_metadata(candles)
    features = list(NUMERIC + CATEGORICAL)
    candidates = []
    for sessions in HOLD_SESSIONS:
        frame = add_outcomes(original, candles, metadata, sessions)
        fit = frame.year.isin(FIT_YEARS)
        estimator = make_model()
        estimator.fit(frame.loc[fit, features], frame.loc[fit, "won"])
        probability = estimator.predict_proba(frame[features])[:, 1]
        selection_scores = probability[frame.year.eq(SELECTION_YEAR).to_numpy()]
        for quantile in QUANTILES:
            threshold = float(np.quantile(selection_scores, quantile))
            value = evaluate(frame, probability, (SELECTION_YEAR,), threshold)
            candidates.append({
                "hold_sessions": sessions,
                "selection_quantile": quantile,
                "threshold": threshold,
                "selection_2022": value,
                "selection_pass": selection_ok(value),
                "score": parent.profile_score(value),
                "_frame": frame, "_probability": probability,
            })
    candidates.sort(key=lambda item: (
        item["selection_pass"], item["score"]), reverse=True)
    frozen = candidates[0] if candidates and candidates[0]["selection_pass"] else None
    validation = test = confirmation = checks = None
    status = "REJECTED_SELECTION"
    if frozen:
        frame, probability = frozen["_frame"], frozen["_probability"]
        threshold = frozen["threshold"]
        validation = evaluate(frame, probability, (VALIDATION_YEAR,), threshold)
        test = evaluate(frame, probability, (TEST_YEAR,), threshold)
        confirmation = {str(year): evaluate(frame, probability, (year,), threshold)
                        for year in CONFIRMATION_YEARS}
        yearly = {str(SELECTION_YEAR): frozen["selection_2022"],
                  str(VALIDATION_YEAR): validation, str(TEST_YEAR): test,
                  **confirmation}
        checks = {year: objective_ok(value) for year, value in yearly.items()}
        status = ("STOCK_MULTISESSION_RESEARCH_CANDIDATE"
                  if all(checks.values()) else "REJECTED_OBJECTIVE")
        selected_rows(frame, probability, threshold).to_pickle(ROWS_OUT)
    report = {
        "status": status, "ready_for_live_orders": False,
        "protocol": {
            "priority": "account return first; win rate diagnostic only",
            "fit": list(FIT_YEARS), "hold_selection": SELECTION_YEAR,
            "frozen_validation": VALIDATION_YEAR, "untouched_test": TEST_YEAR,
            "confirmation_only": list(CONFIRMATION_YEARS),
            "target_pct": TARGET_PCT, "stop_pct": STOP_PCT,
            "hold_sessions": list(HOLD_SESSIONS),
            "entry": "next 15-minute market open", "orders": "market only",
            "money": "3% current-equity margin, fixed 10x gross",
            "objective": ">=30% account return in every month; >=3 trades/calendar day; DD <=10%",
        },
        "candidate_count": len(candidates),
        "top_selection": [public(item) for item in candidates[:15]],
        "frozen_before_2023": public(frozen) if frozen else None,
        "validation_2023": validation, "test_2024": test,
        "confirmation_2025_2026": confirmation,
        "objective_checks": checks,
        "sources": {**sources, str(Path(base.__file__)): digest(Path(base.__file__)),
                    str(Path(multi.__file__)): digest(Path(multi.__file__)),
                    str(Path(__file__)): digest(Path(__file__))},
        "limitations": [
            "Cash-session proxy omits overnight price discovery between regular bars.",
            "Long holds reduce independent opportunity count through exposure controls.",
            "Any historical survivor still needs direct X-Perp shadow transfer.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": status, "candidate_count": len(candidates),
        "frozen": public(frozen) if frozen else None,
        "validation_2023": validation, "test_2024": test,
        "confirmation_2025_2026": confirmation,
        "objective_checks": checks,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
