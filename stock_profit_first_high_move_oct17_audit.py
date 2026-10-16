"""Profit-first stock audit for literal 2.5-4% multi-session moves."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from filter_lab import gate
import stock_15m_multisession_regime_oct6_audit as multi
import stock_cash_literal_move_hgb_oct7_audit as base
import stock_profit_first_literal_runner_oct17_audit as parent
import stock_profit_first_multisession_oct17_audit as session_base
from stock_walk_forward_filter_lab import CATEGORICAL, NUMERIC, make_model


ROOT = Path("reports/audit_2026_09_08")
OUT = ROOT / "stock_profit_first_high_move_oct17.json"
ROWS_OUT = ROOT / "stock_profit_first_high_move_oct17.pkl"
TARGETS_PCT = (2.5, 3.0, 4.0)
HOLD_SESSIONS = (1, 2, 3)
STOP_PCT = 1.0
QUANTILES = (.60, .70, .75, .80, .85, .875, .90, .925, .95)
FIT_YEARS = parent.FIT_YEARS
SELECTION_YEAR = 2022
LATER_YEARS = (2023, 2024, 2025, 2026)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def add_outcomes(frame: pd.DataFrame, candles: dict, metadata: dict,
                 target_pct: float, sessions: int) -> pd.DataFrame:
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
            target_pct, STOP_PCT,
            sessions * int(metadata[symbol]["bars_per_session"]))
        valid["signed_return"] = values
        valid["exit_time"] = exits
        valid["target_hit"] = outcomes == 1
        valid["exit_reason"] = np.where(
            outcomes == 1, 1, np.where(outcomes == -1, 0, 2))
        valid["won"] = valid.target_hit.astype(int)
        valid["net_r"] = values / (STOP_PCT / 100)
        valid["risk_pct"] = STOP_PCT / 100
        valid["literal_target_pct"] = target_pct
        valid["hold_sessions"] = sessions
        pieces.append(valid)
    return pd.concat(pieces, ignore_index=True)


def evaluate(frame: pd.DataFrame, probability: np.ndarray,
             years: tuple[int, ...], threshold: float) -> dict:
    return parent.enrich(base.score(frame, probability, years, threshold))


def selection_ok(value: dict) -> bool:
    return bool(value["return_pct"] > 0 and value["dd_pct"] <= 18
                and value["trades_per_calendar_day"] >= 3)


def objective_ok(value: dict) -> bool:
    return parent.objective_ok(value)


def public(item: dict) -> dict:
    return {key: value for key, value in item.items()
            if key not in {"_probability"}}


def selected_rows(frame: pd.DataFrame, probability: np.ndarray,
                  threshold: float) -> pd.DataFrame:
    mask = probability >= threshold
    chosen = frame.loc[mask].copy()
    chosen["quality"] = probability[mask]
    return pd.DataFrame(gate(chosen.sort_values(
        ["entry_time", "quality", "symbol"], ascending=[True, False, True])
        .to_dict("records"), 4))


def main() -> None:
    candles, sources = base.load()
    proposals = parent.build_target(candles, 2.0)
    metadata = session_base.regular_metadata(candles)
    features = list(NUMERIC + CATEGORICAL)
    candidates = []
    profiles = []
    for target, sessions in itertools.product(TARGETS_PCT, HOLD_SESSIONS):
        frame = add_outcomes(proposals, candles, metadata, target, sessions)
        fit = frame.year.isin(FIT_YEARS)
        estimator = make_model()
        estimator.fit(frame.loc[fit, features], frame.loc[fit, "won"])
        probability = estimator.predict_proba(frame[features])[:, 1]
        selection_scores = probability[frame.year.eq(SELECTION_YEAR).to_numpy()]
        frontier = []
        for quantile in QUANTILES:
            threshold = float(np.quantile(selection_scores, quantile))
            value = evaluate(frame, probability, (SELECTION_YEAR,), threshold)
            item = {
                "target_pct": target, "hold_sessions": sessions,
                "selection_quantile": quantile, "threshold": threshold,
                "selection_2022": value,
                "selection_pass": selection_ok(value),
                "score": parent.profile_score(value),
                "_probability": probability,
            }
            candidates.append(item)
            frontier.append(public(item))
        profile = {
            "target_pct": target, "hold_sessions": sessions,
            "fit_rows": int(fit.sum()),
            "fit_target_prevalence_pct": round(
                100 * float(frame.loc[fit, "target_hit"].mean()), 4),
            "best_2022": max(frontier, key=lambda item: item["score"]),
        }
        profiles.append(profile)
        print(json.dumps(profile, default=float), flush=True)
    candidates.sort(key=lambda item: (item["selection_pass"], item["score"]),
                    reverse=True)
    frozen = candidates[0] if candidates and candidates[0]["selection_pass"] else None
    later = checks = None
    status = "REJECTED_SELECTION"
    if frozen:
        frame = add_outcomes(proposals, candles, metadata,
                             frozen["target_pct"], frozen["hold_sessions"])
        probability = frozen["_probability"]
        later = {str(year): evaluate(frame, probability, (year,), frozen["threshold"])
                 for year in LATER_YEARS}
        all_years = {str(SELECTION_YEAR): frozen["selection_2022"], **later}
        checks = {year: objective_ok(value) for year, value in all_years.items()}
        status = ("STOCK_HIGH_MOVE_RESEARCH_CANDIDATE"
                  if all(checks.values()) else "REJECTED_OBJECTIVE")
        selected_rows(frame, probability, frozen["threshold"]).to_pickle(ROWS_OUT)
    report = {
        "status": status, "ready_for_live_orders": False,
        "protocol": {
            "priority": "account return first; win rate diagnostic only",
            "venue": "Dukascopy US cash-session CFD proxy",
            "fit": list(FIT_YEARS), "profile_and_cutoff_selection": SELECTION_YEAR,
            "frozen_later_years": list(LATER_YEARS),
            "targets_pct": list(TARGETS_PCT), "stop_pct": STOP_PCT,
            "hold_sessions": list(HOLD_SESSIONS),
            "entry": "next 15-minute market open", "orders": "market only",
            "money": "3% current-equity margin, fixed 10x gross",
            "objective": ">=30% account return in every month; >=3 trades/calendar day; DD <=10%",
        },
        "candidate_count": len(candidates), "profiles": profiles,
        "selection_top20": [public(item) for item in candidates[:20]],
        "frozen_before_2023": public(frozen) if frozen else None,
        "later_2023_2026": later, "objective_checks": checks,
        "sources": {**sources, str(Path(base.__file__)): digest(Path(base.__file__)),
                    str(Path(multi.__file__)): digest(Path(multi.__file__)),
                    str(Path(__file__)): digest(Path(__file__))},
        "limitations": [
            "Cash-session data omit overnight path and can only test regular-session exits.",
            "Larger targets reduce the target-hit sample and increase loss streak variance.",
            "Any historical survivor still requires direct X-Perp shadow transfer.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": status,
                      "frozen": public(frozen) if frozen else None,
                      "later_2023_2026": later,
                      "objective_checks": checks}, indent=2), flush=True)


if __name__ == "__main__":
    main()
