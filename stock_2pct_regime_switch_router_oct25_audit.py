"""Causal regime-switch audit for literal two-percent stock moves.

The experiment deliberately starts from the user's economic target.  Every
closed checkpoint proposes both directions; a small tree model chooses whether
the current market state favours continuation or reversal.  Models learn on
2017-2020, thresholds are selected on 2021, and later eras are revealed only
after the preceding gate passes.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import stock_15m_multisession_regime_oct6_audit as source
from stock_module_cluster_audit import portfolio_gate


ROOT = Path("reports/audit_2026_09_08")
OUT = ROOT / "stock_2pct_regime_switch_router_oct25.json"
ROWS_OUT = ROOT / "stock_2pct_regime_switch_router_oct25.csv"
TARGET_PCT = 2.0
STOP_PCT = 1.0
FIT = (2017, 2018, 2019, 2020)
SELECTION = (2021,)
VALIDATION = (2022, 2023)
TEST = (2024,)
CONFIRMATION = (2025, 2026)
HOLDS = (1, 2, 3)
QUANTILES = (.30, .40, .50, .60, .70, .75, .80, .85, .90, .925, .95)
CONFIGS = (
    {"leaves": 7, "leaf": 200, "l2": 4.0},
    {"leaves": 7, "leaf": 400, "l2": 8.0},
    {"leaves": 15, "leaf": 200, "l2": 4.0},
    {"leaves": 15, "leaf": 400, "l2": 8.0},
)
BASE_FEATURES = (
    "ret5", "ret20", "daily_vol20", "gap", "intraday", "last_hour",
    "breadth_intra", "cross_z5", "qqq_ret5", "qqq_ret20", "qqq_gap",
    "qqq_intraday", "qqq_last_hour",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bidirectional(frame: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for direction in (-1, 1):
        part = frame.copy()
        part["direction_n"] = np.int8(direction)
        part["proposal"] = "continuation_or_reversal"
        pieces.append(part)
    return pd.concat(pieces, ignore_index=True)


def schema(frame: pd.DataFrame) -> list[str]:
    symbols = sorted(frame.symbol.unique())
    slots = sorted(frame.slot.unique())
    return [
        *[f"{name}_dir" for name in BASE_FEATURES],
        "abs_ret5", "abs_ret20", "abs_gap", "abs_intraday",
        "trend_alignment", "opening_alignment", "index_alignment",
        *[f"symbol_{name}" for name in symbols],
        *[f"slot_{name}" for name in slots],
    ]


def matrix(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    value = pd.DataFrame(index=frame.index)
    direction = frame.direction_n.to_numpy(float)
    for name in BASE_FEATURES:
        if name == "daily_vol20":
            value[f"{name}_dir"] = frame[name]
        elif name == "breadth_intra":
            value[f"{name}_dir"] = np.where(
                direction > 0, frame[name], 1.0-frame[name])
        else:
            value[f"{name}_dir"] = frame[name]*direction
    for name in ("ret5", "ret20", "gap", "intraday"):
        value[f"abs_{name}"] = frame[name].abs()
    value["trend_alignment"] = np.sign(frame.ret20)*direction
    value["opening_alignment"] = np.sign(frame.intraday)*direction
    value["index_alignment"] = np.sign(frame.qqq_ret20)*direction
    for name in columns:
        if name.startswith("symbol_"):
            value[name] = (frame.symbol == name[7:]).astype(float)
        elif name.startswith("slot_"):
            value[name] = (frame.slot == name[5:]).astype(float)
    return value.reindex(columns=columns).replace(
        [np.inf, -np.inf], np.nan).fillna(0).to_numpy(np.float32)


def accepted(frame: pd.DataFrame, threshold: float) -> list[dict]:
    chosen = frame[frame.quality.ge(threshold)].copy()
    chosen["score"] = chosen.quality
    return portfolio_gate(
        chosen.to_dict("records"), None,
        rank_key=lambda row: (-float(row["quality"]), str(row["symbol"])),
        loss_streak_cap=3,
    )


def evaluate(scored: pd.DataFrame, features: pd.DataFrame,
             years: tuple[int, ...], threshold: float) -> tuple[dict, list[dict]]:
    part = scored[scored.year.isin(years)]
    rows = accepted(part, threshold)
    days = features[features.year.isin(years)].session_date.nunique()
    if not rows:
        return {
            "n": 0, "wr": 0.0, "literal_target_hit_rate": 0.0,
            "return_pct": 0.0, "dd_pct": 0.0, "negative_months": 0,
            "months": 0, "monthly_return_pct": {}, "max_loss_streak": 0,
            "trades_per_market_day": 0.0, "mean_signed_move_pct": 0.0,
            "projected_22_market_day_pct": 0.0,
        }, []
    frame = pd.DataFrame(rows)
    value = source.account_metrics(frame, days)
    mean_move = float(frame.signed_return.mean())
    periods = 22.0*len(rows)/max(days, 1)
    factor = 1.0+.03*10.0*mean_move
    projected = 100.0*(factor**periods-1.0) if factor > 0 else -100.0
    value.update({
        "literal_target_hits": int((frame.outcome_code == 1).sum()),
        "literal_target_hit_rate": round(
            100.0*float((frame.outcome_code == 1).mean()), 4),
        "stops": int((frame.outcome_code == -1).sum()),
        "timeouts": int((frame.outcome_code == 0).sum()),
        "projected_22_market_day_pct": round(projected, 4),
    })
    return value, rows


def annual(scored: pd.DataFrame, features: pd.DataFrame,
           years: tuple[int, ...], threshold: float) -> dict[str, dict]:
    return {str(year): evaluate(scored, features, (year,), threshold)[0]
            for year in years}


def selection_ok(value: dict) -> bool:
    return bool(
        value["n"] >= 350 and value["return_pct"] > 0
        and value["dd_pct"] <= 15
        and value["trades_per_market_day"] >= 1.5
        and value["literal_target_hit_rate"] >= 35
    )


def validation_ok(value: dict, by_year: dict[str, dict]) -> bool:
    return bool(
        value["n"] >= 300 and value["return_pct"] > 0
        and value["dd_pct"] <= 12
        and value["trades_per_market_day"] >= 1.5
        and value["literal_target_hit_rate"] >= 35
        and all(item["return_pct"] > 0 for item in by_year.values())
    )


def later_ok(value: dict) -> bool:
    return bool(
        value["n"] >= 100 and value["return_pct"] > 0
        and value["dd_pct"] <= 10
        and value["trades_per_market_day"] >= 1.5
        and value["literal_target_hit_rate"] >= 35
    )


def objective_ok(value: dict) -> bool:
    return bool(
        value["trades_per_market_day"] >= 3
        and 75 <= value["literal_target_hit_rate"] <= 95
        and value["projected_22_market_day_pct"] >= 30
        and value["dd_pct"] <= 5 and value["negative_months"] == 0
    )


def public(item: dict) -> dict:
    return {key: value for key, value in item.items()
            if key not in ("_model", "_scored", "_rows")}


def main() -> None:
    historical, historical_sources = source.load_history()
    direct, direct_sources = source.load_direct()
    features, feature_meta = source.build_features(historical)
    direct_features, direct_meta = source.build_features(direct)
    proposals = bidirectional(features)
    direct_proposals = bidirectional(direct_features)
    columns = schema(proposals)
    candidates = []
    frontier = []
    for hold in HOLDS:
        resolved = source.add_outcomes(
            proposals, historical, feature_meta, TARGET_PCT, STOP_PCT, hold)
        fit = resolved[resolved.year.isin(FIT)]
        x_fit = matrix(fit, columns)
        y_fit = (fit.outcome_code == 1).astype(np.int8).to_numpy()
        for config in CONFIGS:
            model = HistGradientBoostingClassifier(
                max_iter=180, learning_rate=.05,
                max_leaf_nodes=config["leaves"],
                min_samples_leaf=config["leaf"],
                l2_regularization=config["l2"], random_state=20261025)
            model.fit(x_fit, y_fit)
            scored = resolved.copy()
            scored["quality"] = model.predict_proba(matrix(scored, columns))[:, 1]
            selection_scores = scored[scored.year.isin(SELECTION)].quality
            for quantile in QUANTILES:
                threshold = float(selection_scores.quantile(quantile))
                selection, selection_rows = evaluate(
                    scored, features, SELECTION, threshold)
                item = {
                    "hold_sessions": hold, "model": config,
                    "quantile": quantile, "threshold": threshold,
                    "selection_2021": selection,
                }
                frontier.append(item)
                if not selection_ok(selection):
                    continue
                validation, _ = evaluate(scored, features, VALIDATION, threshold)
                validation_years = annual(
                    scored, features, VALIDATION, threshold)
                item.update({
                    "validation_2022_2023": validation,
                    "validation_years": validation_years,
                    "score": round(
                        min(row["return_pct"] for row in validation_years.values())
                        + .25*validation["return_pct"]-validation["dd_pct"], 6),
                    "_model": model, "_scored": scored,
                    "_rows": selection_rows,
                })
                if validation_ok(validation, validation_years):
                    candidates.append(item)
    candidates.sort(key=lambda item: item["score"], reverse=True)
    frontier.sort(key=lambda item: (
        item["selection_2021"]["projected_22_market_day_pct"],
        item["selection_2021"]["return_pct"]), reverse=True)
    frozen = candidates[0] if candidates else None
    stages = checks = None
    status = "REJECTED_BEFORE_HIDDEN_TEST"
    selected_rows: list[dict] = []
    if frozen:
        scored = frozen["_scored"]
        threshold = float(frozen["threshold"])
        test, test_rows = evaluate(scored, features, TEST, threshold)
        stages = {
            "selection_2021": frozen["selection_2021"],
            "validation_2022_2023": frozen["validation_2022_2023"],
            "test_2024": test,
        }
        selected_rows = test_rows
        status = "REJECTED_TEST_2024"
        if later_ok(test):
            confirmation, confirmation_rows = evaluate(
                scored, features, CONFIRMATION, threshold)
            stages["confirmation_2025_2026"] = confirmation
            selected_rows = confirmation_rows
            status = "REJECTED_CONFIRMATION_2025_2026"
            if later_ok(confirmation):
                direct_resolved = source.add_outcomes(
                    direct_proposals, direct, direct_meta,
                    TARGET_PCT, STOP_PCT, int(frozen["hold_sessions"]))
                direct_resolved["quality"] = frozen["_model"].predict_proba(
                    matrix(direct_resolved, columns))[:, 1]
                direct_result, direct_rows = evaluate(
                    direct_resolved, direct_features, (2026,), threshold)
                stages["direct_xperp_2026"] = direct_result
                selected_rows = direct_rows
                status = "SHADOW_CANDIDATE" if later_ok(
                    direct_result) else "REJECTED_DIRECT_TRANSFER"
        checks = {name: objective_ok(value) for name, value in stages.items()}
    pd.DataFrame(selected_rows).to_csv(ROWS_OUT, index=False)
    report = {
        "status": status,
        "ready_for_live_orders": False,
        "protocol": {
            "fit": list(FIT), "selection": list(SELECTION),
            "validation": list(VALIDATION), "test": list(TEST),
            "confirmation": list(CONFIRMATION),
            "mechanism": (
                "score both directions at three closed session checkpoints; "
                "market-state features let the model switch between continuation "
                "and reversal"),
            "entry": "next 15-minute market open", "market_orders_only": True,
            "target_pct": TARGET_PCT, "stop_pct": STOP_PCT,
            "hold_sessions": list(HOLDS),
            "money": "client-selected 3% current-equity margin; fixed 10x",
            "objective": ">=3/day, 75-95% literal target hits, projected >=30%/22 market days",
        },
        "rows": {"features": len(features), "proposals": len(proposals),
                 "direct_features": len(direct_features)},
        "qualified_selection_and_validation": len(candidates),
        "frozen_before_2024": public(frozen) if frozen else None,
        "stages": stages, "objective_checks": checks,
        "selection_frontier": [public(item) for item in frontier[:20]],
        "sources": {"historical": historical_sources, "direct": direct_sources,
                    str(Path(__file__)): digest(Path(__file__))},
        "limitations": [
            "Cash history establishes cross-era structure but is not an X-Perp fill.",
            "The bounded model and threshold grid still creates selection risk.",
            "OHLC bars use stop-first handling when target and stop share a bar.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": status,
        "qualified_selection_and_validation": len(candidates),
        "frozen": public(frozen) if frozen else None,
        "stages": stages, "objective_checks": checks,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
