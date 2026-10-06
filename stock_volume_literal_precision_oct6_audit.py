"""Precision model for literal +0.5% moves after opening volume dislocations."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier

from stock_module_cluster_audit import portfolio_gate
import stock_15m_multisession_regime_oct6_audit as base
import stock_volume_auction_regime_oct6_audit as volume


ROOT = Path("reports/audit_2026_09_08")
OUT = ROOT / "stock_volume_literal_precision_oct6.json"
ROWS_OUT = ROOT / "stock_volume_literal_precision_oct6.csv"
TARGET_PCT = .5
STOP_PCT = 1.0
HOLD_SESSIONS = 2
ROUTES = ("gap_follow", "gap_fail", "idio_drive", "exhaustion", "breadth_drive")
QUANTILES = (.20, .30, .40, .50, .60, .70, .75, .80, .85, .90, .925, .95)
CONFIGS = (
    {"kind": "hist", "leaves": 15, "leaf": 180, "l2": 3.0},
    {"kind": "hist", "leaves": 31, "leaf": 300, "l2": 5.0},
    {"kind": "extra", "leaf": 80, "features": .7},
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def proposals(frame: pd.DataFrame) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []

    def add(name: str, direction: pd.Series, mask: pd.Series) -> None:
        found = volume.directed(frame, direction.to_numpy(float))
        found = found.loc[mask & found.direction_n.ne(0)].copy()
        found[f"route_{name}"] = 1
        pieces.append(found)

    gap_direction = np.sign(frame.gap_atr)
    add("gap_follow", gap_direction,
        frame.gap_atr.abs().ge(.20) & (frame.move_atr*gap_direction).ge(.05)
        & frame.rvol.ge(.90))
    reverse_gap = -gap_direction
    add("gap_fail", reverse_gap,
        frame.gap_atr.abs().ge(.30) & (frame.move_atr*reverse_gap).ge(.05)
        & frame.rvol.ge(.90))
    residual_direction = np.sign(frame.residual_atr)
    add("idio_drive", residual_direction,
        frame.residual_atr.abs().ge(.18) & frame.range_atr.ge(.25)
        & frame.rvol.ge(.85))
    reverse_residual = -residual_direction
    wick = np.where(reverse_residual > 0, frame.lower_wick_fraction,
                    frame.upper_wick_fraction)
    add("exhaustion", reverse_residual,
        frame.residual_atr.abs().ge(.45) & frame.range_atr.ge(.45)
        & frame.rvol.ge(1.10) & pd.Series(wick, index=frame.index).ge(.12))
    market_direction = np.sign(frame.qqq_move_atr)
    breadth_direction = np.where(market_direction > 0, frame.breadth_up,
                                 1-frame.breadth_up)
    add("breadth_drive", pd.Series(market_direction, index=frame.index),
        frame.qqq_move_atr.abs().ge(.10)
        & pd.Series(breadth_direction, index=frame.index).ge(.57)
        & (frame.move_atr*market_direction).ge(.03) & frame.rvol.ge(.80))

    combined = pd.concat(pieces, ignore_index=True)
    for route in ROUTES:
        column = f"route_{route}"
        if column not in combined:
            combined[column] = 0
    keys = ["symbol", "entry_time", "direction_n"]
    flags = combined.groupby(keys, sort=False)[
        [f"route_{route}" for route in ROUTES]].max().reset_index()
    values = combined.sort_values(keys).drop_duplicates(keys).drop(
        columns=[f"route_{route}" for route in ROUTES])
    return values.merge(flags, on=keys, how="left", validate="one_to_one")


def schema(frame: pd.DataFrame) -> list[str]:
    continuous = (
        "gap_atr_dir", "move_atr_dir", "qqq_gap_atr_dir", "qqq_move_atr_dir",
        "residual_atr_dir", "location_dir", "breadth_dir", "rejection_wick",
        "rvol", "qqq_rvol", "range_atr", "abs_gap_atr", "abs_move_atr",
        "abs_residual_atr", "rvol_residual", "rvol_move",
    )
    return [*continuous, *[f"route_{route}" for route in ROUTES],
            *[f"symbol_{symbol}" for symbol in sorted(frame.symbol.unique())],
            *[f"slot_{slot}" for slot in sorted(frame.slot.unique())]]


def matrix(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    value = pd.DataFrame(index=frame.index)
    for name in ("gap_atr_dir", "move_atr_dir", "qqq_gap_atr_dir",
                 "qqq_move_atr_dir", "residual_atr_dir", "location_dir",
                 "breadth_dir", "rejection_wick", "rvol", "qqq_rvol",
                 "range_atr"):
        value[name] = frame[name].astype(float)
    value["abs_gap_atr"] = frame.gap_atr.abs()
    value["abs_move_atr"] = frame.move_atr.abs()
    value["abs_residual_atr"] = frame.residual_atr.abs()
    value["rvol_residual"] = frame.rvol*frame.residual_atr_dir
    value["rvol_move"] = frame.rvol*frame.move_atr_dir
    for route in ROUTES:
        value[f"route_{route}"] = frame[f"route_{route}"].astype(float)
    for name in columns:
        if name.startswith("symbol_"):
            value[name] = (frame.symbol == name[7:]).astype(float)
        elif name.startswith("slot_"):
            value[name] = (frame.slot == name[5:]).astype(float)
    return value.reindex(columns=columns).replace(
        [np.inf, -np.inf], np.nan).fillna(0).to_numpy(np.float32)


def accepted(frame: pd.DataFrame, years: tuple[int, ...], threshold: float) -> list[dict]:
    chosen = frame[frame.year.isin(years) & frame.quality.ge(threshold)].copy()
    chosen["score"] = chosen.quality
    return portfolio_gate(chosen.to_dict("records"), None,
                          rank_key=lambda row: (-float(row["quality"]),
                                                str(row["symbol"])))


def evaluate(frame: pd.DataFrame, features: pd.DataFrame,
             years: tuple[int, ...], threshold: float) -> dict:
    rows = accepted(frame, years, threshold)
    market_days = features[features.year.isin(years)].session_date.nunique()
    chosen = pd.DataFrame(rows) if rows else frame.iloc[:0].copy()
    value = base.account_metrics(chosen, market_days)
    n = len(rows)
    hits = sum(int(row["outcome_code"]) == 1 for row in rows)
    stops = sum(int(row["outcome_code"]) == -1 for row in rows)
    value.update({"literal_target_hits": hits,
                  "literal_target_hit_rate": round(100*hits/n, 4) if n else 0.0,
                  "stops": stops, "timeouts": n-hits-stops})
    return value


def selection_ok(value: dict) -> bool:
    return bool(value["n"] >= 300 and value["return_pct"] > 0
                and value["literal_target_hit_rate"] >= 70
                and value["dd_pct"] <= 8
                and value["trades_per_market_day"] >= 2.5)


def validation_ok(value: dict) -> bool:
    return bool(value["n"] >= 250 and value["return_pct"] > 0
                and value["literal_target_hit_rate"] >= 68
                and value["dd_pct"] <= 8
                and value["trades_per_market_day"] >= 2.5)


def objective_ok(value: dict, minimum: int) -> bool:
    return bool(value["n"] >= minimum and value["return_pct"] > 0
                and 75 <= value["literal_target_hit_rate"] <= 95
                and value["dd_pct"] <= 5 and value["negative_months"] == 0
                and value["trades_per_market_day"] >= 3
                and value["months_ge_30pct"] == value["months"])


def make_model(config: dict):
    if config["kind"] == "hist":
        return HistGradientBoostingClassifier(
            max_iter=180, learning_rate=.06, max_leaf_nodes=config["leaves"],
            min_samples_leaf=config["leaf"], l2_regularization=config["l2"],
            random_state=20261006)
    return ExtraTreesClassifier(
        n_estimators=300, min_samples_leaf=config["leaf"],
        max_features=config["features"], class_weight="balanced",
        random_state=20261006, n_jobs=-1)


def public(item: dict | None) -> dict | None:
    if item is None:
        return None
    return {key: value for key, value in item.items()
            if key not in ("_model", "_frame")}


def main() -> None:
    historical, historical_sources = base.load_history()
    direct, direct_sources = base.load_direct()
    history_features, history_meta = volume.build_features(historical)
    direct_features, direct_meta = volume.build_features(direct)
    history = base.add_outcomes(proposals(history_features), historical,
                                history_meta, TARGET_PCT, STOP_PCT, HOLD_SESSIONS)
    direct = base.add_outcomes(proposals(direct_features), direct,
                               direct_meta, TARGET_PCT, STOP_PCT, HOLD_SESSIONS)
    fit = history.year.isin((2017, 2018, 2019, 2020)).to_numpy()
    selection_mask = history.year.isin((2021, 2022)).to_numpy()
    columns = schema(history.loc[fit])
    x = matrix(history, columns)
    x_direct = matrix(direct, columns)
    y = history.outcome_code.eq(1).astype(int).to_numpy()
    candidates, frontier = [], []
    for config in CONFIGS:
        model = make_model(config)
        model.fit(x[fit], y[fit])
        scored = history.copy()
        scored["quality"] = model.predict_proba(x)[:, 1]
        for quantile in QUANTILES:
            threshold = float(scored.loc[selection_mask, "quality"].quantile(quantile))
            selection = evaluate(scored, history_features, (2021, 2022), threshold)
            item = {"model": config, "quantile": quantile,
                    "threshold": threshold, "selection_2021_2022": selection}
            frontier.append(item)
            if not selection_ok(selection):
                continue
            validation = evaluate(scored, history_features, (2023, 2024), threshold)
            item["validation_2023_2024"] = validation
            if not validation_ok(validation):
                continue
            item["score"] = round(validation["return_pct"]
                                  -2*validation["dd_pct"]
                                  +.25*validation["literal_target_hit_rate"]
                                  +.1*selection["return_pct"], 8)
            item["_model"], item["_frame"] = model, scored
            candidates.append(item)
    candidates.sort(key=lambda item: item["score"], reverse=True)
    frontier.sort(key=lambda item: (
        item["selection_2021_2022"]["literal_target_hit_rate"],
        item["selection_2021_2022"]["trades_per_market_day"]), reverse=True)
    frozen = candidates[0] if candidates else None
    results = checks = None
    if frozen:
        threshold = frozen["threshold"]
        scored = frozen["_frame"]
        direct_scored = direct.copy()
        direct_scored["quality"] = frozen["_model"].predict_proba(x_direct)[:, 1]
        scored[scored.quality.ge(threshold)].to_csv(ROWS_OUT, index=False)
        results = {
            "selection_2021_2022": frozen["selection_2021_2022"],
            "validation_2023_2024": frozen["validation_2023_2024"],
            "test_2025": evaluate(scored, history_features, (2025,), threshold),
            "observed_2026": evaluate(scored, history_features, (2026,), threshold),
            "direct_xperp": evaluate(direct_scored, direct_features, (2026,), threshold),
        }
        checks = {name: objective_ok(value, 30 if name == "direct_xperp" else 150)
                  for name, value in results.items()}
    status = ("RESEARCH_CANDIDATE_AWAITING_NEW_FORWARD"
              if checks and all(checks.values()) else "REJECTED")
    report = {
        "status": status,
        "protocol": {"fit": "2017-2020", "selection": "2021-2022",
                     "validation": "2023-2024", "test": "2025",
                     "observed": "2026", "direct_transfer": "X-Perp 2026",
                     "entry": "next regular-session market bar",
                     "target_pct": TARGET_PCT, "stop_pct": STOP_PCT,
                     "hold_sessions": HOLD_SESSIONS,
                     "win_definition": "literal target touch before stop/timeout",
                     "money": "3% entry-equity margin, fixed 10x gross"},
        "rows": {"history_features": len(history_features),
                 "history_proposals": len(history),
                 "direct_features": len(direct_features),
                 "direct_proposals": len(direct)},
        "fit_literal_prevalence_pct": round(100*float(y[fit].mean()), 4),
        "qualified_before_test": len(candidates),
        "frozen_before_test": public(frozen),
        "top_before_test": [public(item) for item in candidates[:12]],
        "selection_frontier": [public(item) for item in frontier[:20]],
        "results": results, "checks": checks,
        "sources": {"historical": historical_sources, "direct": direct_sources,
                    str(Path(__file__)): digest(Path(__file__))},
        "limitations": [
            "Cash volume is a long-run proxy and X-Perp volume has different units.",
            "A bounded model/threshold grid creates selection risk.",
            "Any survivor requires a new untouched direct X-Perp forward window.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": status, "rows": report["rows"],
                      "fit_literal_prevalence_pct": report["fit_literal_prevalence_pct"],
                      "qualified_before_test": len(candidates),
                      "frozen": public(frozen), "results": results,
                      "checks": checks}, indent=2), flush=True)


if __name__ == "__main__":
    main()
