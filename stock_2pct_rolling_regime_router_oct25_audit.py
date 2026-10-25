"""Annual causal refit of the frozen literal-2% regime-switch architecture."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import stock_15m_multisession_regime_oct6_audit as source
import stock_2pct_regime_switch_router_oct25_audit as base


ROOT = Path("reports/audit_2026_09_08")
OUT = ROOT / "stock_2pct_rolling_regime_router_oct25.json"
ROWS_OUT = ROOT / "stock_2pct_rolling_regime_router_oct25.csv"
HOLD = 3
CONFIG = {"leaves": 15, "leaf": 200, "l2": 4.0}
QUANTILE = .85
YEARS = tuple(range(2021, 2027))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit_model(resolved: pd.DataFrame, columns: list[str], years: tuple[int, ...]):
    fit = resolved[resolved.year.isin(years)]
    model = HistGradientBoostingClassifier(
        max_iter=180, learning_rate=.05,
        max_leaf_nodes=CONFIG["leaves"], min_samples_leaf=CONFIG["leaf"],
        l2_regularization=CONFIG["l2"], random_state=20261025)
    model.fit(base.matrix(fit, columns),
              (fit.outcome_code == 1).astype(np.int8).to_numpy())
    return model


def combined_metrics(rows: list[dict], features: pd.DataFrame) -> dict:
    if not rows:
        return {"n": 0}
    frame = pd.DataFrame(rows)
    days = features[features.year.isin(YEARS)].session_date.nunique()
    value = source.account_metrics(frame, days)
    mean_move = float(frame.signed_return.mean())
    periods = 22.0*len(frame)/max(days, 1)
    factor = 1.0+.03*10.0*mean_move
    value.update({
        "literal_target_hits": int((frame.outcome_code == 1).sum()),
        "literal_target_hit_rate": round(
            100*float((frame.outcome_code == 1).mean()), 4),
        "stops": int((frame.outcome_code == -1).sum()),
        "timeouts": int((frame.outcome_code == 0).sum()),
        "projected_22_market_day_pct": round(
            100*(factor**periods-1) if factor > 0 else -100.0, 4),
    })
    return value


def main() -> None:
    historical, historical_sources = source.load_history()
    direct, direct_sources = source.load_direct()
    features, metadata = source.build_features(historical)
    direct_features, direct_metadata = source.build_features(direct)
    proposals = base.bidirectional(features)
    resolved = source.add_outcomes(
        proposals, historical, metadata,
        base.TARGET_PCT, base.STOP_PCT, HOLD)
    columns = base.schema(proposals)
    annual = {}
    all_rows = []
    model = None
    threshold = None
    for year in YEARS:
        train_years = tuple(range(year-4, year))
        model = fit_model(resolved, columns, train_years)
        scored = resolved.copy()
        scored["quality"] = model.predict_proba(
            base.matrix(scored, columns))[:, 1]
        calibration = scored[scored.year.eq(year-1)].quality
        threshold = float(calibration.quantile(QUANTILE))
        value, rows = base.evaluate(scored, features, (year,), threshold)
        for row in rows:
            row["refit_year"] = year
            row["quality_threshold"] = threshold
        all_rows.extend(rows)
        annual[str(year)] = {
            "train_years": list(train_years),
            "threshold_from_year": year-1,
            "quality_threshold": threshold,
            "result": value,
        }

    direct_result = None
    direct_rows: list[dict] = []
    if model is not None and threshold is not None:
        direct_proposals = base.bidirectional(direct_features)
        direct_resolved = source.add_outcomes(
            direct_proposals, direct, direct_metadata,
            base.TARGET_PCT, base.STOP_PCT, HOLD)
        direct_resolved["quality"] = model.predict_proba(
            base.matrix(direct_resolved, columns))[:, 1]
        direct_result, direct_rows = base.evaluate(
            direct_resolved, direct_features, (2026,), threshold)

    combined = combined_metrics(all_rows, features)
    pd.DataFrame(all_rows).to_csv(ROWS_OUT, index=False)
    cross_era_positive = all(
        item["result"]["return_pct"] > 0 for item in annual.values())
    objective_checks = {
        year: base.objective_ok(item["result"])
        for year, item in annual.items()
    }
    status = (
        "ROLLING_SHADOW_CANDIDATE"
        if cross_era_positive and direct_result is not None
        and direct_result["return_pct"] > 0
        and all(objective_checks.values())
        else "REJECTED_OBJECTIVE"
    )
    report = {
        "status": status, "ready_for_live_orders": False,
        "protocol": {
            "architecture_source": "frozen static Oct-25 audit",
            "refit": "at each Jan 1, train only the preceding four full years",
            "threshold": (
                "fixed 85th score percentile, measured on the most recent "
                "completed training year"),
            "entry": "next 15-minute market open", "market_orders_only": True,
            "target_pct": base.TARGET_PCT, "stop_pct": base.STOP_PCT,
            "hold_sessions": HOLD,
            "money": "client-selected 3% current-equity margin; fixed 10x",
        },
        "annual_walk_forward": annual,
        "combined_2021_2026": combined,
        "direct_xperp_2026": direct_result,
        "objective_checks": objective_checks,
        "cross_era_positive": cross_era_positive,
        "sources": {"historical": historical_sources, "direct": direct_sources,
                    str(Path(__file__)): digest(Path(__file__))},
        "limitations": [
            "Annual refits require a reproducible year-end training job.",
            "Cash history is a structural proxy and not an X-Perp fill.",
            "The architecture was frozen after the preceding static audit.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": status, "annual_walk_forward": annual,
        "combined_2021_2026": combined,
        "direct_xperp_2026": direct_result,
        "objective_checks": objective_checks,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
