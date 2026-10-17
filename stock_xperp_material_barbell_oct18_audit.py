"""Profit-first material stock X-Perp audit through 2026-10-16.

All entry, model, threshold, and exit choices use data ending before October 7.
The final October 7-16 slice is revealed only after those choices are frozen.
Orders are represented by next-bar market entries and price-triggered exits.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

import stock_cash_xperp_leadlag_oct6_audit as lead
import stock_cash_xperp_tree_filter_oct6_audit as tree


ROOT = Path("reports/audit_2026_09_08")
CASH_MANIFEST = ROOT / "stock_cash_forward_oct17" / "manifest.json"
XPERP_MANIFEST = ROOT / "stock_xperp_forward_oct17" / "manifest.json"
OUT = ROOT / "stock_xperp_material_barbell_oct18.json"
ROWS_OUT = ROOT / "stock_xperp_material_barbell_oct18.csv"
DEVELOPMENT = ("2026-06-15", "2026-08-01")
VALIDATION = ("2026-08-01", "2026-09-01")
CONFIRMATION = ("2026-09-01", "2026-10-07")
FRESH = ("2026-10-07", "2026-10-17")
BASE_TARGETS = (1.0, 1.5, 2.0)
BASE_HOLDS = (16, 32, 64, 128)
RUNNER_TARGETS = (1.5, 2.0, 3.0)
RUNNER_HOLDS = (32, 64, 128)
PARTIAL_FRACTIONS = (.10, .25, .50)
PROBABILITY_THRESHOLDS = tuple(
    round(float(value), 3) for value in np.arange(.35, .701, .025))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_sources() -> tuple[dict, dict, dict]:
    old_cash, old_xperp = lead.CASH_MANIFEST, lead.XPERP_MANIFEST
    try:
        lead.CASH_MANIFEST = CASH_MANIFEST
        lead.XPERP_MANIFEST = XPERP_MANIFEST
        return lead.load_sources()
    finally:
        lead.CASH_MANIFEST, lead.XPERP_MANIFEST = old_cash, old_xperp


def normalized_return(value: dict, days: int) -> float:
    fraction = value["return_pct"] / 100
    if fraction <= -1:
        return -100.0
    return 100 * ((1 + fraction) ** (30 / days) - 1)


def chronological_score(values: dict) -> float:
    rates = (
        normalized_return(values["development"], 47),
        normalized_return(values["validation"], 31),
        normalized_return(values["confirmation"], 36),
    )
    drawdown = max(value["dd_pct"] for value in values.values())
    return round(min(rates) - .3 * drawdown, 8)


def period_ok(values: dict, minimums: dict[str, int]) -> bool:
    return all(value["n"] >= minimums[name]
               and value["return_pct"] > 0
               and value["dd_pct"] <= 10
               for name, value in values.items())


def resolve_barbell(frame: pd.DataFrame, xperp: dict, runner_target: float,
                    hold_bars: int, partial_fraction: float) -> pd.DataFrame:
    """Resolve a 1% partial, delayed break-even, and material runner.

    A stop wins every same-bar ambiguity.  After the first target is touched,
    neither the runner nor break-even can trigger until the following bar.
    """
    output = []
    for symbol, part in frame.groupby("symbol", sort=False):
        candle = xperp[symbol]
        open_, high, low, close = [np.asarray(candle[key], dtype=float)
                                   for key in ("open", "high", "low", "close")]
        times = np.asarray(candle["time"], dtype=np.int64)
        for row in part.to_dict("records"):
            entry_index = int(row["entry_index"])
            direction = int(row["direction_n"])
            entry = float(open_[entry_index])
            near = entry * (1 + direction * .01)
            runner = entry * (1 + direction * runner_target / 100)
            stop = entry * (1 - direction * .01)
            last = min(len(times) - 1, entry_index + hold_bars - 1)
            activated = False
            signed_return = None
            exit_index = last
            for index in range(entry_index, last + 1):
                opening = float(open_[index])
                stop_hit = (float(low[index]) <= stop if direction > 0
                            else float(high[index]) >= stop)
                near_hit = (float(high[index]) >= near if direction > 0
                            else float(low[index]) <= near)
                runner_hit = (float(high[index]) >= runner if direction > 0
                              else float(low[index]) <= runner)
                if not activated:
                    if (opening <= stop if direction > 0 else opening >= stop):
                        signed_return = direction * (opening / entry - 1)
                        exit_index = index
                        break
                    if stop_hit:
                        signed_return = -.01
                        exit_index = index
                        break
                    if (opening >= runner if direction > 0 else opening <= runner):
                        signed_return = (partial_fraction * .01
                                         + (1 - partial_fraction)
                                         * runner_target / 100)
                        exit_index = index
                        break
                    if ((opening >= near if direction > 0 else opening <= near)
                            or near_hit):
                        activated = True
                        continue
                else:
                    break_even_hit = (float(low[index]) <= entry if direction > 0
                                      else float(high[index]) >= entry)
                    if (opening <= entry if direction > 0 else opening >= entry):
                        remainder = direction * (opening / entry - 1)
                        signed_return = (partial_fraction * .01
                                         + (1 - partial_fraction) * remainder)
                        exit_index = index
                        break
                    if break_even_hit:  # pessimistic if runner also touched
                        signed_return = partial_fraction * .01
                        exit_index = index
                        break
                    if ((opening >= runner if direction > 0 else opening <= runner)
                            or runner_hit):
                        signed_return = (partial_fraction * .01
                                         + (1 - partial_fraction)
                                         * runner_target / 100)
                        exit_index = index
                        break
            if signed_return is None:
                remainder = direction * (float(close[last]) / entry - 1)
                signed_return = (partial_fraction * .01
                                 + (1 - partial_fraction) * remainder
                                 if activated else remainder)
            row.update({
                "direction": "LONG" if direction > 0 else "SHORT",
                "signed_return": float(signed_return),
                "exit_time": int(times[exit_index]) + 900,
                "outcome": "TP" if signed_return > 0 else "SL",
                "risk_pct": .01, "net_r": float(signed_return / .01),
                "target_pct": runner_target, "stop_pct": 1.0,
                "hold_bars": hold_bars,
                "near_target_pct": 1.0,
                "near_fraction": partial_fraction,
                "module": "cash_xperp_material_barbell",
                "module_priority": 0,
            })
            output.append(row)
    return pd.DataFrame(output)


def evaluate_periods(frame: pd.DataFrame) -> dict:
    return {
        "development": lead.evaluate(frame, DEVELOPMENT),
        "validation": lead.evaluate(frame, VALIDATION),
        "confirmation": lead.evaluate(frame, CONFIRMATION),
    }


def public(item: dict) -> dict:
    return {key: value for key, value in item.items() if key != "_rows"}


def main() -> None:
    cash, xperp, sources = load_sources()
    raw = lead.raw_features(cash, xperp)

    base_candidates = []
    for spec in lead.SPECS:
        entries = lead.apply_spec(raw, spec)
        for lookback in lead.LOOKBACKS:
            selected = entries[entries.lookback.eq(lookback)]
            if len(selected) < 100:
                continue
            for target, hold in itertools.product(BASE_TARGETS, BASE_HOLDS):
                resolved = lead.add_outcomes(selected, xperp, target, 1.0, hold)
                values = evaluate_periods(resolved)
                if not period_ok(values, {"development": 80,
                                          "validation": 40,
                                          "confirmation": 40}):
                    continue
                base_candidates.append({
                    "entry": spec, "lookback_bars": lookback,
                    "target_pct": target, "hold_bars": hold,
                    "score": chronological_score(values), "periods": values,
                    "_rows": resolved,
                })
    base_candidates.sort(key=lambda item: item["score"], reverse=True)
    base = base_candidates[0] if base_candidates else None
    if base is None:
        raise ValueError("no material base route survived pre-fresh periods")

    train = tree.inside(base["_rows"], DEVELOPMENT, require_closed=True)
    train_x, columns = tree.matrix(train)
    model = GradientBoostingClassifier(
        random_state=261018, n_estimators=60, learning_rate=.05,
        max_depth=2, min_samples_leaf=30, subsample=.8,
    )
    model.fit(train_x, train.signed_return.gt(0).astype(int))
    all_x, _ = tree.matrix(base["_rows"], columns)
    modeled = base["_rows"].copy()
    modeled["probability"] = model.predict_proba(all_x)[:, 1]

    threshold_candidates = []
    for threshold in PROBABILITY_THRESHOLDS:
        selected = modeled[modeled.probability.ge(threshold)]
        values = {
            "validation": lead.evaluate(selected, VALIDATION),
            "confirmation": lead.evaluate(selected, CONFIRMATION),
        }
        if not period_ok(values, {"validation": 20, "confirmation": 20}):
            continue
        score = min(values["validation"]["return_pct"] * 30 / 31,
                    values["confirmation"]["return_pct"] * 30 / 36)
        score -= .3 * max(value["dd_pct"] for value in values.values())
        threshold_candidates.append({"threshold": threshold,
                                     "score": round(score, 8),
                                     "periods": values})
    threshold_candidates.sort(key=lambda item: item["score"], reverse=True)
    threshold = threshold_candidates[0] if threshold_candidates else None
    if threshold is None:
        raise ValueError("no causal probability threshold survived confirmation")

    selected_keys = set(zip(
        modeled.loc[modeled.probability.ge(threshold["threshold"]), "symbol"],
        modeled.loc[modeled.probability.ge(threshold["threshold"]), "entry_time"],
    ))
    raw_entries = lead.apply_spec(raw, base["entry"])
    raw_entries = raw_entries[raw_entries.lookback.eq(base["lookback_bars"])]

    exit_candidates = []
    for runner, hold, partial in itertools.product(
            RUNNER_TARGETS, RUNNER_HOLDS, PARTIAL_FRACTIONS):
        resolved = resolve_barbell(raw_entries, xperp, runner, hold, partial)
        resolved = resolved[[key in selected_keys for key in zip(
            resolved.symbol, resolved.entry_time)]]
        values = evaluate_periods(resolved)
        if not period_ok(values, {"development": 20,
                                  "validation": 20,
                                  "confirmation": 20}):
            continue
        exit_candidates.append({
            "runner_target_pct": runner, "hold_bars": hold,
            "near_fraction": partial,
            "score": chronological_score(values), "periods": values,
            "_rows": resolved,
        })
    exit_candidates.sort(key=lambda item: item["score"], reverse=True)
    frozen = exit_candidates[0] if exit_candidates else None
    if frozen is None:
        raise ValueError("no barbell exit survived pre-fresh periods")

    fresh = lead.evaluate(frozen["_rows"], FRESH)
    full = lead.evaluate(frozen["_rows"], (DEVELOPMENT[0], FRESH[1]))
    frozen["_rows"].to_csv(ROWS_OUT, index=False)
    objective = {
        "all_pre_fresh_periods_positive": all(
            value["return_pct"] > 0 for value in frozen["periods"].values()),
        "fresh_positive": fresh["return_pct"] > 0,
        "fresh_at_least_3_trades_per_day":
            fresh["trades_per_calendar_day"] >= 3,
        "fresh_drawdown_at_most_5pct": fresh["dd_pct"] <= 5,
        "fresh_return_at_least_30pct": fresh["return_pct"] >= 30,
    }
    ready = all(objective.values())
    importances = sorted(zip(columns, model.feature_importances_),
                         key=lambda item: item[1], reverse=True)
    report = {
        "status": ("DIRECT_XPERP_MATERIAL_CANDIDATE"
                   if ready else "REJECTED_MONTHLY_OBJECTIVE"),
        "ready_for_live_orders": False,
        "protocol": {
            "development": DEVELOPMENT, "validation": VALIDATION,
            "confirmation": CONFIRMATION,
            "untouched_fresh_reveal": FRESH,
            "entry": "next 15-minute X-Perp market open",
            "orders": "market only",
            "base_targets_pct": list(BASE_TARGETS),
            "runner_targets_pct": list(RUNNER_TARGETS),
            "stop_pct": 1.0,
            "money": "3% current-equity margin, fixed 10x gross",
            "selection_priority": "account return; win rate diagnostic only",
        },
        "matched_feature_rows": len(raw),
        "base_candidate_count": len(base_candidates),
        "base_top20": [public(item) for item in base_candidates[:20]],
        "frozen_base": public(base),
        "threshold_candidates": threshold_candidates,
        "frozen_threshold": threshold,
        "feature_importance": [
            {"feature": name, "importance": round(float(value), 8)}
            for name, value in importances
        ],
        "barbell_candidate_count": len(exit_candidates),
        "barbell_top20": [public(item) for item in exit_candidates[:20]],
        "frozen_before_oct7": public(frozen),
        "fresh_oct7_oct16": fresh,
        "full_direct_window": full,
        "objective_checks": objective,
        "sources": {**sources, str(CASH_MANIFEST): digest(CASH_MANIFEST),
                    str(XPERP_MANIFEST): digest(XPERP_MANIFEST),
                    str(Path(__file__)): digest(Path(__file__))},
        "limitations": [
            "The direct matched venue history begins in June 2026.",
            "Only October 7-16 is untouched by final route, threshold, and exit selection.",
            "The fresh sample has eight active days and is too short for live activation.",
            "Gross returns exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "base": public(base),
        "threshold": threshold, "barbell": public(frozen),
        "fresh": fresh, "full": full, "objective_checks": objective,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
