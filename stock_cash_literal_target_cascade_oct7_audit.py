"""Frozen 0.5/1/1.5/2% target cascade for deep stock cash-session entries."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

import stock_cash_literal_move_hgb_oct7_audit as base
from stock_walk_forward_filter_lab import CATEGORICAL, NUMERIC, make_model


ROOT = Path("reports/audit_2026_09_08")
OUT = ROOT / "stock_cash_literal_target_cascade_oct7.json"
ROWS_OUT = ROOT / "stock_cash_literal_target_cascade_oct7.pkl"
TARGETS = (.5, 1.0, 1.5, 2.0)
BASE_THRESHOLD = .64327171372379
QUANTILES = (.75, .85, .90, .94, .97, .985, .995)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def add_target(frame: pd.DataFrame, candles: dict, target_pct: float) -> None:
    suffix = str(target_pct).replace(".", "_")
    target_move = target_pct / 100.0
    stop_move = base.STOP_PCT / 100.0
    signed = np.empty(len(frame), dtype=float)
    exits = np.empty(len(frame), dtype=np.int64)
    hits = np.zeros(len(frame), dtype=bool)
    reasons = np.full(len(frame), 2, dtype=np.int8)
    for row in frame.itertuples():
        position = int(row.Index)
        candle = candles[row.symbol]
        first, last = int(row.entry_bar), int(row.session_end)
        entry = float(candle["open"][first])
        direction = 1 if row.direction == "LONG" else -1
        target = entry * (1 + direction * target_move)
        stop = entry * (1 - direction * stop_move)
        value = 0.0
        exit_bar = first
        resolved = False
        for index in range(first, last + 1):
            exit_bar = index
            stop_hit = (float(candle["low"][index]) <= stop if direction > 0
                        else float(candle["high"][index]) >= stop)
            target_hit = (float(candle["high"][index]) >= target if direction > 0
                          else float(candle["low"][index]) <= target)
            if stop_hit:
                opening = float(candle["open"][index])
                fill = min(opening, stop) if direction > 0 else max(opening, stop)
                value = direction * (fill / entry - 1)
                reasons[position] = 0
                resolved = True
                break
            if target_hit:
                value = target_move
                reasons[position] = 1
                hits[position] = True
                resolved = True
                break
        if not resolved:
            value = direction * (float(candle["close"][last]) / entry - 1)
            value = max(-stop_move, min(target_move, value))
        signed[position] = value
        exits[position] = int(candle["time"][exit_bar]) + 900
    frame[f"signed_{suffix}"] = signed
    frame[f"exit_{suffix}"] = exits
    frame[f"hit_{suffix}"] = hits
    frame[f"reason_{suffix}"] = reasons


def routed(frame: pd.DataFrame, thresholds: dict[float, float]) -> pd.DataFrame:
    found = frame[frame.quality_0_5.ge(BASE_THRESHOLD)].copy()
    target = np.full(len(found), .5, dtype=float)
    for value in (1.0, 1.5, 2.0):
        column = f"quality_{str(value).replace('.', '_')}"
        target[found[column].to_numpy() >= thresholds[value]] = value
    found["selected_target_pct"] = target
    found["signed_return"] = 0.0
    found["exit_time"] = 0
    found["target_hit"] = False
    found["exit_reason"] = 2
    for value in TARGETS:
        suffix = str(value).replace(".", "_")
        mask = found.selected_target_pct.eq(value)
        found.loc[mask, "signed_return"] = found.loc[mask, f"signed_{suffix}"]
        found.loc[mask, "exit_time"] = found.loc[mask, f"exit_{suffix}"]
        found.loc[mask, "target_hit"] = found.loc[mask, f"hit_{suffix}"]
        found.loc[mask, "exit_reason"] = found.loc[mask, f"reason_{suffix}"]
    found["net_r"] = found.signed_return / (base.STOP_PCT / 100.0)
    found["outcome"] = np.where(found.signed_return < 0, "SL", "TP")
    return found


def evaluate(frame: pd.DataFrame, years: tuple[int, ...],
             thresholds: dict[float, float]) -> dict:
    selected = routed(frame[frame.year.isin(years)], thresholds)
    ordered = selected.sort_values(
        ["entry_time", "quality_0_5", "symbol"],
        ascending=[True, False, True]).to_dict("records")
    accepted = base.gate(ordered, 4)
    result = base.account(accepted, already_gated=True)
    result["target_distribution"] = {
        str(value): sum(float(row["selected_target_pct"]) == value
                        for row in accepted) for value in TARGETS}
    result["average_selected_target_pct"] = (
        round(sum(float(row["selected_target_pct"]) for row in accepted)
              / len(accepted), 4) if accepted else 0.0)
    return result


def acceptable(candidate: dict, fixed: dict, selection: bool) -> bool:
    return bool(candidate["n"] >= (1200 if selection else 900)
                and candidate["return_pct"] > 0
                and candidate["return_pct"] > fixed["return_pct"]
                and candidate["dd_pct"] <= 8.0
                and candidate["dd_pct"] <= fixed["dd_pct"] + 1.0
                and candidate["literal_target_hit_rate"] >= 60
                and candidate["trades_per_calendar_day"] >= 3.0)


def main() -> None:
    candles, sources = base.load()
    frame = base.build(candles).reset_index(drop=True)
    # Replace the base outcome too so every target is produced by one resolver.
    for target in TARGETS:
        add_target(frame, candles, target)
    features = list(NUMERIC + CATEGORICAL)
    fit = frame.year.between(2017, 2021)
    for target in TARGETS:
        suffix = str(target).replace(".", "_")
        model = make_model()
        model.fit(frame.loc[fit, features], frame.loc[fit, f"hit_{suffix}"].astype(int))
        frame[f"quality_{suffix}"] = model.predict_proba(frame[features])[:, 1]
    selection = frame[frame.year.eq(2022) & frame.quality_0_5.ge(BASE_THRESHOLD)]
    grid = {}
    for target in (1.0, 1.5, 2.0):
        column = f"quality_{str(target).replace('.', '_')}"
        grid[target] = sorted(set(float(selection[column].quantile(q)) for q in QUANTILES))
    fixed_thresholds = {1.0: math.inf, 1.5: math.inf, 2.0: math.inf}
    fixed_2022 = evaluate(frame, (2022,), fixed_thresholds)
    frontier = []
    choices = []
    for values in itertools.product(*(grid[value] for value in (1.0, 1.5, 2.0))):
        thresholds = dict(zip((1.0, 1.5, 2.0), values))
        value = evaluate(frame, (2022,), thresholds)
        item = {"thresholds": {str(k): v for k, v in thresholds.items()},
                "selection_2022": value}
        frontier.append(item)
        if acceptable(value, fixed_2022, True):
            rank = ((value["return_pct"] - fixed_2022["return_pct"])
                    - 2 * max(0, value["dd_pct"] - fixed_2022["dd_pct"]))
            choices.append((rank, thresholds, value))
    choices.sort(key=lambda item: item[0], reverse=True)
    frozen = choices[0] if choices else None
    v23 = test24 = later = None
    status = "REJECTED_SELECTION"
    public_frozen = None
    if frozen:
        rank, thresholds, selection_value = frozen
        public_frozen = {"rank": rank,
                         "thresholds": {str(k): v for k, v in thresholds.items()},
                         "fixed_2022": fixed_2022,
                         "selection_2022": selection_value}
        fixed23 = evaluate(frame, (2023,), fixed_thresholds)
        candidate23 = evaluate(frame, (2023,), thresholds)
        v23 = {"fixed": fixed23, "cascade": candidate23}
        if acceptable(candidate23, fixed23, False):
            fixed24 = evaluate(frame, (2024,), fixed_thresholds)
            candidate24 = evaluate(frame, (2024,), thresholds)
            test24 = {"fixed": fixed24, "cascade": candidate24}
            if acceptable(candidate24, fixed24, False):
                later = {}
                for year in (2025, 2026):
                    fixed_value = evaluate(frame, (year,), fixed_thresholds)
                    candidate = evaluate(frame, (year,), thresholds)
                    later[str(year)] = {"fixed": fixed_value, "cascade": candidate}
                status = ("STOCK_CASH_CASCADE_CANDIDATE"
                          if all(acceptable(value["cascade"], value["fixed"], False)
                                 for value in later.values())
                          else "REJECTED_LATER")
                if status == "STOCK_CASH_CASCADE_CANDIDATE":
                    routed(frame, thresholds).to_pickle(ROWS_OUT)
            else:
                status = "REJECTED_2024"
        else:
            status = "REJECTED_2023"
    report = {
        "status": status,
        "protocol": {"fit": [2017, 2018, 2019, 2020, 2021],
                     "selection": 2022, "validation": 2023,
                     "confirmation": 2024, "hidden_if_confirmation_passes": [2025, 2026],
                     "targets_pct": list(TARGETS), "stop_pct": 1.0,
                     "hold": "same cash session only", "base_threshold": BASE_THRESHOLD,
                     "money": "3% current equity margin, fixed 10x gross"},
        "rows": len(frame), "selection_combinations": len(frontier),
        "frozen_before_2023": public_frozen, "validation_2023": v23,
        "confirmation_2024": test24, "later": later,
        "frontier_top20": sorted(frontier, key=lambda item: (
            item["selection_2022"]["return_pct"]
            - 2 * item["selection_2022"]["dd_pct"]), reverse=True)[:20],
        "sources": {**sources, str(Path(__file__)): digest(Path(__file__))},
    }
    if status != "STOCK_CASH_CASCADE_CANDIDATE" and ROWS_OUT.exists():
        ROWS_OUT.unlink()
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "status", "rows", "selection_combinations", "frozen_before_2023",
        "validation_2023", "confirmation_2024", "later")}, indent=2), flush=True)


if __name__ == "__main__":
    import math
    main()
