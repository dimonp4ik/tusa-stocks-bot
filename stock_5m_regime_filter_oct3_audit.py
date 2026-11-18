"""Causal regime-specific filter search on the stock five-minute families."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from stock_forward_monitor import frozen_rows
from stock_module_cluster_audit import portfolio_gate
from stock_5m_family_oct3_audit import (
    DEVELOPMENT, HIDDEN, PROFILE, PRODUCTION_SOURCE, ROWS_OUTPUT,
    VALIDATION, combine, improves, route_metrics, stage_rows, stages,
)


ROOT = Path("reports/audit_2026_09_08")
OUTPUT = ROOT / "stock_5m_regime_filter_oct3.json"
SELECTED_OUTPUT = ROOT / "stock_5m_regime_filter_oct3_rows.csv"
BASE_KEYS = ("family", "direction", "target_r", "stop_atr_mult")
FEATURES = (
    "signal_minutes", "efficiency", "abs_zscore", "signed_zscore",
    "volume_ratio", "trend_dir_atr", "qqq_trend_dir_5m_atr",
    "qqq_ret12_dir_5m_atr",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    sign = result["direction"].map({"LONG": 1.0, "SHORT": -1.0})
    result["abs_zscore"] = result["zscore"].abs()
    result["signed_zscore"] = sign * result["zscore"]
    return result


def cut(frame: pd.DataFrame, period: tuple[str, str]) -> pd.DataFrame:
    start = pd.Timestamp(period[0], tz="UTC").timestamp()
    end = pd.Timestamp(period[1], tz="UTC").timestamp()
    return frame[(frame["entry_time"] >= start) & (frame["entry_time"] < end)]


def atoms(development: pd.DataFrame) -> list[tuple]:
    output = []
    for field in FEATURES:
        values = pd.to_numeric(development[field], errors="coerce").dropna()
        if len(values) < 30:
            continue
        thresholds = sorted(set(float(value) for value in
                                values.quantile([.15, .3, .45, .55, .7, .85])))
        for threshold in thresholds:
            output.append((field, "ge", threshold))
            output.append((field, "le", threshold))
    return output


def condition_mask(frame: pd.DataFrame, conditions: tuple[tuple, ...]) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for field, operator, threshold in conditions:
        values = pd.to_numeric(frame[field], errors="coerce")
        mask &= values.ge(threshold) if operator == "ge" else values.le(threshold)
    return mask.fillna(False)


def select(frame: pd.DataFrame, candidate: dict) -> pd.DataFrame:
    mask = pd.Series(True, index=frame.index)
    for key, value in candidate["route"].items():
        mask &= frame[key] == value
    conditions = tuple(tuple(value) for value in candidate["conditions"])
    return frame[mask & condition_mask(frame, conditions)]


def base_group(frame: pd.DataFrame, route: dict) -> pd.DataFrame:
    result = frame
    for key, value in route.items():
        result = result[result[key] == value]
    return result


def discovery(frame: pd.DataFrame) -> tuple[list[dict], int]:
    development = cut(frame, DEVELOPMENT)
    validation = cut(frame, VALIDATION)
    public_atoms = atoms(development)
    found = []
    evaluated = 0
    grouped = development.groupby(list(BASE_KEYS), observed=True, sort=True)
    for route_values, dev_group in grouped:
        route = dict(zip(BASE_KEYS, route_values))
        val_group = base_group(validation, route)
        if len(dev_group) < 15 or len(val_group) < 12:
            continue
        ranked = []
        for atom in public_atoms:
            chosen = dev_group[condition_mask(dev_group, (atom,))]
            if len(chosen) < 10:
                continue
            value = route_metrics(chosen.to_dict("records"))
            ranked.append((value["mean_r"] + value["wr"] / 10000, atom))
        strongest = [atom for _, atom in sorted(ranked, reverse=True)[:20]]
        rules = [(atom,) for atom in public_atoms]
        rules += [pair for pair in itertools.combinations(strongest, 2)
                  if pair[0][0] != pair[1][0]]
        seen = set()
        target = float(route["target_r"])
        required_wr = 84.0 if target == .25 else 75.0
        for conditions in rules:
            rule_mask = condition_mask(dev_group, conditions)
            signature = np.packbits(rule_mask.to_numpy()).tobytes()
            if signature in seen:
                continue
            seen.add(signature)
            evaluated += 1
            dev = route_metrics(dev_group[rule_mask].to_dict("records"))
            val = route_metrics(
                val_group[condition_mask(val_group, conditions)].to_dict("records"))
            if not (dev["n"] >= 10 and val["n"] >= 8
                    and dev["symbols"] >= 4 and val["symbols"] >= 4
                    and dev["wr"] >= required_wr and val["wr"] >= required_wr
                    and dev["net_r"] > 0 and val["net_r"] > 0
                    and dev["max_loss_streak"] <= 3
                    and val["max_loss_streak"] <= 3):
                continue
            score = (min(dev["mean_r"], val["mean_r"])
                     + .25 * (dev["mean_r"] + val["mean_r"])
                     + .0001 * min(dev["n"], val["n"]))
            found.append({
                "route": route,
                "conditions": [list(value) for value in conditions],
                "development": dev,
                "validation": val,
                "score": score,
            })
    found.sort(key=lambda item: item["score"], reverse=True)
    return found, evaluated


def supplement_rows(frame: pd.DataFrame, candidates: list[dict]) -> list[dict]:
    output = []
    for priority, candidate in enumerate(candidates):
        for row in select(frame, candidate).to_dict("records"):
            output.append({**row,
                           "module": f"shadow_stock_5m_regime_{priority + 1}",
                           "module_priority": 100 + priority,
                           "candidate_priority": priority})
    output.sort(key=lambda row: (
        row["candidate_priority"], row["entry_time"], row["symbol"]))
    unique = {}
    for row in output:
        unique.setdefault((row["symbol"], row["entry_time"]), row)
    return list(unique.values())


def accepted_supplement_metrics(production: list[dict], supplement: list[dict],
                                period: tuple[str, str]) -> dict:
    rows = stage_rows(combine(production, supplement), period)
    accepted = portfolio_gate(
        rows, None, rank_key=lambda row: int(row["module_priority"]))
    return route_metrics([
        row for row in accepted
        if str(row.get("module", "")).startswith("shadow_stock_5m_regime_")
    ])


def main() -> None:
    frame = prepare(pd.read_csv(ROWS_OUTPUT))
    production = frozen_rows(pd.read_csv(PRODUCTION_SOURCE), PROFILE)
    baseline = stages(production)
    raw_candidates, evaluated = discovery(frame)
    exact = []
    for candidate in raw_candidates[:500]:
        extra = supplement_rows(frame, [candidate])
        values = stages(combine(production, extra))
        if (improves(values["development"], baseline["development"], 4)
                and improves(values["validation"], baseline["validation"], 4)):
            exact.append({**candidate, "portfolio": {
                "development": values["development"],
                "validation": values["validation"],
            }, "exact_score": (
                values["development"]["return_pct"]
                - baseline["development"]["return_pct"]
                + 2 * (values["validation"]["return_pct"]
                       - baseline["validation"]["return_pct"])
            )})
    exact.sort(key=lambda item: item["exact_score"], reverse=True)
    frozen = []
    used = set()
    for candidate in exact:
        key = (candidate["route"]["family"], candidate["route"]["direction"])
        if key in used:
            continue
        frozen.append(candidate)
        used.add(key)
        if len(frozen) == 3:
            break
    supplement = supplement_rows(frame, frozen)
    pd.DataFrame(supplement).to_csv(SELECTED_OUTPUT, index=False)
    candidate_stages = stages(combine(production, supplement))
    accepted_supplement = {
        "development": accepted_supplement_metrics(
            production, supplement, DEVELOPMENT),
        "validation": accepted_supplement_metrics(
            production, supplement, VALIDATION),
        "hidden": accepted_supplement_metrics(
            production, supplement, HIDDEN),
    }
    hidden_raw = route_metrics(stage_rows(supplement, HIDDEN))
    hidden_accepted = accepted_supplement["hidden"]
    hidden_pass = bool(frozen) and (
        hidden_accepted["n"] >= 8 and hidden_accepted["wr"] >= 75
        and hidden_accepted["net_r"] > 0
        and improves(candidate_stages["hidden"], baseline["hidden"], 5))
    report = {
        "status": ("SHADOW_CANDIDATE_AWAITING_NEW_FORWARD"
                   if hidden_pass else "REJECTED"),
        "protocol": {
            "development": DEVELOPMENT,
            "validation": VALIDATION,
            "hidden_revealed_after_freeze": HIDDEN,
            "search": "one or two causal price/QQQ conditions per family/direction/target/stop",
            "entry": "next 5-minute market open proxy",
            "portfolio": "production priority and frozen stock portfolio gate",
            "money": "3% current-equity margin, 10x, gross",
        },
        "rules_evaluated": evaluated,
        "raw_qualified": len(raw_candidates),
        "exact_qualified_before_hidden": len(exact),
        "frozen_before_hidden": frozen,
        "baseline": baseline,
        "candidate": candidate_stages,
        "accepted_supplement": accepted_supplement,
        "hidden_supplement_raw": hidden_raw,
        "hidden_pass": hidden_pass,
        "activation_gate": {"new_forward_trades": 30,
                            "positive_new_months": 1, "met": False},
        "sources": {str(ROWS_OUTPUT): digest(ROWS_OUTPUT),
                    str(PRODUCTION_SOURCE): digest(PRODUCTION_SOURCE)},
        "limitations": [
            "Five-minute X-Perp history spans about sixty-two days.",
            "A finite two-condition search creates multiple-testing risk.",
            "Any pass remains shadow-only pending genuinely new forward data.",
            "Fees are excluded at the user's request.",
        ],
    }
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "rules_evaluated": evaluated,
        "raw_qualified": len(raw_candidates),
        "exact_qualified": len(exact),
        "frozen": [{"route": row["route"], "conditions": row["conditions"]}
                   for row in frozen],
        "baseline": baseline, "candidate": candidate_stages,
        "accepted_supplement": accepted_supplement,
        "hidden_raw": hidden_raw, "hidden_pass": hidden_pass,
    }, indent=2))


if __name__ == "__main__":
    main()
