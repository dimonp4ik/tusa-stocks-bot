"""Causal OI audit for an independent stock X-Perp shadow supplement.

The rule library is selected on an early direct-X-Perp block, checked on a
later block, and only then revealed on September/October.  Open interest is
lagged by one hour and measured in contracts, so the filter cannot use an
incomplete snapshot or confuse a price move with an OI move.
"""
from __future__ import annotations

from bisect import bisect_right
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from stock_forward_monitor import frozen_rows
from stock_module_cluster_audit import portfolio_gate
from src.stock_venue_router import (
    ROBUST_FREQUENCY_EXCLUDED_SYMBOLS,
    _session_bucket,
)


ROOT = Path("reports/audit_2026_09_08")
SOURCE = ROOT / "stock_forward_oct3_families.csv"
OI_SOURCE = ROOT / "stock_xperp_open_interest_1h.json"
OUTPUT = ROOT / "stock_xperp_oi_supplement_oct3.json"
ROWS_OUTPUT = ROOT / "stock_xperp_oi_supplement_oct3_rows.csv"
PROFILE = "robust_dynamic_entry_filtered"
TARGET_R = .25
MARGIN_FRACTION = .03
LEVERAGE = 10.0
DEVELOPMENT = ("2026-08-04", "2026-08-20")
VALIDATION = ("2026-08-20", "2026-09-05")
HIDDEN = ("2026-09-05", "2026-10-03")
OI_HORIZONS = (6, 24, 72, 168)
ROUTE_KEYS = ("family", "direction")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cut(frame: pd.DataFrame, period: tuple[str, str]) -> pd.DataFrame:
    start = pd.Timestamp(period[0], tz="UTC").timestamp()
    end = pd.Timestamp(period[1], tz="UTC").timestamp()
    return frame[(frame["entry_time"] >= start) & (frame["entry_time"] < end)]


def load_oi() -> dict[str, tuple[list[int], list[float]]]:
    payload = json.loads(OI_SOURCE.read_text(encoding="utf-8"))
    output = {}
    for symbol, value in payload["symbols"].items():
        rows = value["rows"]
        output[symbol] = (
            [int(row[0]) // 1000 for row in rows],
            [float(row[1]) for row in rows],
        )
    return output


def oi_changes(series: tuple[list[int], list[float]], entry_time: float) -> dict:
    times, values = series
    index = bisect_right(times, int(entry_time) - 3600) - 1
    result = {}
    for horizon in OI_HORIZONS:
        key = f"oi_change_{horizon}h"
        if index < horizon or values[index - horizon] <= 0:
            result[key] = math.nan
        else:
            result[key] = values[index] / values[index - horizon] - 1
    return result


def add_oi_context(frame: pd.DataFrame) -> pd.DataFrame:
    series = load_oi()
    cache: dict[tuple[str, int], dict] = {}

    def features(row: pd.Series) -> pd.Series:
        symbol, entry = str(row["symbol"]), int(row["entry_time"])
        key = (symbol, entry)
        if key not in cache:
            own = oi_changes(series[symbol], entry)
            qqq = oi_changes(series["QQQUSDT"], entry)
            values = {}
            for horizon in OI_HORIZONS:
                own_key = f"oi_change_{horizon}h"
                values[f"own_{own_key}"] = own[own_key]
                values[f"qqq_{own_key}"] = qqq[own_key]
                values[f"relative_{own_key}"] = own[own_key] - qqq[own_key]
            cache[key] = values
        return pd.Series(cache[key])

    return pd.concat([frame.reset_index(drop=True),
                      frame.apply(features, axis=1).reset_index(drop=True)], axis=1)


def raw_metrics(frame: pd.DataFrame) -> dict:
    pnl = pd.to_numeric(frame["net_r"], errors="coerce").dropna().to_numpy(float)
    return {
        "n": int(len(pnl)),
        "wr": round(float(100 * np.mean(pnl > 0)), 2) if len(pnl) else 0.0,
        "net_r": round(float(pnl.sum()), 6),
        "mean_r": round(float(pnl.mean()), 6) if len(pnl) else 0.0,
        "symbols": int(frame["symbol"].nunique()) if len(frame) else 0,
    }


def condition_mask(frame: pd.DataFrame, conditions: tuple[tuple, ...]) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for field, operator, threshold in conditions:
        values = pd.to_numeric(frame[field], errors="coerce")
        mask &= values.ge(threshold) if operator == "ge" else values.le(threshold)
    return mask.fillna(False)


def select_rule(frame: pd.DataFrame, candidate: dict) -> pd.DataFrame:
    mask = pd.Series(True, index=frame.index)
    for key, value in candidate["route"].items():
        mask &= frame[key] == value
    conditions = tuple(tuple(value) for value in candidate["conditions"])
    return frame[mask & condition_mask(frame, conditions)]


def atoms(development: pd.DataFrame) -> list[tuple]:
    fields = [f"{prefix}_oi_change_{horizon}h"
              for prefix in ("own", "qqq", "relative")
              for horizon in OI_HORIZONS]
    result = []
    for field in fields:
        values = pd.to_numeric(development[field], errors="coerce").dropna()
        if len(values) < 20:
            continue
        for threshold in sorted(set(float(value) for value in
                                    values.quantile([.2, .35, .5, .65, .8]))):
            result.append((field, "ge", threshold))
            result.append((field, "le", threshold))
    return result


def discover(development: pd.DataFrame, validation: pd.DataFrame) -> tuple[list[dict], int]:
    public_atoms = atoms(development)
    found = []
    evaluated = 0
    for route_values, group in development.groupby(list(ROUTE_KEYS), observed=True,
                                                     sort=True):
        route = dict(zip(ROUTE_KEYS, route_values))
        validation_group = validation.copy()
        for key, value in route.items():
            validation_group = validation_group[validation_group[key] == value]
        if len(group) < 10 or len(validation_group) < 8:
            continue
        ranked = []
        for atom in public_atoms:
            chosen = group[condition_mask(group, (atom,))]
            if len(chosen) >= 8:
                value = raw_metrics(chosen)
                ranked.append((value["mean_r"] + value["wr"] / 10000, atom))
        strongest = [atom for _, atom in sorted(ranked, reverse=True)[:14]]
        rules = [(atom,) for atom in public_atoms]
        rules += [pair for pair in itertools.combinations(strongest, 2)
                  if pair[0][0] != pair[1][0]]
        seen = set()
        for conditions in rules:
            rule_mask = condition_mask(group, conditions)
            chosen = group[rule_mask]
            signature = np.packbits(rule_mask.to_numpy()).tobytes()
            if signature in seen:
                continue
            seen.add(signature)
            evaluated += 1
            development_value = raw_metrics(chosen)
            validation_value = raw_metrics(
                validation_group[condition_mask(validation_group, conditions)])
            if not (development_value["n"] >= 8
                    and validation_value["n"] >= 6
                    and development_value["symbols"] >= 3
                    and validation_value["symbols"] >= 3
                    and development_value["wr"] >= 82
                    and validation_value["wr"] >= 82
                    and development_value["net_r"] > 0
                    and validation_value["net_r"] > 0):
                continue
            score = (min(development_value["mean_r"], validation_value["mean_r"])
                     + .25 * (development_value["mean_r"]
                              + validation_value["mean_r"])
                     + .0001 * min(development_value["n"], validation_value["n"]))
            found.append({
                "route": route,
                "conditions": [list(value) for value in conditions],
                "development": development_value,
                "validation": validation_value,
                "score": score,
            })
    found.sort(key=lambda item: item["score"], reverse=True)
    return found, evaluated


def supplement_rows(frame: pd.DataFrame, candidates: list[dict]) -> list[dict]:
    selected = []
    for priority, candidate in enumerate(candidates):
        rows = select_rule(frame, candidate).copy()
        rows["module"] = f"shadow_stock_oi_{priority + 1}"
        rows["module_priority"] = 100 + priority
        rows["oi_candidate_priority"] = priority
        selected.append(rows)
    if not selected:
        return []
    combined = pd.concat(selected, ignore_index=True)
    combined = combined.sort_values(["oi_candidate_priority", "entry_time", "symbol"])
    combined = combined.drop_duplicates(["symbol", "entry_time"], keep="first")
    return combined.to_dict("records")


def combine(production: list[dict], supplement: list[dict]) -> list[dict]:
    rows = pd.concat([pd.DataFrame(production), pd.DataFrame(supplement)],
                     ignore_index=True)
    rows = rows.sort_values(["module_priority", "entry_time", "symbol"])
    return rows.drop_duplicates(["symbol", "entry_time"], keep="first").to_dict("records")


def money(rows: list[dict]) -> dict:
    accepted = portfolio_gate(rows, None,
                              rank_key=lambda row: int(row["module_priority"]))
    if not accepted:
        return {"n": 0, "wr": 0.0, "return_pct": 0.0, "dd_pct": 0.0,
                "negative_months": 0, "max_loss_streak": 0,
                "trades_per_calendar_day": 0.0, "trades_per_active_day": 0.0}
    ordered = sorted(accepted, key=lambda row: (
        float(row["exit_time"]), float(row["entry_time"]), str(row["symbol"])))
    equity = peak = 1.0
    drawdown = 0.0
    wins = streak = max_streak = 0
    monthly_start: dict[str, float] = {}
    monthly_end: dict[str, float] = {}
    active_days = set()
    for row in ordered:
        exit_stamp = pd.Timestamp(float(row["exit_time"]), unit="s", tz="UTC")
        month = exit_stamp.strftime("%Y-%m")
        monthly_start.setdefault(month, equity)
        equity *= (1 + MARGIN_FRACTION * LEVERAGE
                   * float(row["risk_pct"]) * float(row["net_r"]))
        monthly_end[month] = equity
        peak = max(peak, equity)
        drawdown = max(drawdown, (peak - equity) / peak)
        won = float(row["net_r"]) > 0
        wins += int(won)
        streak = 0 if won else streak + 1
        max_streak = max(max_streak, streak)
        active_days.add(pd.Timestamp(float(row["entry_time"]), unit="s", tz="UTC")
                        .strftime("%Y-%m-%d"))
    first = pd.Timestamp(min(float(row["entry_time"]) for row in accepted),
                         unit="s", tz="UTC")
    last = pd.Timestamp(max(float(row["entry_time"]) for row in accepted),
                        unit="s", tz="UTC")
    calendar_days = max(1, (last.date() - first.date()).days + 1)
    monthly = {key: round(100 * (monthly_end[key] / monthly_start[key] - 1), 4)
               for key in sorted(monthly_start)}
    return {
        "n": len(accepted), "wr": round(100 * wins / len(accepted), 2),
        "return_pct": round(100 * (equity - 1), 4),
        "dd_pct": round(100 * drawdown, 4),
        "negative_months": sum(value < 0 for value in monthly.values()),
        "monthly_return_pct": monthly,
        "max_loss_streak": max_streak,
        "active_days": len(active_days),
        "trades_per_calendar_day": round(len(accepted) / calendar_days, 3),
        "trades_per_active_day": round(len(accepted) / len(active_days), 3),
    }


def stage_rows(rows: list[dict], period: tuple[str, str]) -> list[dict]:
    start = pd.Timestamp(period[0], tz="UTC").timestamp()
    end = pd.Timestamp(period[1], tz="UTC").timestamp()
    return [row for row in rows if start <= float(row["entry_time"]) < end]


def stages(production: list[dict], supplement: list[dict]) -> dict:
    combined = combine(production, supplement) if supplement else production
    return {name: money(stage_rows(combined, period)) for name, period in (
        ("development", DEVELOPMENT), ("validation", VALIDATION), ("hidden", HIDDEN))}


def improves(value: dict, baseline: dict, minimum_added: int) -> bool:
    return (value["n"] >= baseline["n"] + minimum_added
            and value["return_pct"] > baseline["return_pct"]
            and value["wr"] >= max(75.0, baseline["wr"] - 5.0)
            and value["dd_pct"] <= baseline["dd_pct"] + .35
            and value["max_loss_streak"] <= baseline["max_loss_streak"] + 1)


def main() -> None:
    frame = pd.read_csv(SOURCE)
    frame = frame[(frame["target_r"] - TARGET_R).abs() < 1e-9].copy()
    frame = frame[~frame["symbol"].isin(ROBUST_FREQUENCY_EXCLUDED_SYMBOLS)].copy()
    frame["session_bucket"] = frame["signal_minutes"].map(_session_bucket)
    frame = add_oi_context(frame)
    frame = frame.dropna(subset=["own_oi_change_24h", "qqq_oi_change_24h"])
    development, validation = cut(frame, DEVELOPMENT), cut(frame, VALIDATION)
    production = frozen_rows(pd.read_csv(SOURCE), PROFILE)
    baseline = stages(production, [])
    raw_candidates, evaluated = discover(development, validation)

    exact = []
    for candidate in raw_candidates[:300]:
        extra = supplement_rows(frame, [candidate])
        values = stages(production, extra)
        if (improves(values["development"], baseline["development"], 3)
                and improves(values["validation"], baseline["validation"], 3)):
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
    frozen = exact[:3]
    supplement = supplement_rows(frame, frozen)
    candidate_stages = stages(production, supplement)
    hidden_pass = bool(frozen) and improves(
        candidate_stages["hidden"], baseline["hidden"], 5)
    status = ("SHADOW_CANDIDATE_AWAITING_NEW_FORWARD"
              if hidden_pass else "REJECTED")

    if supplement:
        pd.DataFrame(supplement).to_csv(ROWS_OUTPUT, index=False)
    else:
        ROWS_OUTPUT.write_text("", encoding="utf-8")
    report = {
        "status": status,
        "protocol": {
            "development": DEVELOPMENT,
            "validation": VALIDATION,
            "hidden_revealed_after_freeze": HIDDEN,
            "target_r": TARGET_R,
            "entry": "next 15-minute market open",
            "oi": "X-Perp contracts, newest usable snapshot lagged one hour",
            "money": "3% of current equity margin, 10x, gross",
            "production_priority": True,
        },
        "rows_with_oi": len(frame),
        "rules_evaluated": evaluated,
        "raw_qualified": len(raw_candidates),
        "exact_qualified_before_hidden": len(exact),
        "frozen_before_hidden": frozen,
        "baseline": baseline,
        "candidate": candidate_stages,
        "hidden_pass": hidden_pass,
        "activation_gate": {
            "required_new_forward_trades": 30,
            "required_positive_new_months": 1,
            "met": False,
        },
        "sources": {str(SOURCE): digest(SOURCE), str(OI_SOURCE): digest(OI_SOURCE)},
        "limitations": [
            "Public venue OI covers only about sixty days.",
            "A finite rule search creates multiple-testing risk.",
            "Any passing result remains shadow-only until genuinely new data pass the gate.",
            "Fees are excluded at the user's request.",
        ],
    }
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": status, "rules_evaluated": evaluated,
        "raw_qualified": len(raw_candidates),
        "exact_qualified": len(exact),
        "frozen": [{"route": row["route"], "conditions": row["conditions"]}
                   for row in frozen],
        "baseline": baseline, "candidate": candidate_stages,
        "hidden_pass": hidden_pass,
    }, indent=2))


if __name__ == "__main__":
    main()
