"""Freeze causal ranking and a daily loss-streak guard for native X-Perps.

Candidates are screened on July-August and must confirm in September.  The
October 7-16 slice is revealed only for the selected policy.  The guard uses
only closed outcomes and resets at the next UTC day.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

import pandas as pd

from stock_module_cluster_audit import portfolio_gate
import stock_cash_xperp_leadlag_oct6_audit as lead
import stock_xperp_native_cross_section_oct18_audit as parent
import stock_xperp_native_economics_oct18_audit as economics
import stock_xperp_native_growth_profile_oct18_audit as growth
import stock_xperp_native_volatility_guard_oct18_audit as volatility


OUT = parent.ROOT / "stock_xperp_native_quality_guard_oct18.json"
ROWS_OUT = parent.ROOT / "stock_xperp_native_quality_guard_oct18.csv"
RANKERS = ("symbol", "z_high", "move_high", "atr_low", "market_support_high")
LOSS_STREAK_CAPS = (1, 2, 3)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ranker(name: str) -> Callable[[dict], object] | None:
    if name == "symbol":
        return None
    if name == "z_high":
        return lambda row: -abs(float(row["cross_z"]))
    if name == "move_high":
        return lambda row: -abs(float(row["signal_return"]))
    if name == "atr_low":
        return lambda row: float(row["atr14_fraction"])
    if name == "market_support_high":
        return lambda row: -float(row["market_support"])
    raise ValueError(f"unknown ranker: {name}")


def prepare() -> tuple[pd.DataFrame, dict, dict]:
    profile = json.loads(growth.OUT.read_text(encoding="utf-8"))
    parameters = profile["selection"]["selected"]["parameters"]
    volatility_report = json.loads(volatility.OUT.read_text(encoding="utf-8"))
    threshold = float(volatility_report["selection"][
        "selected_maximum_atr14_fraction"])
    candles, sources = parent.load_sources()
    raw = volatility.add_atr14(
        growth.rebuild(parameters, candles, {}), candles)
    raw = raw[raw.atr14_fraction.le(threshold)].copy()
    raw["market_support"] = raw.direction_n * raw.cross_mean
    return raw, parameters, sources


def gated(frame: pd.DataFrame, bounds: tuple[str, str], rank_name: str,
          loss_streak_cap: int) -> list[dict]:
    rows = lead.period_rows(frame, bounds).to_dict("records")
    return portfolio_gate(
        rows, None, rank_key=ranker(rank_name),
        loss_streak_cap=loss_streak_cap)


def evaluate(frame: pd.DataFrame, bounds: tuple[str, str], rank_name: str,
             loss_streak_cap: int, margin: float = .05) -> dict:
    return economics.simulate(
        gated(frame, bounds, rank_name, loss_streak_cap), margin)


def period_values(frame: pd.DataFrame, rank_name: str,
                  loss_streak_cap: int) -> dict:
    return {
        month: evaluate(frame, bounds, rank_name, loss_streak_cap)
        for month, bounds in growth.COMPLETE_MONTHS.items()
    }


def selection_ok(values: dict) -> bool:
    minimum_trades = {"2026-07": 81, "2026-08": 93}
    return all(
        values[month]["return_pct"] >= 30
        and values[month]["dd_pct"] <= 8
        and values[month]["n"] >= minimum_trades[month]
        for month in ("2026-07", "2026-08")
    )


def confirmation_ok(values: dict) -> bool:
    september = values["2026-09"]
    return (september["return_pct"] >= 30
            and september["dd_pct"] <= 8
            and september["n"] >= 90)


def outcome_diagnostics(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0, "profitable_exit_rate_pct": 0.0,
                "literal_stop_rate_pct": 0.0}
    return {
        "n": len(rows),
        "profitable_exit_rate_pct": round(
            100 * sum(float(row["signed_return"]) > 0 for row in rows)
            / len(rows), 4),
        "literal_stop_rate_pct": round(
            100 * sum(int(row["outcome_code"]) == -1 for row in rows)
            / len(rows), 4),
    }


def main() -> None:
    frame, parameters, sources = prepare()
    candidates = []
    for loss_cap in LOSS_STREAK_CAPS:
        for rank_name in RANKERS:
            values = period_values(frame, rank_name, loss_cap)
            candidates.append({
                "ranker": rank_name,
                "loss_streak_cap": loss_cap,
                "selection_ok": selection_ok(values),
                "confirmation_ok": confirmation_ok(values),
                "complete_months": values,
            })
    survivors = [item for item in candidates
                 if item["selection_ok"] and item["confirmation_ok"]]
    if not survivors:
        raise ValueError("no causal quality guard survived confirmation")
    # Safety first: choose the earliest daily pause that retains the objective.
    survivors.sort(key=lambda item: (
        item["loss_streak_cap"],
        -min(item["complete_months"][month]["return_pct"]
             for month in ("2026-07", "2026-08")),
        -item["complete_months"]["2026-09"]["return_pct"],
        item["ranker"],
    ))
    selected = survivors[0]
    rank_name = selected["ranker"]
    loss_cap = int(selected["loss_streak_cap"])

    full_rows = gated(frame, parent.FULL, rank_name, loss_cap)
    pd.DataFrame(full_rows).to_csv(ROWS_OUT, index=False)
    periods = {
        **growth.COMPLETE_MONTHS,
        "pre_fresh": growth.PRE_FRESH,
        "fresh": parent.FRESH,
        "full": parent.FULL,
    }
    scenarios = {
        f"margin_{round(margin * 100)}pct": {
            name: evaluate(frame, bounds, rank_name, loss_cap, margin)
            for name, bounds in periods.items()
        }
        for margin in (.03, .05)
    }
    fresh = scenarios["margin_5pct"]["fresh"]
    fresh_projection = round(
        100 * ((1 + fresh["return_pct"] / 100) ** (22 / 10) - 1), 4)
    diagnostics = {
        name: outcome_diagnostics(gated(frame, bounds, rank_name, loss_cap))
        for name, bounds in periods.items()
    }
    report = {
        "status": "QUALITY_GUARD_CONFIRMED_FRESH_MONTH_INCOMPLETE",
        "ready_for_live_orders": False,
        "selection": {
            "selected_without_fresh": True,
            "screened_on": "July-August 2026",
            "confirmed_on": "September 2026",
            "policy": (
                "among policies with every complete month >=30%, DD <=8%, "
                "and >=3 trades per day, choose the smallest causal daily "
                "loss-streak cap; rank simultaneous signals using closed data"
            ),
            "selected_ranker": rank_name,
            "selected_loss_streak_cap": loss_cap,
            "parameters": parameters,
            "candidates": candidates,
        },
        "margin_scenarios": scenarios,
        "diagnostics": diagnostics,
        "fresh_22_session_projection_pct": fresh_projection,
        "objective_checks": {
            "every_complete_month_at_least_30pct": all(
                scenarios["margin_5pct"][month]["return_pct"] >= 30
                for month in growth.COMPLETE_MONTHS),
            "every_complete_month_drawdown_at_most_8pct": all(
                scenarios["margin_5pct"][month]["dd_pct"] <= 8
                for month in growth.COMPLETE_MONTHS),
            "fresh_positive": fresh["return_pct"] > 0,
            "fresh_projection_at_least_30pct": fresh_projection >= 30,
            "complete_fresh_month_available": False,
        },
        "sources": {
            **sources,
            str(Path(portfolio_gate.__code__.co_filename)):
                digest(Path(portfolio_gate.__code__.co_filename)),
            str(volatility.OUT): digest(volatility.OUT),
            str(growth.OUT): digest(growth.OUT),
            str(Path(__file__)): digest(Path(__file__)),
        },
        "limitations": [
            "The base route parameters used September before this guard audit.",
            "October 7-16 is ten calendar days, not a complete month.",
            "The 22-session projection is context, not proof.",
            "Five-percent margin is a client choice and raises drawdown.",
            "Native stock X-Perp history begins in 2026.",
            "Gross returns exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "ranker": rank_name,
        "loss_streak_cap": loss_cap,
        "complete_months": {
            month: scenarios["margin_5pct"][month]
            for month in growth.COMPLETE_MONTHS
        },
        "fresh": fresh,
        "fresh_projection": fresh_projection,
        "checks": report["objective_checks"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
