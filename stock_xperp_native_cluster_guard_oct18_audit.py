"""Freeze a rolling-hour concentration guard for the native growth profile.

Only July 5 through October 6 is used to choose the guard.  October 7-16 and
the public-data forward snapshot are diagnostics after the policy is frozen.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from stock_module_cluster_audit import portfolio_gate
import stock_cash_xperp_leadlag_oct6_audit as lead
import stock_xperp_native_cross_section_oct18_audit as parent
import stock_xperp_native_economics_oct18_audit as economics
import stock_xperp_native_growth_profile_oct18_audit as growth


OUT = parent.ROOT / "stock_xperp_native_cluster_guard_oct18.json"
ROWS_OUT = parent.ROOT / "stock_xperp_native_cluster_guard_oct18.csv"
CANDIDATE_CAPS = (1, 2, 3, 4, 6)
MARGIN = .05
MAX_PRE_FRESH_DD_PCT = 8.0


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gate(frame: pd.DataFrame, bounds: tuple[str, str],
         cap: int | None) -> list[dict]:
    rows = lead.period_rows(frame, bounds).to_dict("records")
    return portfolio_gate(rows, None, rolling_hour_cap=cap)


def evaluate(frame: pd.DataFrame, bounds: tuple[str, str],
             cap: int | None) -> dict:
    return economics.simulate(gate(frame, bounds, cap), MARGIN)


def policy_values(frame: pd.DataFrame, cap: int | None) -> dict:
    periods = {
        **growth.COMPLETE_MONTHS,
        "pre_fresh": growth.PRE_FRESH,
        "fresh": parent.FRESH,
        "full": parent.FULL,
    }
    values = {name: evaluate(frame, bounds, cap)
              for name, bounds in periods.items()}
    fresh = values["fresh"]
    values["fresh_22_session_projection_pct"] = round(
        100 * ((1 + fresh["return_pct"] / 100) ** (22 / 10) - 1), 4)
    return values


def qualifies(values: dict, baseline: dict) -> bool:
    return (
        all(values[month]["return_pct"] >= 30
            for month in growth.COMPLETE_MONTHS)
        and values["pre_fresh"]["negative_months"] == 0
        and values["pre_fresh"]["dd_pct"] <= MAX_PRE_FRESH_DD_PCT
        and values["pre_fresh"]["return_pct"]
            >= baseline["pre_fresh"]["return_pct"]
    )


def main() -> None:
    profile = json.loads(growth.OUT.read_text(encoding="utf-8"))
    parameters = profile["selection"]["selected"]["parameters"]
    candles, sources = parent.load_sources()
    raw = growth.rebuild(parameters, candles, {})

    baseline = policy_values(raw, None)
    candidates = []
    for cap in CANDIDATE_CAPS:
        values = policy_values(raw, cap)
        candidates.append({
            "rolling_hour_cap": cap,
            "qualifies_pre_fresh": qualifies(values, baseline),
            "values": values,
        })
    survivors = [item for item in candidates if item["qualifies_pre_fresh"]]
    if not survivors:
        raise ValueError("no rolling-hour guard satisfied the frozen rule")
    # The smallest surviving cap is the strongest concentration guard.
    selected = min(survivors, key=lambda item: item["rolling_hour_cap"])
    cap = int(selected["rolling_hour_cap"])
    full_rows = gate(raw, parent.FULL, cap)
    pd.DataFrame(full_rows).to_csv(ROWS_OUT, index=False)

    before = baseline["pre_fresh"]
    after = selected["values"]["pre_fresh"]
    report = {
        "status": "PRE_FRESH_CLUSTER_GUARD_PASS_FRESH_INCOMPLETE",
        "ready_for_live_orders": False,
        "selection": {
            "selected_before_fresh": True,
            "candidate_caps": list(CANDIDATE_CAPS),
            "policy": (
                "choose the smallest finite rolling-60-minute cap with every "
                "complete Jul-Sep month >=30% at client 5% margin, no negative "
                "pre-fresh month, pre-fresh DD <=8%, and pre-fresh return no "
                "lower than the uncapped baseline"
            ),
            "selected_rolling_hour_cap": cap,
            "qualified_caps": [item["rolling_hour_cap"] for item in survivors],
            "parameters": parameters,
            "candidates": candidates,
            "baseline": baseline,
        },
        "selected_values": selected["values"],
        "effect_pre_fresh": {
            "return_pct_before": before["return_pct"],
            "return_pct_after": after["return_pct"],
            "dd_pct_before": before["dd_pct"],
            "dd_pct_after": after["dd_pct"],
            "trades_before": before["n"],
            "trades_after": after["n"],
        },
        "objective_checks": {
            "every_complete_pre_fresh_month_at_least_30pct": all(
                selected["values"][month]["return_pct"] >= 30
                for month in growth.COMPLETE_MONTHS),
            "pre_fresh_drawdown_at_most_8pct": after["dd_pct"] <= 8,
            "pre_fresh_return_not_lower_than_baseline":
                after["return_pct"] >= before["return_pct"],
            "fresh_positive": selected["values"]["fresh"]["return_pct"] > 0,
            "complete_fresh_month_available": False,
        },
        "sources": {
            **sources,
            str(Path(portfolio_gate.__code__.co_filename)):
                digest(Path(portfolio_gate.__code__.co_filename)),
            str(growth.OUT): digest(growth.OUT),
            str(parent.OUT): digest(parent.OUT),
            str(Path(__file__)): digest(Path(__file__)),
        },
        "limitations": [
            "October 7-16 is diagnostic and is not used to choose the cap.",
            "October is incomplete; its projection is not proof of a full month.",
            "Five-percent margin is a client choice and raises drawdown.",
            "Native stock X-Perp history begins in 2026.",
            "Gross returns exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "selected_rolling_hour_cap": cap,
        "qualified_caps": report["selection"]["qualified_caps"],
        "complete_months": {
            month: selected["values"][month]
            for month in growth.COMPLETE_MONTHS
        },
        "pre_fresh": after,
        "fresh": selected["values"]["fresh"],
        "fresh_projection": selected["values"]["fresh_22_session_projection_pct"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
