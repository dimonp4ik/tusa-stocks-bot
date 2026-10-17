"""Select a high-growth native stock X-Perp profile before October 7.

The shortlist comes from the parent pre-fresh search.  This audit keeps only
literal targets no larger than 2%, requires every complete July-September
month to exceed 30% at a client-selected 5% margin, caps pre-fresh drawdown at
10%, and ranks survivors by pre-fresh return.  October 7-16 is not used for
selection.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

import stock_cash_xperp_leadlag_oct6_audit as lead
import stock_xperp_native_cross_section_oct18_audit as parent
import stock_xperp_native_economics_oct18_audit as economics


OUT = parent.ROOT / "stock_xperp_native_growth_profile_oct18.json"
ROWS_OUT = parent.ROOT / "stock_xperp_native_growth_profile_oct18.csv"
PRE_FRESH = ("2026-07-05", "2026-10-07")
COMPLETE_MONTHS = {
    "2026-07": ("2026-07-05", "2026-08-01"),
    "2026-08": ("2026-08-01", "2026-09-01"),
    "2026-09": ("2026-09-01", "2026-10-01"),
}
GROWTH_MARGIN = .05


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def accepted(frame: pd.DataFrame, bounds: tuple[str, str]) -> list[dict]:
    return lead.accepted(lead.period_rows(frame, bounds).to_dict("records"))


def rebuild(item: dict, candles: dict,
            features: dict[int, pd.DataFrame]) -> pd.DataFrame:
    lookback = int(item["lookback_bars"])
    features.setdefault(lookback, parent.feature_rows(candles, lookback))
    entries = parent.select_entries(
        features[lookback], float(item["z_threshold"]),
        float(item["minimum_move_pct"]) / 100)
    rows = lead.add_outcomes(
        entries, candles, float(item["target_pct"]),
        float(item["stop_pct"]), int(item["hold_bars"]))
    rows["module"] = "native_xperp_growth_profile"
    rows["module_priority"] = 0
    return rows


def qualifies(months: dict, pre_fresh: dict) -> bool:
    return (all(value["return_pct"] >= 30 for value in months.values())
            and pre_fresh["dd_pct"] <= 10
            and pre_fresh["negative_months"] == 0)


def margin_values(frame: pd.DataFrame, margin: float) -> dict:
    periods = {**COMPLETE_MONTHS, "pre_fresh": PRE_FRESH,
               "fresh": parent.FRESH, "full": parent.FULL}
    values = {
        name: economics.simulate(accepted(frame, bounds), margin)
        for name, bounds in periods.items()
    }
    values["fresh_22_session_projection_pct"] = round(
        100 * ((1 + values["fresh"]["return_pct"] / 100) ** (22 / 10) - 1), 4)
    return values


def public(item: dict) -> dict:
    return {key: value for key, value in item.items() if key != "_rows"}


def main() -> None:
    parent_report = json.loads(parent.OUT.read_text(encoding="utf-8"))
    candles, sources = parent.load_sources()
    features, survivors = {}, []
    for parameters in parent_report["top20"]:
        if float(parameters["target_pct"]) > 2:
            continue
        rows = rebuild(parameters, candles, features)
        growth = margin_values(rows, GROWTH_MARGIN)
        months = {name: growth[name] for name in COMPLETE_MONTHS}
        if not qualifies(months, growth["pre_fresh"]):
            continue
        survivors.append({
            "parameters": {key: parameters[key] for key in (
                "lookback_bars", "z_threshold", "minimum_move_pct",
                "target_pct", "stop_pct", "hold_bars", "score")},
            "growth_margin": growth,
            "_rows": rows,
        })
    survivors.sort(key=lambda item: (
        -item["growth_margin"]["pre_fresh"]["return_pct"],
        item["growth_margin"]["pre_fresh"]["dd_pct"]))
    if not survivors:
        raise ValueError("no pre-fresh high-growth profile survived")
    selected = survivors[0]
    rows = selected["_rows"]
    full_accepted = accepted(rows, parent.FULL)
    full_frame = pd.DataFrame(full_accepted)
    full_frame.to_csv(ROWS_OUT, index=False)
    scenarios = {
        f"margin_{round(100 * margin)}pct": margin_values(rows, margin)
        for margin in economics.MARGINS
    }
    three_pct = {
        "development": lead.evaluate(rows, parent.DEVELOPMENT),
        "validation": lead.evaluate(rows, parent.VALIDATION),
        "confirmation": lead.evaluate(rows, parent.CONFIRMATION),
        "fresh": lead.evaluate(rows, parent.FRESH),
        "full": lead.evaluate(rows, parent.FULL),
    }
    five = scenarios["margin_5pct"]
    checks = {
        "selected_before_fresh": True,
        "target_at_most_2pct":
            selected["parameters"]["target_pct"] <= 2,
        "every_complete_pre_fresh_month_at_least_30pct": all(
            five[month]["return_pct"] >= 30 for month in COMPLETE_MONTHS),
        "pre_fresh_drawdown_at_most_10pct":
            five["pre_fresh"]["dd_pct"] <= 10,
        "fresh_positive": five["fresh"]["return_pct"] > 0,
        "fresh_projection_at_least_30pct":
            five["fresh_22_session_projection_pct"] >= 30,
        "complete_fresh_month_available": False,
    }
    report = {
        "status": "PRE_FRESH_MONTHLY_OBJECTIVE_PASS_FRESH_INCOMPLETE",
        "ready_for_live_orders": False,
        "selection": {
            "shortlist_source": "parent pre-fresh top20",
            "qualified_profiles": len(survivors),
            "policy": (
                "target <=2%; every complete Jul-Sep month >=30% at client "
                "5% margin; pre-fresh DD <=10%; maximize pre-fresh return"),
            "selected": public(selected),
            "all_survivors": [public(item) for item in survivors],
        },
        "three_pct_reference": three_pct,
        "margin_scenarios": scenarios,
        "diagnostics": economics.diagnostics(full_frame),
        "objective_checks": checks,
        "sources": {
            **sources, str(parent.OUT): digest(parent.OUT),
            str(Path(__file__)): digest(Path(__file__)),
        },
        "limitations": [
            "Five-percent margin is a client choice and raises drawdown.",
            "October is incomplete and its projection is not proof.",
            "Native stock X-Perp history begins in 2026.",
            "The fresh period has already been observed in earlier research; it is diagnostic here.",
            "Gross returns exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "parameters": selected["parameters"],
        "qualified_profiles": len(survivors),
        "three_pct": three_pct, "five_pct": five,
        "checks": checks,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
