"""Select a closed-bar volatility guard for the native growth profile.

The ATR threshold is selected on July and August only.  September is a
confirmation slice for this guard, and October 7-16 is reported after the
selection is frozen.  All features are known before the next-bar market entry.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from stock_module_cluster_audit import portfolio_gate
import stock_cash_xperp_leadlag_oct6_audit as lead
import stock_xperp_native_cross_section_oct18_audit as parent
import stock_xperp_native_economics_oct18_audit as economics
import stock_xperp_native_growth_profile_oct18_audit as growth


OUT = parent.ROOT / "stock_xperp_native_volatility_guard_oct18.json"
ROWS_OUT = parent.ROOT / "stock_xperp_native_volatility_guard_oct18.csv"
THRESHOLDS = (.0035, .004, .005, .006, .0075, .01, .0125, .015, .02)
SELECTION_PERIODS = {
    "2026-07": growth.COMPLETE_MONTHS["2026-07"],
    "2026-08": growth.COMPLETE_MONTHS["2026-08"],
}
MIN_TRADES = {"2026-07": 81, "2026-08": 93}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def add_atr14(frame: pd.DataFrame, candles: dict) -> pd.DataFrame:
    """Attach the 14-bar normalized ATR from bars closed before entry."""
    output = []
    for row in frame.to_dict("records"):
        candle = candles[row["symbol"]]
        index = int(row["entry_index"]) - 1
        if index < 14:
            continue
        high = np.asarray(candle["high"], dtype=float)
        low = np.asarray(candle["low"], dtype=float)
        close = np.asarray(candle["close"], dtype=float)
        current_high = high[index - 13:index + 1]
        current_low = low[index - 13:index + 1]
        previous_close = close[index - 14:index]
        true_range = np.maximum(
            current_high - current_low,
            np.maximum(abs(current_high - previous_close),
                       abs(current_low - previous_close)))
        row["atr14_fraction"] = float(np.mean(true_range) / close[index])
        output.append(row)
    return pd.DataFrame(output)


def gated(frame: pd.DataFrame, bounds: tuple[str, str]) -> list[dict]:
    rows = lead.period_rows(frame, bounds).to_dict("records")
    return portfolio_gate(rows, None)


def evaluate(frame: pd.DataFrame, bounds: tuple[str, str],
             margin: float = .05) -> dict:
    return economics.simulate(gated(frame, bounds), margin)


def selection_values(frame: pd.DataFrame) -> dict:
    return {name: evaluate(frame, bounds)
            for name, bounds in SELECTION_PERIODS.items()}


def qualifies(values: dict) -> bool:
    return all(
        values[name]["return_pct"] >= 30
        and values[name]["dd_pct"] <= 8
        and values[name]["n"] >= MIN_TRADES[name]
        for name in SELECTION_PERIODS
    )


def main() -> None:
    profile = json.loads(growth.OUT.read_text(encoding="utf-8"))
    parameters = profile["selection"]["selected"]["parameters"]
    candles, sources = parent.load_sources()
    raw = add_atr14(growth.rebuild(parameters, candles, {}), candles)

    candidates = []
    for threshold in THRESHOLDS:
        filtered = raw[raw.atr14_fraction.le(threshold)]
        values = selection_values(filtered)
        candidates.append({
            "maximum_atr14_fraction": threshold,
            "qualifies_selection": qualifies(values),
            "selection_values": values,
        })
    survivors = [item for item in candidates if item["qualifies_selection"]]
    if not survivors:
        raise ValueError("no volatility threshold satisfied the selection rule")
    survivors.sort(key=lambda item: (
        -min(value["return_pct"]
             for value in item["selection_values"].values()),
        -sum(value["return_pct"]
             for value in item["selection_values"].values()),
        item["maximum_atr14_fraction"],
    ))
    selected = survivors[0]
    threshold = float(selected["maximum_atr14_fraction"])
    filtered = raw[raw.atr14_fraction.le(threshold)]
    full_rows = gated(filtered, parent.FULL)
    pd.DataFrame(full_rows).to_csv(ROWS_OUT, index=False)

    periods = {
        **growth.COMPLETE_MONTHS,
        "pre_fresh": growth.PRE_FRESH,
        "fresh": parent.FRESH,
        "full": parent.FULL,
    }
    scenarios = {
        f"margin_{round(margin * 100)}pct": {
            name: evaluate(filtered, bounds, margin)
            for name, bounds in periods.items()
        }
        for margin in (.03, .05)
    }
    fresh = scenarios["margin_5pct"]["fresh"]
    fresh_projection = round(
        100 * ((1 + fresh["return_pct"] / 100) ** (22 / 10) - 1), 4)
    confirmation = scenarios["margin_5pct"]["2026-09"]
    report = {
        "status": "VOLATILITY_GUARD_CONFIRMED_FRESH_MONTH_INCOMPLETE",
        "ready_for_live_orders": False,
        "selection": {
            "threshold_selected_on": "July-August 2026 only",
            "candidate_thresholds": list(THRESHOLDS),
            "policy": (
                "require each selection month >=30% at client 5% margin, "
                "DD <=8%, and >=3 trades per calendar day; maximize the "
                "weaker monthly return, then combined return"
            ),
            "selected_maximum_atr14_fraction": threshold,
            "parameters": parameters,
            "candidates": candidates,
        },
        "margin_scenarios": scenarios,
        "fresh_22_session_projection_pct": fresh_projection,
        "objective_checks": {
            "selected_without_september_or_fresh": True,
            "september_at_least_30pct": confirmation["return_pct"] >= 30,
            "september_drawdown_at_most_8pct": confirmation["dd_pct"] <= 8,
            "fresh_positive": fresh["return_pct"] > 0,
            "fresh_projection_at_least_30pct": fresh_projection >= 30,
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
            "The base route parameters predate this guard but used September.",
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
        "threshold": threshold,
        "selection": selected["selection_values"],
        "confirmation": confirmation,
        "fresh": fresh,
        "fresh_projection": fresh_projection,
        "checks": report["objective_checks"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
