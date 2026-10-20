"""Audit a causal rolling-hour entry cap on the frozen backing-level route."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from stock_module_cluster_audit import portfolio_gate
import stock_cash_xperp_leadlag_oct6_audit as lead
import stock_xperp_backing_level_guard_oct20_audit as backing
import stock_xperp_native_cross_section_oct18_audit as parent
import stock_xperp_native_economics_oct18_audit as economics
import stock_xperp_native_growth_profile_oct18_audit as growth
import stock_xperp_native_quality_guard_oct18_audit as quality


OUT = parent.ROOT / "stock_xperp_backing_hour_cap_oct20.json"
ROWS_OUT = parent.ROOT / "stock_xperp_backing_hour_cap_oct20.csv"
CAPS = (2, 3, 4, 5, 6)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows_for(frame: pd.DataFrame, bounds: tuple[str, str], rank_name: str,
             loss_cap: int, rolling_hour_cap: int) -> list[dict]:
    rows = lead.period_rows(frame, bounds).to_dict("records")
    return portfolio_gate(
        rows, None, rolling_hour_cap=rolling_hour_cap,
        rank_key=quality.ranker(rank_name), loss_streak_cap=loss_cap)


def evaluate(frame: pd.DataFrame, bounds: tuple[str, str], rank_name: str,
             loss_cap: int, rolling_hour_cap: int, margin: float) -> dict:
    rows = rows_for(frame, bounds, rank_name, loss_cap, rolling_hour_cap)
    result = economics.simulate(rows, margin)
    if rows:
        days = len({pd.Timestamp(float(row["entry_time"]), unit="s", tz="UTC").date()
                    for row in rows})
        result.update({
            "active_market_days": days,
            "trades_per_market_day": round(len(rows) / days, 4),
            "profitable_exit_rate_pct": round(
                100 * sum(float(row["signed_return"]) > 0 for row in rows)
                / len(rows), 4),
            "literal_tp_rate_pct": round(
                100 * sum(int(row["outcome_code"]) == 1 for row in rows)
                / len(rows), 4),
            "maximum_entries_rolling_hour": max(
                sum(1 for other in rows
                    if float(row["entry_time"]) - 3600
                    < float(other["entry_time"]) <= float(row["entry_time"]))
                for row in rows),
        })
    else:
        result.update({"active_market_days": 0, "trades_per_market_day": 0.0,
                       "profitable_exit_rate_pct": 0.0,
                       "literal_tp_rate_pct": 0.0,
                       "maximum_entries_rolling_hour": 0})
    return result


def period_ok(value: dict) -> bool:
    return bool(value["return_pct"] >= 30 and value["negative_months"] == 0
                and value["dd_pct"] <= 8
                and value["trades_per_market_day"] >= 3)


def main() -> None:
    raw, parameters, sources = quality.prepare()
    candles, candle_sources = parent.load_sources()
    sources.update(candle_sources)
    frame = backing.guarded(backing.add_backing_distance(
        raw, candles, backing.LOOKBACKS[1]))
    frozen_quality = json.loads(quality.OUT.read_text(encoding="utf-8"))
    rank_name = str(frozen_quality["selection"]["selected_ranker"])
    loss_cap = int(frozen_quality["selection"]["selected_loss_streak_cap"])

    candidates = []
    for cap in CAPS:
        july = evaluate(frame, growth.COMPLETE_MONTHS["2026-07"],
                        rank_name, loss_cap, cap, 0.05)
        august = evaluate(frame, growth.COMPLETE_MONTHS["2026-08"],
                          rank_name, loss_cap, cap, 0.05)
        candidates.append({
            "rolling_hour_cap": cap,
            "selection": {"2026-07": july, "2026-08": august},
            "selection_pass": period_ok(july) and period_ok(august),
            "selection_floor_pct": min(july["return_pct"], august["return_pct"]),
            "selection_max_dd_pct": max(july["dd_pct"], august["dd_pct"]),
        })
    survivors = [item for item in candidates if item["selection_pass"]]
    if survivors:
        # Minimize one-hour concentration first; return and DD break ties only.
        survivors.sort(key=lambda item: (
            item["rolling_hour_cap"], -item["selection_floor_pct"],
            item["selection_max_dd_pct"]))
        selected = survivors[0]
    else:
        candidates.sort(key=lambda item: (
            item["selection_floor_pct"], -item["selection_max_dd_pct"]),
            reverse=True)
        selected = candidates[0]
    cap = int(selected["rolling_hour_cap"])
    september = evaluate(frame, growth.COMPLETE_MONTHS["2026-09"],
                         rank_name, loss_cap, cap, 0.05)
    validation_pass = bool(selected["selection_pass"] and period_ok(september))

    periods = {**growth.COMPLETE_MONTHS, "fresh": parent.FRESH,
               "full": parent.FULL}
    scenarios = {
        f"margin_{int(100 * margin)}pct": {
            name: evaluate(frame, bounds, rank_name, loss_cap, cap, margin)
            for name, bounds in periods.items()
        } for margin in (0.03, 0.05)
    }
    full_rows = rows_for(frame, parent.FULL, rank_name, loss_cap, cap)
    pd.DataFrame(full_rows).to_csv(ROWS_OUT, index=False)
    five = scenarios["margin_5pct"]
    objective = bool(validation_pass and all(
        period_ok(five[month]) for month in growth.COMPLETE_MONTHS))
    report = {
        "status": ("ROLLING_HOUR_CAP_CONFIRMED_FRESH_INCOMPLETE" if objective
                   else "ROLLING_HOUR_CAP_REJECTED_OBJECTIVE"),
        "ready_for_live_orders": False,
        "protocol": {
            "route": "frozen native stock X-Perp route with 32-bar backing guard",
            "selection": ["2026-07", "2026-08"],
            "validation": "2026-09",
            "fresh": list(parent.FRESH),
            "candidate_caps": list(CAPS),
            "decision": "smallest rolling-hour cap retaining the complete-month objective",
            "money": "client 3% or 5% current-equity margin; fixed 10x",
            "orders": "market only",
        },
        "base_parameters": parameters,
        "ranker": rank_name,
        "loss_streak_cap": loss_cap,
        "backing_level_lookback_bars": 32,
        "candidates": candidates,
        "selected_rolling_hour_cap": cap,
        "validation": september,
        "validation_pass": validation_pass,
        "margin_scenarios": scenarios,
        "objective_complete_months_pass": objective,
        "sources": {
            **sources,
            str(backing.OUT): digest(backing.OUT),
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "Native stock X-Perp history begins in July 2026.",
            "The fresh interval is only three resolved calendar days.",
            "Gross returns exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "selected_cap": cap,
        "candidates": candidates, "validation": september,
        "scenarios": scenarios, "objective": objective,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
