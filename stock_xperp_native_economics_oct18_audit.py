"""Economic and concentration diagnostics for the frozen native X-Perp route."""
from __future__ import annotations

import collections
import hashlib
import heapq
import json
from pathlib import Path

import pandas as pd

import stock_xperp_native_cross_section_oct18_audit as parent


OUT = parent.ROOT / "stock_xperp_native_economics_oct18.json"
MARGINS = (.02, .03, .05)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def simulate(rows: list[dict], margin_fraction: float,
             leverage: float = 10.0) -> dict:
    if not rows:
        return {"n": 0, "return_pct": 0.0, "dd_pct": 0.0,
                "monthly_return_pct": {}, "negative_months": 0}
    equity = peak = 1.0
    drawdown = 0.0
    pending: list[tuple[float, int, float, dict]] = []
    completed = []
    sequence = 0

    def realize(until: float | None) -> None:
        nonlocal equity, peak, drawdown
        while pending and (until is None or pending[0][0] < until):
            exit_time, order, pnl, row = heapq.heappop(pending)
            equity += pnl
            peak = max(peak, equity)
            drawdown = max(drawdown, (peak - equity) / peak)
            completed.append((exit_time, order, pnl, row))

    for row in sorted(rows, key=lambda item: (
            float(item["entry_time"]), int(item["module_priority"]),
            str(item["symbol"]))):
        realize(float(row["entry_time"]))
        pnl = (equity * margin_fraction * leverage
               * float(row["signed_return"]))
        sequence += 1
        heapq.heappush(pending, (
            float(row["exit_time"]), sequence, pnl, row))
    realize(None)
    completed.sort(key=lambda item: (item[0], item[1]))
    ledger = 1.0
    monthly_start, monthly_end = {}, {}
    for exit_time, _, pnl, _ in completed:
        month = pd.Timestamp(
            exit_time, unit="s", tz="UTC").strftime("%Y-%m")
        monthly_start.setdefault(month, ledger)
        ledger += pnl
        monthly_end[month] = ledger
    monthly = {
        month: round(100 * (monthly_end[month] / start - 1), 4)
        for month, start in monthly_start.items()
    }
    return {
        "n": len(rows), "return_pct": round(100 * (equity - 1), 4),
        "dd_pct": round(100 * drawdown, 4),
        "monthly_return_pct": monthly,
        "negative_months": sum(value < 0 for value in monthly.values()),
        "months_ge_30pct": sum(value >= 30 for value in monthly.values()),
    }


def diagnostics(frame: pd.DataFrame) -> dict:
    scans = frame.groupby("entry_time").size()
    hours = frame.assign(
        clock_hour=(frame.entry_time // 3600).astype(int)
    ).groupby("clock_hour").size()
    symbols = frame.groupby("symbol").size().sort_values(ascending=False)
    total = max(len(frame), 1)
    codes = collections.Counter(frame.outcome_code.astype(int))
    return {
        "n": len(frame),
        "literal_tp_rate_pct": round(100 * codes[1] / total, 4),
        "literal_stop_rate_pct": round(100 * codes[-1] / total, 4),
        "timed_exit_rate_pct": round(100 * codes[0] / total, 4),
        "max_entries_same_15m": int(scans.max()),
        "entries_in_multi_signal_scans_pct": round(
            100 * int(scans[scans > 1].sum()) / total, 4),
        "max_entries_same_clock_hour": int(hours.max()),
        "entries_in_hours_with_3plus_pct": round(
            100 * int(hours[hours >= 3].sum()) / total, 4),
        "largest_symbol_share_pct": round(100 * int(symbols.iloc[0]) / total, 4),
        "largest_symbol": str(symbols.index[0]),
        "top5_symbol_share_pct": round(100 * int(symbols.iloc[:5].sum()) / total, 4),
    }


def period(frame: pd.DataFrame, bounds: tuple[str, str]) -> pd.DataFrame:
    start = pd.Timestamp(bounds[0], tz="UTC").timestamp()
    end = pd.Timestamp(bounds[1], tz="UTC").timestamp()
    return frame[frame.entry_time.between(start, end - 1)].copy()


def main() -> None:
    report = json.loads(parent.OUT.read_text(encoding="utf-8"))
    frame = pd.read_csv(parent.ROWS_OUT)
    periods = {
        "development": parent.DEVELOPMENT,
        "validation": parent.VALIDATION,
        "confirmation": parent.CONFIRMATION,
        "fresh": parent.FRESH,
        "full": parent.FULL,
    }
    scenarios = {}
    for margin in MARGINS:
        name = f"margin_{round(100 * margin):.0f}pct"
        scenarios[name] = {
            key: simulate(period(frame, bounds).to_dict("records"), margin)
            for key, bounds in periods.items()
        }
        fresh = scenarios[name]["fresh"]
        scenarios[name]["fresh_22_session_projection_pct"] = round(
            100 * ((1 + fresh["return_pct"] / 100) ** (22 / 10) - 1), 4)
    base = scenarios["margin_3pct"]["full"]
    expected = report["full_jul5_oct16"]
    if (base["return_pct"] != expected["return_pct"]
            or base["dd_pct"] != expected["dd_pct"]):
        raise ValueError("3% economic replay does not match parent audit")
    result = {
        "status": "REJECTED_MONTHLY_OBJECTIVE",
        "ready_for_live_orders": False,
        "diagnostics": {
            key: diagnostics(period(frame, bounds))
            for key, bounds in periods.items()
        },
        "margin_scenarios": scenarios,
        "objective_checks": {
            name: {
                "no_negative_observed_months":
                    values["full"]["negative_months"] == 0,
                "every_observed_month_at_least_30pct":
                    values["full"]["months_ge_30pct"]
                    == len(values["full"]["monthly_return_pct"]),
                "fresh_projection_at_least_30pct":
                    values["fresh_22_session_projection_pct"] >= 30,
            } for name, values in scenarios.items()
        },
        "sources": {
            str(parent.OUT): digest(parent.OUT),
            str(parent.ROWS_OUT): digest(parent.ROWS_OUT),
            str(Path(__file__)): digest(Path(__file__)),
        },
        "limitations": [
            "Five-percent margin increases drawdown and is only a client scenario.",
            "October is partial and its 22-session projection is not evidence.",
            "Literal TP rate excludes profitable timed exits.",
            "Gross returns exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
