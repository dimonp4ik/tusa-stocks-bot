"""Strict holdout audit for a high-payoff stock 5m complement.

The route is selected only on Aug 3-20.  The single top development route is
then checked on Aug 21-Sep 7; there is no validation fallback.  Sep 8-Oct 2 is
revealed only after validation passes.
"""
from __future__ import annotations

import collections
import hashlib
import heapq
import json
import pickle
from pathlib import Path

import pandas as pd

import stock_5m_family_oct3_audit as base
from stock_5m_regime_filter_oct3_audit import prepare
from stock_forward_monitor import frozen_rows
from stock_module_cluster_audit import portfolio_gate


ROOT = Path("reports/audit_2026_09_08")
CURRENT_REPORT = ROOT / "stock_5m_regime_filter_oct3.json"
CURRENT_ROWS = ROOT / "stock_5m_regime_filter_oct3_rows.csv"
OUTPUT = ROOT / "stock_5m_high_payoff_oct3.json"
ROWS_OUTPUT = ROOT / "stock_5m_high_payoff_oct3_rows.csv"
HIGH_TARGETS = (1.5, 2.0, 3.0, 4.0)
ROUTE_KEYS = (
    "family", "direction", "target_r", "stop_atr_mult",
    "session_bucket_5m", "qqq_regime_5m",
)
WR_FLOORS = {1.5: 48.0, 2.0: 43.0, 3.0: 34.0, 4.0: 28.0}
MARGIN_FRACTION = 0.03
LEVERAGE = 10.0


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def accepted(rows: list[dict]) -> list[dict]:
    return portfolio_gate(
        rows, None, rank_key=lambda row: int(row["module_priority"]))


def money(rows: list[dict]) -> dict:
    """Use equity known at entry, so overlapping trades cannot compound early."""
    rows = accepted(rows)
    if not rows:
        return {"n": 0, "wr": 0.0, "return_pct": 0.0, "dd_pct": 0.0,
                "negative_months": 0, "max_loss_streak": 0,
                "trades_per_calendar_day": 0.0,
                "trades_per_active_day": 0.0}
    equity = peak = 1.0
    drawdown = 0.0
    pending: list[tuple[float, int, float, dict]] = []
    completed: list[tuple[float, float, dict]] = []
    sequence = 0

    def realize(until: float | None) -> None:
        nonlocal equity, peak, drawdown
        while pending and (until is None or pending[0][0] < until):
            exit_time, _, pnl, row = heapq.heappop(pending)
            equity += pnl
            peak = max(peak, equity)
            drawdown = max(drawdown, (peak - equity) / peak)
            completed.append((exit_time, pnl, row))

    for row in sorted(rows, key=lambda item: (
            float(item["entry_time"]), int(item["module_priority"]),
            str(item["symbol"]))):
        realize(float(row["entry_time"]))
        pnl = (equity * MARGIN_FRACTION * LEVERAGE
               * float(row["risk_pct"]) * float(row["net_r"]))
        sequence += 1
        heapq.heappush(pending, (float(row["exit_time"]), sequence, pnl, row))
    realize(None)

    completed.sort(key=lambda item: (item[0], float(item[2]["entry_time"]),
                                     str(item[2]["symbol"])))
    wins = streak = max_streak = 0
    active_days = set()
    monthly_start: dict[str, float] = {}
    monthly_end: dict[str, float] = {}
    ledger_equity = 1.0
    modules = collections.Counter()
    for exit_time, pnl, row in completed:
        won = pnl > 0
        wins += int(won)
        streak = 0 if won else streak + 1
        max_streak = max(max_streak, streak)
        month = pd.Timestamp(exit_time, unit="s", tz="UTC").strftime("%Y-%m")
        monthly_start.setdefault(month, ledger_equity)
        ledger_equity += pnl
        monthly_end[month] = ledger_equity
        active_days.add(pd.Timestamp(
            float(row["entry_time"]), unit="s", tz="UTC").strftime("%Y-%m-%d"))
        modules[str(row["module"])] += 1
    first = pd.Timestamp(min(float(row["entry_time"]) for row in rows),
                         unit="s", tz="UTC")
    last = pd.Timestamp(max(float(row["entry_time"]) for row in rows),
                        unit="s", tz="UTC")
    calendar_days = max(1, (last.date() - first.date()).days + 1)
    monthly = {
        key: round(100 * (monthly_end[key] / monthly_start[key] - 1), 4)
        for key in sorted(monthly_start)
    }
    return {
        "n": len(rows), "wr": round(100 * wins / len(rows), 2),
        "return_pct": round(100 * (equity - 1), 4),
        "dd_pct": round(100 * drawdown, 4),
        "negative_months": sum(value < 0 for value in monthly.values()),
        "monthly_return_pct": monthly,
        "max_loss_streak": max_streak,
        "active_days": len(active_days),
        "trades_per_calendar_day": round(len(rows) / calendar_days, 3),
        "trades_per_active_day": round(len(rows) / len(active_days), 3),
        "by_module_count": dict(sorted(modules.items())),
    }


def stage(rows: list[dict], period: tuple[str, str]) -> list[dict]:
    return base.stage_rows(rows, period)


def evaluate(rows: list[dict], period: tuple[str, str]) -> dict:
    selected = stage(rows, period)
    accepted_rows = accepted(selected)
    high = [row for row in accepted_rows
            if str(row.get("module", "")).startswith(
                "shadow_stock_5m_high_payoff")]
    return {
        **money(selected),
        "accepted_high_payoff": len(high),
        "high_payoff_wr": (round(100 * sum(float(row["net_r"]) > 0 for row in high)
                                  / len(high), 2) if high else 0.0),
        "high_payoff_net_r": round(sum(float(row["net_r"]) for row in high), 6),
    }


def select(rows: list[dict], route: dict) -> list[dict]:
    return [row for row in rows if all(row[key] == value
                                       for key, value in route.items())]


def route_metrics(rows: list[dict]) -> dict:
    return base.route_metrics(rows)


def high_supplement(rows: list[dict], route: dict) -> list[dict]:
    return [{**row, "module": "shadow_stock_5m_high_payoff",
             "module_priority": 102, "candidate_priority": 2}
            for row in select(rows, route)]


def development_ok(value: dict, baseline: dict, raw: dict,
                   target: float) -> bool:
    target = float(target)
    return bool(
        raw["n"] >= 8 and raw["symbols"] >= 3
        and raw["wr"] >= WR_FLOORS[target]
        and raw["net_r"] > 0 and raw["max_loss_streak"] <= 4
        and value["accepted_high_payoff"] >= 4
        and value["return_pct"] >= baseline["return_pct"] + 0.05
        and value["wr"] >= 75.0
        and value["dd_pct"] <= baseline["dd_pct"] + 0.35
        and value["max_loss_streak"] <= baseline["max_loss_streak"] + 1
        and value["negative_months"] == 0
    )


def forward_ok(value: dict, baseline: dict, raw: dict, target: float,
               minimum_accepted: int) -> bool:
    target = float(target)
    return bool(
        raw["n"] >= 6 and raw["symbols"] >= 3
        and raw["wr"] >= WR_FLOORS[target] - 3.0 and raw["net_r"] > 0
        and value["accepted_high_payoff"] >= minimum_accepted
        and value["return_pct"] > baseline["return_pct"]
        and value["wr"] >= 75.0
        and value["dd_pct"] <= baseline["dd_pct"] + 0.35
        and value["max_loss_streak"] <= baseline["max_loss_streak"] + 1
        and value["negative_months"] == 0
    )


def main() -> None:
    current_report = json.loads(CURRENT_REPORT.read_text(encoding="utf-8"))
    if current_report["status"] != "SHADOW_CANDIDATE_AWAITING_NEW_FORWARD":
        raise ValueError("current 5m stock supplement is not a frozen shadow candidate")
    production = frozen_rows(pd.read_csv(base.PRODUCTION_SOURCE), base.PROFILE)
    current_supplement = prepare(pd.read_csv(CURRENT_ROWS)).to_dict("records")
    current = base.combine(production, current_supplement)

    caches = {symbol: pickle.loads(
        (base.CACHE / f"{symbol}_5min_current.pkl").read_bytes())
        for symbol in base.SYMBOLS}
    original_targets = base.TARGETS
    try:
        base.TARGETS = HIGH_TARGETS
        high_rows = base.build_rows(caches)
    finally:
        base.TARGETS = original_targets
    pd.DataFrame(high_rows).to_csv(ROWS_OUTPUT, index=False)

    periods = {
        "development": base.DEVELOPMENT,
        "validation": base.VALIDATION,
        "hidden": base.HIDDEN,
    }
    baseline = {name: evaluate(current, period)
                for name, period in periods.items()}
    development = stage(high_rows, base.DEVELOPMENT)
    routes = []
    seen = set()
    for row in development:
        key = tuple(row[field] for field in ROUTE_KEYS)
        if key in seen:
            continue
        seen.add(key)
        routes.append(dict(zip(ROUTE_KEYS, key)))

    qualified = []
    for route in routes:
        raw = route_metrics(select(development, route))
        supplement = high_supplement(high_rows, route)
        combined = base.combine(current, supplement)
        value = evaluate(combined, base.DEVELOPMENT)
        if not development_ok(
                value, baseline["development"], raw,
                float(route["target_r"])):
            continue
        score = (value["return_pct"] - baseline["development"]["return_pct"]
                 - 2 * max(0.0, value["dd_pct"]
                           - baseline["development"]["dd_pct"])
                 + value["accepted_high_payoff"] / 1000)
        qualified.append({"route": route, "score": round(score, 8),
                          "raw_development": raw,
                          "portfolio_development": value})
    qualified.sort(key=lambda item: (
        item["score"], item["portfolio_development"]["return_pct"]),
        reverse=True)

    frozen = qualified[0] if qualified else None
    validation = validation_raw = None
    validation_pass = False
    supplement = []
    if frozen:
        supplement = high_supplement(high_rows, frozen["route"])
        combined = base.combine(current, supplement)
        validation_raw = route_metrics(select(
            stage(high_rows, base.VALIDATION), frozen["route"]))
        validation = evaluate(combined, base.VALIDATION)
        validation_pass = forward_ok(
            validation, baseline["validation"], validation_raw,
            float(frozen["route"]["target_r"]), 3)

    hidden = hidden_raw = None
    hidden_pass = False
    if validation_pass:
        combined = base.combine(current, supplement)
        hidden_raw = route_metrics(select(
            stage(high_rows, base.HIDDEN), frozen["route"]))
        hidden = evaluate(combined, base.HIDDEN)
        hidden_pass = forward_ok(
            hidden, baseline["hidden"], hidden_raw,
            float(frozen["route"]["target_r"]), 5)

    report = {
        "status": ("SHADOW_CANDIDATE_AWAITING_NEW_FORWARD"
                   if hidden_pass else "REJECTED"),
        "protocol": {
            "development": base.DEVELOPMENT,
            "validation": base.VALIDATION,
            "hidden": base.HIDDEN,
            "selection": "rank on development; freeze top one; no validation fallback",
            "targets_r": list(HIGH_TARGETS),
            "stop_atr_multipliers": list(base.STOP_ATR_MULTS),
            "entry": "next 5-minute market open proxy",
            "exit": "stop first on ambiguous OHLC bar; otherwise session close",
            "portfolio": "production and current 5m regimes rank before complement",
            "money": "3% of equity known at entry, 10x, gross",
        },
        "routes_evaluated": len(routes),
        "development_pass": len(qualified),
        "frozen_before_validation": frozen,
        "validation_raw": validation_raw,
        "validation": validation,
        "validation_pass": validation_pass,
        "hidden_raw": hidden_raw,
        "hidden": hidden,
        "hidden_pass": hidden_pass,
        "baseline": baseline,
        "sources": {
            str(CURRENT_REPORT): digest(CURRENT_REPORT),
            str(CURRENT_ROWS): digest(CURRENT_ROWS),
            **{str(base.CACHE / f"{symbol}_5min_current.pkl"):
               digest(base.CACHE / f"{symbol}_5min_current.pkl")
               for symbol in base.SYMBOLS},
        },
        "limitations": [
            "Direct five-minute X-Perp history spans about sixty-two days.",
            "A passing route would remain shadow-only until genuinely new forward data pass an activation gate.",
            "Open-trade mark-to-market drawdown is unavailable in outcome rows.",
            "Fees are excluded at the user's request.",
        ],
    }
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "routes_evaluated": report["routes_evaluated"],
        "development_pass": report["development_pass"],
        "frozen": frozen,
        "validation_raw": validation_raw,
        "validation": validation,
        "validation_pass": validation_pass,
        "hidden_raw": hidden_raw,
        "hidden": hidden,
        "hidden_pass": hidden_pass,
        "baseline": baseline,
    }, indent=2))


if __name__ == "__main__":
    main()
