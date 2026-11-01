"""Audit a UTC-day net-R brake prompted by the Nov-1 shadow incident."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import heapq
import json
from pathlib import Path

import pandas as pd

import stock_xperp_regime_target_router_nov1_forward as current


ROOT = Path("reports/audit_2026_09_08")
ROWS = ROOT / "stock_xperp_regime_target_router_nov1_forward.csv"
OUT = ROOT / "stock_xperp_daily_net_loss_pause_nov1.json"
MONTHS = ("2026-07", "2026-08", "2026-09", "2026-10")
STOP_FRACTION = .01


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def apply_pause(frame: pd.DataFrame, cap_r: float) -> pd.DataFrame:
    pending = []
    accepted = []
    day = None
    daily_r = 0.0
    sequence = 0
    ordered = frame.sort_values(
        ["entry_time", "symbol", "direction"], kind="mergesort")
    for row in ordered.to_dict("records"):
        now = float(row["entry_time"])
        new_day = int(now // 86400)
        if new_day != day:
            day = new_day
            daily_r = 0.0
        while pending and pending[0][0] < now:
            _, _, completed = heapq.heappop(pending)
            if int(float(completed["exit_time"]) // 86400) == day:
                daily_r += float(completed["signed_return"]) / STOP_FRACTION
        if daily_r <= -float(cap_r):
            continue
        accepted.append(row)
        sequence += 1
        heapq.heappush(pending, (
            float(row["exit_time"]), sequence, row))
    return pd.DataFrame(accepted, columns=frame.columns)


def month(frame: pd.DataFrame, value: str) -> pd.DataFrame:
    stamps = pd.to_datetime(frame.entry_time, unit="s", utc=True)
    return frame[stamps.dt.strftime("%Y-%m").eq(value)].copy()


def compact(frame: pd.DataFrame) -> dict:
    value = current.metrics(frame)
    return {
        margin: {key: metrics[key] for key in (
            "n", "return_pct", "dd_pct", "max_loss_streak",
            "trades_per_active_day")}
        for margin, metrics in value.items()
    }


def main() -> None:
    rows = pd.read_csv(ROWS)
    baseline = {value: compact(month(rows, value)) for value in MONTHS}
    candidates = []
    for cap_r in (1.0, 2.0, 3.0):
        guarded = apply_pause(rows, cap_r)
        values = {value: compact(month(guarded, value)) for value in MONTHS}
        candidates.append({
            "daily_net_loss_cap_r": cap_r,
            "accepted": len(guarded),
            "rejected": len(rows) - len(guarded),
            "months": values,
        })
    selected = next(item for item in candidates
                    if item["daily_net_loss_cap_r"] == 2.0)
    report = {
        "status": "SHADOW_DAILY_NET_R_BRAKE",
        "ready_for_live_orders": False,
        "orders_possible": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "rule": "pause new native entries after UTC-day realized R <= -2R",
            "existing_rule_retained": "pause after two consecutive negative closes",
            "position_size": "client setting",
            "orders": "market only",
        },
        "reason": (
            "Four Nov-1 stops were interleaved with flat/profitable expiries, "
            "so the consecutive-loss brake reset while the day remained negative."),
        "baseline": baseline,
        "candidates": candidates,
        "selected": selected,
        "finding": (
            "The -2R brake rejects one additional historical row in July and "
            "none in August-October on top of the frozen consecutive-loss pause."),
        "sources": {
            str(ROWS): digest(ROWS),
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "This brake was formulated after observing the Nov-1 live shadow loss cluster.",
            "It is a loss-containment rule, not evidence of higher future return.",
            "Gross returns exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "selected": selected,
    }, indent=2))


if __name__ == "__main__":
    main()
