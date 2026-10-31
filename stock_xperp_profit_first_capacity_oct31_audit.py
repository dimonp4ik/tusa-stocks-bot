"""Audit wider native X-Perp capacity without using hidden periods to select it."""
from __future__ import annotations

import hashlib
import heapq
import json
from pathlib import Path

import pandas as pd

from stock_module_cluster_audit import portfolio_gate
import stock_xperp_backing_hour_cap_oct20_audit as hour_cap
import stock_xperp_backing_level_guard_oct20_audit as backing
import stock_xperp_native_cross_section_oct18_audit as parent
import stock_xperp_native_economics_oct18_audit as economics
import stock_xperp_native_quality_guard_oct18_audit as quality


OUT = parent.ROOT / "stock_xperp_profit_first_capacity_oct31.json"
ROWS = backing.ROWS_OUT
DIRECTION_CAPS = (4, 5, 6, 8)
ROLLING_HOUR_CAPS = (4, 5, 6, 8)
PERIODS = {
    "development_2026_07": ("2026-07-01", "2026-08-01"),
    "validation_2026_08": ("2026-08-01", "2026-09-01"),
    "hidden_2026_09": ("2026-09-01", "2026-10-01"),
    "hidden_2026_10_01_17": ("2026-10-01", "2026-10-17"),
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def raw_guarded_rows() -> tuple[list[dict], str, int]:
    raw, _, _ = quality.prepare()
    candles, _ = parent.load_sources()
    frame = backing.guarded(backing.add_backing_distance(raw, candles, 32))
    report = json.loads(quality.OUT.read_text(encoding="utf-8"))
    return (frame.to_dict("records"),
            str(report["selection"]["selected_ranker"]),
            int(report["selection"]["selected_loss_streak_cap"]))


def window(rows: list[dict], bounds: tuple[str, str]) -> list[dict]:
    start = pd.Timestamp(bounds[0], tz="UTC").timestamp()
    end = pd.Timestamp(bounds[1], tz="UTC").timestamp()
    return [row for row in rows if start <= float(row["entry_time"]) < end]


def concentration(rows: list[dict]) -> dict:
    pending: list[tuple[float, int, str]] = []
    open_direction = {"LONG": 0, "SHORT": 0}
    peak_open = peak_direction = 0
    recent: list[float] = []
    peak_hour = 0
    for order, row in enumerate(sorted(rows, key=lambda item: (
            float(item["entry_time"]), str(item["symbol"]))), 1):
        now = float(row["entry_time"])
        while pending and pending[0][0] < now:
            _, _, direction = heapq.heappop(pending)
            open_direction[direction] -= 1
        recent = [stamp for stamp in recent if stamp > now - 3600]
        direction = str(row["direction"])
        open_direction[direction] += 1
        heapq.heappush(pending, (float(row["exit_time"]), order, direction))
        recent.append(now)
        peak_open = max(peak_open, sum(open_direction.values()))
        peak_direction = max(peak_direction, open_direction[direction])
        peak_hour = max(peak_hour, len(recent))
    return {
        "peak_open_positions": peak_open,
        "peak_same_direction_positions": peak_direction,
        "peak_rolling_hour_entries": peak_hour,
        "peak_margin_pct_at_3pct": round(3 * peak_open, 4),
        "peak_margin_pct_at_5pct": round(5 * peak_open, 4),
    }


def metrics(rows: list[dict], margin: float = .03) -> dict:
    value = economics.simulate(rows, margin)
    streak = maximum = 0
    ordered = sorted(rows, key=lambda item: (
        float(item["exit_time"]), float(item["entry_time"]),
        str(item["symbol"])))
    for row in ordered:
        streak = streak + 1 if float(row["signed_return"]) <= 0 else 0
        maximum = max(maximum, streak)
    days = len({pd.Timestamp(float(row["entry_time"]), unit="s", tz="UTC").date()
                for row in rows})
    value.update({
        "win_rate_pct": round(100 * sum(float(row["signed_return"]) > 0
                                         for row in rows) / len(rows), 4)
                        if rows else 0.0,
        "max_loss_streak": maximum,
        "active_days": days,
        "trades_per_active_day": round(len(rows) / days, 4) if days else 0.0,
        **concentration(rows),
    })
    return value


def run_policy(raw: list[dict], rank_name: str, loss_cap: int,
               direction_cap: int, hour_cap_value: int) -> dict:
    result = {}
    for name, bounds in PERIODS.items():
        gated = portfolio_gate(
            window(raw, bounds), None,
            rolling_hour_cap=hour_cap_value,
            rank_key=quality.ranker(rank_name),
            loss_streak_cap=loss_cap,
            direction_cap=direction_cap)
        result[name] = metrics(gated)
    return result


def development_ok(candidate: dict, baseline: dict) -> bool:
    return bool(
        candidate["return_pct"] >= baseline["return_pct"]
        and candidate["negative_months"] == 0
        and candidate["dd_pct"] <= max(8.0, baseline["dd_pct"] + 1.0)
        and candidate["max_loss_streak"] <= baseline["max_loss_streak"] + 1)


def validation_ok(candidate: dict, baseline: dict) -> bool:
    return bool(
        candidate["return_pct"] >= baseline["return_pct"]
        and candidate["negative_months"] == 0
        and candidate["dd_pct"] <= max(8.0, baseline["dd_pct"] + 1.0)
        and candidate["max_loss_streak"] <= baseline["max_loss_streak"] + 1)


def hidden_ok(candidate: dict, baseline: dict) -> bool:
    return bool(
        candidate["return_pct"] >= baseline["return_pct"]
        and candidate["negative_months"] <= baseline["negative_months"]
        and candidate["dd_pct"] <= baseline["dd_pct"] + 1.5
        and candidate["max_loss_streak"] <= baseline["max_loss_streak"] + 2)


def main() -> None:
    raw, rank_name, loss_cap = raw_guarded_rows()
    policies = []
    for direction_cap in DIRECTION_CAPS:
        for rolling_hour_cap in ROLLING_HOUR_CAPS:
            periods = run_policy(raw, rank_name, loss_cap,
                                 direction_cap, rolling_hour_cap)
            policies.append({
                "direction_cap": direction_cap,
                "rolling_hour_cap": rolling_hour_cap,
                "periods": periods,
            })
    baseline = next(item for item in policies
                    if item["direction_cap"] == 4
                    and item["rolling_hour_cap"] == 4)
    base_dev = baseline["periods"]["development_2026_07"]
    base_val = baseline["periods"]["validation_2026_08"]
    eligible = [item for item in policies
                if development_ok(item["periods"]["development_2026_07"], base_dev)
                and validation_ok(item["periods"]["validation_2026_08"], base_val)]
    eligible.sort(key=lambda item: (
        min(item["periods"]["development_2026_07"]["return_pct"],
            item["periods"]["validation_2026_08"]["return_pct"]),
        -max(item["periods"]["development_2026_07"]["dd_pct"],
             item["periods"]["validation_2026_08"]["dd_pct"]),
        -item["direction_cap"], -item["rolling_hour_cap"]), reverse=True)
    selected = eligible[0] if eligible else baseline
    hidden_periods = ("hidden_2026_09", "hidden_2026_10_01_17")
    hidden_pass = all(hidden_ok(selected["periods"][name],
                              baseline["periods"][name])
                      for name in hidden_periods)
    changed = (selected["direction_cap"], selected["rolling_hour_cap"]) != (4, 4)
    accepted = bool(changed and hidden_pass)
    report = {
        "status": "CAPACITY_ACCEPTED" if accepted else "CAPACITY_REJECTED",
        "ready_for_live_orders": False,
        "protocol": {
            "route": "frozen native X-Perp backing-level route",
            "development": "2026-07",
            "validation": "2026-08",
            "hidden_reveal": ["2026-09", "2026-10-01..16"],
            "candidate_direction_caps": list(DIRECTION_CAPS),
            "candidate_rolling_hour_caps": list(ROLLING_HOUR_CAPS),
            "fixed_per_scan_cap": 2,
            "money": "3% current-equity margin per trade; fixed 10x",
        },
        "ranker": rank_name,
        "loss_streak_cap": loss_cap,
        "baseline": baseline,
        "selected_before_hidden": selected,
        "hidden_pass": hidden_pass,
        "accepted": accepted,
        "policies": policies,
        "decision": (
            "widen runtime caps only when development and validation improve return "
            "without material DD/streak regression, then both hidden periods pass"),
        "sources": {
            str(ROWS): digest(ROWS),
            str(quality.OUT): digest(quality.OUT),
            str(hour_cap.OUT): digest(hour_cap.OUT),
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "Direct native X-Perp history begins in July 2026.",
            "The base strategy parameters had already used September.",
            "October is an incomplete hidden period.",
            "Gross returns exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "baseline": baseline,
        "selected_before_hidden": selected,
        "hidden_pass": hidden_pass,
        "accepted": accepted,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
