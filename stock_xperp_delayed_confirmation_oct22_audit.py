"""Causal delayed-confirmation control for the frozen X-Perp reversal route."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from stock_module_cluster_audit import portfolio_gate
import stock_xperp_backing_hour_cap_oct20_audit as hour_cap
import stock_xperp_backing_level_guard_oct20_audit as backing
import stock_xperp_native_cross_section_oct18_audit as parent
import stock_xperp_native_economics_oct18_audit as economics
import stock_xperp_native_growth_forward_monitor as resolver
import stock_xperp_native_growth_profile_oct18_audit as growth
import stock_xperp_native_quality_guard_oct18_audit as quality
import stock_xperp_native_volatility_guard_oct18_audit as volatility


OUT = parent.ROOT / "stock_xperp_delayed_confirmation_oct22.json"
KINDS = ("directional_close", "previous_bar_break")
WAITS = (2, 4, 8)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def confirmation_index(candles: dict, entry_index: int, direction: int,
                       kind: str, wait_bars: int) -> int | None:
    """Find a closed confirmation bar; its next bar is the market entry."""
    last = min(entry_index + wait_bars - 1, len(candles["time"]) - 2)
    for index in range(entry_index, last + 1):
        if (index <= 0
                or int(candles["time"][index])
                - int(candles["time"][index - 1]) != 900
                or int(candles["time"][index + 1])
                - int(candles["time"][index]) != 900):
            continue
        close = float(candles["close"][index])
        if kind == "directional_close":
            valid = (direction * (close - float(candles["open"][index])) > 0
                     and direction * (
                         close - float(candles["close"][index - 1])) > 0)
        elif kind == "previous_bar_break":
            valid = (close > float(candles["high"][index - 1])
                     if direction > 0
                     else close < float(candles["low"][index - 1]))
        else:
            raise ValueError(f"unknown confirmation kind: {kind}")
        if valid:
            return index
    return None


def prepare_raw(candles: dict, parameters: dict, maximum_atr: float,
                backing_lookback: int) -> pd.DataFrame:
    features = parent.feature_rows(candles, int(parameters["lookback_bars"]))
    entries = parent.select_entries(
        features, float(parameters["z_threshold"]),
        float(parameters["minimum_move_pct"]) / 100)
    entries = volatility.add_atr14(entries, candles)
    entries = entries[entries.atr14_fraction.le(maximum_atr)]
    return backing.guarded(backing.add_backing_distance(
        entries, candles, backing_lookback))


def resolve_confirmed(frame: pd.DataFrame, candles: dict, parameters: dict,
                      kind: str, wait_bars: int,
                      backing_lookback: int) -> list[dict]:
    output = []
    for row in frame.to_dict("records"):
        symbol = str(row["symbol"])
        source = candles[symbol]
        direction = int(row["direction_n"])
        confirmed = confirmation_index(
            source, int(row["entry_index"]), direction, kind, wait_bars)
        if confirmed is None:
            continue
        entry_index = confirmed + 1
        entry = float(source["open"][entry_index])
        start = max(0, entry_index - backing_lookback)
        prior_low = min(float(value) for value in
                        source["low"][start:entry_index])
        prior_high = max(float(value) for value in
                         source["high"][start:entry_index])
        distance = (entry / prior_low - 1 if direction > 0
                    else prior_high / entry - 1)
        if distance < 0:
            continue
        shifted = {
            **row,
            "entry_index": entry_index,
            "entry_time": int(source["time"][entry_index]),
            "confirmation_bar_time": int(source["time"][confirmed]),
            "confirmation_kind": kind,
            "confirmation_wait_bars": wait_bars,
            "backing_level_distance": distance,
        }
        output.append(resolver.resolve_row(
            shifted, source, float(parameters["target_pct"]),
            float(parameters["stop_pct"]), int(parameters["hold_bars"])))
    return output


def period_metrics(rows: list[dict], bounds: tuple[str, str]) -> dict:
    low = pd.Timestamp(bounds[0], tz="UTC").timestamp()
    high = pd.Timestamp(bounds[1], tz="UTC").timestamp()
    selected = [row for row in rows
                if low <= float(row["entry_time"]) < high
                and row["outcome"] != "PENDING"]
    result = economics.simulate(selected, .05)
    days = len({pd.Timestamp(float(row["entry_time"]), unit="s", tz="UTC").date()
                for row in selected})
    result["trades_per_market_day"] = round(len(selected) / days, 4) if days else 0.0
    result["profitable_exit_rate_pct"] = (round(
        100 * sum(float(row["signed_return"]) > 0 for row in selected)
        / len(selected), 4) if selected else 0.0)
    return result


def development_pass(value: dict) -> bool:
    return bool(value["return_pct"] >= 30
                and value["negative_months"] == 0
                and value["dd_pct"] <= 8
                and value["trades_per_market_day"] >= 3)


def main() -> None:
    profile = json.loads(growth.OUT.read_text(encoding="utf-8"))
    parameters = profile["selection"]["selected"]["parameters"]
    maximum_atr = float(json.loads(
        volatility.OUT.read_text(encoding="utf-8"))["selection"][
            "selected_maximum_atr14_fraction"])
    quality_report = json.loads(quality.OUT.read_text(encoding="utf-8"))
    rank_name = str(quality_report["selection"]["selected_ranker"])
    loss_cap = int(quality_report["selection"]["selected_loss_streak_cap"])
    cap_report = json.loads(hour_cap.OUT.read_text(encoding="utf-8"))
    rolling_cap = int(cap_report["selected_rolling_hour_cap"])
    backing_lookback = int(cap_report["backing_level_lookback_bars"])
    candles, sources = parent.load_sources()
    raw = prepare_raw(candles, parameters, maximum_atr, backing_lookback)
    full_start = pd.Timestamp(parent.FULL[0], tz="UTC").timestamp()
    fresh_start = pd.Timestamp(parent.FULL[1], tz="UTC").timestamp()
    raw = raw[raw.entry_time.ge(full_start) & raw.entry_time.lt(fresh_start)]

    candidates = []
    for kind in KINDS:
        for wait_bars in WAITS:
            resolved = resolve_confirmed(
                raw, candles, parameters, kind, wait_bars, backing_lookback)
            gated = portfolio_gate(
                resolved, None, rolling_hour_cap=rolling_cap,
                rank_key=quality.ranker(rank_name), loss_streak_cap=loss_cap)
            july = period_metrics(gated, growth.COMPLETE_MONTHS["2026-07"])
            august = period_metrics(gated, growth.COMPLETE_MONTHS["2026-08"])
            candidates.append({
                "confirmation": kind,
                "wait_bars": wait_bars,
                "selection_months": {"2026-07": july, "2026-08": august},
                "selection_pass": development_pass(july) and development_pass(august),
                "selection_floor_pct": min(july["return_pct"], august["return_pct"]),
                "resolved_before_portfolio": len(resolved),
                "accepted_before_fresh": len(gated),
            })
    candidates.sort(key=lambda row: (
        row["selection_pass"], row["selection_floor_pct"]), reverse=True)
    qualified = [row for row in candidates if row["selection_pass"]]
    report = {
        "status": "REJECTED_DEVELOPMENT",
        "ready_for_live_orders": False,
        "orders_possible": False,
        "hypothesis": (
            "wait for a closed directional reversal confirmation before the "
            "next-bar market entry instead of fading the residual immediately"),
        "protocol": {
            "candidate_confirmation_kinds": list(KINDS),
            "candidate_wait_bars": list(WAITS),
            "selection": ["2026-07", "2026-08"],
            "validation": "2026-09 only if development passes",
            "entry": "market open after the closed confirmation bar",
            "exit": "frozen 2% target, 1% stop, 64 bars, stop-first",
            "money": "client 5% current-equity margin; fixed 10x",
            "portfolio": {
                "rolling_hour_cap": rolling_cap,
                "loss_streak_cap": loss_cap,
                "ranker": rank_name,
            },
        },
        "raw_frozen_candidates": len(raw),
        "candidates": candidates,
        "qualified_development_candidates": len(qualified),
        "selected_before_validation": qualified[0] if qualified else None,
        "validation_2026_09": None,
        "october_reveal": None,
        "sources": {
            **sources,
            str(growth.OUT): digest(growth.OUT),
            str(volatility.OUT): digest(volatility.OUT),
            str(quality.OUT): digest(quality.OUT),
            str(hour_cap.OUT): digest(hour_cap.OUT),
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "No candidate passed development, so later periods are not used.",
            "Native X-Perp history starts in July 2026.",
            "The hypothesis was motivated by already observed October stop paths.",
            "Gross returns exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "qualified_development_candidates": len(qualified),
        "top_development": candidates[:3],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
