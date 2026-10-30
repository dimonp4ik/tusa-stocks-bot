"""Discover and freeze a quality-conditioned stock target challenger.

The incumbent 2/2.5/3% target route is preserved.  This audit explores one
simple override at a time using only causal entry features.  A rule that looks
better across all currently visible months is registered for genuinely new
forward observation; discovery on those months cannot authorize runtime use.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path

import pandas as pd

import stock_xperp_basket_trend_forward_oct30_audit as forward
import stock_xperp_regime_target_router_oct30_audit as router
import stock_xperp_short_exhaustion_guard_oct30_audit as guard
import stock_xperp_short_stack_guard_oct30_forward as source


ROOT = Path("reports/audit_2026_09_08")
MANIFEST = ROOT / "stock_xperp_forward_oct30_1000" / "manifest.json"
OUT = ROOT / "stock_xperp_quality_target_challenger_oct30.json"
FEATURES = {
    "abs_cross_z": (2.5, 3.0, 3.5, 4.0),
    "atr14_fraction": (.002, .004, .006),
    "backing_level_distance": (.0025, .005, .01, .02),
}
DIRECTIONS = ("ALL", "LONG", "SHORT")
OPERATORS = ("at_least", "at_most")
PERIODS = {
    "july": ("2026-07-01", "2026-08-01"),
    "august": ("2026-08-01", "2026-09-01"),
    "september": ("2026-09-01", "2026-10-01"),
    "october": ("2026-10-01", "2026-11-01"),
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(row: dict) -> tuple[str, int, str]:
    return str(row["symbol"]), int(row["entry_time"]), str(row["direction"])


def feature_value(row: dict, feature: str) -> float:
    if feature == "abs_cross_z":
        return abs(float(row["cross_z"]))
    return float(row[feature])


def matches(row: dict, config: dict) -> bool:
    if config["direction"] != "ALL" and row["direction"] != config["direction"]:
        return False
    value = feature_value(row, config["feature"])
    threshold = float(config["threshold"])
    return value >= threshold if config["operator"] == "at_least" else value <= threshold


def metrics(frame: pd.DataFrame) -> dict:
    value = forward.metrics(frame, .03)
    return {key: value[key] for key in (
        "return_pct", "dd_pct", "resolved", "pending",
        "trades_per_active_day", "literal_tp_rate_pct", "max_loss_streak")}


def evaluate(frame: pd.DataFrame) -> dict:
    stamp = pd.to_datetime(frame.entry_time, unit="s", utc=True)
    return {
        name: metrics(frame[
            stamp.ge(pd.Timestamp(start, tz="UTC"))
            & stamp.lt(pd.Timestamp(end, tz="UTC"))].copy())
        for name, (start, end) in PERIODS.items()
    }


def main() -> None:
    raw, sources, candles = source.candidates(MANIFEST)
    entries = guard.apply_guard(raw, -.01)
    resolved = {target: router.resolve(entries, candles, target)
                for target in router.TARGETS}
    lookups = {
        target: {identity(row): row for row in frame.to_dict("records")}
        for target, frame in resolved.items()
    }

    def route(config: dict | None = None) -> pd.DataFrame:
        output = []
        for entry in entries.to_dict("records"):
            hour = int(float(entry["entry_time"]) // 3600) % 24
            target = (2.0 if float(entry["basket_trend_32"]) <= -.04 else
                      3.0 if 6 <= hour < 12 else 2.5)
            if config is not None and matches(entry, config):
                target = float(config["target_pct"])
            output.append(lookups[target][identity(entry)])
        return guard.accept(pd.DataFrame(output))

    baseline = evaluate(route())
    candidates = []
    for feature, thresholds in FEATURES.items():
        for threshold, operator, direction, target in itertools.product(
                thresholds, OPERATORS, DIRECTIONS, router.TARGETS):
            config = {
                "feature": feature, "threshold": threshold,
                "operator": operator, "direction": direction,
                "target_pct": target,
            }
            changed = sum(matches(row, config)
                          for row in entries.to_dict("records"))
            values = evaluate(route(config))
            deltas = {
                name: {
                    "return_delta_pp": round(
                        value["return_pct"]-baseline[name]["return_pct"], 4),
                    "dd_delta_pp": round(value["dd_pct"]-baseline[name]["dd_pct"], 4),
                }
                for name, value in values.items()
            }
            passes_visible = bool(all(
                deltas[name]["return_delta_pp"] >= 0
                and deltas[name]["dd_delta_pp"] <= .10
                for name in PERIODS)
                and any(deltas[name]["return_delta_pp"] > 0 for name in PERIODS))
            candidates.append({
                "config": config, "matched_raw_entries": int(changed),
                "visible_periods_pass": passes_visible,
                "minimum_return_delta_pp": min(
                    item["return_delta_pp"] for item in deltas.values()),
                "total_return_delta_pp": round(sum(
                    item["return_delta_pp"] for item in deltas.values()), 4),
                "deltas": deltas, "metrics": values,
            })
    qualified = [item for item in candidates if item["visible_periods_pass"]]
    qualified.sort(key=lambda item: (
        -item["minimum_return_delta_pp"], -item["total_return_delta_pp"],
        json.dumps(item["config"], sort_keys=True)))
    challenger = qualified[0] if qualified else None
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    freeze_end = min(int(item["end"]) for item in manifest["symbols"].values()
                     if item.get("end") is not None)
    status = ("SHADOW_CHALLENGER_AWAIT_GENUINELY_NEW_FORWARD"
              if challenger else "REJECTED_NO_CROSS_MONTH_CHALLENGER")
    report = {
        "status": status,
        "ready_for_live_orders": False, "orders_possible": False,
        "runtime_change_allowed": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "incumbent": ("2% when 32-bar basket <= -4%; otherwise 3% from "
                          "06:00-12:00 UTC; otherwise 2.5%"),
            "candidate_count": len(candidates),
            "features": FEATURES, "directions": list(DIRECTIONS),
            "operators": list(OPERATORS), "targets_pct": list(router.TARGETS),
            "visible_discovery_periods": list(PERIODS),
            "entry": "unchanged next-15-minute market open",
            "position_fraction": "client setting", "orders": "market only",
            "hold_bars": 64, "stop_pct": 1.0,
        },
        "baseline": baseline,
        "qualified_visible_candidates": len(qualified),
        "frozen_challenger": challenger,
        "forward_activation_gate": {
            "freeze_candle_end_utc": pd.Timestamp(
                freeze_end, unit="s", tz="UTC").isoformat(),
            "minimum_new_resolved_trades": 30,
            "requirements": [
                "candidate return strictly above incumbent on new outcomes",
                "candidate drawdown no higher than incumbent on new outcomes",
                "overall frequency remains at least 3 trades per active day",
            ],
            "observed_new_resolved_trades": 0, "met": False,
        },
        "top_visible_candidates": sorted(
            candidates, key=lambda item: (-item["total_return_delta_pp"],
                                          -item["minimum_return_delta_pp"]))[:15],
        "sources": {**sources, str(MANIFEST): digest(MANIFEST),
                    str(Path(__file__).resolve()): digest(Path(__file__).resolve())},
        "limitations": [
            "The challenger was discovered using every currently visible month and requires new forward data.",
            "Venue-native stock X-Perp history begins in late June 2026.",
            "Public OHLC candles cannot reproduce latency or intrabar order sequence.",
            "Gross returns exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": status, "baseline": baseline,
        "qualified_visible_candidates": len(qualified),
        "frozen_challenger": challenger,
        "forward_activation_gate": report["forward_activation_gate"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
