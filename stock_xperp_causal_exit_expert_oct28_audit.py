"""Causal regime-specific exit expert for the frozen stock X-Perp entries.

Twenty-four literal market-exit policies are scored from virtual outcomes that
finished strictly before each new entry.  Router hyperparameters are selected
on July-August, tested on September and confirmed on October.  No entry filter,
position size or limit order is introduced by this audit.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.stock_xperp_native_shadow import AUDITED_SYMBOLS
from stock_module_cluster_audit import portfolio_gate
import stock_xperp_basket_trend_forward_oct28_audit as forward
import stock_xperp_native_growth_forward_monitor as monitor
import stock_xperp_native_quality_guard_oct18_audit as quality
import stock_xperp_session_target_router_oct27_audit as base


ROOT = Path("reports/audit_2026_09_08")
MANIFEST = ROOT / "stock_xperp_forward_oct27_2156" / "manifest.json"
OUT = ROOT / "stock_xperp_causal_exit_expert_oct28.json"
ROWS_OUT = ROOT / "stock_xperp_causal_exit_expert_oct28.csv"
TARGETS = (1.5, 2.0, 2.5, 3.0)
STOPS = (.75, 1.0)
HOLDS = (32, 64, 96)
POLICIES = tuple(itertools.product(TARGETS, STOPS, HOLDS))
BASE_POLICY = (2.0, 1.0, 64)
MONTHS = ("2026-07", "2026-08", "2026-09", "2026-10")
REGIME_SPECS = (
    ("direction",),
    ("direction", "session"),
    ("direction", "trend_bucket"),
    ("direction", "session", "trend_bucket"),
    ("direction", "atr_bucket"),
    ("direction", "trend_bucket", "atr_bucket"),
)
WINDOWS = (60, 120)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def policy_id(policy: tuple[float, float, int]) -> str:
    target, stop, hold = policy
    return f"t{target:g}_s{stop:g}_h{hold}"


def identity(row: dict) -> tuple[str, int, str]:
    return str(row["symbol"]), int(row["entry_time"]), str(row["direction"])


def add_regimes(entries: pd.DataFrame) -> pd.DataFrame:
    result = entries.copy()
    result["direction"] = np.where(result.direction_n.gt(0), "LONG", "SHORT")
    stamp = pd.to_datetime(result.entry_time, unit="s", utc=True)
    hour = stamp.dt.hour
    result["session"] = np.select(
        [hour.le(2), hour.le(7), hour.le(15)],
        ["utc_00_02", "utc_03_07", "utc_08_15"],
        default="utc_16_23")
    alignment = result.direction_n * result.basket_trend_96
    result["trend_bucket"] = np.select(
        [alignment.lt(0), alignment.lt(.005)],
        ["opposed", "mild"], default="strong")
    result["atr_bucket"] = np.select(
        [result.atr14_fraction.le(.004), result.atr14_fraction.le(.007)],
        ["low", "medium"], default="high")
    return result


def regime(row: dict, spec: tuple[str, ...]) -> tuple:
    return tuple(str(row[field]) for field in spec)


def load_inputs() -> tuple[pd.DataFrame, dict, dict]:
    candles, sources, _ = monitor.load_manifest(MANIFEST)
    candles = {symbol: candles[symbol] for symbol in AUDITED_SYMBOLS}
    entries = add_regimes(base.prepare_entries(candles))
    entries.sort_values(["entry_time", "symbol", "direction"], inplace=True)
    entries.reset_index(drop=True, inplace=True)
    return entries, candles, sources


def resolve_policies(entries: pd.DataFrame, candles: dict) -> dict[str, pd.DataFrame]:
    output = {}
    for policy in POLICIES:
        target, stop, hold = policy
        rows = monitor.resolve_entries(entries, candles, {
            "target_pct": target, "stop_pct": stop, "hold_bars": hold})
        frame = pd.DataFrame(rows)
        frame["policy_id"] = policy_id(policy)
        frame["policy_target_pct"] = target
        frame["policy_stop_pct"] = stop
        frame["policy_hold_bars"] = hold
        output[policy_id(policy)] = frame
    return output


def tail_stats(cumulative: list[float], cumulative_sq: list[float],
               window: int) -> tuple[int, float, float]:
    n = min(window, len(cumulative)-1)
    if n <= 0:
        return 0, 0.0, 0.0
    total = cumulative[-1] - cumulative[-1-n]
    total_sq = cumulative_sq[-1] - cumulative_sq[-1-n]
    mean = total/n
    variance = max(0.0, (total_sq-total*total/n)/(n-1)) if n > 1 else 0.0
    return n, mean, math.sqrt(variance)


def append_history(history: tuple[list[float], list[float]], value: float) -> None:
    cumulative, cumulative_sq = history
    cumulative.append(cumulative[-1] + value)
    cumulative_sq.append(cumulative_sq[-1] + value*value)


def precompute_stats(entries: pd.DataFrame, resolved: dict[str, pd.DataFrame]) -> tuple[dict, dict, list[str]]:
    """Return dense causal statistics and cutoff-integrity counters."""
    entry_rows = entries.to_dict("records")
    policy_ids = list(resolved)
    stats = {
        (spec, window): np.zeros((len(entries), len(policy_ids), 4), dtype=float)
        for spec in REGIME_SPECS for window in WINDOWS
    }
    maximum_violation = 0
    maximum_history = 0
    for spec in REGIME_SPECS:
        for policy_index, pid in enumerate(policy_ids):
            frame = resolved[pid]
            events = [row for row in frame.to_dict("records")
                      if str(row.get("outcome")) != "PENDING"
                      and pd.notna(row.get("exit_time"))]
            events.sort(key=lambda row: (float(row["exit_time"]),
                                         float(row["entry_time"]), str(row["symbol"])))
            pointer = 0
            last_added_exit = -math.inf
            global_history = ([0.0], [0.0])
            local_history: dict[tuple, tuple[list[float], list[float]]] = defaultdict(
                lambda: ([0.0], [0.0]))
            for index, row in enumerate(entry_rows):
                now = float(row["entry_time"])
                while pointer < len(events) and float(events[pointer]["exit_time"]) < now:
                    event = events[pointer]
                    value = float(event["signed_return"])
                    append_history(global_history, value)
                    append_history(local_history[regime(event, spec)], value)
                    last_added_exit = float(event["exit_time"])
                    pointer += 1
                maximum_violation = max(maximum_violation, int(last_added_exit >= now))
                local = local_history[regime(row, spec)]
                maximum_history = max(maximum_history, len(local[0])-1)
                for window in WINDOWS:
                    n, mean, std = tail_stats(*local, window)
                    global_n, global_mean, _ = tail_stats(*global_history, 4*window)
                    stats[(spec, window)][index, policy_index] = (
                        n, mean, std, global_mean if global_n else 0.0)
    integrity = {
        "strictly_prior_exit_violations": maximum_violation,
        "maximum_local_history": maximum_history,
        "policies": len(POLICIES), "entries": len(entries),
    }
    return stats, integrity, policy_ids


def outcome_lookups(resolved: dict[str, pd.DataFrame]) -> dict[str, dict]:
    return {
        pid: {identity(row): row for row in frame.to_dict("records")}
        for pid, frame in resolved.items()
    }


def choose_rows(entries: pd.DataFrame, lookups: dict[str, dict], stats: dict,
                policy_ids: list[str], config: dict) -> pd.DataFrame:
    spec = tuple(config["regime_fields"])
    output = []
    baseline_pid = policy_id(BASE_POLICY)
    for index, entry in enumerate(entries.to_dict("records")):
        key = identity(entry)
        scores = {}
        matrix = stats[(spec, config["window"])]
        for policy_index, pid in enumerate(policy_ids):
            n, mean, std, global_mean = matrix[index, policy_index]
            if n < config["minimum_history"]:
                continue
            shrink = float(config["shrinkage"])
            estimate = (n*mean + shrink*global_mean) / (n+shrink)
            scores[pid] = estimate - float(config["uncertainty_penalty"])*std/math.sqrt(n)
        chosen = baseline_pid
        if scores:
            candidate = max(scores, key=lambda pid: (scores[pid], pid == baseline_pid, pid))
            baseline_score = scores.get(baseline_pid, -math.inf)
            if scores[candidate] >= baseline_score + float(config["minimum_uplift"]):
                chosen = candidate
        row = dict(lookups[chosen][key])
        row["expert_regime"] = "|".join(regime(entry, spec))
        row["expert_score"] = scores.get(chosen)
        row["module"] = "native_xperp_causal_exit_expert"
        row["module_priority"] = 0
        output.append(row)
    combined = pd.DataFrame(output).sort_values(["entry_time", "symbol"])
    accepted = portfolio_gate(
        combined.to_dict("records"), None, rolling_hour_cap=4,
        rank_key=quality.ranker("z_high"), loss_streak_cap=2)
    return pd.DataFrame(accepted, columns=combined.columns)


def fixed_rows(resolved: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frame = resolved[policy_id(BASE_POLICY)].copy()
    frame["module"] = "native_xperp_fixed_exit"
    frame["module_priority"] = 0
    rows = portfolio_gate(
        frame.sort_values(["entry_time", "symbol"]).to_dict("records"), None,
        rolling_hour_cap=4, rank_key=quality.ranker("z_high"), loss_streak_cap=2)
    return pd.DataFrame(rows, columns=frame.columns)


def month_rows(frame: pd.DataFrame, month: str) -> pd.DataFrame:
    values = pd.to_datetime(frame.entry_time, unit="s", utc=True).dt.strftime("%Y-%m")
    return frame[values.eq(month)]


def evaluate(frame: pd.DataFrame) -> dict:
    return {month: {
        "margin_3pct": forward.metrics(month_rows(frame, month), .03),
        "margin_5pct": forward.metrics(month_rows(frame, month), .05),
    } for month in MONTHS}


def selection_ok(values: dict) -> bool:
    return all(
        values[month]["margin_3pct"]["return_pct"] > 0
        and values[month]["margin_3pct"]["dd_pct"] <= 5
        and values[month]["margin_3pct"]["trades_per_active_day"] >= 3
        and values[month]["margin_3pct"]["negative_months"] == 0
        for month in ("2026-07", "2026-08"))


def selection_score(values: dict) -> float:
    returns = [values[month]["margin_3pct"]["return_pct"]
               for month in ("2026-07", "2026-08")]
    drawdown = max(values[month]["margin_3pct"]["dd_pct"]
                   for month in ("2026-07", "2026-08"))
    return round(min(returns) + .2*sum(returns) - .5*drawdown, 8)


def dominates(value: dict, baseline: dict) -> bool:
    return bool(
        value["return_pct"] >= baseline["return_pct"]
        and value["dd_pct"] <= baseline["dd_pct"] + .25
        and value["trades_per_active_day"] >= 3
        and value["negative_months"] == 0)


def main() -> None:
    entries, candles, sources = load_inputs()
    print(json.dumps({"candidate_entries": len(entries)}), flush=True)
    resolved = resolve_policies(entries, candles)
    print(json.dumps({"exit_policies_resolved": len(resolved)}), flush=True)
    stats, causal_integrity, policy_ids = precompute_stats(entries, resolved)
    print(json.dumps({"causal_statistics_ready": causal_integrity}), flush=True)
    lookups = outcome_lookups(resolved)
    baseline_rows = fixed_rows(resolved)
    baseline = evaluate(baseline_rows)
    candidates = []
    configs = itertools.product(
        REGIME_SPECS, (12, 24), WINDOWS, (10, 30), (0.0, .5), (0.0, .0005))
    for config_index, (spec, minimum, window, shrinkage, penalty, uplift) in enumerate(configs, 1):
        config = {
            "regime_fields": list(spec), "minimum_history": minimum,
            "window": window, "shrinkage": shrinkage,
            "uncertainty_penalty": penalty, "minimum_uplift": uplift,
        }
        rows = choose_rows(entries, lookups, stats, policy_ids, config)
        values = evaluate(rows)
        candidates.append({
            "config": config, "selection_ok": selection_ok(values),
            "selection_score": selection_score(values),
            "selection": {month: values[month]
                          for month in ("2026-07", "2026-08")},
        })
        if config_index % 24 == 0:
            print(json.dumps({"router_configs_completed": config_index}), flush=True)
    eligible = [item for item in candidates if item["selection_ok"]]
    eligible.sort(key=lambda item: (-item["selection_score"],
                                   json.dumps(item["config"], sort_keys=True)))
    selected = eligible[0]
    selected_rows = choose_rows(
        entries, lookups, stats, policy_ids, selected["config"])
    values = evaluate(selected_rows)
    selected_rows.to_csv(ROWS_OUT, index=False)
    test_pass = dominates(values["2026-09"]["margin_3pct"],
                          baseline["2026-09"]["margin_3pct"])
    confirmation_pass = bool(
        test_pass and dominates(values["2026-10"]["margin_3pct"],
                                baseline["2026-10"]["margin_3pct"]))
    objective = all(
        values[month]["margin_3pct"]["return_pct"] >= 30
        and values[month]["margin_3pct"]["negative_months"] == 0
        and values[month]["margin_3pct"]["trades_per_active_day"] >= 3
        for month in MONTHS)
    report = {
        "status": ("SHADOW_CANDIDATE_REQUIRES_GENUINELY_NEW_FORWARD"
                   if confirmation_pass else "REJECTED_OUT_OF_SAMPLE"),
        "ready_for_live_orders": False, "orders_possible": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "entry_route": "frozen basket-trend route",
            "entry": "next 15-minute market open", "limit_orders": False,
            "virtual_training_cutoff": "policy exit_time strictly before entry_time",
            "selection": ["2026-07", "2026-08"],
            "test": "2026-09", "confirmation": "2026-10",
            "targets_pct": list(TARGETS), "stops_pct": list(STOPS),
            "hold_bars": list(HOLDS), "candidate_router_configs": len(candidates),
        },
        "integrity": {**causal_integrity, "eligible_configs": len(eligible)},
        "baseline": baseline,
        "selected": {**selected, "all_months": values},
        "test_pass": test_pass, "confirmation_pass": confirmation_pass,
        "objective_30pct_every_month_at_3pct": objective,
        "selection_top20": eligible[:20],
        "sources": {**sources, str(MANIFEST): digest(MANIFEST),
                    str(Path(__file__).resolve()): digest(Path(__file__).resolve())},
        "limitations": [
            "All four months were already observed before this audit; a passing route still needs new forward data.",
            "Native stock X-Perp history covers only late June through October 2026.",
            "Virtual counterfactual exits use public OHLC and cannot reproduce latency or intrabar order sequence.",
            "Gross returns exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "integrity": report["integrity"],
        "selected_config": selected["config"], "baseline": baseline,
        "selected_all_months": values, "test_pass": test_pass,
        "confirmation_pass": confirmation_pass,
        "objective_30pct_every_month_at_3pct": objective,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
