"""Test 1.5-5% stock X-Perp targets without selecting on holdouts.

Exact target outcomes are expensive to resolve, so each target is cached once.
The policy grid is ranked only on July-August.  September and October are
opened after selection, and an October-oracle diagnostic is reported only to
prove whether the requested 30% was present anywhere in the bounded grid.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path

import pandas as pd

import stock_xperp_regime_target_router_oct30_audit as current
import stock_xperp_short_exhaustion_guard_oct30_audit as guard
import stock_xperp_short_stack_guard_oct30_forward as source


ROOT = Path("reports/audit_2026_09_08")
OUT = ROOT / "stock_xperp_extended_target_router_oct31.json"
TARGETS = (1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0)
EXPANDED_TARGETS = (3.0, 3.5, 4.0, 5.0)
THRESHOLDS = (-.06, -.04, -.02)
WINDOWS = ((0, 6), (6, 12), (12, 18), (18, 24), (6, 18))
CACHE_NAMES = {
    1.5: "1p5", 2.0: "2", 2.5: "2p5", 3.0: "3",
    3.5: "3p5", 4.0: "4", 5.0: "5",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cache_path(target: float) -> Path:
    return ROOT / f"_tmp_stock_target_{CACHE_NAMES[target]}_oct31.pkl"


def target_frames() -> tuple[dict[float, pd.DataFrame], dict]:
    missing = [target for target in TARGETS if not cache_path(target).exists()]
    sources = {}
    if missing:
        raw, sources, candles = source.candidates(current.MANIFEST)
        entries = guard.apply_guard(raw, -.01)
        for target in missing:
            current.resolve(entries, candles, target).to_pickle(cache_path(target))
    frames = {target: pd.read_pickle(cache_path(target)) for target in TARGETS}
    identities = {
        target: set(map(current.identity, frame.to_dict("records")))
        for target, frame in frames.items()
    }
    reference = identities[2.5]
    if any(value != reference for value in identities.values()):
        raise ValueError("target caches do not contain the same entries")
    return frames, sources


def route(frames: dict[float, pd.DataFrame], expanded: float,
          threshold: float, window: tuple[int, int]) -> pd.DataFrame:
    lookups = {
        target: {current.identity(row): row for row in frame.to_dict("records")}
        for target, frame in frames.items()
    }
    start, end = window
    output = []
    for entry in frames[2.5].to_dict("records"):
        hour = int(float(entry["entry_time"]) // 3600) % 24
        if float(entry["basket_trend_32"]) <= threshold:
            target = 2.0
        elif start <= hour < end:
            target = expanded
        else:
            target = 2.5
        output.append(lookups[target][current.identity(entry)])
    return guard.accept(pd.DataFrame(output))


def periods(frame: pd.DataFrame) -> dict[str, dict]:
    boundaries = {
        "selection_july": ("2026-07-01", "2026-08-01"),
        "selection_august": ("2026-08-01", "2026-09-01"),
        "test_september": ("2026-09-01", "2026-10-01"),
        "confirmation_october": ("2026-10-01", "2026-11-01"),
    }
    return {
        name: current.forward.metrics(current.slice_time(frame, start, end), .03)
        for name, (start, end) in boundaries.items()
    }


def compact(value: dict) -> dict:
    keys = ("n", "return_pct", "dd_pct", "profitable_exit_rate_pct",
            "max_loss_streak", "trades_per_active_day")
    return {key: value[key] for key in keys}


def selection_score(values: dict[str, dict]) -> float:
    july = values["selection_july"]
    august = values["selection_august"]
    return round(
        min(july["return_pct"], august["return_pct"])
        + .25 * (july["return_pct"] + august["return_pct"])
        - .5 * max(july["dd_pct"], august["dd_pct"]), 8)


def main() -> None:
    frames, regenerated_sources = target_frames()
    fixed = {
        str(target): periods(guard.accept(frame))
        for target, frame in frames.items()
    }
    candidates = []
    candidate_frames = {}
    for expanded, threshold, window in itertools.product(
            EXPANDED_TARGETS, THRESHOLDS, WINDOWS):
        name = f"expanded_{expanded:g}_threshold_{threshold:g}_hours_{window[0]}_{window[1]}"
        frame = route(frames, expanded, threshold, window)
        values = periods(frame)
        candidate_frames[name] = frame
        candidates.append({
            "name": name, "expanded_target_pct": expanded,
            "exhausted_target_pct": 2.0, "base_target_pct": 2.5,
            "exhaustion_threshold": threshold, "expansion_window": list(window),
            "selection_score": selection_score(values),
            "periods": {key: compact(value) for key, value in values.items()},
        })
    candidates.sort(key=lambda item: (-item["selection_score"], item["name"]))
    selected = candidates[0]
    runtime_name = "expanded_3_threshold_-0.04_hours_6_12"
    runtime = next(item for item in candidates if item["name"] == runtime_name)
    selected_holdout_pass = bool(
        selected["periods"]["test_september"]["return_pct"]
            >= runtime["periods"]["test_september"]["return_pct"]
        and selected["periods"]["test_september"]["dd_pct"]
            <= runtime["periods"]["test_september"]["dd_pct"] + .25
        and selected["periods"]["confirmation_october"]["return_pct"]
            >= runtime["periods"]["confirmation_october"]["return_pct"]
        and selected["periods"]["confirmation_october"]["dd_pct"]
            <= runtime["periods"]["confirmation_october"]["dd_pct"] + .25)
    october_oracle = max(
        candidates,
        key=lambda item: item["periods"]["confirmation_october"]["return_pct"])
    report = {
        "status": "REJECTED_EXTENDED_TARGET_ROUTER",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "ready_for_live_orders": False, "orders_possible": False,
        "runtime_change_allowed": False,
        "protocol": {
            "selection": ["2026-07", "2026-08"],
            "sealed_holdouts": ["2026-09", "2026-10"],
            "targets_pct": list(TARGETS),
            "expanded_targets_pct": list(EXPANDED_TARGETS),
            "exhaustion_thresholds": list(THRESHOLDS),
            "expansion_windows_utc": [list(value) for value in WINDOWS],
            "margin_fraction": .03, "leverage": 10,
            "position_fraction": "client setting", "orders": "market only",
        },
        "fixed_target_diagnostics": {
            target: {key: compact(value) for key, value in periods_.items()}
            for target, periods_ in fixed.items()
        },
        "selected_before_holdouts": selected,
        "current_runtime_policy": runtime,
        "selected_holdout_pass": selected_holdout_pass,
        "october_oracle_diagnostic_not_selectable": october_oracle,
        "october_grid_max_return_pct": october_oracle[
            "periods"]["confirmation_october"]["return_pct"],
        "october_grid_reaches_30pct": bool(october_oracle[
            "periods"]["confirmation_october"]["return_pct"] >= 30),
        "candidate_count": len(candidates),
        "candidates": candidates,
        "sources": {
            str(current.MANIFEST): digest(current.MANIFEST),
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
            **regenerated_sources,
        },
        "limitations": [
            "The X-Perp history starts in late June 2026.",
            "No October-aware policy is eligible for selection.",
            "No tested target route reached 30% in October at 3% margin.",
            "Gross results exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "selected": selected, "runtime": runtime,
        "selected_holdout_pass": selected_holdout_pass,
        "october_oracle": october_oracle,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
