"""Sequential audit of the stock X-Perp 2/2.5/3% shadow target router."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pandas as pd

import stock_xperp_basket_trend_forward_oct30_audit as forward
import stock_xperp_native_growth_forward_monitor as monitor
import stock_xperp_short_exhaustion_guard_oct30_audit as guard
import stock_xperp_short_stack_guard_oct30_forward as source
from src.stock_xperp_native_shadow import (
    TARGET_EXHAUSTION_THRESHOLD, TARGET_EXPANSION_END_HOUR,
    TARGET_EXPANSION_START_HOUR, routed_target_fraction,
)


ROOT = Path("reports/audit_2026_09_08")
MANIFEST = ROOT / "stock_xperp_forward_oct30" / "manifest.json"
FREEZE_MANIFEST = ROOT / "stock_xperp_forward_oct27_2156" / "manifest.json"
OUT = ROOT / "stock_xperp_regime_target_router_oct30.json"
ROWS_OUT = ROOT / "stock_xperp_regime_target_router_oct30.csv"
TARGETS = (2.0, 2.5, 3.0)
EXHAUSTION_THRESHOLDS = (-.02, -.04)
EXPANSION_WINDOWS = ((0, 6), (6, 12), (12, 18), (18, 24))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(row: dict) -> tuple[str, int, str]:
    return str(row["symbol"]), int(row["entry_time"]), str(row["direction"])


def resolve(entries: pd.DataFrame, candles: dict, target: float) -> pd.DataFrame:
    frame = pd.DataFrame(monitor.resolve_entries(entries, candles, {
        "target_pct": target, "stop_pct": 1.0, "hold_bars": 64}))
    frame["module"] = "native_xperp_regime_target_router"
    frame["module_priority"] = 0
    frame["audited_target_pct"] = target
    return frame


def route(entries: pd.DataFrame, resolved: dict[float, pd.DataFrame],
          threshold: float, expansion: tuple[int, int]) -> pd.DataFrame:
    lookups = {
        target: {identity(row): row for row in frame.to_dict("records")}
        for target, frame in resolved.items()}
    start, end = expansion
    output = []
    for entry in entries.to_dict("records"):
        hour = int(float(entry["entry_time"]) // 3600) % 24
        if float(entry["basket_trend_32"]) <= threshold:
            target = 2.0
        elif start <= hour < end:
            target = 3.0
        else:
            target = 2.5
        output.append(lookups[target][identity(entry)])
    return guard.accept(pd.DataFrame(output))


def fixed(resolved: dict[float, pd.DataFrame], target: float) -> pd.DataFrame:
    return guard.accept(resolved[target])


def slice_time(frame: pd.DataFrame, start: str,
               end: str | None = None) -> pd.DataFrame:
    stamp = pd.to_datetime(frame.entry_time, unit="s", utc=True)
    mask = stamp.ge(pd.Timestamp(start, tz="UTC"))
    if end is not None:
        mask &= stamp.lt(pd.Timestamp(end, tz="UTC"))
    return frame[mask].copy()


def evaluate(frame: pd.DataFrame, freeze_end: int) -> dict:
    periods = {
        "selection_july": slice_time(frame, "2026-07-01", "2026-08-01"),
        "selection_august": slice_time(frame, "2026-08-01", "2026-09-01"),
        "test_september": slice_time(frame, "2026-09-01", "2026-10-01"),
        "confirmation_october_to_freeze": frame[
            frame.entry_time.ge(pd.Timestamp("2026-10-01", tz="UTC").timestamp())
            & frame.entry_time.le(freeze_end)].copy(),
        "new_resolved_after_freeze": frame[
            frame.outcome.ne("PENDING") & frame.exit_time.gt(freeze_end)].copy(),
        "october_to_date": slice_time(frame, "2026-10-01", "2026-11-01"),
    }
    return {name: {
        "margin_3pct": forward.metrics(rows, .03),
        "margin_5pct": forward.metrics(rows, .05),
    } for name, rows in periods.items()}


def score(values: dict) -> float:
    july = values["selection_july"]["margin_3pct"]
    august = values["selection_august"]["margin_3pct"]
    return round(july["return_pct"] + august["return_pct"]
                 - .5 * max(july["dd_pct"], august["dd_pct"]), 8)


def dominates(value: dict, baseline: dict, dd_tolerance: float = .30) -> bool:
    return bool(value["return_pct"] >= baseline["return_pct"]
                and value["dd_pct"] <= baseline["dd_pct"] + dd_tolerance
                and value["trades_per_active_day"] >= 3
                and value["negative_months"] == 0)


def compact(values: dict) -> dict:
    return {name: {
        "return_pct": value["margin_3pct"]["return_pct"],
        "dd_pct": value["margin_3pct"]["dd_pct"],
        "resolved": value["margin_3pct"]["resolved"],
        "pending": value["margin_3pct"]["pending"],
    } for name, value in values.items()}


def main() -> None:
    raw, sources, candles = source.candidates(MANIFEST)
    entries = guard.apply_guard(raw, -.01)
    resolved = {target: resolve(entries, candles, target) for target in TARGETS}
    freeze = json.loads(FREEZE_MANIFEST.read_text(encoding="utf-8"))
    freeze_end = min(int(item["end"]) for item in freeze["symbols"].values()
                     if item.get("end") is not None)
    baseline_rows = fixed(resolved, 2.5)
    baseline = evaluate(baseline_rows, freeze_end)
    candidates = []
    frames = {}
    for threshold in EXHAUSTION_THRESHOLDS:
        for expansion in EXPANSION_WINDOWS:
            name = f"threshold_{threshold:g}_hours_{expansion[0]}_{expansion[1]}"
            frame = route(entries, resolved, threshold, expansion)
            values = evaluate(frame, freeze_end)
            frames[name] = frame
            candidates.append({
                "name": name, "threshold": threshold,
                "expansion": list(expansion), "score": score(values),
                "selection": {key: values[key] for key in (
                    "selection_july", "selection_august")},
            })
    candidates.sort(key=lambda item: (-item["score"], item["name"]))
    selected_config = candidates[0]
    selected_rows = frames[selected_config["name"]]
    selected = evaluate(selected_rows, freeze_end)
    expected = {
        "threshold": TARGET_EXHAUSTION_THRESHOLD,
        "expansion": [TARGET_EXPANSION_START_HOUR, TARGET_EXPANSION_END_HOUR],
    }
    runtime_parity = bool(
        selected_config["threshold"] == expected["threshold"]
        and selected_config["expansion"] == expected["expansion"]
        and routed_target_fraction(-.05, 8 * 3600) == .02
        and routed_target_fraction(-.02, 8 * 3600) == .03
        and routed_target_fraction(-.02, 14 * 3600) == .025)
    test_pass = dominates(
        selected["test_september"]["margin_3pct"],
        baseline["test_september"]["margin_3pct"])
    confirmation_pass = bool(test_pass and dominates(
        selected["confirmation_october_to_freeze"]["margin_3pct"],
        baseline["confirmation_october_to_freeze"]["margin_3pct"])
    )
    forward_pass = bool(confirmation_pass and dominates(
        selected["new_resolved_after_freeze"]["margin_3pct"],
        baseline["new_resolved_after_freeze"]["margin_3pct"], 0.0))
    accepted = bool(runtime_parity and forward_pass)
    selected_rows.to_csv(ROWS_OUT, index=False)
    report = {
        "status": "SHADOW_ONLY_FORWARD_EXPERIMENT" if accepted else "REJECTED",
        "ready_for_live_orders": False, "orders_possible": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "selection": ["2026-07", "2026-08"], "test": "2026-09",
            "confirmation": "2026-10 through Oct-27 freeze",
            "forward_diagnostic": "resolved outcomes after Oct-27 freeze",
            "entry": "next 15-minute market open", "orders": "market only",
            "stop_pct": 1.0, "hold_bars": 64,
            "position_fraction": "client setting",
        },
        "selected": selected_config, "runtime_parity": runtime_parity,
        "test_pass": test_pass, "confirmation_pass": confirmation_pass,
        "forward_pass": forward_pass,
        "baseline_fixed_2_5pct": baseline, "selected_router": selected,
        "candidate_selection": candidates,
        "sources": {**sources, str(MANIFEST): digest(MANIFEST),
                    str(FREEZE_MANIFEST): digest(FREEZE_MANIFEST),
                    str(Path(__file__).resolve()): digest(Path(__file__).resolve())},
        "limitations": [
            "Native X-Perp history begins in late June 2026.",
            "The post-freeze sample contains only two active days.",
            "The target router remains paper-only pending more forward outcomes.",
            "Gross returns exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "selected": selected_config,
        "runtime_parity": runtime_parity, "test_pass": test_pass,
        "confirmation_pass": confirmation_pass, "forward_pass": forward_pass,
        "baseline": compact(baseline), "router": compact(selected),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
