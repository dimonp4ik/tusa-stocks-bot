"""Sequential target audit for the stock SHORT-exhaustion shadow route."""
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
from src.stock_xperp_native_shadow import TARGET_FRACTION


ROOT = Path("reports/audit_2026_09_08")
MANIFEST = ROOT / "stock_xperp_forward_oct30" / "manifest.json"
FREEZE_MANIFEST = ROOT / "stock_xperp_forward_oct27_2156" / "manifest.json"
OUT = ROOT / "stock_xperp_short_exhaustion_target_oct30.json"
ROWS_OUT = ROOT / "stock_xperp_short_exhaustion_target_oct30.csv"
TARGETS = (1.5, 2.0, 2.5, 3.0)
HOLDS_FOR_SELECTED_TARGET = (32, 64, 96)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(entries: pd.DataFrame, candles: dict,
            target: float, hold: int) -> pd.DataFrame:
    rows = monitor.resolve_entries(entries, candles, {
        "target_pct": target, "stop_pct": 1.0, "hold_bars": hold})
    frame = pd.DataFrame(rows)
    frame["module"] = "native_xperp_short_exhaustion_target"
    frame["module_priority"] = 0
    frame["audited_target_pct"] = target
    frame["audited_hold_bars"] = hold
    return guard.accept(frame)


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
    return {
        name: {
            "margin_3pct": forward.metrics(rows, .03),
            "margin_5pct": forward.metrics(rows, .05),
        }
        for name, rows in periods.items()
    }


def selection_score(values: dict) -> float:
    july = values["selection_july"]["margin_3pct"]
    august = values["selection_august"]["margin_3pct"]
    if (july["return_pct"] <= 0 or august["return_pct"] <= 0
            or july["trades_per_active_day"] < 3
            or august["trades_per_active_day"] < 3):
        return float("-inf")
    return round(july["return_pct"] + august["return_pct"]
                 - .5 * max(july["dd_pct"], august["dd_pct"]), 8)


def compare(value: dict, baseline: dict, *, return_tolerance: float = 0.0,
            dd_tolerance: float = .25) -> bool:
    return bool(
        value["return_pct"] >= baseline["return_pct"] - return_tolerance
        and value["dd_pct"] <= baseline["dd_pct"] + dd_tolerance
        and value["trades_per_active_day"] >= 3
        and value["negative_months"] == 0)


def compact(values: dict) -> dict:
    return {
        name: {
            "return_pct": item["margin_3pct"]["return_pct"],
            "dd_pct": item["margin_3pct"]["dd_pct"],
            "resolved": item["margin_3pct"]["resolved"],
            "pending": item["margin_3pct"]["pending"],
        }
        for name, item in values.items()
    }


def main() -> None:
    raw, sources, candles = source.candidates(MANIFEST)
    entries = guard.apply_guard(raw, -.01)
    freeze = json.loads(FREEZE_MANIFEST.read_text(encoding="utf-8"))
    freeze_end = min(int(item["end"]) for item in freeze["symbols"].values()
                     if item.get("end") is not None)

    target_rows = {target: resolve(entries, candles, target, 64)
                   for target in TARGETS}
    target_values = {target: evaluate(rows, freeze_end)
                     for target, rows in target_rows.items()}
    ranked = sorted(TARGETS, key=lambda target: (
        -selection_score(target_values[target]), target))
    selected_target = ranked[0]
    if selected_target != 2.5:
        raise ValueError(f"unexpected selected target: {selected_target}")

    hold_rows = {64: target_rows[selected_target]}
    for hold in HOLDS_FOR_SELECTED_TARGET:
        if hold != 64:
            hold_rows[hold] = resolve(entries, candles, selected_target, hold)
    hold_values = {hold: evaluate(rows, freeze_end)
                   for hold, rows in hold_rows.items()}
    selected_hold = max(HOLDS_FOR_SELECTED_TARGET, key=lambda hold: (
        selection_score(hold_values[hold]), -abs(hold - 64)))
    if selected_hold != 64:
        raise ValueError(f"unexpected selected hold: {selected_hold}")

    baseline = target_values[2.0]
    selected = hold_values[selected_hold]
    test_pass = compare(
        selected["test_september"]["margin_3pct"],
        baseline["test_september"]["margin_3pct"])
    confirmation_pass = bool(test_pass and compare(
        selected["confirmation_october_to_freeze"]["margin_3pct"],
        baseline["confirmation_october_to_freeze"]["margin_3pct"],
        return_tolerance=1.0, dd_tolerance=0.0))
    forward_pass = bool(confirmation_pass and compare(
        selected["new_resolved_after_freeze"]["margin_3pct"],
        baseline["new_resolved_after_freeze"]["margin_3pct"],
        dd_tolerance=0.0))
    runtime_parity = bool(abs(TARGET_FRACTION * 100 - selected_target) < 1e-12)
    accepted = bool(forward_pass and runtime_parity)
    hold_rows[selected_hold].to_csv(ROWS_OUT, index=False)
    report = {
        "status": ("SHADOW_ONLY_FORWARD_EXPERIMENT" if accepted else "REJECTED"),
        "ready_for_live_orders": False, "orders_possible": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "selection": ["2026-07", "2026-08"],
            "test": "2026-09",
            "confirmation": "2026-10 through Oct-27 freeze",
            "forward_diagnostic": "resolved outcomes after Oct-27 freeze",
            "target_candidates_pct": list(TARGETS),
            "hold_candidates_for_selected_target": list(HOLDS_FOR_SELECTED_TARGET),
            "stop_pct": 1.0, "entry": "next 15-minute market open",
            "orders": "market only", "position_fraction": "client setting",
        },
        "selected": {"target_pct": selected_target, "hold_bars": selected_hold},
        "runtime_target_parity": runtime_parity,
        "test_pass": test_pass, "confirmation_pass": confirmation_pass,
        "forward_pass": forward_pass,
        "baseline_2pct": baseline, "selected_2_5pct": selected,
        "target_selection": {
            str(target): {"score": selection_score(values),
                          "periods": compact(values)}
            for target, values in target_values.items()},
        "hold_robustness": {
            str(hold): {"score": selection_score(values),
                        "periods": compact(values)}
            for hold, values in hold_values.items()},
        "sources": {
            **sources, str(MANIFEST): digest(MANIFEST),
            str(FREEZE_MANIFEST): digest(FREEZE_MANIFEST),
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "Native X-Perp history begins in late June 2026.",
            "The post-freeze sample contains only two active days.",
            "A 1 percentage-point confirmation return tolerance was fixed for this audit.",
            "The route remains paper-only until more genuinely new forward outcomes arrive.",
            "Gross returns exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "selected": report["selected"],
        "test_pass": test_pass, "confirmation_pass": confirmation_pass,
        "forward_pass": forward_pass, "baseline": compact(baseline),
        "selected_values": compact(selected),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
