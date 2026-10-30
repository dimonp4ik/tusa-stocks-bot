"""Reject a post-hoc multi-horizon stock SHORT exhaustion veto.

The filter family was motivated by five outcomes already visible in the latest
snapshot.  Thresholds are selected on July-August and checked on September and
the older October prefix, but the candidate remains ineligible for runtime use
because its hypothesis was formulated after seeing the latest outcomes.
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
PREVIOUS_MANIFEST = ROOT / "stock_xperp_forward_oct30" / "manifest.json"
OUT = ROOT / "stock_xperp_multihorizon_short_veto_oct30.json"
B32 = (-.02, -.04, -.06)
B64 = (-.04, -.06, -.08)
B96 = (-.04, -.06, -.08)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(frame: pd.DataFrame) -> dict:
    value = forward.metrics(frame, .03)
    return {key: value[key] for key in (
        "return_pct", "dd_pct", "resolved", "pending",
        "trades_per_active_day", "literal_tp_rate_pct", "max_loss_streak")}


def period(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    stamp = pd.to_datetime(frame.entry_time, unit="s", utc=True)
    return frame[stamp.ge(pd.Timestamp(start, tz="UTC"))
                 & stamp.lt(pd.Timestamp(end, tz="UTC"))].copy()


def evaluate(frame: pd.DataFrame, previous_end: int) -> dict:
    return {
        "selection_july": metrics(period(frame, "2026-07-01", "2026-08-01")),
        "selection_august": metrics(period(frame, "2026-08-01", "2026-09-01")),
        "test_september": metrics(period(frame, "2026-09-01", "2026-10-01")),
        "confirmation_old_october": metrics(frame[
            frame.entry_time.ge(pd.Timestamp("2026-10-01", tz="UTC").timestamp())
            & frame.entry_time.le(previous_end)].copy()),
        "visible_post_hypothesis_slice": metrics(frame[
            frame.outcome.ne("PENDING") & frame.exit_time.gt(previous_end)].copy()),
    }


def compact_delta(value: dict, baseline: dict) -> dict:
    return {
        name: {
            "return_delta_pp": round(
                item["return_pct"]-baseline[name]["return_pct"], 4),
            "dd_delta_pp": round(item["dd_pct"]-baseline[name]["dd_pct"], 4),
            "candidate": item, "baseline": baseline[name],
        }
        for name, item in value.items()
    }


def main() -> None:
    raw, sources, candles = source.candidates(MANIFEST)
    if "basket_trend_64" not in raw.columns:
        raise ValueError("64-bar basket trend missing from causal candidates")
    base_entries = guard.apply_guard(raw, -.01)
    resolved = {target: router.resolve(base_entries, candles, target)
                for target in router.TARGETS}

    def route(entries: pd.DataFrame) -> pd.DataFrame:
        admitted = guard.apply_guard(entries, -.01)
        return router.route(admitted, resolved, -.04, (6, 12))

    previous = json.loads(PREVIOUS_MANIFEST.read_text(encoding="utf-8"))
    previous_end = min(int(item["end"]) for item in previous["symbols"].values()
                       if item.get("end") is not None)
    baseline = evaluate(route(raw), previous_end)
    candidates = []
    frames = {}
    for b32, b64, b96 in itertools.product(B32, B64, B96):
        veto = (raw.direction.eq("SHORT")
                & raw.basket_trend_32.le(b32)
                & raw.basket_trend_64.le(b64)
                & raw.basket_trend_96.le(b96))
        frame = route(raw[~veto].copy())
        values = evaluate(frame, previous_end)
        selection_gain = sum(
            values[name]["return_pct"]-baseline[name]["return_pct"]
            for name in ("selection_july", "selection_august"))
        config = {"basket_32_at_most": b32, "basket_64_at_most": b64,
                  "basket_96_at_most": b96}
        name = json.dumps(config, sort_keys=True)
        frames[name] = frame
        candidates.append({
            "config": config, "vetoed_raw_candidates": int(veto.sum()),
            "selection_return_gain_pp": round(selection_gain, 4),
            "values": values,
        })
    candidates.sort(key=lambda item: (
        -item["selection_return_gain_pp"], item["vetoed_raw_candidates"],
        json.dumps(item["config"], sort_keys=True)))
    selected = candidates[0]
    values = selected["values"]
    test_pass = bool(
        values["test_september"]["return_pct"]
        >= baseline["test_september"]["return_pct"]
        and values["test_september"]["dd_pct"]
        <= baseline["test_september"]["dd_pct"])
    confirmation_pass = bool(
        values["confirmation_old_october"]["return_pct"]
        >= baseline["confirmation_old_october"]["return_pct"]
        and values["confirmation_old_october"]["dd_pct"]
        <= baseline["confirmation_old_october"]["dd_pct"])
    report = {
        "status": "REJECTED_CONFIRMATION_AND_POSTHOC",
        "ready_for_live_orders": False, "orders_possible": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hypothesis": {
            "rule": ("veto SHORT when closed basket trends at 32, 64 and 96 "
                     "bars are all below selected thresholds"),
            "motivation_contamination": (
                "filter family was formulated after the five latest outcomes were visible"),
            "entry": "unchanged next-15-minute market open",
            "position_fraction": "client setting", "orders": "market only",
        },
        "protocol": {
            "grid_size": len(candidates),
            "selection": ["2026-07", "2026-08"],
            "test": "2026-09", "confirmation": "2026-10 through previous snapshot",
            "latest_slice": "diagnostic only; prohibited from acceptance decision",
        },
        "baseline": baseline,
        "selected_on_july_august": selected,
        "selected_deltas": compact_delta(values, baseline),
        "test_pass": test_pass, "confirmation_pass": confirmation_pass,
        "runtime_change_allowed": False,
        "rejection_reasons": [
            "The selected rule lowers return and raises drawdown on the older October confirmation prefix.",
            "The hypothesis was created after seeing the latest five outcomes.",
            "It removes the latest profitable trade together with the four losses.",
        ],
        "top_selection_candidates": candidates[:10],
        "sources": {**sources, str(MANIFEST): digest(MANIFEST),
                    str(PREVIOUS_MANIFEST): digest(PREVIOUS_MANIFEST),
                    str(Path(__file__).resolve()): digest(Path(__file__).resolve())},
        "limitations": [
            "Venue-native stock X-Perp history begins in late June 2026.",
            "The visible latest slice has only five resolved trades.",
            "Gross returns exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "selected": selected,
        "test_pass": test_pass, "confirmation_pass": confirmation_pass,
        "selected_deltas": report["selected_deltas"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
