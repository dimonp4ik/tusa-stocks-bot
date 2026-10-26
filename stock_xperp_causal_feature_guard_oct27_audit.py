"""Mine simple causal feature guards on the frozen basket-trend stock route.

Only one entry-time feature, one threshold and optionally one side may be used.
July and August select the rule, September validates it, and already-observed
October is a diagnostic.  Exit geometry and portfolio rails stay frozen.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import stock_xperp_basket_trend_forward_oct27_audit as forward
import stock_xperp_current_barbell_oct27_audit as current
import stock_xperp_native_growth_forward_monitor as monitor


ROOT = Path("reports/audit_2026_09_08")
OUT = ROOT / "stock_xperp_causal_feature_guard_oct27.json"
ROWS_OUT = ROOT / "stock_xperp_causal_feature_guard_oct27.csv"
QUANTILES = (.1, .2, .3, .4, .5, .6, .7, .8, .9)
SIDES = ("ALL", "LONG", "SHORT")
RELATIONS = ("low", "high")
FEATURES = (
    "abs_cross_z", "abs_signal_return", "cross_std", "atr14_fraction",
    "backing_level_distance", "signal_atr_ratio", "market_alignment",
    "basket_alignment_16", "basket_alignment_32", "basket_alignment_64",
    "basket_alignment_96",
)
PERIODS = {
    "selection_july": ("2026-07-01", "2026-08-01"),
    "selection_august": ("2026-08-01", "2026-09-01"),
    "validation_september": ("2026-09-01", "2026-10-01"),
    "observed_october": ("2026-10-01", "2026-11-01"),
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare() -> tuple[pd.DataFrame, dict]:
    entries, candles, sources = current.prepare_entries()
    outcomes = pd.DataFrame(monitor.resolve_entries(entries, candles, {
        "target_pct": 2.0, "stop_pct": 1.0, "hold_bars": 64}))
    outcomes["module"] = "native_xperp_causal_feature_guard"
    outcomes["module_priority"] = 0
    outcomes["abs_cross_z"] = outcomes.cross_z.abs()
    outcomes["abs_signal_return"] = outcomes.signal_return.abs()
    outcomes["signal_atr_ratio"] = (
        outcomes.abs_signal_return / outcomes.atr14_fraction.replace(0, np.nan))
    outcomes["market_alignment"] = outcomes.direction_n * outcomes.cross_mean
    for horizon in (16, 32, 64, 96):
        outcomes[f"basket_alignment_{horizon}"] = (
            outcomes.direction_n * outcomes[f"basket_trend_{horizon}"])
    return outcomes, sources


def apply(frame: pd.DataFrame, feature: str, relation: str,
          side: str, threshold: float) -> pd.DataFrame:
    condition = (frame[feature].le(threshold) if relation == "low"
                 else frame[feature].ge(threshold))
    if side != "ALL":
        condition = frame.direction.ne(side) | condition
    return frame[condition.fillna(False)].copy()


def gated(frame: pd.DataFrame) -> pd.DataFrame:
    return current.gated(frame)


def bounds(frame: pd.DataFrame, period: tuple[str, str]) -> pd.DataFrame:
    lo = pd.Timestamp(period[0], tz="UTC").timestamp()
    hi = pd.Timestamp(period[1], tz="UTC").timestamp()
    return frame[frame.entry_time.ge(lo) & frame.entry_time.lt(hi)]


def evaluate(frame: pd.DataFrame) -> dict:
    return {name: {"margin_3pct": forward.metrics(bounds(frame, span), .03),
                   "margin_5pct": forward.metrics(bounds(frame, span), .05)}
            for name, span in PERIODS.items()}


def public(item: dict) -> dict:
    return {key: value for key, value in item.items()
            if not key.startswith("_")}


def main() -> None:
    raw, sources = prepare()
    baseline_rows = gated(raw)
    baseline = evaluate(baseline_rows)
    selection = pd.concat([
        bounds(raw, PERIODS["selection_july"]),
        bounds(raw, PERIODS["selection_august"]),
    ], ignore_index=True)
    candidates = []
    seen = set()
    for feature, relation, side, quantile in itertools.product(
            FEATURES, RELATIONS, SIDES, QUANTILES):
        threshold = float(selection[feature].quantile(quantile))
        filtered = apply(raw, feature, relation, side, threshold)
        signature = (feature, side, tuple(filtered.index))
        if signature in seen:
            continue
        seen.add(signature)
        accepted = gated(filtered)
        values = evaluate(accepted)
        improvements = []
        selection_ok = True
        for month in ("selection_july", "selection_august"):
            value = values[month]["margin_3pct"]
            base = baseline[month]["margin_3pct"]
            improvements.append(value["return_pct"] - base["return_pct"])
            selection_ok &= bool(
                value["return_pct"] > base["return_pct"]
                and value["dd_pct"] <= min(5.0, base["dd_pct"] + .5)
                and value["trades_per_active_day"] >= 3)
        score = min(improvements) - .25 * max(
            values[month]["margin_3pct"]["dd_pct"]
            for month in ("selection_july", "selection_august"))
        candidates.append({
            "feature": feature, "relation": relation, "side": side,
            "selection_quantile": quantile, "threshold": threshold,
            "selection_score": round(score, 8),
            "selection_ok": selection_ok, "values": values,
            "_rows": accepted,
        })
    eligible = [item for item in candidates if item["selection_ok"]]
    eligible.sort(key=lambda item: item["selection_score"], reverse=True)
    selected = eligible[0] if eligible else None
    validation_pass = october_pass = False
    if selected:
        validation = selected["values"]["validation_september"]["margin_3pct"]
        validation_base = baseline["validation_september"]["margin_3pct"]
        validation_pass = bool(
            validation["return_pct"] > validation_base["return_pct"]
            and validation["dd_pct"] <= 5
            and validation["trades_per_active_day"] >= 3)
        october = selected["values"]["observed_october"]["margin_3pct"]
        october_base = baseline["observed_october"]["margin_3pct"]
        october_pass = bool(
            october["return_pct"] > october_base["return_pct"]
            and october["dd_pct"] <= 5
            and october["trades_per_active_day"] >= 3)
        selected["_rows"].to_csv(ROWS_OUT, index=False)
    accepted = bool(selected and validation_pass and october_pass)
    report = {
        "status": ("SHADOW_CANDIDATE_REQUIRES_GENUINELY_NEW_FORWARD"
                   if accepted else "REJECTED_NO_ROBUST_IMPROVEMENT"),
        "ready_for_live_orders": False, "orders_possible": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "mechanism": "one causal feature threshold, optionally on one side",
            "selection": ["2026-07", "2026-08"],
            "validation": "2026-09",
            "already_observed_diagnostic": "2026-10 through snapshot end",
            "unchanged_exit": "2% target / 1% stop / 64 bars, stop first",
            "entry": "next 15-minute market open", "orders": "market only",
            "money": "client-selected 3% or 5% current-equity margin; fixed 10x",
        },
        "grid": {"features": list(FEATURES), "relations": list(RELATIONS),
                 "sides": list(SIDES), "quantiles": list(QUANTILES),
                 "unique_candidates": len(candidates),
                 "selection_survivors": len(eligible)},
        "baseline": baseline,
        "selected": public(selected) if selected else None,
        "top20_selection": [public(item) for item in eligible[:20]],
        "checks": {"validation_september": validation_pass,
                   "observed_october": october_pass,
                   "accepted": accepted},
        "sources": {**sources,
                    str(current.MANIFEST): digest(current.MANIFEST),
                    str(Path(__file__).resolve()): digest(Path(__file__).resolve())},
        "limitations": [
            "October was already observed and is diagnostic, not fresh evidence.",
            "The native X-Perp history begins in July 2026.",
            "Threshold search creates multiple-testing risk despite simple rules.",
            "Public OHLC cannot reproduce live fill latency.",
            "Gross returns exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": report["status"], "grid": report["grid"],
                      "selected": report["selected"],
                      "checks": report["checks"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
