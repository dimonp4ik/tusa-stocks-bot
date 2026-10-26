"""Chronological basket-trend guard for native X-Perp reversal longs."""
from __future__ import annotations

import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import stock_xperp_native_cross_section_oct18_audit as parent
import stock_xperp_native_economics_oct18_audit as economics
import stock_xperp_native_growth_forward_monitor as monitor


ROOT = Path("reports/audit_2026_09_08")
MANIFEST = ROOT / "stock_xperp_forward_oct26" / "manifest.json"
REFERENCE = ROOT / "stock_xperp_backing_level_guard_oct20.csv"
FORWARD = ROOT / "stock_xperp_backing_hour_cap_forward_oct26.csv"
OUT = ROOT / "stock_xperp_basket_trend_long_guard_oct26.json"
ROWS_OUT = ROOT / "stock_xperp_basket_trend_long_guard_oct26.csv"
HORIZONS = (16, 32, 64, 96)
QUANTILES = (.1, .2, .3, .4, .5, .6, .7, .8)
SIDES = ("LONG", "SHORT")
PERIODS = {
    "selection_july": ("2026-07-01", "2026-08-01"),
    "selection_august": ("2026-08-01", "2026-09-01"),
    "validation_september": ("2026-09-01", "2026-10-01"),
    "observed_oct1_oct16": ("2026-10-01", "2026-10-17"),
    "observed_oct17_oct26": ("2026-10-17", "2026-10-27"),
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def add_context(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    candles, sources, _ = monitor.load_manifest(MANIFEST)
    features = parent.feature_rows(candles, 2)
    context = features.groupby("entry_time", as_index=False).cross_mean.first()
    context.sort_values("entry_time", inplace=True)
    context.rename(columns={"cross_mean": "basket_bar_return"}, inplace=True)
    for horizon in HORIZONS:
        context[f"basket_trend_{horizon}"] = context.basket_bar_return.rolling(
            horizon, min_periods=horizon).sum()
    return frame.merge(context, on="entry_time", how="left",
                       validate="many_to_one"), sources


def bounds(frame: pd.DataFrame, period: tuple[str, str]) -> pd.DataFrame:
    lo = pd.Timestamp(period[0], tz="UTC").timestamp()
    hi = pd.Timestamp(period[1], tz="UTC").timestamp()
    return frame[frame.entry_time.ge(lo) & frame.entry_time.lt(hi)]


def apply(frame: pd.DataFrame, horizon: int, side: str,
          minimum_alignment: float) -> pd.DataFrame:
    alignment = frame.direction_n * frame[f"basket_trend_{horizon}"]
    return frame[(frame.direction.ne(side)) | alignment.ge(minimum_alignment)].copy()


def metrics(frame: pd.DataFrame, margin: float) -> dict:
    value = economics.simulate(frame.to_dict("records"), margin)
    days = max(1, pd.to_datetime(
        frame.entry_time, unit="s", utc=True).dt.date.nunique())
    value["trades_per_active_day"] = round(len(frame) / days, 4)
    value["mean_signed_move_pct"] = round(
        100 * float(frame.signed_return.mean()), 6) if len(frame) else 0.0
    value["profitable_exit_rate_pct"] = round(
        100 * float(frame.signed_return.gt(0).mean()), 4) if len(frame) else 0.0
    return value


def evaluate(frame: pd.DataFrame) -> dict:
    return {name: {"margin_3pct": metrics(bounds(frame, span), .03),
                   "margin_5pct": metrics(bounds(frame, span), .05)}
            for name, span in PERIODS.items()}


def main() -> None:
    raw = pd.concat([pd.read_csv(REFERENCE), pd.read_csv(FORWARD)],
                    ignore_index=True)
    full, sources = add_context(raw)
    selection = pd.concat([
        bounds(full, PERIODS["selection_july"]),
        bounds(full, PERIODS["selection_august"])])
    candidates = []
    for horizon, side in itertools.product(HORIZONS, SIDES):
        alignment = selection.direction_n * selection[f"basket_trend_{horizon}"]
        for quantile in QUANTILES:
            threshold = float(alignment.quantile(quantile))
            filtered = apply(full, horizon, side, threshold)
            values = evaluate(filtered)
            july = values["selection_july"]["margin_3pct"]
            august = values["selection_august"]["margin_3pct"]
            score = min(july["return_pct"], august["return_pct"]) - .5 * max(
                july["dd_pct"], august["dd_pct"])
            candidates.append({
                "horizon_bars": horizon, "guarded_side": side,
                "selection_quantile": quantile,
                "minimum_alignment": threshold, "selection_score": score,
                "values": values, "_rows": filtered,
            })
    eligible = [item for item in candidates if all(
        item["values"][name]["margin_3pct"]["return_pct"] > 0
        and item["values"][name]["margin_3pct"]["trades_per_active_day"] >= 3
        for name in ("selection_july", "selection_august"))]
    eligible.sort(key=lambda item: item["selection_score"], reverse=True)
    selected = eligible[0]
    validation = selected["values"]["validation_september"]["margin_3pct"]
    fresh = selected["values"]["observed_oct17_oct26"]["margin_3pct"]
    validation_pass = validation["return_pct"] > 0 and validation["trades_per_active_day"] >= 3
    observed_pass = fresh["return_pct"] > 0 and fresh["dd_pct"] <= 3
    selected["_rows"].to_csv(ROWS_OUT, index=False)
    public = {key: value for key, value in selected.items() if not key.startswith("_")}
    report = {
        "status": ("SHADOW_CANDIDATE_REQUIRES_GENUINELY_NEW_FORWARD"
                   if validation_pass and observed_pass else "REJECTED"),
        "ready_for_live_orders": False, "orders_possible": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "mechanism": "guard one reversal side with the sum of prior closed cross-sectional basket returns",
            "selection": ["2026-07", "2026-08"],
            "validation": "2026-09", "already_observed_diagnostic": "2026-10-01..26",
            "entry": "unchanged next-15m market entry", "orders": "market only",
            "money": "client-selected 3% or 5% current-equity margin; fixed 10x",
        },
        "grid": {"horizons": list(HORIZONS), "quantiles": list(QUANTILES),
                 "guarded_sides": list(SIDES), "candidates": len(candidates),
                 "eligible": len(eligible)},
        "baseline": evaluate(full), "selected": public,
        "validation_pass": validation_pass, "observed_oct17_oct26_pass": observed_pass,
        "sources": {**sources, str(MANIFEST): digest(MANIFEST),
                    str(REFERENCE): digest(REFERENCE), str(FORWARD): digest(FORWARD),
                    str(Path(__file__).resolve()): digest(Path(__file__).resolve())},
        "limitations": [
            "October was already observed before this audit and cannot activate live orders.",
            "The audit conservatively post-filters portfolio-admitted rows.",
            "Native history begins in July 2026.",
            "Gross returns exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": report["status"], "selected": public,
                      "validation_pass": validation_pass,
                      "observed_pass": observed_pass}, indent=2), flush=True)


if __name__ == "__main__":
    main()
