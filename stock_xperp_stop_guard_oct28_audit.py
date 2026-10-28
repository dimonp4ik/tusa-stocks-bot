"""Causal pre-entry stop-guard search for the frozen native stock route.

Rules use only fields known at entry.  July and August must both improve before
September and October are inspected.  Filtering happens before the portfolio
gate so removed losses may be replaced by later valid signals.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from stock_module_cluster_audit import portfolio_gate
import stock_xperp_basket_trend_long_guard_oct26_audit as frozen
import stock_xperp_native_quality_guard_oct18_audit as quality
import stock_xperp_third_target_oct28_audit as common


ROOT = Path("reports/audit_2026_09_08")
OUT = ROOT / "stock_xperp_stop_guard_oct28.json"
ROWS_OUT = ROOT / "stock_xperp_stop_guard_oct28.csv"
MONTHS = ("2026-07", "2026-08", "2026-09", "2026-10")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def admit(frame: pd.DataFrame) -> pd.DataFrame:
    accepted = portfolio_gate(
        frame.to_dict("records"), None, rolling_hour_cap=4,
        rank_key=quality.ranker("z_high"), loss_streak_cap=2)
    return pd.DataFrame(accepted, columns=frame.columns)


def metric(frame: pd.DataFrame) -> dict:
    value = common.forward.metrics(frame, .03)
    resolved = frame[frame.outcome.ne("PENDING")]
    stops = int(resolved.outcome.eq("SL").sum())
    value.update({
        "stops": stops,
        "stop_rate_pct": round(100*stops/len(resolved), 4) if len(resolved) else 0.0,
    })
    return value


def periods(frame: pd.DataFrame) -> dict:
    labels = pd.to_datetime(frame.entry_time, unit="s", utc=True).dt.strftime("%Y-%m")
    return {month: metric(frame[labels.eq(month)]) for month in MONTHS}


def compact(values: dict) -> dict:
    keys = ("n", "return_pct", "dd_pct", "profitable_exit_rate_pct",
            "trades_per_active_day", "max_loss_streak", "stops", "stop_rate_pct")
    return {month: {key: item[key] for key in keys}
            for month, item in values.items()}


def rule_masks(frame: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    output = []
    absolute_z = frame.cross_z.abs()
    absolute_move = frame.signal_return.abs()
    for value in (2.25, 2.5, 2.75, 3.0):
        output.append((f"abs_z>={value}", absolute_z.ge(value)))
    for value in (.005, .006, .008, .010, .0125):
        output.append((f"abs_move>={100*value:.2f}%", absolute_move.ge(value)))
    for value in (.004, .005, .006, .0075, .009):
        output.append((f"atr14<={100*value:.2f}%", frame.atr14_fraction.le(value)))
    for value in (.005, .010, .015, .020):
        output.append((f"backing_distance>={100*value:.2f}%",
                       frame.backing_level_distance.ge(value)))
    for value in (.02, .03, .04, .05):
        output.append((f"backing_distance<={100*value:.2f}%",
                       frame.backing_level_distance.le(value)))
    directional_basket = frame.direction_n * frame.basket_trend_96
    for value in (-.02, -.01, -.005, 0.0, .005):
        output.append((f"directional_basket96>={100*value:.2f}%",
                       directional_basket.ge(value)))
    hours = pd.to_datetime(frame.entry_time, unit="s", utc=True).dt.hour
    for start in (0, 6, 12, 18):
        output.append((f"exclude_utc_{start:02d}_{start+5:02d}",
                       ~hours.between(start, start+5)))
    output.extend([
        ("long_only", frame.direction_n.eq(1)),
        ("short_only", frame.direction_n.eq(-1)),
        ("abs_z>=2.25_and_backing>=0.50%",
         absolute_z.ge(2.25) & frame.backing_level_distance.ge(.005)),
        ("abs_z>=2.25_and_atr<=0.75%",
         absolute_z.ge(2.25) & frame.atr14_fraction.le(.0075)),
        ("move>=0.60%_and_atr<=0.75%",
         absolute_move.ge(.006) & frame.atr14_fraction.le(.0075)),
    ])
    return output


def main() -> None:
    entries, candles, sources = common.raw_entries()
    resolved = pd.DataFrame(
        common.resolve(row, candles[row["symbol"]], 1.0, 3.0)
        for row in entries.to_dict("records"))
    guarded = frozen.apply(resolved, 96, "LONG", -0.005626823134606253)
    baseline_rows = admit(guarded)
    baseline = periods(baseline_rows)
    candidates = []
    for name, mask in rule_masks(guarded):
        rows = admit(guarded[mask].copy())
        values = periods(rows)
        development_pass = all(
            values[month]["return_pct"] > baseline[month]["return_pct"]
            and values[month]["dd_pct"] <= baseline[month]["dd_pct"] + .25
            and values[month]["stop_rate_pct"] < baseline[month]["stop_rate_pct"]
            and values[month]["trades_per_active_day"] >= 3
            for month in ("2026-07", "2026-08"))
        candidates.append({
            "rule": name, "development_pass": development_pass,
            "development_min_return_gain": min(
                values[m]["return_pct"]-baseline[m]["return_pct"]
                for m in ("2026-07", "2026-08")),
            "periods": compact(values), "_rows": rows,
        })
    qualified = [item for item in candidates if item["development_pass"]]
    qualified.sort(key=lambda item: item["development_min_return_gain"], reverse=True)
    selected = qualified[0] if qualified else None
    holdout_pass = bool(selected and all(
        selected["periods"][month]["return_pct"] >= baseline[month]["return_pct"]
        and selected["periods"][month]["dd_pct"] <= baseline[month]["dd_pct"] + .25
        and selected["periods"][month]["stop_rate_pct"] < baseline[month]["stop_rate_pct"]
        for month in ("2026-09", "2026-10")))
    if selected:
        selected["_rows"].to_csv(ROWS_OUT, index=False)
    report = {
        "status": ("RESEARCH_CANDIDATE_AWAIT_FRESH_FORWARD" if holdout_pass
                   else "REJECTED_HOLDOUT" if selected
                   else "REJECTED_DEVELOPMENT"),
        "ready_for_live_orders": False, "orders_possible": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "entry_and_exit": "unchanged frozen 2% target / 1% stop / 64 bars",
            "rule_information": "closed candles and cross-section known before entry",
            "filter_order": "before portfolio gate with causal backfill",
            "selection": ["2026-07", "2026-08"],
            "holdouts": ["2026-09", "2026-10"],
            "money": "3% current-equity margin; fixed 10x; market only",
        },
        "grid": {"candidate_rules": len(candidates)},
        "baseline": compact(baseline),
        "selected_before_holdouts": (None if selected is None else
            {key: value for key, value in selected.items() if key != "_rows"}),
        "holdout_pass": holdout_pass,
        "top_development": [
            {key: value for key, value in item.items() if key != "_rows"}
            for item in sorted(candidates, key=lambda x: x["development_min_return_gain"],
                               reverse=True)[:10]],
        "sources": {str(common.MANIFEST): digest(common.MANIFEST),
                    **sources, str(Path(__file__).resolve()): digest(Path(__file__).resolve())},
        "limitations": [
            "Native X-Perp history begins in July 2026.",
            "October was already observed and cannot activate a rule.",
            "The finite grid still creates multiple-testing risk.",
            "Gross results exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "status", "grid", "baseline", "selected_before_holdouts",
        "holdout_pass")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
