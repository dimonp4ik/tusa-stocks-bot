"""Audit a causal market-wide stale-price pause before US premarket opens.

The Nov-3 stop cluster exposed a venue-mechanics failure mode: an equity
X-Perp can print only sporadically overnight, then reprice through the fixed
stop when US premarket price discovery begins.  This audit tests one narrow
rule derived from that mechanism, rather than a symbol blacklist:

* a trigger occurs during 03:15-04:00 America/New_York;
* the closed 64-bar basket decline is at least 6%; and
* no more than 55% of the universe has volume in 6 of its last 8 closed bars.

Once triggered, the whole route is paused through the rest of the New York
calendar day so rejected rows cannot be replaced by later candidates from the
same failed reversal regime.

America/New_York makes the UTC window follow US daylight-saving time.  The
November outcomes are diagnostic only; July-October determine whether the
guard is safe enough for the shadow runtime.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

import stock_xperp_basket_trend_forward_oct30_audit as forward
import stock_xperp_regime_target_router_oct30_audit as router
import stock_xperp_short_exhaustion_guard_oct30_audit as short_guard
import stock_xperp_short_stack_guard_oct30_forward as source


ROOT = Path("reports/audit_2026_09_08")
MANIFEST = ROOT / "stock_xperp_forward_nov3_2015_stop_review" / "manifest.json"
PRIOR_MANIFEST = ROOT / "stock_xperp_forward_nov3_stop_review" / "manifest.json"
OUT = ROOT / "stock_xperp_premarket_liquidity_guard_nov3.json"
ROWS_OUT = ROOT / "stock_xperp_premarket_liquidity_guard_nov3.csv"
BASELINE_OUT = ROOT / "stock_xperp_premarket_liquidity_guard_nov3_baseline.csv"
NY = ZoneInfo("America/New_York")
LOOKBACK_BARS = 8
MINIMUM_ACTIVE_BARS = 6
MAXIMUM_LIQUID_UNIVERSE_RATIO = .55
MAXIMUM_BASKET_TREND_64 = -.06
WINDOW_START_MINUTE = 3 * 60 + 15
WINDOW_END_MINUTE = 4 * 60
PAUSE_END_MINUTE = 24 * 60


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(row: dict) -> tuple[str, int, str]:
    return str(row["symbol"]), int(row["entry_time"]), str(row["direction"])


def in_premarket_risk_window(entry_time: float) -> bool:
    local = datetime.fromtimestamp(float(entry_time), tz=timezone.utc).astimezone(NY)
    minute = local.hour * 60 + local.minute
    return WINDOW_START_MINUTE <= minute < WINDOW_END_MINUTE


def active_volume_bars(candles: dict, signal_index: int,
                       lookback: int = LOOKBACK_BARS) -> int | None:
    """Count closed bars with actual contracts traded, ending at signal bar."""
    index = int(signal_index)
    if lookback < 1 or index < lookback - 1:
        return None
    volumes = candles.get("volume") or []
    if index >= len(volumes):
        return None
    try:
        return sum(float(volumes[offset]) > 0
                   for offset in range(index - lookback + 1, index + 1))
    except (TypeError, ValueError):
        return None


def add_liquidity(frame: pd.DataFrame, candles: dict) -> pd.DataFrame:
    output = frame.copy()
    indexes = {
        symbol: {int(stamp): index for index, stamp in enumerate(series["time"])}
        for symbol, series in candles.items()
    }
    ratios = {}
    for entry_time in output.entry_time.astype("int64").unique():
        signal_time = int(entry_time) - 900
        liquid = available = 0
        for symbol, series in candles.items():
            signal_index = indexes[symbol].get(signal_time)
            if signal_index is None:
                continue
            count = active_volume_bars(series, signal_index)
            if count is None:
                continue
            available += 1
            liquid += count >= MINIMUM_ACTIVE_BARS
        ratios[int(entry_time)] = (
            float(liquid) / available if available else float("nan"))
    output["liquid_universe_ratio_8x6"] = output.entry_time.astype(
        "int64").map(ratios)
    return output


def apply_guard(frame: pd.DataFrame) -> pd.DataFrame:
    trigger_times: dict[object, float] = {}
    for row in frame.sort_values("entry_time").to_dict("records"):
        stamp = float(row["entry_time"])
        local = datetime.fromtimestamp(stamp, tz=timezone.utc).astimezone(NY)
        if (in_premarket_risk_window(stamp)
                and float(row["basket_trend_64"]) <= MAXIMUM_BASKET_TREND_64
                and float(row.get("liquid_universe_ratio_8x6") or 0)
                <= MAXIMUM_LIQUID_UNIVERSE_RATIO):
            trigger_times.setdefault(local.date(), stamp)
    keep = []
    for row in frame.to_dict("records"):
        stamp = float(row["entry_time"])
        local = datetime.fromtimestamp(stamp, tz=timezone.utc).astimezone(NY)
        minute = local.hour * 60 + local.minute
        triggered_at = trigger_times.get(local.date())
        keep.append(not (triggered_at is not None
                         and stamp >= triggered_at
                         and minute < PAUSE_END_MINUTE))
    return frame[pd.Series(keep, index=frame.index)].copy()


def routed_raw(entries: pd.DataFrame,
               resolved: dict[float, pd.DataFrame]) -> pd.DataFrame:
    lookups = {
        target: {identity(row): row for row in values.to_dict("records")}
        for target, values in resolved.items()
    }
    rows = []
    for entry in entries.to_dict("records"):
        hour = int(float(entry["entry_time"]) // 3600) % 24
        if float(entry["basket_trend_32"]) <= -.04:
            target = 2.0
        elif 6 <= hour < 12:
            target = 3.0
        else:
            target = 2.5
        rows.append(lookups[target][identity(entry)])
    return pd.DataFrame(rows, columns=next(iter(resolved.values())).columns)


def compact(frame: pd.DataFrame) -> dict:
    values = forward.metrics(frame, .03)
    return {key: values[key] for key in (
        "n", "resolved", "pending", "return_pct", "dd_pct",
        "profitable_exit_rate_pct", "literal_tp_rate_pct",
        "max_loss_streak", "trades_per_active_day",
    )}


def monthly(frame: pd.DataFrame) -> dict:
    stamps = pd.to_datetime(frame.entry_time, unit="s", utc=True)
    return {
        month: compact(frame[stamps.dt.strftime("%Y-%m").eq(month)])
        for month in ("2026-07", "2026-08", "2026-09", "2026-10", "2026-11")
    }


def delta(candidate: dict, baseline: dict) -> dict:
    return {
        month: {
            "return_delta_pp": round(
                candidate[month]["return_pct"] - baseline[month]["return_pct"], 4),
            "dd_delta_pp": round(
                candidate[month]["dd_pct"] - baseline[month]["dd_pct"], 4),
            "trade_delta": candidate[month]["n"] - baseline[month]["n"],
        }
        for month in baseline
    }


def main() -> None:
    raw, sources, candles = source.candidates(MANIFEST)
    entries = short_guard.apply_guard(raw, -.0075)
    resolved = {target: router.resolve(raw, candles, target)
                for target in router.TARGETS}
    routed = add_liquidity(routed_raw(entries, resolved), candles)
    baseline = short_guard.accept(routed)
    guarded = short_guard.accept(apply_guard(routed))
    baseline.to_csv(BASELINE_OUT, index=False)
    guarded.to_csv(ROWS_OUT, index=False)

    baseline_monthly = monthly(baseline)
    guarded_monthly = monthly(guarded)
    changes = delta(guarded_monthly, baseline_monthly)
    base_ids = {identity(row) for row in baseline.to_dict("records")}
    guarded_ids = {identity(row) for row in guarded.to_dict("records")}
    base_lookup = {identity(row): row for row in baseline.to_dict("records")}
    guard_lookup = {identity(row): row for row in guarded.to_dict("records")}
    changed_rows = []
    for label, ids, lookup in (
        ("removed", base_ids - guarded_ids, base_lookup),
        ("added_by_portfolio_replacement", guarded_ids - base_ids, guard_lookup),
    ):
        for key in sorted(ids, key=lambda value: (value[1], value[0])):
            row = lookup[key]
            changed_rows.append({
                "change": label,
                "symbol": row["symbol"],
                "direction": row["direction"],
                "entry_utc": pd.Timestamp(
                    row["entry_time"], unit="s", tz="UTC").isoformat(),
                "entry_new_york": datetime.fromtimestamp(
                    float(row["entry_time"]), tz=timezone.utc
                ).astimezone(NY).isoformat(),
                "liquid_universe_ratio_8x6": (
                    None if pd.isna(row.get("liquid_universe_ratio_8x6"))
                    else float(row["liquid_universe_ratio_8x6"])),
                "basket_trend_64": float(row["basket_trend_64"]),
                "outcome": row["outcome"],
                "net_r": None if pd.isna(row.get("net_r")) else float(row["net_r"]),
            })

    historical_gate = {
        "july_return_not_lower":
            changes["2026-07"]["return_delta_pp"] >= 0,
        "august_return_not_lower":
            changes["2026-08"]["return_delta_pp"] >= 0,
        "september_return_not_lower":
            changes["2026-09"]["return_delta_pp"] >= 0,
        "october_return_not_lower":
            changes["2026-10"]["return_delta_pp"] >= 0,
        "no_complete_month_dd_worse": all(
            changes[month]["dd_delta_pp"] <= 0
            for month in ("2026-07", "2026-08", "2026-09", "2026-10")),
        "frequency_above_three_each_complete_month": all(
            guarded_monthly[month]["trades_per_active_day"] >= 3
            for month in ("2026-07", "2026-08", "2026-09", "2026-10")),
    }
    historical_gate["passes"] = all(historical_gate.values())
    november_three = lambda frame: frame[
        pd.to_datetime(frame.entry_time, unit="s", utc=True)
        .dt.strftime("%Y-%m-%d").eq("2026-11-03")]
    fresh = {
        "baseline": compact(november_three(baseline)),
        "guarded": compact(november_three(guarded)),
    }
    fresh["stop_delta"] = int(
        november_three(guarded).outcome.eq("SL").sum()
        - november_three(baseline).outcome.eq("SL").sum())
    fresh["return_delta_pp"] = round(
        fresh["guarded"]["return_pct"] - fresh["baseline"]["return_pct"], 4)
    operational_gate = {
        "november_3_stop_count_not_higher": fresh["stop_delta"] <= 0,
        "november_3_return_not_lower": fresh["return_delta_pp"] >= 0,
        "november_3_drawdown_not_higher": (
            fresh["guarded"]["dd_pct"] <= fresh["baseline"]["dd_pct"]),
    }
    operational_gate["passes"] = all(operational_gate.values())
    accepted = bool(historical_gate["passes"] and operational_gate["passes"])

    report = {
        "status": ("FORWARD_ONLY_SHADOW_EXPERIMENT"
                   if accepted
                   else ("REJECTED_HISTORICAL_GATE"
                         if not historical_gate["passes"]
                         else "REJECTED_OPERATIONAL_GATE")),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "ready_for_live_orders": False,
        "orders_possible": False,
        "runtime_change_allowed": accepted,
        "rule": {
            "local_timezone": "America/New_York",
            "trigger_window": "03:15 <= local entry < 04:00",
            "latched_pause_end": "next America/New_York calendar day",
            "basket_trend_64_at_or_below": MAXIMUM_BASKET_TREND_64,
            "per_symbol_lookback_closed_15m_bars": LOOKBACK_BARS,
            "per_symbol_minimum_nonzero_volume_bars": MINIMUM_ACTIVE_BARS,
            "maximum_liquid_universe_ratio": MAXIMUM_LIQUID_UNIVERSE_RATIO,
            "otherwise": "route unchanged",
        },
        "protocol": {
            "mechanism": "stale overnight X-Perp repricing at US premarket price discovery",
            "acceptance_data": ["2026-07", "2026-08", "2026-09", "2026-10"],
            "november_3": "diagnostic only; excluded from historical gate",
            "portfolio_replayed_after_filter": True,
            "position_fraction": "client setting",
            "leverage": 10,
            "orders": "market only",
        },
        "historical_gate": historical_gate,
        "operational_gate": operational_gate,
        "baseline_monthly": baseline_monthly,
        "guarded_monthly": guarded_monthly,
        "monthly_delta": changes,
        "november_3_diagnostic": fresh,
        "changed_rows": changed_rows,
        "integrity": {
            "raw_candle_overlap": forward.candle_overlap(
                PRIOR_MANIFEST, MANIFEST),
        },
        "sources": {
            **sources,
            str(MANIFEST): digest(MANIFEST),
            str(PRIOR_MANIFEST): digest(PRIOR_MANIFEST),
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "The rule was formulated after observing the Nov-3 cluster; Nov-3 is not acceptance evidence.",
            "Native stock X-Perp history begins in late June 2026.",
            "OHLCV replay cannot reproduce live ticker latency within a 15-minute bar.",
            "Gross results exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "historical_gate": historical_gate,
        "monthly_delta": changes,
        "november_3_diagnostic": fresh,
        "changed_rows": changed_rows,
        "integrity": report["integrity"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
