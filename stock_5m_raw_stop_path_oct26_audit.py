"""Stop-path attribution for all post-freeze raw stock shadow proposals.

The production materiality gate is left untouched.  This diagnostic asks what
the rejected raw setups would have done with a literal 0.5% target and fixed
0.5%/1% stops, then attributes stops using only context closed before entry.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import stock_5m_material_forward_oct8_audit as audit
import stock_5m_material_forward_oct13_audit as loader
from src.research_profit_gate import economic_gate


ROOT = Path("reports/audit_2026_09_08")
MANIFEST = ROOT / "stock_xperp_5m_oct25_forward" / "manifest.json"
OUT = ROOT / "stock_5m_raw_stop_path_oct26.json"
ROWS_OUT = ROOT / "stock_5m_raw_stop_path_oct26.csv"
TARGET_VALUES = (.5, 1.0, 1.5, 2.0)
STOP_VALUES = (.5, 1.0)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def causal_context(row: dict, candle: dict, lookup: dict[int, int]) -> dict:
    first = lookup[int(float(row["entry_time"]))]
    signal = lookup[int(float(row["signal_bar_ts"]))]
    entry = float(row["entry_price"])
    direction = 1 if str(row["direction"]) == "LONG" else -1
    high = np.asarray(candle["high"], dtype=float)
    low = np.asarray(candle["low"], dtype=float)
    open_ = np.asarray(candle["open"], dtype=float)
    close = np.asarray(candle["close"], dtype=float)
    volume = np.asarray(candle["volume"], dtype=float)
    previous = close[signal-1]
    true_range = max(
        high[signal]-low[signal], abs(high[signal]-previous),
        abs(low[signal]-previous), 1e-12)
    span = max(high[signal]-low[signal], 1e-12)
    rejection = ((min(open_[signal], close[signal])-low[signal])/span
                 if direction > 0 else
                 (high[signal]-max(open_[signal], close[signal]))/span)
    result = {
        "entry_index": first,
        "signal_body_dir_atr": direction*(close[signal]-open_[signal])/true_range,
        "directed_rejection_wick": float(rejection),
        "signal_volume_ratio_72": float(volume[signal]) / max(
            float(np.median(volume[max(0, signal-72):signal])), 1e-12),
        "qqq_adverse": float(row.get("qqq_trend_dir_atr", 0)) < 0,
        "late_session": float(row.get("signal_minutes", 0)) > 150,
    }
    for window in (72, 288):
        start = max(0, signal-window)
        opposing = (float(np.max(high[start:signal])) if direction > 0
                    else float(np.min(low[start:signal])))
        distance = direction*(opposing-entry)/entry
        result[f"opposing_level_{window}_pct"] = 100*distance
    prior_open = open_[signal-1]
    prior_close = close[signal-1]
    result["opposing_order_block_proxy"] = bool(
        (direction > 0 and prior_close < prior_open)
        or (direction < 0 and prior_close > prior_open))
    if direction > 0:
        prior_level = float(np.max(high[max(0, signal-72):signal]))
        result["failed_breakout"] = bool(
            high[signal] > prior_level and close[signal] < prior_level)
    else:
        prior_level = float(np.min(low[max(0, signal-72):signal]))
        result["failed_breakout"] = bool(
            low[signal] < prior_level and close[signal] > prior_level)
    return result


def resolve(row: dict, candle: dict, context: dict,
            target_pct: float, stop_pct: float) -> dict:
    first = int(context["entry_index"])
    session_last = int(audit.context(candle)["session_last"][first])
    original_bars = max(session_last-first+1, 1)
    extended_last = min(len(candle["time"])-1,
                        first+2*original_bars-1)
    entry = float(row["entry_price"])
    direction = 1 if str(row["direction"]) == "LONG" else -1
    target = entry*(1+direction*target_pct/100)
    stop = entry*(1-direction*stop_pct/100)
    hard = entry*(1-direction*2*stop_pct/100)
    outcome = "TIMEOUT"
    exit_index = session_last
    signed_return = direction*(float(candle["close"][session_last])/entry-1)
    stop_index = None
    for index in range(first, session_last+1):
        # Stop-first treatment is deliberately pessimistic for OHLC ties.
        stop_hit = (float(candle["low"][index]) <= stop if direction > 0
                    else float(candle["high"][index]) >= stop)
        target_hit = (float(candle["high"][index]) >= target if direction > 0
                      else float(candle["low"][index]) <= target)
        if stop_hit:
            outcome, exit_index, stop_index = "SL", index, index
            signed_return = -stop_pct/100
            break
        if target_hit:
            outcome, exit_index = "TP", index
            signed_return = target_pct/100
            break
    signed_return = max(-stop_pct/100, min(target_pct/100, signed_return))
    result = {
        "literal_target_pct": target_pct,
        "literal_stop_pct": stop_pct,
        "literal_outcome": outcome,
        "literal_signed_return": signed_return,
        "literal_exit_time": int(candle["time"][exit_index])+300,
        "stop_closed_back_inside": False,
        "later_target_same_session": False,
        "later_target_double_horizon": False,
        "hard_reversal_proxy": False,
    }
    if stop_index is None:
        return result

    close_at_stop = float(candle["close"][stop_index])
    result["stop_closed_back_inside"] = bool(
        close_at_stop > stop if direction > 0 else close_at_stop < stop)

    def target_after(last: int) -> bool:
        if last <= stop_index:
            return False
        if direction > 0:
            return bool(np.max(candle["high"][stop_index+1:last+1],
                               initial=-np.inf) >= target)
        return bool(np.min(candle["low"][stop_index+1:last+1],
                           initial=np.inf) <= target)

    result["later_target_same_session"] = target_after(session_last)
    result["later_target_double_horizon"] = target_after(extended_last)
    result["hard_reversal_proxy"] = bool(
        float(np.min(candle["low"][stop_index:extended_last+1])) <= hard
        if direction > 0 else
        float(np.max(candle["high"][stop_index:extended_last+1])) >= hard)
    return result


def maximum_loss_streak(rows: pd.DataFrame) -> int:
    streak = maximum = 0
    for value in rows.sort_values(
            ["literal_exit_time", "entry_time", "symbol"]
    ).literal_signed_return:
        streak = streak+1 if float(value) <= 0 else 0
        maximum = max(maximum, streak)
    return maximum


def drawdown(factors: np.ndarray) -> float:
    if not len(factors):
        return 0.0
    equity = np.cumprod(factors)
    peaks = np.maximum.accumulate(np.maximum(equity, 1.0))
    return 100*float(np.max((peaks-equity)/peaks))


def metrics(frame: pd.DataFrame) -> dict:
    ordered = frame.sort_values(
        ["literal_exit_time", "entry_time", "symbol"]).copy()
    signed = ordered.literal_signed_return.to_numpy(float)
    factors = 1+.03*10*signed
    dates = pd.to_datetime(ordered.entry_time, unit="s", utc=True).dt.date
    days = max(int(pd.Series(dates).nunique()), 1)
    frequency = len(ordered)/days
    gate = economic_gate(
        signed.tolist(), frequency, margin_fraction=.03,
        market_days=22, required_trades_per_day=3)
    months = pd.to_datetime(
        ordered.literal_exit_time, unit="s", utc=True).dt.strftime("%Y-%m")
    monthly = pd.Series(factors, index=months).groupby(level=0).prod()
    return {
        "n": len(ordered),
        "outcomes": dict(sorted(Counter(ordered.literal_outcome).items())),
        "literal_tp_rate_pct": round(
            100*float(ordered.literal_outcome.eq("TP").mean()), 4),
        "profitable_exit_rate_pct": round(100*float((signed > 0).mean()), 4),
        "return_pct": round(100*(float(np.prod(factors))-1), 4),
        "dd_pct": round(drawdown(factors), 4),
        "max_loss_streak": maximum_loss_streak(ordered),
        "trades_per_market_day": round(frequency, 4),
        "mean_signed_price_return_pct": round(100*float(signed.mean()), 6),
        "monthly_return_pct": {
            str(month): round(100*(float(value)-1), 4)
            for month, value in monthly.items()},
        "negative_months": int((monthly < 1).sum()),
        "economic_gate": gate,
    }


def rate_lifts(frame: pd.DataFrame, fields: list[str]) -> dict:
    baseline = float(frame.literal_outcome.eq("SL").mean())
    result = {}
    for field in fields:
        mask = frame[field].astype(bool)
        rate = float(frame.loc[mask].literal_outcome.eq("SL").mean()) \
            if mask.any() else 0.0
        result[field] = {
            "present": int(mask.sum()),
            "stops": int(frame.loc[mask].literal_outcome.eq("SL").sum()),
            "stop_rate_pct": round(100*rate, 4),
            "stop_rate_lift": round(rate/baseline, 4) if baseline else None,
        }
    return result


def path_summary(stops: pd.DataFrame) -> dict:
    fields = (
        "stop_closed_back_inside", "later_target_same_session",
        "later_target_double_horizon", "hard_reversal_proxy")
    return {field: {
        "count": int(stops[field].sum()),
        "pct_of_stops": round(100*float(stops[field].mean()), 4),
    } for field in fields}


def main() -> None:
    _, candles = loader.verified_manifest(MANIFEST)
    raw, _ = audit.replay(candles)
    fresh = [row for row in raw
             if int(float(row["entry_time"])) > audit.MATERIAL_FREEZE]
    cluster_counts = Counter((
        int(float(row["entry_time"]))//3600, str(row["direction"]))
        for row in fresh)
    enriched = []
    for row in fresh:
        candle = candles[str(row["symbol"])]
        lookup = {int(value): index for index, value in enumerate(candle["time"])}
        causal = causal_context(row, candle, lookup)
        cluster_size = cluster_counts[(
            int(float(row["entry_time"]))//3600, str(row["direction"]))]
        for target_pct in TARGET_VALUES:
            for stop_pct in STOP_VALUES:
                causal_geometry = dict(causal)
                for window in (72, 288):
                    distance = float(
                        causal_geometry[f"opposing_level_{window}_pct"])
                    causal_geometry[
                        f"opposing_level_{window}_before_target"] = bool(
                            0 <= distance <= target_pct)
                path = resolve(
                    row, candle, causal_geometry, target_pct, stop_pct)
                enriched.append({
                    **row, **causal_geometry, **path,
                    "same_direction_hour_size": cluster_size,
                    "clustered_same_direction_hour": cluster_size > 1,
                })
    frame = pd.DataFrame(enriched)
    flags = [
        "opposing_level_72_before_target",
        "opposing_level_288_before_target",
        "opposing_order_block_proxy", "failed_breakout", "qqq_adverse",
        "late_session", "clustered_same_direction_hour",
    ]
    geometries = {}
    for target_pct in TARGET_VALUES:
        for stop_pct in STOP_VALUES:
            part = frame[
                frame.literal_target_pct.eq(target_pct)
                & frame.literal_stop_pct.eq(stop_pct)].copy()
            stops = part[part.literal_outcome.eq("SL")]
            key = f"tp_{target_pct:g}_sl_{stop_pct:g}"
            geometries[key] = {
                "metrics_margin_3pct": metrics(part),
                "causal_stop_rate_lifts": rate_lifts(part, flags),
                "post_stop_path": path_summary(stops),
                "clustered": {
                    "rows": int(part.clustered_same_direction_hour.sum()),
                    "stops": int(part.loc[
                        part.clustered_same_direction_hour,
                        "literal_outcome"].eq("SL").sum()),
                    "max_same_direction_hour": int(
                        part.same_direction_hour_size.max()),
                },
            }
    report = {
        "status": "DIAGNOSTIC_ONLY_INSUFFICIENT_SAMPLE",
        "ready_for_live_orders": False, "orders_possible": False,
        "protocol": {
            "source": "all causal raw runtime proposals after material freeze",
            "material_gate_changed": False,
            "entry": "next 5m market-open proxy", "orders": "market only",
            "target_pct": list(TARGET_VALUES), "stop_pct": list(STOP_VALUES),
            "exit": "cash-session close; stop first on OHLC ties",
            "money": "3% current-equity margin, fixed 10x",
            "no_filter_fitted": True,
        },
        "sample": {
            "raw_events": len(fresh),
            "start": pd.to_datetime(min(float(row["entry_time"])
                                        for row in fresh), unit="s", utc=True).isoformat(),
            "end": pd.to_datetime(max(float(row["entry_time"])
                                      for row in fresh), unit="s", utc=True).isoformat(),
            "directions": dict(sorted(Counter(
                str(row["direction"]) for row in fresh).items())),
            "families": dict(sorted(Counter(
                str(row["family"]) for row in fresh).items())),
        },
        "geometries": geometries,
        "decision": (
            "Do not fit a runtime filter from 45 one-sided raw events. Use the "
            "attribution only to predeclare a rule for genuinely new outcomes."),
        "sources": {
            str(MANIFEST): digest(MANIFEST),
            **{str(Path(meta["path"])): meta["sha256"]
               for meta in json.loads(MANIFEST.read_text())["symbols"].values()},
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "All 45 fresh raw proposals are shorts and cannot validate long logic.",
            "The raw proposals failed the production 0.5% material-target gate.",
            "Post-stop paths are diagnostic and unavailable at entry time.",
            "A single October interval cannot establish a stable filter.",
        ],
    }
    frame.to_csv(ROWS_OUT, index=False)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
