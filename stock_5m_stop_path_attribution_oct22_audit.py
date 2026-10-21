"""Causal stop-path attribution for the frozen stock five-minute route.

This is diagnostic only.  It does not fit a filter from the small 2026 sample.
Every structural field is computed from bars closed before the market entry.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import stock_5m_barbell_runner_oct3_audit as barbell


ROOT = Path("reports/audit_2026_09_08")
SOURCE = ROOT / "stock_5m_regime_filter_oct3_rows.csv"
OUT = ROOT / "stock_5m_stop_path_attribution_oct22.json"
ROWS_OUT = ROOT / "stock_5m_stop_path_attribution_oct22.csv"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def causal_context(row: dict, context: dict) -> dict:
    candle = context["candles"]
    first = context["index"][int(float(row["entry_time"]))]
    signal = first - 1
    entry = float(candle["open"][first])
    risk = entry * float(row["risk_pct"])
    direction = 1 if str(row["direction"]) == "LONG" else -1
    high = np.asarray(candle["high"], dtype=float)
    low = np.asarray(candle["low"], dtype=float)
    open_ = np.asarray(candle["open"], dtype=float)
    close = np.asarray(candle["close"], dtype=float)
    volume = np.asarray(candle["volume"], dtype=float)
    previous = close[signal - 1]
    true_range = max(high[signal] - low[signal],
                     abs(high[signal] - previous), abs(low[signal] - previous))
    bar_range = max(high[signal] - low[signal], 1e-12)
    upper_wick = (high[signal] - max(open_[signal], close[signal])) / bar_range
    lower_wick = (min(open_[signal], close[signal]) - low[signal]) / bar_range
    directed_rejection = lower_wick if direction > 0 else upper_wick
    result = {
        "entry_price": entry, "risk_price": risk,
        "signal_body_atr": direction * (close[signal] - open_[signal])
                           / max(true_range, 1e-12),
        "directed_rejection_wick": directed_rejection,
        "prior_bar_volume_ratio": volume[signal] /
            max(float(np.median(volume[max(0, signal-72):signal])), 1e-12),
        "qqq_adverse": float(row.get("qqq_trend_dir_5m_atr", 0)) < 0,
        "late_session": float(row.get("signal_minutes", 0)) > 150,
    }
    target_r = float(row["target_r"])
    for window in (72, 288):
        start = max(0, signal - window)
        if direction > 0:
            opposing = float(np.max(high[start:signal]))
            distance = (opposing - entry) / risk
        else:
            opposing = float(np.min(low[start:signal]))
            distance = (entry - opposing) / risk
        result[f"opposing_level_{window}_r"] = distance
        result[f"opposing_level_{window}_before_target"] = (
            0 <= distance <= target_r)
    prior_open = open_[signal - 1]
    prior_close = close[signal - 1]
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


def stop_path(row: dict, context: dict, causal: dict) -> dict:
    if str(row["outcome"]) != "SL":
        return {}
    candle = context["candles"]
    first = context["index"][int(float(row["entry_time"]))]
    exit_open = int(float(row["exit_time"])) - 300
    exit_index = context["index"].get(exit_open)
    if exit_index is None:
        raise ValueError(f"missing stop bar: {row['symbol']} {exit_open}")
    session_last = int(context["session_last"][first])
    original_bars = max(session_last - first + 1, 1)
    extended_last = min(len(candle["time"]) - 1,
                        first + 2 * original_bars - 1)
    entry = causal["entry_price"]
    risk = causal["risk_price"]
    direction = 1 if str(row["direction"]) == "LONG" else -1
    target = entry + direction * risk * float(row["target_r"])
    stop = entry - direction * risk
    hard = entry - direction * 2 * risk

    def hit_target(last: int) -> bool:
        if direction > 0:
            return bool(np.max(candle["high"][exit_index+1:last+1], initial=-np.inf)
                        >= target)
        return bool(np.min(candle["low"][exit_index+1:last+1], initial=np.inf)
                    <= target)

    if direction > 0:
        closed_inside = float(candle["close"][exit_index]) > stop
        hard_reversal = float(np.min(
            candle["low"][exit_index:extended_last+1])) <= hard
    else:
        closed_inside = float(candle["close"][exit_index]) < stop
        hard_reversal = float(np.max(
            candle["high"][exit_index:extended_last+1])) >= hard
    return {
        "stop_closed_back_inside": bool(closed_inside),
        "later_target_same_session": hit_target(session_last),
        "later_target_double_horizon": hit_target(extended_last),
        "hard_reversal_proxy": bool(hard_reversal),
    }


def rate_lifts(frame: pd.DataFrame, fields: list[str]) -> dict:
    baseline = float(frame.is_stop.mean())
    output = {}
    for field in fields:
        mask = frame[field].astype(bool)
        rate = float(frame.loc[mask, "is_stop"].mean()) if mask.any() else 0.0
        output[field] = {
            "present": int(mask.sum()), "stops": int(frame.loc[mask, "is_stop"].sum()),
            "stop_rate_pct": round(100 * rate, 4),
            "stop_rate_lift": round(rate / baseline, 4) if baseline else None,
        }
    return output


def main() -> None:
    rows = pd.read_csv(SOURCE).to_dict("records")
    contexts = barbell.cache_context()
    enriched = []
    for row in rows:
        context = contexts[str(row["symbol"])]
        causal = causal_context(row, context)
        path = stop_path(row, context, causal)
        enriched.append({**row, **causal, **path,
                         "is_stop": str(row["outcome"]) == "SL"})
    frame = pd.DataFrame(enriched)
    flags = [
        "opposing_level_72_before_target",
        "opposing_level_288_before_target",
        "opposing_order_block_proxy", "failed_breakout", "qqq_adverse",
        "late_session",
    ]
    stops = frame[frame.is_stop]
    path = {
        key: {"count": int(stops[key].fillna(False).sum()),
              "pct_of_stops": round(100 * float(stops[key].fillna(False).mean()), 4)}
        for key in ("stop_closed_back_inside", "later_target_same_session",
                    "later_target_double_horizon", "hard_reversal_proxy")
    }
    report = {
        "status": "DIAGNOSTIC_ONLY_INSUFFICIENT_SAMPLE",
        "ready_for_live_orders": False, "orders_possible": False,
        "protocol": {
            "source": "exact frozen stock 5m selected rows",
            "causality": "all structural fields use bars closed before entry",
            "entry": "next five-minute market open", "orders": "market only",
            "money": "client-selected margin; baseline 3%, fixed 10x",
            "no_filter_fitted": True,
        },
        "sample": {
            "rows": len(frame), "stops": int(frame.is_stop.sum()),
            "start": pd.to_datetime(frame.entry_time.min(), unit="s", utc=True).isoformat(),
            "end": pd.to_datetime(frame.entry_time.max(), unit="s", utc=True).isoformat(),
            "families": frame.family.value_counts().to_dict(),
        },
        "causal_stop_rate_lifts": rate_lifts(frame, flags),
        "post_stop_path": path,
        "by_family": {
            family: {"rows": len(part), "stops": int(part.is_stop.sum()),
                     "stop_rate_pct": round(100 * float(part.is_stop.mean()), 4)}
            for family, part in frame.groupby("family")
        },
        "decision": (
            "Do not fit a production rule from 28 stops. Continue shadow collection; "
            "predeclare any later confirmation rule from these causal diagnostics."),
        "sources": {str(SOURCE): digest(SOURCE),
                    str(Path(barbell.__file__).resolve()): digest(Path(barbell.__file__).resolve()),
                    **{str(barbell.base.CACHE / f"{symbol}_5min_current.pkl"):
                       digest(barbell.base.CACHE / f"{symbol}_5min_current.pkl")
                       for symbol in sorted(barbell.base.SYMBOLS)},
                    str(Path(__file__).resolve()): digest(Path(__file__).resolve())},
        "limitations": [
            "Only 28 stops exist in the frozen direct X-Perp sample.",
            "Post-stop paths are diagnostic and were not available at entry.",
            "No live-order or deployment decision can be supported by this sample.",
        ],
    }
    frame.to_csv(ROWS_OUT, index=False)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
