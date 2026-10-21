"""Attribute frozen stock X-Perp forward stops without fitting a new rule."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import pickle
from pathlib import Path


ROOT = Path("reports/audit_2026_09_08")
FORWARD = ROOT / "stock_xperp_backing_hour_cap_forward_oct22.json"
MANIFEST = ROOT / "stock_xperp_forward_oct22" / "manifest.json"
OUT = ROOT / "stock_xperp_forward_loss_attribution_oct22.json"
ROWS_OUT = ROOT / "stock_xperp_forward_loss_attribution_oct22.csv"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _touched(high: float, low: float, level: float, direction: int,
             favorable: bool) -> bool:
    if favorable:
        return high >= level if direction > 0 else low <= level
    return low <= level if direction > 0 else high >= level


def attribute_stop(row: dict, candles: dict) -> dict:
    """Describe the path after the first literal stop bar.

    The stop bar itself is excluded from post-stop recovery tests because OHLC
    cannot establish the order of movements inside that bar.
    """
    entry_index = int(row["entry_index"])
    hold_bars = int(row["hold_bars"])
    direction = 1 if str(row["direction"]) == "LONG" else -1
    entry = float(row["entry"])
    stop_fraction = float(row["stop_pct"]) / 100.0
    target_fraction = float(row["target_pct"]) / 100.0
    stop = entry * (1 - direction * stop_fraction)
    target = entry * (1 + direction * target_fraction)
    second_adverse = entry * (1 - direction * 2 * stop_fraction)
    last = min(entry_index + hold_bars - 1, len(candles["time"]) - 1)
    stop_index = None
    for index in range(entry_index, last + 1):
        if _touched(float(candles["high"][index]),
                    float(candles["low"][index]), stop, direction, False):
            stop_index = index
            break
    if stop_index is None:
        raise ValueError("reported stop is absent from its candle path")

    after = range(stop_index + 1, last + 1)
    highs = [float(candles["high"][index]) for index in after]
    lows = [float(candles["low"][index]) for index in after]
    target_after = any(_touched(high, low, target, direction, True)
                       for high, low in zip(highs, lows))
    entry_reclaimed = any(_touched(high, low, entry, direction, True)
                          for high, low in zip(highs, lows))
    another_r_adverse = any(
        _touched(high, low, second_adverse, direction, False)
        for high, low in zip(highs, lows))
    if highs:
        maximum_favorable = ((max(highs) / entry - 1) if direction > 0
                             else (1 - min(lows) / entry))
        maximum_adverse = ((min(lows) / entry - 1) if direction > 0
                           else (1 - max(highs) / entry))
    else:
        maximum_favorable = maximum_adverse = 0.0
    horizon_close = float(candles["close"][last])
    horizon_signed = direction * (horizon_close / entry - 1)
    return {
        **row,
        "stop_bar_time": int(candles["time"][stop_index]),
        "bars_after_stop": max(0, last - stop_index),
        "original_target_hit_after_stop": target_after,
        "entry_reclaimed_after_stop": entry_reclaimed,
        "another_1r_adverse_after_stop": another_r_adverse,
        "maximum_favorable_after_stop_pct": 100 * maximum_favorable,
        "maximum_adverse_after_stop_pct": 100 * maximum_adverse,
        "horizon_signed_return_pct": 100 * horizon_signed,
        "backing_inside_stop_distance": (
            float(row["backing_level_distance"]) < stop_fraction),
    }


def rate(rows: list[dict], key: str) -> float:
    return (round(100 * sum(bool(row[key]) for row in rows) / len(rows), 4)
            if rows else 0.0)


def main() -> None:
    import pandas as pd

    report = json.loads(FORWARD.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    stop_rows = [row for row in report["fresh"] if row["outcome"] == "SL"]
    required_symbols = {str(row["symbol"]) for row in stop_rows}
    candles = {}
    sources = {str(FORWARD): digest(FORWARD), str(MANIFEST): digest(MANIFEST)}
    for symbol, item in manifest["symbols"].items():
        if symbol not in required_symbols:
            continue
        path = Path(item["path"])
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != item["sha256"]:
            raise ValueError(f"source hash mismatch: {symbol}")
        candles[symbol] = pickle.loads(raw)
        sources[str(path)] = hashlib.sha256(raw).hexdigest()
    if set(candles) != required_symbols:
        raise ValueError("manifest does not contain every stopped symbol")
    stops = [attribute_stop(row, candles[str(row["symbol"])])
             for row in stop_rows]
    result = {
        "status": "FORWARD_STOP_PATH_ATTRIBUTION_ONLY",
        "ready_for_live_orders": False,
        "orders_possible": False,
        "protocol": {
            "sample": "all literal stops in the frozen Oct-17+ forward route",
            "post_stop_window": "remaining frozen 64-bar horizon",
            "ambiguity": "stop bar excluded from recovery tests",
            "purpose": "diagnosis only; no threshold or runtime fitting",
        },
        "summary": {
            "stops": len(stops),
            "direction_count": dict(Counter(row["direction"] for row in stops)),
            "original_target_hit_after_stop_pct": rate(
                stops, "original_target_hit_after_stop"),
            "entry_reclaimed_after_stop_pct": rate(
                stops, "entry_reclaimed_after_stop"),
            "another_1r_adverse_after_stop_pct": rate(
                stops, "another_1r_adverse_after_stop"),
            "backing_inside_stop_distance_pct": rate(
                stops, "backing_inside_stop_distance"),
            "mean_horizon_signed_return_pct": round(sum(
                float(row["horizon_signed_return_pct"]) for row in stops)
                / len(stops), 6),
        },
        "by_entry_day": {},
        "sources": {
            **sources,
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            f"Only {len(stops)} independent forward stops are available.",
            "Public OHLC cannot order movements within one candle.",
            "Post-stop paths diagnose failure; they do not validate a filter.",
        ],
    }
    for day, frame in pd.DataFrame(stops).assign(
            day=lambda value: pd.to_datetime(
                value.entry_time, unit="s", utc=True).dt.strftime("%Y-%m-%d")
            ).groupby("day"):
        rows = frame.to_dict("records")
        result["by_entry_day"][day] = {
            "stops": len(rows),
            "target_after_stop_pct": rate(
                rows, "original_target_hit_after_stop"),
            "another_1r_adverse_pct": rate(
                rows, "another_1r_adverse_after_stop"),
        }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True),
                   encoding="utf-8")
    pd.DataFrame(stops).to_csv(ROWS_OUT, index=False)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
