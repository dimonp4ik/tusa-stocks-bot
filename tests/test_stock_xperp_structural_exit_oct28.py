import json
from pathlib import Path

import numpy as np

import stock_xperp_structural_exit_oct28_audit as audit


def candle_for_exit(direction: int) -> tuple[dict, dict, dict]:
    entry = 100.0
    candle = {
        "time": np.array([0, 900, 1800], dtype=np.int64),
        "open": np.array([entry, entry, entry]),
        "high": np.array([102.5, entry, entry]),
        "low": np.array([98.5, entry, entry]),
        "close": np.array([entry, entry, entry]),
    }
    row = {"symbol": "TEST", "entry_index": 0, "entry_time": 0,
           "direction_n": direction}
    geometry = {
        "entry": entry,
        "stop_price": 99.0 if direction > 0 else 101.0,
        "target_price": 102.0 if direction > 0 else 98.0,
        "stop_fraction": .01, "target_fraction": .02,
        "structural_stop_valid": True, "structural_target_valid": True,
    }
    return candle, row, geometry


def test_same_bar_ambiguity_is_stop_first_for_both_directions():
    for direction in (1, -1):
        candle, row, geometry = candle_for_exit(direction)
        resolved = audit.resolve_geometry(row, candle, geometry)
        assert resolved["outcome"] == "SL"
        assert resolved["signed_return"] == -.01


def test_structural_levels_ignore_entry_and_future_candles():
    size, index = 110, 100
    base = {
        "time": np.arange(size, dtype=np.int64) * 900,
        "open": np.full(size, 100.0),
        "high": np.full(size, 100.5),
        "low": np.full(size, 99.5),
        "close": np.full(size, 100.0),
    }
    base["low"][94] = 98.0
    base["high"][94] = 102.0
    changed = {key: value.copy() for key, value in base.items()}
    changed["high"][index:] = 999.0
    changed["low"][index:] = 1.0
    row = {"symbol": "TEST", "entry_index": index,
           "entry_time": int(base["time"][index]), "direction_n": 1,
           "atr14_fraction": .005}
    config = {"lookback": 64, "buffer_atr": .1,
              "stop_min": .005, "stop_max": .03,
              "target_min": .01, "target_max": .03,
              "minimum_rr": 1.0, "fallback": "fixed", "mode": "both"}
    left = audit.structural_geometry(row, base, config)
    right = audit.structural_geometry(row, changed, config)
    assert left == right


def test_report_is_sequential_and_never_enables_orders():
    report = json.loads(Path(audit.OUT).read_text(encoding="utf-8"))
    assert report["protocol"]["selection"] == "2026-07"
    assert report["protocol"]["validation"] == "2026-08"
    assert report["protocol"]["test"] == "2026-09"
    assert report["protocol"]["confirmation"] == "2026-10"
    assert report["protocol"]["candidate_count"] == 12
    assert report["ready_for_live_orders"] is False
    assert report["orders_possible"] is False
    assert report["integrity"]["future_bars_used_for_levels"] == 0
