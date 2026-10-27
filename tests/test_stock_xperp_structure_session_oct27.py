import copy

import stock_xperp_native_structure_guard_oct27_audit as structure
import stock_xperp_session_target_router_oct27_audit as sessions


def candle(count=40):
    return {
        "time": [1_000 + 900*i for i in range(count)],
        "open": [100 + .1*i for i in range(count)],
        "high": [100.4 + .1*i for i in range(count)],
        "low": [99.6 + .1*i for i in range(count)],
        "close": [100.1 + .1*i for i in range(count)],
        "volume": [100 + i for i in range(count)],
    }


def test_structure_features_ignore_candles_after_entry():
    values = candle()
    row = {
        "symbol": "TEST", "entry_index": 25,
        "entry_time": values["time"][25], "entry": values["open"][25],
        "direction": "LONG", "stop_pct": 1.0, "target_pct": 2.0,
    }
    before = structure.causal_features(row, values)
    changed = copy.deepcopy(values)
    for field in ("open", "high", "low", "close", "volume"):
        changed[field][26:] = [9_999] * (len(changed[field])-26)
    after = structure.causal_features(row, changed)
    fields = (
        "target_side_swing_r", "stop_side_swing_r", "supportive_sweep",
        "breakout_hold", "rejection_wick_fraction", "volume_ratio_20",
        "opposing_order_block", "structure_conflict",
    )
    assert {key: before[key] for key in fields} == {
        key: after[key] for key in fields}


def test_session_boundaries_are_complete_and_disjoint():
    base = 1_798_243_200  # 2026-12-25 00:00:00 UTC
    assert sessions.session_name(base + 2*3600) == "utc_00_02"
    assert sessions.session_name(base + 3*3600) == "utc_03_07"
    assert sessions.session_name(base + 8*3600) == "utc_08_15"
    assert sessions.session_name(base + 16*3600) == "utc_16_23"
