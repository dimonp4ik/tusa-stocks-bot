import json

import stock_xperp_stop_guard_oct28_audit as stop_guard
import stock_xperp_third_target_oct28_audit as third


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def row():
    return {
        "symbol": "TESTUSDT", "entry_index": 0, "entry_time": 1000,
        "direction_n": 1, "cross_z": -3.0, "signal_return": -.01,
        "atr14_fraction": .005, "backing_level_distance": .02,
        "basket_trend_96": 0.0,
    }


def candles(second_bar=False):
    value = {
        "time": [1000], "open": [100.0], "high": [105.0],
        "low": [100.5], "close": [104.0],
    }
    if second_bar:
        value = {key: values + extra for key, values, extra in (
            ("time", value["time"], [1900]),
            ("open", value["open"], [102.5]),
            ("high", value["high"], [105.0]),
            ("low", value["low"], [101.0]),
            ("close", value["close"], [104.0]),
        )}
    return value


def test_tp3_is_not_awarded_on_ambiguous_activation_candle():
    result = third.resolve(row(), candles(), .5, 4.0)
    assert result["outcome"] == "PENDING"
    assert result["activation_bar_time"] == 1000


def test_tp3_can_resolve_from_the_next_bar():
    result = third.resolve(row(), candles(True), .5, 4.0)
    assert result["outcome"] == "TP3"
    assert result["signed_return"] == .03


def test_third_target_grid_is_rejected_with_exact_baseline_parity():
    value = load(third.OUT)
    assert value["integrity"]["baseline_row_parity"] is True
    assert value["grid"]["candidates"] == 15
    assert value["selected_before_holdouts"] is None
    assert value["status"] == "REJECTED_DEVELOPMENT"
    assert value["ready_for_live_orders"] is False


def test_stop_guard_has_no_two_month_development_survivor():
    value = load(stop_guard.OUT)
    assert value["grid"]["candidate_rules"] == 36
    assert value["selected_before_holdouts"] is None
    assert value["status"] == "REJECTED_DEVELOPMENT"
    assert value["orders_possible"] is False
