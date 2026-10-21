import numpy as np

import stock_vol_scaled_residual_cross_section_oct21_audit as audit


def candles(size=40):
    times = np.arange(size, dtype=np.int64) * 900
    output = {}
    for index in range(8):
        close = 100 + .05 * np.sin(np.arange(size))
        output[f"S{index}USDT"] = {
            "time": times.copy(), "open": close.copy(),
            "high": close + .2, "low": close - .2,
            "close": close.copy(), "volume": np.ones(size),
        }
    output["S0USDT"]["close"][20] = 102
    return output


def test_scaled_feature_is_closed_bar_and_next_open():
    source = candles()
    frame = audit.feature_rows(source, 2)
    entry_time = int(source["S0USDT"]["time"][21])
    row = frame[(frame.symbol == "S0USDT")
                & (frame.entry_time == entry_time)].iloc[0]

    assert row.entry_index == 21
    assert row.scaled_residual > 0
    assert row.direction_n == -1
    assert row.atr14_fraction > 0


def test_signal_window_cannot_cross_a_session_gap():
    source = candles()
    for candle in source.values():
        candle["time"][20:] += 3600
    frame = audit.feature_rows(source, 2)

    assert not (frame.entry_time == int(source["S0USDT"]["time"][21])).any()


def test_future_bars_do_not_change_scaled_feature():
    source = candles()
    original = audit.feature_rows(source, 2)
    changed = {
        symbol: {key: value.copy() for key, value in candle.items()}
        for symbol, candle in source.items()
    }
    for candle in changed.values():
        for key in ("open", "high", "low", "close", "volume"):
            candle[key][22:] *= 5
    revised = audit.feature_rows(changed, 2)
    cutoff = int(source["S0USDT"]["time"][22])
    columns = ["symbol", "entry_index", "entry_time", "signal_return",
               "atr14_fraction", "scaled_return", "scaled_residual",
               "cross_z", "direction_n"]
    left = original[original.entry_time < cutoff][columns].reset_index(drop=True)
    right = revised[revised.entry_time < cutoff][columns].reset_index(drop=True)

    assert left.equals(right)
