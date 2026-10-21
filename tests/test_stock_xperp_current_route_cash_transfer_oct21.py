import numpy as np

import stock_xperp_current_route_cash_transfer_oct21_audit as audit


def candles(size=20):
    times = np.arange(size, dtype=np.int64) * 900
    output = {}
    for index in range(8):
        close = 100 + np.arange(size, dtype=float) * .01
        output[f"S{index}USDT"] = {
            "time": times.copy(), "open": close.copy(),
            "high": close + .1, "low": close - .1,
            "close": close, "volume": np.ones(size),
        }
    # One completed signal bar is a large positive cross-sectional outlier.
    output["S0USDT"]["close"][10] = 102.0
    return output


def test_feature_uses_closed_bar_and_enters_next_open():
    source = candles()
    frame = audit.feature_rows(source, 2)
    entry_time = int(source["S0USDT"]["time"][11])
    row = frame[(frame.symbol == "S0USDT")
                & (frame.entry_time == entry_time)].iloc[0]

    assert row.entry_index == 11
    assert row.direction_n == -1
    assert row.universe_n == 8


def test_future_bars_do_not_change_prior_features():
    source = candles()
    original = audit.feature_rows(source, 2)
    changed = {
        symbol: {key: value.copy() for key, value in candle.items()}
        for symbol, candle in source.items()
    }
    for candle in changed.values():
        for key in ("open", "high", "low", "close", "volume"):
            candle[key][12:] *= 10
    revised = audit.feature_rows(changed, 2)
    cutoff = int(source["S0USDT"]["time"][12])
    columns = ["symbol", "entry_index", "entry_time", "signal_return",
               "universe_n", "cross_mean", "cross_std", "cross_z",
               "direction_n"]
    left = original[original.entry_time < cutoff][columns].reset_index(drop=True)
    right = revised[revised.entry_time < cutoff][columns].reset_index(drop=True)

    assert left.equals(right)


def test_selection_accepts_eight_aligned_equities():
    frame = audit.feature_rows(candles(), 2)
    selected = audit.select_entries(frame, 2.0, .004)

    assert not selected.empty
    assert selected.universe_n.min() == 8
