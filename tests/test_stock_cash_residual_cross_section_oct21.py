import numpy as np

import stock_cash_residual_cross_section_oct21_audit as audit


def candles(size=20):
    times = np.arange(size, dtype=np.int64) * 900
    output = {}
    for index in range(8):
        close = np.full(size, 100.0)
        close[10:] = 97.0
        output[f"S{index}USDT"] = {
            "time": times.copy(), "open": close.copy(),
            "high": close + .1, "low": close - .1,
            "close": close, "volume": np.ones(size),
        }
    # The stock fell in absolute terms, but less than all its peers.  Its
    # residual is positive, so residual reversion must SHORT it.
    output["S0USDT"]["close"][10:] = 99.0
    return output


def test_direction_reverses_residual_not_absolute_return():
    source = candles()
    frame = audit.feature_rows(source, 2)
    entry_time = int(source["S0USDT"]["time"][11])
    row = frame[(frame.symbol == "S0USDT")
                & (frame.entry_time == entry_time)].iloc[0]

    assert row.signal_return < 0
    assert row.residual_return > 0
    assert row.direction_n == -1
    assert row.entry_index == 11


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
               "universe_n", "cross_mean", "cross_std", "residual_return",
               "cross_z", "direction_n"]
    left = original[original.entry_time < cutoff][columns].reset_index(drop=True)
    right = revised[revised.entry_time < cutoff][columns].reset_index(drop=True)

    assert left.equals(right)


def test_mandatory_monthly_economics_are_exact():
    assert round(audit.required_mean_move(3, .03), 6) == 1.327709
    assert round(audit.required_mean_move(3, .05), 6) == .796626
