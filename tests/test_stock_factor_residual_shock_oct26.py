import numpy as np

import stock_factor_residual_shock_oct26_audit as audit


def candles(close):
    count = len(close)
    return {
        "time": [1_700_000_000 + 900 * i for i in range(count)],
        "open": list(close), "high": list(np.asarray(close) * 1.001),
        "low": list(np.asarray(close) * .999), "close": list(close),
        "volume": [100.0] * count,
    }


def test_signal_bar_does_not_change_its_own_rolling_beta():
    index = np.arange(120, dtype=float)
    factor_close = 100 * np.exp(.0005 * index + .001 * np.sin(index / 4))
    stock_close = 50 * np.exp(.0007 * index + .0015 * np.sin(index / 4))
    stocks = {"TESTUSDT": candles(stock_close)}
    original = audit.feature_rows(stocks, candles(factor_close), 2, 26)

    changed = factor_close.copy()
    changed[90] *= 1.10
    mutated = audit.feature_rows(stocks, candles(changed), 2, 26)
    entry_time = 1_700_000_000 + 900 * 91
    before = original[original.entry_time.eq(entry_time)].iloc[0]
    after = mutated[mutated.entry_time.eq(entry_time)].iloc[0]

    assert before.beta == after.beta
    assert before.factor_residual != after.factor_residual


def test_style_direction_uses_factor_residual_sign():
    frame = np.array([-0.02, 0.03])
    import pandas as pd
    features = pd.DataFrame({
        "residual_z": [-3.0, 3.0], "factor_residual": frame,
    })
    reversal = audit.select(features, 2.0, .002, "reversion")
    continuation = audit.select(features, 2.0, .002, "continuation")
    assert reversal.direction_n.tolist() == [1, -1]
    assert continuation.direction_n.tolist() == [-1, 1]
