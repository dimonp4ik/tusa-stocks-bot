import numpy as np
import pandas as pd

import stock_xperp_material_barbell_oct18_audit as audit


def candle(open_, high, low, close):
    return {"time": np.array([100, 1000]),
            "open": np.array(open_, dtype=float),
            "high": np.array(high, dtype=float),
            "low": np.array(low, dtype=float),
            "close": np.array(close, dtype=float)}


def entry(direction=1):
    return pd.DataFrame([{"symbol": "TEST", "entry_index": 0,
                          "direction_n": direction, "entry_time": 100}])


def test_stop_wins_same_bar_ambiguity_before_partial():
    candles = {"TEST": candle([100, 100], [102, 100], [98, 100], [100, 100])}
    found = audit.resolve_barbell(entry(), candles, 1.5, 2, .5).iloc[0]
    assert found.signed_return == -.01


def test_break_even_is_delayed_until_bar_after_partial():
    candles = {"TEST": candle([100, 100], [101.2, 100.2], [99.5, 99.8], [101, 100])}
    found = audit.resolve_barbell(entry(), candles, 1.5, 2, .5).iloc[0]
    assert found.signed_return == .005


def test_score_uses_weakest_period_and_not_win_rate():
    values = {
        "development": {"return_pct": 47.0, "dd_pct": 2.0, "wr": 10},
        "validation": {"return_pct": 3.1, "dd_pct": 1.0, "wr": 20},
        "confirmation": {"return_pct": 3.6, "dd_pct": 1.5, "wr": 30},
    }
    assert audit.chronological_score(values) > 2.0
