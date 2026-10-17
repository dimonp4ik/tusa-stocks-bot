import numpy as np
import pandas as pd
import pytest

import stock_xperp_native_cross_section_oct18_audit as audit


def candles(values):
    return {
        "time": [0, 900, 1800, 2700],
        "close": values,
    }


def test_feature_rows_enter_next_closed_bar_and_reverse_extreme():
    frame = audit.feature_rows({
        "AUSDT": candles([100, 100, 110, 109]),
        "BUSDT": candles([100, 100, 100, 100]),
        "CUSDT": candles([100, 100, 99, 99]),
    }, 1)
    row = frame[(frame.symbol == "AUSDT") & (frame.entry_time == 2700)].iloc[0]
    assert row.entry_index == 3
    assert row.signal_return == pytest.approx(.1)
    assert row.direction_n == -1
    assert np.isfinite(row.cross_z)


def test_entry_filter_requires_breadth_and_material_move():
    frame = pd.DataFrame({
        "cross_z": [2.1, 2.1, 2.1],
        "signal_return": [.005, .003, .005],
        "universe_n": [10, 10, 9],
    })
    assert audit.select_entries(frame, 2.0, .004).index.tolist() == [0]


def test_candidate_gate_uses_profit_drawdown_and_density_not_win_rate():
    values = {name: {
        "n": 100, "return_pct": 1.0, "dd_pct": 5.0,
        "trades_per_calendar_day": 3.0, "wr": 1.0,
    } for name in audit.PERIOD_DAYS}
    assert audit.candidate_ok(values)
    values["confirmation"]["return_pct"] = 0
    assert not audit.candidate_ok(values)
