import numpy as np
import pytest
from stock_xperp_native_growth_forward_monitor import resolve_row


@pytest.mark.parametrize("direction", [-1,1])
@pytest.mark.parametrize("opens,highs,lows", [
    ([100,100],[100.1,103],[99.9,99.5]),
    ([100,98],[100.1,98.5],[99.9,97]),
    ([100,100],[100.1,104],[99.9,96]),
])
def test_preallocated_arrays_preserve_gap_and_stop_first_results(direction,opens,highs,lows):
    candle = dict(time=[0,900],open=opens,high=highs,low=lows,close=[100,100])
    arrays = {k:np.asarray(v) for k,v in candle.items()}
    entry = dict(symbol="TEST",entry_index=0,direction_n=direction)
    assert resolve_row(entry,candle,2.5,1,2) == resolve_row(entry,arrays,2.5,1,2)
