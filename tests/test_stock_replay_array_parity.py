import numpy as np
import pytest
from stock_xperp_native_growth_forward_monitor import resolve_row,resolve_entries
import pandas as pd
from stock_xperp_native_volatility_guard_oct18_audit import add_atr14


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
    assert resolve_entries(pd.DataFrame([entry]),{'TEST':candle},dict(target_pct=2.5,stop_pct=1,hold_bars=2))==[
        resolve_row(entry,candle,2.5,1,2)]


def test_atr_batch_is_closed_bar_only_and_matches_hand_calculation():
    close=[100+i for i in range(20)]
    c=dict(high=[p+1 for p in close],low=[p-1 for p in close],close=close)
    rows=pd.DataFrame([dict(symbol='TEST',entry_index=15),dict(symbol='TEST',entry_index=16)])
    actual=add_atr14(rows,{'TEST':c})
    assert actual.atr14_fraction.tolist()==[2/114,2/115]
    c['high'][15]=1000
    assert add_atr14(rows.iloc[:1],{'TEST':c}).atr14_fraction.iloc[0]==2/114
