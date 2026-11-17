import copy
import pytest
from stock_closed_activity_nov17 import closed_activity


def test_entry_bar_and_future_activity_cannot_change_closed_activity():
    c=dict(time=[i*900 for i in range(15)],volume=[0,1,0,1,1,1,1,0]+[100]*7)
    expected=closed_activity(c,8)
    assert expected==dict(last_bar_active=False,last_two_active=False,active_eight=5)
    altered=copy.deepcopy(c);altered['volume'][8:]=[float('nan')]*7
    assert closed_activity(altered,8)==expected


def test_missing_history_and_gaps_are_not_counted_as_traded_bars():
    c=dict(time=[i*900 for i in range(9)],volume=[1]*9)
    assert closed_activity(c,7) is None
    c['time'][4]+=1
    assert closed_activity(c,8) is None
    c['time'][4]-=1;c['volume'][4]=-1
    with pytest.raises(ValueError):closed_activity(c,8)
