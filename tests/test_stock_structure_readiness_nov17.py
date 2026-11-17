import pandas as pd
from stock_structure_geometry_nov17_audit import readiness_masks


def test_variable_stop_r_can_be_positive_while_price_profit_is_negative():
    rows=[]
    for i in range(20):
        win=i%2==0
        rows.append(dict(symbol=str(i),entry_time=i*20,exit_time=i*20+10,
            net_r=2.5 if win else -1.,signed_return=.005 if win else -.02))
    rows.append(dict(symbol='current',entry_time=1000,exit_time=2000,net_r=100.,signed_return=1.))
    values=readiness_masks(pd.DataFrame(rows))
    assert bool(values.iloc[-1].r_ready) is True
    assert bool(values.iloc[-1].price_ready) is False


def test_outcome_at_entry_time_or_later_is_not_completed_evidence():
    rows=[dict(symbol=str(i),entry_time=0,exit_time=100+i,net_r=1.,signed_return=.01) for i in range(20)]
    rows.append(dict(symbol='current',entry_time=119,exit_time=200,net_r=-100.,signed_return=-1.))
    values=readiness_masks(pd.DataFrame(rows))
    assert bool(values.iloc[-1].r_ready) is False
    assert bool(values.iloc[-1].price_ready) is False
