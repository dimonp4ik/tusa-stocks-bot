import copy
import numpy as np
import pandas as pd
from stock_mark_conditional_features_nov19 import FeatureContext,FEATURES,mature_before


def dataset():
    n=200;t=[1783382400+j*900 for j in range(n)]
    mark=dict(time=t,open=[100.]*n,high=[100.2]*n,low=[99.8]*n,close=[100.]*n)
    trade=copy.deepcopy(mark);trade['volume']=[1.]*n
    return {s:(copy.deepcopy(mark),copy.deepcopy(trade)) for s in ('TEST','QQQUSDT','SPYUSDT')}


def entry():
    return dict(symbol='TEST',entry_time=1783382400+141*900,direction_n=1,module='mark_break16',
        risk_pct=.003,target_fraction=.015,signal_strength=1.)


def test_future_asset_and_peer_prices_and_volumes_cannot_change_entry_features():
    data=dataset();before=FeatureContext(data).row(entry())
    changed=copy.deepcopy(data)
    for mark,trade in changed.values():
        for f in (mark,trade):
            for k in ('open','high','low','close'):f[k][141:]=[1000.]*59
        trade['volume'][141:]=[1e9]*59
    np.testing.assert_array_equal(before,FeatureContext(changed).row(entry()))
    assert len(before)==len(FEATURES)


def test_future_outcome_columns_never_enter_the_explicit_feature_schema():
    context=FeatureContext(dataset());row=entry();before=context.row(row)
    row.update(net_r=1e20,exit_time=1e30,signed_return=999.,outcome='TP',realized_pnl=1e20)
    assert context.row(row)==before
    assert not set(FEATURES)&{'net_r','exit_time','signed_return','outcome','realized_pnl'}


def test_training_cut_purges_the_whole_unfinished_horizon_even_for_a_quick_exit():
    cutoff=1000000.
    frame=pd.DataFrame([dict(entry_time=cutoff-40*900,exit_time=cutoff-39*900,outcome_known=True),
        dict(entry_time=cutoff-900,exit_time=cutoff-10,outcome_known=True),
        dict(entry_time=cutoff-40*900,exit_time=cutoff,outcome_known=True),
        dict(entry_time=cutoff-40*900,exit_time=cutoff-39*900,outcome_known=False)])
    assert mature_before(frame,cutoff).tolist()==[True,False,False,False]
