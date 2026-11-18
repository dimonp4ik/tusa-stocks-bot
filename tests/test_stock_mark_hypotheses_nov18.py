import copy
import numpy as np
import pytest
from stock_mark_hypotheses_nov18 import candidates,finite_ema,resolve


def source():
    n=200;t=[1783382400+j*900 for j in range(n)]
    mark=dict(time=t,open=[100.]*n,high=[100.1]*n,low=[99.9]*n,close=[100.]*n)
    trade=copy.deepcopy(mark);trade['volume']=[1.]*n
    for f in (mark,trade):f['high'][140]=100.5;f['close'][140]=100.4
    return mark,trade


def selected(mark,trade):
    return candidates(mark,trade,'TEST',kind='break16',target_fraction=.015,regular_only=False)


def test_entry_uses_latest_closed_boundary_without_needing_the_next_candle():
    mark,trade=source();prefix=lambda f:{k:v[:141] for k,v in f.items()}
    rows=selected(prefix(mark),prefix(trade))
    assert len(rows)==1 and rows[0]['entry_index']==141
    assert rows[0]['entry_time']==mark['time'][140]+900
    assert rows[0]['entry']==trade['close'][140]


def test_future_ohlc_does_not_change_a_known_entry_or_its_levels():
    mark,trade=source();initial=selected(mark,trade)[0]
    for f in (mark,trade):
        for k in ('open','high','low','close'):f[k][141:]=[1000.]*59
    assert selected(mark,trade)[0]==initial


def test_disagreeing_or_untraded_last_price_cannot_be_used_as_a_clean_reference():
    mark,trade=source();trade['close'][140]=100.9;trade['high'][140]=101.
    assert selected(mark,trade)==[]
    mark,trade=source();trade['volume'][140]=0
    assert selected(mark,trade)==[]


def test_fixed_window_ema_is_seeded_from_closed_data_only():
    x=np.arange(200,dtype=float)+100
    ema=finite_ema(x,64)
    prefix=finite_ema(x[:150],64)
    np.testing.assert_allclose(ema[:150],prefix,equal_nan=True)
    assert np.isnan(ema[126]) and np.isfinite(ema[127])


def test_exit_does_not_score_a_touch_from_a_zero_volume_carried_bar():
    mark,trade=source();r=selected(mark,trade)[0];i=r['entry_index']
    trade['high'][i]=r['tp1']+1;trade['low'][i]=r['sl']-1;trade['volume'][i]=0
    for j in range(i+1,i+32):
        trade['open'][j]=trade['close'][j]=r['entry'];trade['high'][j]=r['entry']+.01;trade['low'][j]=r['entry']-.01
    result=resolve(r,trade)
    assert result['outcome']=='TIME' and result['net_r']==pytest.approx(0.)


def test_opening_loss_gap_is_preserved_and_same_bar_ambiguity_is_stop_first():
    mark,trade=source();r=selected(mark,trade)[0];i=r['entry_index']
    trade['high'][i]=r['tp1']+1;trade['low'][i]=r['sl']-1
    assert resolve(r,trade)['outcome']=='SL'
    trade['open'][i]=r['sl']-0.5
    assert resolve(r,trade)['net_r']<-1


def test_a_last_trade_spike_is_not_a_mark_trigger_or_a_verified_mark_fill():
    mark,trade=source()
    rows=candidates(mark,trade,'TEST',kind='break16',target_fraction=.015,regular_only=False,trigger_kind='mark')
    r=rows[0];i=r['entry_index'];trade['high'][i]=r['tp1']+.1
    result=resolve(r,trade,mark=mark)
    assert result['outcome']=='TIME' and result['exit_price_kind']=='hypothetical_mark_trigger_market_fill'
    with pytest.raises(ValueError,match='mark_trigger_history_required'):resolve(r,trade)


def test_short_stop_priority_and_adverse_opening_gap_match_market_stop_direction():
    t=[900+i*900 for i in range(32)]
    trade=dict(time=t,open=[100.]*32,high=[102.]*32,low=[97.]*32,close=[100.]*32,volume=[1.]*32)
    row=dict(entry_time=900,entry_index=0,direction_n=-1,entry=100.,sl=101.,tp1=98.,risk_pct=.01)
    assert resolve(row,trade)['net_r']==pytest.approx(-1.)
    trade['open'][0]=101.5
    assert resolve(row,trade)['net_r']==pytest.approx(-1.5)
