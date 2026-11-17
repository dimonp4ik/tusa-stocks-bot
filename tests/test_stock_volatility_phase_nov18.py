import copy
import numpy as np
import pandas as pd
import pytest
from stock_volatility_phase_nov18 import feature_pool,phase_direction


def candles(scale=1.):
    t=np.arange(100)*900
    c=100+scale*np.arange(100)*.03
    return dict(time=t.tolist(),open=(c-.01*scale).tolist(),high=(c+.1*scale).tolist(),
                low=(c-.1*scale).tolist(),close=c.tolist(),volume=[1.]*100)


def test_entry_features_do_not_read_any_following_open_range_or_volume():
    sources={f'S{i}':candles(i+1) for i in range(12)}
    before=feature_pool(sources)
    changed=copy.deepcopy(sources)
    for c in changed.values():
        for key in ('open','high','low','close'):c[key][80:]=[1e5]*20
        c['volume'][80:]=[1e10]*20
    after=feature_pool(changed)
    pd.testing.assert_frame_equal(before[before.entry_time.le(80*900)].reset_index(drop=True),
        after[after.entry_time.le(80*900)].reset_index(drop=True))


def test_pre_impulse_volatility_is_not_inflated_by_the_impulse_it_normalizes():
    c=candles();sources={f'S{i}':copy.deepcopy(c) for i in range(12)}
    # Change only the two impulse candles, leaving the older ATR untouched.
    sources['S0']['close'][79]=105.;sources['S0']['high'][79]=105.1
    f=feature_pool(sources);r=f[(f.symbol=='S0')&f.entry_time.eq(80*900)].iloc[0]
    expected=(105./c['close'][77]-1)/(.2/c['close'][77])
    assert r.normalized_move==pytest.approx(expected)
    assert r.atr14_fraction>.002


@pytest.mark.parametrize('impulse',[1,-1])
def test_closed_phase_distinguishes_supported_continuation_and_wick_rejection(impulse):
    row=dict(signal_return=impulse*.01,body_fraction=impulse*.002,
        close_location=.85 if impulse>0 else .15,own_trend_16=impulse*.02,
        basket_trend_16=impulse*.01,upper_wick=.1,lower_wick=.1,active_bars_8=8,last_volume=1.)
    assert phase_direction(row,'closed_phase')==impulse
    row.update(body_fraction=-impulse*.001,upper_wick=.5,lower_wick=.5)
    assert phase_direction(row,'closed_phase')==-impulse
    row.update(body_fraction=0.,upper_wick=0.,lower_wick=0.)
    assert phase_direction(row,'closed_phase') is None


def test_zero_cross_section_dispersion_is_unknown_and_bad_ohlc_is_rejected():
    source={f'S{i}':candles() for i in range(12)}
    f=feature_pool(source)
    assert f.raw_z.isna().all() and f.robust_normalized_z.isna().all()
    source['S0']['low'][70]=1000
    with pytest.raises(ValueError,match='invalid source arrays'):feature_pool(source)
