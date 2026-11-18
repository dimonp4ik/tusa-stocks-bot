import copy
from stock_mark_retest_nov18 import candidates,retest_signals


def fixture():
    n=200;mark=dict(time=[1783382400+j*900 for j in range(n)],open=[100.]*n,
        high=[100.1]*n,low=[99.9]*n,close=[100.]*n)
    mark['close'][140]=100.4;mark['high'][140]=100.5
    mark['open'][142]=100.08;mark['low'][142]=100.05;mark['high'][142]=100.25;mark['close'][142]=100.2
    trade=copy.deepcopy(mark);trade['volume']=[1.]*n
    return mark,trade


def test_breakout_alone_is_not_a_retest_and_confirmation_uses_known_close():
    mark,trade=fixture()
    prefix=lambda f:{k:v[:141] for k,v in f.items()}
    assert retest_signals(prefix(mark))==[]
    rows=candidates(mark,trade,'TEST',target_fraction=.015,regular_only=False)
    assert len(rows)==1 and rows[0]['entry_time']==mark['time'][142]+900
    assert rows[0]['entry']==trade['close'][142] and rows[0]['breakout_at']<rows[0]['entry_time']


def test_future_prices_do_not_change_retest_confirmation_or_its_stop():
    mark,trade=fixture();first=candidates(mark,trade,'TEST',target_fraction=.015,regular_only=False)[0]
    for f in (mark,trade):
        for k in ('open','high','low','close'):f[k][143:]=[1000.]*57
    assert candidates(mark,trade,'TEST',target_fraction=.015,regular_only=False)[0]==first


def test_failed_level_cannot_be_recovered_with_future_hindsight():
    mark,trade=fixture();mark['close'][141]=99.8;mark['low'][141]=99.7
    assert retest_signals(mark)==[]


def test_short_retest_is_the_price_mirror_of_the_long_setup():
    mark,trade=fixture()
    for frame in (mark,trade):
        old=copy.deepcopy(frame)
        frame['open']=[200-v for v in old['open']];frame['close']=[200-v for v in old['close']]
        frame['high']=[200-v for v in old['low']];frame['low']=[200-v for v in old['high']]
    rows=candidates(mark,trade,'TEST',target_fraction=.015,regular_only=False)
    assert len(rows)==1 and rows[0]['direction']=='SHORT'
    assert rows[0]['tp1']<rows[0]['entry']<rows[0]['sl']


def test_expired_wait_cannot_reappear_as_a_later_retest():
    mark,trade=fixture()
    for k in ('open','high','low','close'):mark[k][142]=100.
    for k in ('open','high','low','close'):mark[k][145]=100.
    # The later reclaim is outside the four-bar pending setup.
    mark['open'][146]=100.08;mark['low'][146]=100.05;mark['high'][146]=100.25;mark['close'][146]=100.2
    assert retest_signals(mark)==[]
