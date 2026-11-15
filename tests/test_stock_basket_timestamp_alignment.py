import math
import pytest
from src.stock_xperp_native_shadow import AUDITED_SYMBOLS, basket_trends


def series(end=119):
    return dict(time=[i*900 for i in range(end+1)],close=[100.]*(end+1))


def test_lagging_instrument_still_contributes_at_its_available_timestamp():
    symbols=sorted(AUDITED_SYMBOLS)[:11]
    data={s:series() for s in symbols[:10]}
    stale=series(118);stale['close'][-1]=102.
    data[symbols[-1]]=stale
    # At 119 only ten rows exist (all flat). At 118 the lagging eleventh
    # instrument contributes a two-bar move of 2%, divided by eleven.
    assert basket_trends(data,119*900,(2,))[2]==pytest.approx(.02/11)


def test_past_basket_is_unchanged_by_future_bars_even_with_invalid_future_price():
    data={s:series(118) for s in sorted(AUDITED_SYMBOLS)[:11]}
    for item in data.values():
        item['close'][-1]=101.
    before=basket_trends(data,118*900,(2,4))
    for item in data.values():
        item['time'] += [119*900,120*900]
        item['close'] += [1e9,float('nan')]
    assert basket_trends(data,118*900,(2,4))==before


def test_insufficient_actual_overlap_cannot_be_filled_with_stale_returns():
    data={s:series(118) for s in sorted(AUDITED_SYMBOLS)[:11]}
    assert basket_trends(data,119*900,(2,)) is None


def test_aligned_histories_preserve_closed_bar_cross_sectional_mean():
    data={s:series(220) for s in sorted(AUDITED_SYMBOLS)[:11]}
    for number,item in enumerate(data.values()):
        item['close']=[100+number+math.sin(i/13+number) for i in range(221)]
    expected=sum(sum(v['close'][i]/v['close'][i-2]-1 for v in data.values())/len(data)
        for i in range(220-31,221))
    assert basket_trends(data,220*900,(32,))[32]==pytest.approx(expected,abs=1e-12)
