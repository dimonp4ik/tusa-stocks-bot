from stock_breakout_retest_research import retest_entries


def candle():
    close=[100+i*.002 for i in range(112)]
    c=dict(time=[i*900 for i in range(112)],open=close.copy(),close=close,
        high=[p+.1 for p in close],low=[p-.1 for p in close],volume=[100]*112)
    c['close'][100]=100.9;c['high'][100]=100.92;c['low'][100]=100.17;c['volume'][100]=300
    c['open'][101]=100.9;c['low'][101]=100.22;c['high'][101]=100.55;c['close'][101]=100.45
    c['open'][102]=100.65
    return c


def anchor_rows(c):
    return [r for r in retest_entries(c,'TEST') if r['breakout_signal_time']==100*900]


def test_first_retest_enters_next_open_without_reading_that_bars_outcome():
    c=candle();expected=anchor_rows(c)
    assert len(expected)==1 and expected[0]['entry_index']==102
    for k,v in [('high',1000),('low',1),('close',999),('volume',0)]:
        changed={key:list(values) for key,values in c.items()};changed[k][102]=v
        assert anchor_rows(changed)==expected


def test_close_back_through_level_invalidates_setup():
    c=candle();c['close'][101]=100.1
    assert not anchor_rows(c)


def test_invalid_first_market_open_is_not_replaced_with_future_better_entry():
    c=candle();c['open'][102]=100.1
    assert not anchor_rows(c)
