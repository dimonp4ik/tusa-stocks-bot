from stock_structural_breakout_research import entries


def sample():
    close = [100+i*.002 for i in range(110)]
    close[100]=100.9
    candle = dict(time=[i*900 for i in range(110)],open=close.copy(),close=close,
        high=[p+.1 for p in close],low=[p-.1 for p in close],
        volume=[100 if i!=100 else 300 for i in range(110)])
    candle["high"][100]=100.92
    candle["low"][100]=100.17
    candle["open"][101]=100.9
    return candle


def test_next_candle_high_low_close_and_volume_cannot_create_signal():
    c = sample()
    baseline = [r for r in entries(c,"TEST") if r["entry_index"]==101]
    assert len(baseline)==1
    for key,value in (("high",1000),("low",1),("close",999),("volume",0)):
        changed={k:list(v) for k,v in c.items()}
        changed[key][101]=value
        assert [r for r in entries(changed,"TEST") if r["entry_index"]==101]==baseline


def test_zero_historical_volume_cannot_be_used_as_breakout_confirmation():
    c=sample()
    c["volume"][:100]=[0]*100
    assert not [r for r in entries(c,"TEST") if r["entry_index"]==101]
