from unittest.mock import Mock
import pytest
from src.entry_window import clip_native_entry_minute,EntryWindowUnavailable
from src import binance_client as market


def seconds(events=None):
    values=[]
    events=events or {}
    for stamp in range(60):
        o,h,l,c,v=events.get(stamp,(100,100,100,100,0))
        values.append([str(stamp*1000),str(o),str(h),str(l),str(c),str(v),str(v),'0','1'])
    return values


def frame(raw):
    traded=[r for r in raw if float(r[5])>0]
    o,h,l,c=(float(traded[0][1]),max(float(r[2]) for r in traded),
        min(float(r[3]) for r in traded),float(traded[-1][4])) if traded else (100,100,100,100)
    return dict(time=[0,60],open=[o,100],high=[h,100],low=[l,100],close=[c,100],
        volume=[sum(float(r[6]) for r in raw),0],confirmed=[1,1])


def clip(raw,*,opened=10.5,ticks=(),**kwargs):
    return clip_native_entry_minute(frame(raw),opened_at=opened,direction='LONG',
        sl=99,tp1=102,tp2=102,now=120,load_seconds=lambda _:raw,
        load_second_trades=lambda _:ticks,**kwargs)


def test_pre_entry_target_and_stop_wicks_cannot_close_the_signal():
    raw=seconds({5:(100,104,98,100,1)})
    result=clip(raw)
    assert max(result['high'])==100 and min(result['low'])==100
    assert min(result['time'])==11


def test_post_entry_gap_is_preserved_in_chronological_seconds():
    raw=seconds({0:(100,100,100,100,1),20:(98,98,98,98,1)})
    result=clip(raw)
    i=result['time'].index(20)
    assert result['open'][i]==98 and result['low'][i]==98


def test_boundary_second_uses_exact_publication_milliseconds():
    raw=seconds({0:(100,100,100,100,1),10:(103,103,100,100,2)})
    ticks=[dict(ts='10100',tradeId='1',px='103',sz='1'),
        dict(ts='10600',tradeId='2',px='100',sz='1')]
    result=clip(raw,ticks=ticks)
    assert max(result['high'])==100 and result['time'][0]==10.6


def test_post_publication_boundary_trade_is_retained():
    raw=seconds({0:(100,100,100,100,1),10:(100,103,100,103,2)})
    ticks=[dict(ts='10100',tradeId='1',px='100',sz='1'),
        dict(ts='10600',tradeId='2',px='103',sz='1')]
    assert clip(raw,ticks=ticks)['high'][0]==103


def test_missing_or_revised_seconds_never_invent_a_result():
    raw=seconds({5:(100,103,100,100,1)})
    with pytest.raises(EntryWindowUnavailable):clip(raw[:-1])
    bad=raw+[list(raw[10])];bad[-1][2]='101'
    with pytest.raises(EntryWindowUnavailable):clip(bad)


def test_trade_volume_must_reconcile_before_filtering_publication_second():
    raw=seconds({0:(100,100,100,100,1),10:(100,103,100,103,2)})
    ticks=[dict(ts='10100',tradeId='1',px='100',sz='1'),
        dict(ts='10600',tradeId='2',px='103',sz='.5')]
    with pytest.raises(EntryWindowUnavailable):clip(raw,ticks=ticks)


def test_no_extra_requests_if_the_entry_minute_cannot_hit_a_bracket():
    load=Mock(side_effect=AssertionError('unexpected extra request'))
    original=frame(seconds())
    assert clip_native_entry_minute(original,opened_at=10.5,direction='LONG',
        sl=99,tp1=102,tp2=102,now=120,load_seconds=load,load_second_trades=load) is original
    load.assert_not_called()


def test_forming_entry_minute_waits_without_loading_or_scoring_old_extrema():
    raw=seconds({5:(100,103,100,100,1)});original=frame(raw);original['confirmed'][0]=0
    load=Mock()
    with pytest.raises(EntryWindowUnavailable):
        clip_native_entry_minute(original,opened_at=10.5,direction='LONG',sl=99,tp1=102,tp2=102,
            now=50,load_seconds=load,load_second_trades=load)
    load.assert_not_called()


def test_public_trade_pagination_uses_trade_ids_after_timestamp_anchor(monkeypatch):
    market.get_xperp_entry_second_trades.cache_clear()
    monkeypatch.setattr(market,'get_xperp_instruments',lambda:{'TEST':'TEST-X'})
    api=Mock(side_effect=[dict(data=[dict(ts='10500',tradeId='20',px='100',sz='1')]),
        dict(data=[dict(ts='10400',tradeId='19',px='101',sz='1'),
            dict(ts='9900',tradeId='18',px='99',sz='1')])])
    monkeypatch.setattr(market,'_okx_get',api)
    result=market.get_xperp_entry_second_trades('TESTUSDT',10)
    assert len(result)==2
    assert api.call_args_list[0].args[1]['type']=='2'
    assert api.call_args_list[1].args[1]['type']=='1'
    assert api.call_args_list[1].args[1]['after']=='20'
    market.get_xperp_entry_second_trades.cache_clear()


def test_closed_one_minute_cache_expires_at_the_next_minute_boundary(monkeypatch):
    market._kl_cache.clear();clock=[122.]
    monkeypatch.setattr(market.time,'time',lambda:clock[0])
    monkeypatch.setattr(market,'get_xperp_instruments',lambda:{'TEST':'TEST-X'})
    def row(stamp,price,confirmed):
        return [str(stamp*1000),str(price),str(price),str(price),str(price),'1','1','1',confirmed]
    api=Mock(side_effect=[dict(data=[row(120,102,'0'),row(60,101,'1'),row(0,100,'1')]),
        dict(data=[row(180,103,'0'),row(120,102,'1'),row(60,101,'1')])])
    monkeypatch.setattr(market,'_okx_get',api)
    assert market.get_klines_xperp('TESTUSDT',limit=2,interval='1m')['close'][-1]==101
    clock[0]=184.
    assert market.get_klines_xperp('TESTUSDT',limit=2,interval='1m')['close'][-1]==102
    assert api.call_count==2
    market._kl_cache.clear()
