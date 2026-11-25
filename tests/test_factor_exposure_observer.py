import logging
import math
from queue import Queue
from types import SimpleNamespace

from src import factor_exposure_observer as observer
from src import okx_trader
from src.shared_capacity_runtime import SharedCapacityRuntime


def bar(stamp, close, lots='1', confirmed='1'):
    return [str(stamp*1000), '1', '1', '1', str(close), lots, '1', '1', confirmed]


def test_only_traded_confirmed_past_bars_produce_returns():
    data = [bar(0,1), bar(900,2), bar(1800,3,'0'), bar(2700,4), bar(3600,8),
            bar(4500,16,confirmed='0'), bar(5400,32), bar(6300,64)]
    assert set(observer.closed_returns(reversed(data),6300)) == {900,3600}
    assert observer.closed_returns(data,6300)[900] == math.log(2)


def histories(epoch, *, opposed=False, volatile=False):
    btc = {}
    for offset in range(96,0,-1):
        n = .0002*((offset % 5)-2)
        if offset<=16:
            n = -.002 + n if opposed else .002 + n if volatile else .0001 + n
        btc[epoch-offset*900] = n
    return {'BTC-X':btc,'ETH-X':dict(btc),'MSTR-X':{k:-v for k,v in btc.items()}}


def test_shadow_uses_shared_direction_and_condition():
    epoch = 100*900
    held = (('BTC-X','long'),('MSTR-X','short'))
    for opposed,volatile,expected in ((True,False,'opposed'),(False,True,'volatile'),(False,False,'calm')):
        data = histories(epoch,opposed=opposed,volatile=volatile)
        result = observer.evaluate('ETH-X','LONG',held,epoch,'BTC-X',lambda i,e:data[i])
        assert result == ('would_allow' if expected=='calm' else 'would_block',expected,2,0)
    data = histories(epoch,opposed=True)
    missed = observer.evaluate('ETH-X','LONG',(('BTC-X','long'),('MSTR-X',None)),epoch,
                               'BTC-X',lambda i,e:data[i])
    assert missed == ('unknown','incomplete_positions',1,1)
    hedged = observer.evaluate('ETH-X','LONG',(('BTC-X','long'),('MSTR-X','long')),epoch,
                               'BTC-X',lambda i,e:data[i])
    assert hedged == ('would_allow','below_factor_limit',1,0)


def test_observer_keeps_the_cap_if_phase_data_is_missing():
    epoch = 100*900
    data = histories(epoch)
    data = {i:{k:v for k,v in values.items() if k<epoch-16*900} for i,values in data.items()}
    result = observer.evaluate('ETH-X','LONG',(('BTC-X','long'),('MSTR-X','short')),epoch,
                               'BTC-X',lambda i,e:data[i])
    assert result == ('would_block','unknown',2,0)


def test_enqueued_snapshot_has_directions_and_no_sensitive_fields(monkeypatch):
    jobs = Queue()
    monkeypatch.setattr(observer,'_JOBS',jobs)
    monkeypatch.setattr(observer,'_worker',SimpleNamespace(is_alive=lambda:True))
    snapshot = dict(count=2,inst_ids=('BTC-X','MSTR-X'),side_by_inst={'BTC-X':'long','MSTR-X':'short'},
                    margin_usd=123,account_key='private')
    observer.record(11,{'id':4,'strategy_entry_bar_ts':900,'direction':'LONG'},'ETH-X',snapshot,913)
    assert jobs.get_nowait() == (11,4,'ETH-X','LONG',900,(('BTC-X','long'),('MSTR-X','short')))


def test_cheap_observation_logs_allow_and_stale_snapshot_logs_unknown(caplog):
    with caplog.at_level(logging.INFO,logger=observer.__name__):
        snapshot = dict(count=1,inst_ids=('BTC-X',),side_by_inst={'BTC-X':'long'})
        signal = {'id':4,'strategy_entry_bar_ts':900,'direction':'LONG'}
        observer.record(11,signal,'ETH-X',snapshot,913)
        observer.record(12,signal,'ETH-X',snapshot,1800)
    assert 'decision=would_allow basis=less_than_two_positions' in caplog.text
    assert 'decision=unknown basis=snapshot_context' in caplog.text
    assert 'BTC-X' not in caplog.text


def test_account_snapshot_infers_net_sides_without_sending_them_to_coordinator(monkeypatch):
    rows = [dict(instId='BTC-X',pos='2',posSide='net',instType='FUTURES',margin='1'),
            dict(instId='MSTR-X',pos='-2',posSide='net',instType='FUTURES',margin='2'),
            dict(instId='ETH-X',pos='1',posSide='n/a',margin='3')]
    monkeypatch.setattr(okx_trader,'pending_entries_clear',lambda _: (True,None))
    monkeypatch.setattr(okx_trader,'_request',lambda *_: (True,rows))
    ok,snapshot = okx_trader.get_open_position_margin({})
    assert ok and snapshot['margin_usd']==6 and snapshot['short_count']==1
    assert snapshot['side_by_inst'] == {'BTC-X':'long','MSTR-X':'short','ETH-X':None}
    assert snapshot['unknown_side_count'] == 1
    assert 'side_by_inst' not in SharedCapacityRuntime('crypto')._snapshot(snapshot,130,913)
