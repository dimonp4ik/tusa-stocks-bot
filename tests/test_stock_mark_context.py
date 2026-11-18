import copy
import json
from unittest.mock import patch

import pytest
from src.stock_mark_context import mark_frame,MarkContextSource,StockMarkContextObserver,PROFILE,HISTORY_BARS
from src.stock_structure_observer import PROFILE as ORIGINAL_PROFILE
from src.structure_observation_store import ObservationStore,closed_frame

BUCKET=1795033800
NOW=BUCKET+40


def mark_rows():
    return [[str(t*1000),'100','100.2','99.8','100','1']
        for t in range(BUCKET-HISTORY_BARS*900,BUCKET,900)]


def trade_frame():
    n=HISTORY_BARS
    return dict(time=list(range(BUCKET-n*900,BUCKET,900)),open=[101.]*n,high=[101.2]*n,
        low=[100.8]*n,close=[101.]*n,volume=[2.]*n,confirmed=[1]*n)


class Source:
    def inventory(self):return dict(QQQ='QQQ-X'),{'QQQ-X':dict(state='live',ctVal='1',lotSz='.01',minSz='.01',tickSz='.01')}
    def mark_candles(self,instrument,bucket):return mark_frame(mark_rows(),bucket)
    def candles(self,symbol):return trade_frame()
    def book(self,instrument):return dict(ts=NOW*1000,bids=[['99.99','100']],asks=[['100.01','100']])
    def place_order(self,*args):raise AssertionError('research cannot trade')


def test_mark_and_trade_series_remain_distinct_and_mark_does_not_invent_volume(tmp_path):
    store=ObservationStore(tmp_path/'observer.db');store.initialize()
    mark=mark_frame(mark_rows(),BUCKET)
    trade=closed_frame(trade_frame(),BUCKET,HISTORY_BARS)
    first=store.put_frame('QQQ-X',trade)
    second=store.put_frame('QQQ-X',mark,kind='mark')
    assert store.reconstruct('QQQ-X',first)==trade
    assert store.reconstruct('QQQ-X',second)==mark
    assert 'volume' not in mark and second['kind']=='mark' and 'kind' not in first


def test_forming_mark_is_ignored_and_future_confirmed_mark_is_rejected():
    rows=mark_rows()+[[str(BUCKET*1000),'1','1000','1','900','0']]
    assert mark_frame(rows,BUCKET)['close'][-1]==100.
    rows[-1][-1]='1'
    with pytest.raises(ValueError,match='future_confirmed_mark_bar'):mark_frame(rows,BUCKET)


def test_mark_adapter_honors_documented_hundred_row_limit_and_paginates():
    source=MarkContextSource();rows=list(reversed(mark_rows()))
    forming=[str(BUCKET*1000),'100','100','100','100','0']
    data=[forming]+rows
    calls=[]
    def get(path,params,timeout):
        calls.append(params.copy())
        available=[r for r in data if 'after' not in params or int(r[0])<int(params['after'])]
        return dict(data=available[:params['limit']])
    with patch('src.binance_client._okx_get',side_effect=get),patch.object(source,'_pace'):
        result=source.mark_candles('QQQ-X',BUCKET)
    assert result['time'][-1]==BUCKET-900 and len(result['time'])==HISTORY_BARS
    assert len(calls)==2 and all(c['limit']<=100 for c in calls)


def test_context_capture_is_immutable_exports_both_histories_and_preserves_original_profile(tmp_path):
    observer=StockMarkContextObserver(tmp_path/'observer.db',source=Source(),clock=lambda:NOW,symbols=['QQQUSDT'])
    result=observer.run()
    assert result['status']=='complete' and result['mark_loaded']==result['trade_loaded']==result['quotes']==1
    assert observer.run()['status']=='already_recorded'
    assert observer.store.summary(ORIGINAL_PROFILE)['scans']==0
    row=observer.store.export(PROFILE)['scans'][0]['observations'][0]
    assert row['closed_frame']['close'][-1]==100. and 'volume' not in row['closed_frame']
    assert row['trade_closed_frame']['close'][-1]==101.
    assert row['payload']['trade_to_mark_fraction']==pytest.approx(.01)
    assert row['payload']['mark_volume_available'] is False


def test_failed_mark_source_is_unknown_instead_of_falling_back_to_trade_prices(tmp_path):
    source=Source();source.mark_candles=lambda *args: (_ for _ in ()).throw(ValueError('unavailable'))
    observer=StockMarkContextObserver(tmp_path/'observer.db',source=source,clock=lambda:NOW,symbols=['QQQUSDT'])
    result=observer.run()
    assert result['status']=='partial' and result['mark_loaded']==0 and result['trade_loaded']==1
    row=observer.store.export(PROFILE)['scans'][0]['observations'][0]
    assert 'closed_frame' not in row and row['trade_closed_frame']['close'][-1]==101.


def test_context_job_never_reads_client_connections_or_modifies_main_database(tmp_path):
    import main
    database=tmp_path/'main.db';database.write_bytes(b'unchanged')
    def factory(path,revision):
        return StockMarkContextObserver(path,source=Source(),clock=lambda:NOW,symbols=['QQQUSDT'],revision=revision)
    with (patch.dict('os.environ',{'DB_PATH':str(database),'STOCK_MARK_CONTEXT_ENABLED':'1'}),
          patch.object(main,'_liquid_mark_observer',None),
          patch.object(main.autotrader._shared_capacity,'persistent_storage_ready',return_value=True),
          patch('src.stock_mark_context.StockMarkContextObserver',side_effect=factory),
          patch.object(main.autotrader,'at_get_active_traders') as clients):
        assert main._run_liquid_mark_context()['status']=='complete'
        assert 'mark 1/1' in main._liquid_mark_status_text()
    clients.assert_not_called();assert database.read_bytes()==b'unchanged'
