import copy
import json
import math
import sqlite3
from unittest.mock import patch
import pytest

from src import stock_xperp_native_shadow as native
from src.stock_structure_observer import (StockStructureObserver,PROFILE,FROZEN_AT,
    PublicSource,book_evidence,closed_minute_activity,signal_geometry,observer_path)
from src.structure_observation_store import ObservationStore,closed_frame

BUCKET=math.ceil(FROZEN_AT/900)*900
NOW=BUCKET+20


def frame():
    n=native.REQUIRED_HISTORY_BARS
    return dict(time=list(range(BUCKET-n*900,BUCKET,900)),open=[100.]*n,high=[100.2]*n,
        low=[99.8]*n,close=[100.]*n,volume=[0.]*n,volume_contracts=[.05]*n,confirmed=[1]*n)


def book():return dict(ts=NOW*1000,bids=[['99.99','10']],asks=[['100.01','10']])


def minutes():
    return [[str(stamp * 1000), '100', '100', '100', '100',
             '.01' if i in (1, 4, 8, 10, 13) else '0', '0', '0', '1']
            for i, stamp in enumerate(range(BUCKET - 900, BUCKET, 60))]


class Source:
    def __init__(self):self.calls=0;self.candidate_calls=0
    def inventory(self):
        return dict(AAA='AAA-X',BBB='BBB-X'),{s:dict(state='live',ctVal=1,lotSz=1,minSz=1,tickSz=.01)
            for s in ('AAA-X','BBB-X')}
    def candles(self,symbol):return frame()
    def book(self,instrument):self.calls+=1;return book()
    def minute_rows(self, instrument, bucket):
        assert (instrument, bucket, self.calls) == ('AAA-X', BUCKET, 2)
        return list(reversed(minutes()))
    def candidates(self,frames,now,diagnostics):
        self.candidate_calls+=1;diagnostics['result']='candidates'
        assert all(series['volume'] == frame()['volume_contracts'] for series in frames.values())
        return [dict(symbol='AAAUSDT',direction='LONG',entry=100.,atr14_fraction=.004,
            target_r=2.5,signal_bar_ts=BUCKET-900,entry_time=BUCKET)]
    def place_order(self,*args):raise AssertionError('collector must never trade')


def test_collection_is_idempotent_durable_and_does_not_create_signal_tables(tmp_path):
    path=tmp_path/'observer.sqlite';source=Source()
    observer=StockStructureObserver(path,source=source,clock=lambda:NOW,symbols=['AAAUSDT','BBBUSDT'])
    result=observer.run()
    assert result['status']=='complete' and result['quotes']==2 and result['candidates']==1
    assert result['books_within_spread']==2 and result['native_candidates_within_spread']==1
    assert result['minute_activity']['AAAUSDT']['active_minutes'] == 5
    assert result['minute_activity']['AAAUSDT']['newest_traded_minute'] == 2
    assert observer.run()['status']=='already_recorded' and source.calls==2
    restarted=ObservationStore(path);summary=restarted.summary(PROFILE)
    assert summary['quotes']==2 and summary['candidates']==1
    assert json.loads(summary['last']['summary'])['minute_activity'] == result['minute_activity']
    with sqlite3.connect(path) as c:
        tables={r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert not tables & {'signals','autotrade_users','autotrade_positions'}
        payload=json.loads(c.execute("SELECT payload FROM observations WHERE symbol='AAAUSDT'").fetchone()[0])
    assert payload['quote_reference_price']==100.01 and payload['market_reference_only']
    assert payload['geometry']['signal_entry']==100.
    rebuilt=restarted.reconstruct('AAA-X',payload['frame'])
    assert rebuilt['close']==frame()['close']
    assert rebuilt['volume']==frame()['volume_contracts']
    assert payload['frame']['kind']=='trade_contracts'


def test_observer_counts_wide_books_as_valid_but_not_within_entry_spread(tmp_path):
    source=Source()
    wide=dict(ts=NOW*1000,bids=[['99','10']],asks=[['101','10']])
    source.book=lambda instrument:wide if instrument=='AAA-X' else book()
    result=StockStructureObserver(tmp_path/'observer.sqlite',source=source,clock=lambda:NOW,
                                  sleep=lambda _: None, symbols=['AAAUSDT','BBBUSDT']).run()
    assert result['quotes']==2 and result['raw_setups']==1
    assert result['books_within_spread']==1 and result['native_candidates_within_spread']==0


def test_wide_native_observation_records_two_independent_books_without_a_signal(tmp_path):
    source = Source()
    now = [NOW]

    def sleep(seconds):
        now[0] += seconds

    def quote(instrument):
        source.calls += 1
        spread = .1 if now[0] == NOW and instrument == 'AAA-X' else .07
        if instrument == 'BBB-X':
            spread = .02
        return {'ts': now[0] * 1000, 'bids': [[100 - spread / 2, 5]], 'asks': [[100 + spread / 2, 5]]}

    source.book = quote
    observer = StockStructureObserver(tmp_path / 'observer.sqlite', source=source, clock=lambda: now[0], sleep=sleep,
                                      symbols=['AAAUSDT', 'BBBUSDT'])
    result = observer.run()
    probe = result['spread_rechecks']['AAAUSDT']
    assert probe['distinct_pair'] and probe['max_spread_bps'] == pytest.approx(7)
    assert probe['initial_received_at'] == NOW
    assert [row['received_at'] for row in probe['samples']] == [NOW + 5, NOW + 10]
    assert source.calls == 4
    assert result['native_candidates_within_spread'] == 0
    assert result['status'] == 'complete'


def test_revised_closed_candle_is_preserved_without_overwriting_old_evidence(tmp_path):
    store=ObservationStore(tmp_path/'observer.sqlite');store.initialize()
    old=closed_frame(frame(),BUCKET,native.REQUIRED_HISTORY_BARS)
    first=store.put_frame('A',old)
    changed=copy.deepcopy(old);changed['close'][-1]=100.1
    second=store.put_frame('A',changed)
    assert second['overrides'] and not first['overrides']
    assert store.reconstruct('A',first)==old
    assert store.reconstruct('A',second)==changed


def test_forming_or_future_bar_cannot_create_a_candidate(tmp_path):
    source=Source();bad=frame();bad['time'][-1]=BUCKET;bad['confirmed'][-1]=0
    source.candles=lambda symbol:bad
    observer=StockStructureObserver(tmp_path/'observer.sqlite',source=source,clock=lambda:NOW,
                                    symbols=['AAAUSDT','BBBUSDT'])
    result=observer.run()
    assert result['loaded']==0 and result['candidates']==0 and source.candidate_calls==0
    assert result['status']=='partial' and result['quotes']==2


def test_stale_or_failed_book_is_not_invented_as_a_valid_quote(tmp_path):
    source=Source();source.book=lambda instrument:None
    result=StockStructureObserver(tmp_path/'observer.sqlite',source=source,clock=lambda:NOW,
                                  sleep=lambda _: None, symbols=['AAAUSDT','BBBUSDT']).run()
    assert result['quotes']==0 and result['status']=='partial'
    old=book();old['ts']-=4000
    assert not book_evidence(old,NOW)['ok']
    assert book_evidence(old,NOW)['rejection']=='book_age'
    crossed=book();crossed['asks']=[['99','1']]
    assert not book_evidence(crossed,NOW)['ok']


def test_minute_activity_rejects_disagreement_and_incomplete_windows(tmp_path):
    source = Source()
    source.minute_rows = lambda instrument, bucket: minutes()[:-1]
    result = StockStructureObserver(tmp_path / 'observer.sqlite', source=source, clock=lambda: NOW,
                                    symbols=['AAAUSDT', 'BBBUSDT']).run()
    assert result['status'] == 'complete' and result['native_candidates_within_spread'] == 1
    assert result['minute_activity'] == {'AAAUSDT': {'status': 'unverified'}}
    with pytest.raises(ValueError, match='minute_volume_disagreement'):
        closed_minute_activity(minutes(), BUCKET, .06)
    rows = minutes()
    rows[-1][0] = str(BUCKET * 1000)
    with pytest.raises(ValueError, match='minute_coverage'):
        closed_minute_activity(rows, BUCKET, .05)


def test_wrong_contract_book_cannot_be_labelled_as_the_requested_market():
    source=PublicSource()
    answer=dict(data=[dict(instId='OTHER-X',**book())])
    with patch('src.binance_client._okx_get',return_value=answer):
        with pytest.raises(ValueError,match='book_instrument_mismatch'):
            source.book('AAA-X')


def test_single_instrument_books_accept_actual_okx_payload_without_echoed_id():
    source=PublicSource()
    row=dict(seqId=42,**book())
    with patch('src.binance_client._okx_get',return_value=dict(code='0',data=[row])) as request:
        assert source.book('AAA-X')==row
    assert request.call_args.args[1]['instId']=='AAA-X'


def test_freeze_and_closed_geometry_do_not_use_future_quotes_or_prices(tmp_path):
    source=Source()
    value=StockStructureObserver(tmp_path/'observer.sqlite',source=source,clock=lambda:FROZEN_AT-1,
                                  symbols=['AAAUSDT']).run()
    assert value['status']=='before_freeze' and source.calls==0
    g=signal_geometry(dict(direction='LONG',entry=100,atr14_fraction=.004,target_r=2.5),frame())
    assert g['stop']==pytest.approx(99.7) and g['target']==pytest.approx(102.5)
    assert g['signal_risk_fraction']==.003
    main=tmp_path/'main.db';main.touch()
    assert observer_path(main).parent==main.parent
    assert observer_path(main)!=main


def test_read_only_status_does_not_create_or_modify_the_archive(tmp_path):
    path=tmp_path/'missing.sqlite'
    assert ObservationStore(path).summary(PROFILE)['status']=='not_started'
    assert not path.exists()


def test_data_cleanup_preserves_the_indicator_prefix_of_retained_scans(tmp_path):
    store=ObservationStore(tmp_path/'observer.sqlite');store.initialize()
    f=closed_frame(frame(),BUCKET,native.REQUIRED_HISTORY_BARS)
    ref=store.put_frame('A',f)
    store.claim(PROFILE,BUCKET,NOW,'test')
    store.finish(PROFILE,BUCKET,NOW+29*86400,'complete',{})
    assert store.reconstruct('A',ref)==f


def test_expired_interrupted_scan_is_cleaned_before_a_new_capture_can_fail(tmp_path):
    observer=StockStructureObserver(tmp_path/'observer.sqlite',source=Source(),clock=lambda:NOW,
        symbols=['AAAUSDT','BBBUSDT'])
    observer.store.initialize()
    observer.store.claim(PROFILE,BUCKET-40*86400,NOW-40*86400,'interrupted')
    with patch.object(observer.store,'claim',side_effect=sqlite3.OperationalError('disk full')):
        result=observer.run()
    assert result['status']=='archive_unavailable'
    assert observer.store.summary(PROFILE)['scans']==0


def test_application_collection_is_independent_of_trading_pause_and_client_database(tmp_path):
    import main
    source=Source();main_database=tmp_path/'main.db';main_database.write_bytes(b'untouched')
    def factory(path,revision):
        return StockStructureObserver(path,source=source,clock=lambda:NOW,
            symbols=['AAAUSDT','BBBUSDT'],revision=revision)
    with (patch.dict('os.environ',{'STOCK_STRUCTURE_OBSERVER_ENABLED':'1','DB_PATH':str(main_database)}),
          patch.object(main,'_structure_quote_observer',None),
          patch.object(main.autotrader._shared_capacity,'persistent_storage_ready',return_value=True),
          patch('src.stock_structure_observer.StockStructureObserver',side_effect=factory),
          patch.object(main,'_native_xperp_daily_loss_pause',return_value=True) as paused,
          patch.object(main.autotrader,'at_get_active_traders') as clients):
        assert main._run_structure_quote_observer()['quotes']==2
        assert 'valid books 2' in main._structure_observer_status_text()
        assert 'spread within 5 bp: 2/2 books, 1/1 native setups' in main._structure_observer_status_text()
        assert 'minute activity verified 1/1 native setups, 1 with at least 3 traded minutes' in main._structure_observer_status_text()
    assert main_database.read_bytes()==b'untouched'
    paused.assert_not_called();clients.assert_not_called()
