import os
os.environ['PYTHON_DOTENV_DISABLED']='1'
from contextlib import ExitStack
from unittest.mock import patch
import sqlite3
import time
import pytest
from src.tp1_plan import plan_tp1_close
from src import db
from src import autotrader as at
import main


@pytest.mark.parametrize('position,pct,lot,minimum,size',[
    (3,50,1,1,1),(2,50,1,1,1),(1,50,1,1,1),
    (5,50,1,3,5),(20,10,1,3,3),(2.9,50,.1,.1,1.4),
    (.5,50,.1,.3,.5),(5,0,1,1,0),(5,5,1,1,1)])
def test_explicit_minimum_mode_places_a_valid_quantity_and_no_dust(position,pct,lot,minimum,size):
    p=plan_tp1_close(position,pct,dict(lotSz=lot,minSz=minimum),ensure_minimum=True)
    assert p['size']==pytest.approx(size)
    assert 0<=p['size']<=position
    if 0<p['size']<position:
        assert p['size']>=minimum
        assert position-p['size']>=minimum-1e-12


def test_legacy_five_percent_and_disabled_tp1_are_never_silently_rounded_up():
    assert plan_tp1_close(3,5,dict(lotSz=1,minSz=1))['size']==0
    assert plan_tp1_close(1,0,dict(lotSz=1,minSz=1),ensure_minimum=True)['size']==0


def test_tp1_is_reduce_only_market_on_trigger_not_a_limit_entry():
    with patch.object(at.okx,'_request',return_value=(True,[{'algoId':'tp1','sCode':'0'}])) as request:
        ok,_=at.okx.place_tp1_partial({},'TEST-X','LONG',101,1)
    assert ok
    payload=request.call_args.kwargs['body']
    assert payload['reduceOnly']=='true'
    assert payload['tpOrdPx']=='-1'
    assert payload['sz']=='1'


def test_migration_preserves_old_preferences_and_only_clicked_user_changes(tmp_path,monkeypatch):
    path=tmp_path/'tp1.sqlite3';monkeypatch.setattr(db,'DB_PATH',str(path))
    db.init_db()
    with sqlite3.connect(path) as c:
        c.execute('ALTER TABLE autotrade_users DROP COLUMN tp1_minimum_close')
        for uid in (7,8):
            c.execute("INSERT INTO autotrade_users(user_id,allowed,size_mode,size_value,tp1_close_pct) VALUES(?,1,'percent',3,5)",(uid,))
    db.init_db()
    assert db.at_get(7)['tp1_close_pct']==5
    assert db.at_get(7)['tp1_minimum_close']==0
    db.at_set_tp1_close_pct(7,50,minimum_close=True)
    assert db.at_get(7)['tp1_minimum_close']==1
    assert db.at_get(7)['size_value']==3
    assert db.at_get(8)['tp1_close_pct']==5
    assert db.at_get(8)['tp1_minimum_close']==0
    db.at_set_tp1_close_pct(7,25)
    assert db.at_get(7)['tp1_minimum_close']==0


def test_auto_button_changes_requesting_user_setting_and_issues_no_exchange_order():
    with (patch.object(main,'at_get',return_value=dict(allowed=1,tp1_close_pct=5)),
          patch.object(main,'at_set_tp1_close_pct') as save,
          patch.object(main,'_answer_callback'),patch.object(main,'_reply') as reply,
          patch.object(main,'_at_show_menu'),patch.object(at.okx,'place_tp1_partial') as order):
        assert main._at_handle_callback('cb',99,7,'at_tp1auto50') is True
    save.assert_called_once_with(7,50.,minimum_close=True)
    order.assert_not_called()
    assert 'целиком' in reply.call_args.args[1]
    assert 'Уже выставленные ордера не меняются' in reply.call_args.args[1]


def test_one_flat_tp1_response_does_not_retire_the_position_or_cancel_its_stop():
    pos=dict(id=1,user_id=7,inst_id='TEST-X',sz=1,tp1_algo_id='tp1',sl_algo_id='sl',
        direction='LONG',entry_px=100,sl_px=100)
    with (patch.object(at,'AUTOTRADE_ENABLED',True),
          patch.object(at,'at_open_positions_for_signal',return_value=[pos]),
          patch.object(at,'at_get',return_value={'user_id':7}),
          patch.object(at,'_creds_of',return_value={'test':True}),
          patch.object(at,'_dm'),
          patch.object(at.okx,'get_xperp_spec',return_value={'tickSz':1}),
          patch.object(at.okx,'get_position_size',return_value=(True,0)),
          patch.object(at,'at_close_position') as close,
          patch.object(at.okx,'cancel_protection') as cancel):
        at.mirror_transition(dict(id=2,symbol='TESTUSDT'),'TP1_PARTIAL',100.)
    close.assert_not_called();cancel.assert_not_called()


@pytest.mark.parametrize('actual_size,expected',[(5,2),(3,1),(1,1)])
def test_real_entry_uses_actual_fill_for_minimum_tp1_order(actual_size,expected):
    spec=dict(ctVal=1,lotSz=1,minSz=1,tickSz=.1)
    sig=dict(id=11,symbol='BTCUSDT',direction='LONG',entry_price=7.8,sl=7,tp1=8.5,tp2=10)
    messages=[]
    with ExitStack() as stack:
        for name,value in [('_creds_of',{'test':True}),('at_has_open_position',False),
            ('at_set_balance',None),('get_bot_state',None),('set_bot_state',None),
            ('set_signal_size_mult',None),('at_create_entry_intent',91),('at_mark_entry_submitting',True),
            ('at_update_entry_intent_protection',None),('at_activate_entry_intent',True)]:
            stack.enter_context(patch.object(at,name,return_value=value))
        stack.enter_context(patch.object(at,'_dm',side_effect=lambda uid,text:messages.append(text)))
        for name,value in [('validate_autotrade_account',(True,dict(position_mode='net_mode',permissions=('read_only','trade'),account_uid='123'))),
            ('get_balance',(True,130)),('get_open_position_margin',(True,dict(margin_usd=0,count=0,inst_ids=()))),
            ('get_xperp_spec',spec),('get_last_price',7.8),('ensure_leverage',(True,None)),
            ('get_order_book',dict(ts=time.time()*1000,bids=[['7.799','100']],asks=[['7.8','100']])),
            ('place_market_entry',(True,'entry-1')),('get_position_avg_px',7.8),('place_protection_oco',(True,'stop-1'))]:
            stack.enter_context(patch.object(at.okx,name,return_value=value))
        stack.enter_context(patch.object(at.okx,'get_position_size',side_effect=[(True,0),(True,actual_size)]))
        order=stack.enter_context(patch.object(at.okx,'place_tp1_partial',return_value=(True,'tp1-1')))
        at._open_for_user(dict(user_id=7,size_mode='percent',size_value=3,tp1_close_pct=50,tp1_minimum_close=1),
            sig,'BTC-USDT-SWAP','BTC')
    assert order.call_args.args[-1]==expected
    assert messages
    if actual_size==1:assert 'Остатка к TP2 не будет' in messages[-1]
