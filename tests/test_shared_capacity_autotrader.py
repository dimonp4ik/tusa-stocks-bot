import time
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock,patch
import pytest
from src import autotrader as at
from src.shared_capacity_transport import CapacityUnavailable


def entry_harness(prepare=True,reserve_error=False,begin_error=False,expire=False,
                  refresh_ask=None):
    spec=dict(ctVal=1,lotSz=.001,minSz=.001,tickSz=.01)
    sig=dict(id=1,symbol='TEST',direction='LONG',entry_price=100,sl=98,tp1=102,tp2=110)
    shared=SimpleNamespace(enabled=True,prepare=Mock(return_value=prepare),
        reserve=Mock(return_value='a'*32),begin=Mock(),maintenance=Mock())
    if reserve_error:shared.reserve.side_effect=CapacityUnavailable('lost reserve response')
    if begin_error:shared.begin.side_effect=CapacityUnavailable('lost begin response')
    clock=[100.]
    wall = [time.time()]
    if expire:shared.begin.side_effect=lambda *args:clock.__setitem__(0,116.)
    if refresh_ask is not None:
        shared.begin.side_effect=lambda *args:wall.__setitem__(0,wall[0]+4)
    with ExitStack() as stack:
        for name,value in [('_creds_of',{'test':True}),('at_has_open_position',False),
            ('at_set_balance',None),('get_bot_state',None),('set_bot_state',None),
            ('set_signal_size_mult',None),('at_all_open_positions',[]),('_dm',None),('_entry_skip',None)]:
            stack.enter_context(patch.object(at,name,return_value=value))
        for name,value in [('validate_autotrade_account',(True,dict(account_uid='123',position_mode='net_mode'))),
            ('get_balance',(True,128.)),('get_open_position_margin',(True,dict(count=0,inst_ids=(),margin_usd=0.,margin_by_inst={}))),
            ('get_xperp_spec',spec),('get_last_price',100.),('get_position_size',(True,0)),
            ('ensure_leverage',(True,None)),('get_order_book',dict(ts=time.time()*1000,
                bids=[[99.99,1000]],asks=[[100.01,1000]]))]:
            stack.enter_context(patch.object(at.okx,name,return_value=value))
        stack.enter_context(patch.object(at,'_shared_capacity',shared))
        stack.enter_context(patch.object(at.time,'monotonic',side_effect=lambda:clock[0]))
        if refresh_ask is not None:
            stack.enter_context(patch.object(at.time,'time',side_effect=lambda:wall[0]))
            at.okx.get_order_book.side_effect = [
                dict(ts=wall[0]*1000, bids=[[99.99,1000]], asks=[[100.01,1000]]),
                dict(ts=(wall[0]+4)*1000, bids=[[refresh_ask-.02,1000]], asks=[[refresh_ask,1000]]),
            ]
        create=stack.enter_context(patch.object(at,'at_create_entry_intent',return_value=91))
        mark=stack.enter_context(patch.object(at,'at_mark_entry_submitting',return_value=True))
        close=stack.enter_context(patch.object(at,'at_close_position'))
        order=stack.enter_context(patch.object(at.okx,'place_market_entry',return_value=(False,'rejected')))
        at._open_for_user(dict(user_id=7,size_mode='percent',size_value=3),sig,'TEST-X','TEST')
    return shared,create,mark,close,order


@pytest.mark.parametrize('failure',['reserve','begin'])
def test_shared_network_failure_closes_only_unsent_intent_and_never_places_order(failure):
    shared,create,mark,close,order=entry_harness(reserve_error=failure=='reserve',begin_error=failure=='begin')
    create.assert_called_once();mark.assert_not_called();order.assert_not_called()
    close.assert_called_once_with(91,'ENTRY_SHARED_REFUSED')
    if failure=='reserve':shared.begin.assert_not_called()


def test_unready_coordinator_refuses_before_creating_entry():
    _,create,mark,close,order=entry_harness(prepare=False)
    create.assert_not_called();mark.assert_not_called();close.assert_not_called();order.assert_not_called()


def test_snapshot_that_expires_during_shared_claim_is_not_submitted():
    _,create,mark,close,order=entry_harness(expire=True)
    create.assert_called_once();mark.assert_not_called();order.assert_not_called()
    close.assert_called_once_with(91,'ENTRY_SHARED_EXPIRED')


@pytest.mark.parametrize('ask', [100.99, 102.01])
def test_stale_book_is_refreshed_and_reserved_quantity_or_bracket_can_refuse_entry(ask):
    _,create,mark,close,order=entry_harness(refresh_ask=ask)
    create.assert_called_once();mark.assert_not_called();order.assert_not_called()
    close.assert_called_once_with(91,'ENTRY_BOOK_CHANGED')


def test_stale_book_refresh_can_pass_with_the_same_reserved_quantity():
    _,create,mark,_,order=entry_harness(refresh_ask=100.0)
    mark.assert_called_once()
    assert order.call_args.args[3] == create.call_args.args[4] == .383


def test_shared_entry_pause_does_not_stop_management_of_existing_position():
    shared=SimpleNamespace(maintenance=Mock(),prepare=Mock(return_value=False))
    pos=dict(id=1,user_id=7,inst_id='TEST-X',signal_id=2,sl_algo_id='',sz=.01)
    with (patch.object(at,'_shared_capacity',shared),patch.object(at,'AUTOTRADE_ENABLED',True),
          patch.object(at,'at_all_open_positions',return_value=[pos]),
          patch.object(at,'at_get',return_value={'user_id':7}),
          patch.object(at,'_creds_of',return_value={'test':True}),patch.object(at,'_dm'),
          patch.object(at.okx,'get_position_size',return_value=(True,.01)),
          patch.object(at.okx,'close_position_market',return_value=(True,None)) as close,
          patch.object(at,'at_close_position') as record):
        at.poll_exchange_closes()
    shared.maintenance.assert_not_called()  # independent of the TP/SL path
    close.assert_called_once();record.assert_called_once_with(1,'UNPROTECTED_CLOSED')
