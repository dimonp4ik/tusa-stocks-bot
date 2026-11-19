from contextlib import ExitStack
from unittest.mock import patch
import pytest
from src import autotrader as at


def run_flat(position,*,matched=100+1/3,lookup_error=False):
    messages=[]
    with ExitStack() as stack:
        stack.enter_context(patch.object(at,'AUTOTRADE_ENABLED',True))
        for name,value in [('at_all_open_positions',[position]),('at_get',{'user_id':7}),('_creds_of',{'test':True})]:
            stack.enter_context(patch.object(at,name,return_value=value))
        stack.enter_context(patch.object(at.time,'time',return_value=2000.))
        stack.enter_context(patch.object(at,'_dm',side_effect=lambda uid,text:messages.append(text)))
        record=stack.enter_context(patch.object(at,'at_close_position'))
        stack.enter_context(patch.object(at.okx,'get_position_size',return_value=(True,0)))
        stack.enter_context(patch.object(at.okx,'cancel_protection',return_value=(True,None)))
        stack.enter_context(patch.object(at.okx,'get_xperp_spec',return_value={'tickSz':.01}))
        stack.enter_context(patch.object(at.okx,'get_order_by_client_id',return_value=(True,
            dict(instId='TEST',clOrdId='entry1',side='buy',state='filled',accFillSz='3',avgPx='100'))))
        history=stack.enter_context(patch.object(at.okx,'get_protection_order_ids',
            side_effect=RuntimeError('temporary history failure') if lookup_error else lambda creds,inst,algo,**kw:('11',) if algo=='tp1' else ('22',)))
        fills=stack.enter_context(patch.object(at.okx,'get_last_fill_px',
            side_effect=lambda *args,**kwargs:matched if kwargs.get('order_ids') else 100.))
        at.poll_exchange_closes()
    return messages,record,history,fills


def position(**overrides):
    p=dict(id=1,user_id=7,inst_id='TEST',signal_id=2,status='OPEN',opened_at=1.,direction='LONG',
        sl_algo_id='stop',tp1_algo_id='tp1',entry_px=100.,entry_filled_sz=3.,sz=2.,client_order_id='entry1')
    p.update(overrides);return p


def test_flat_notification_includes_tp1_in_volume_weighted_whole_position_result():
    messages,record,_,fills=run_flat(position())
    assert len(messages)==1
    assert '100.33' in messages[0] and '+0.333%' in messages[0]
    assert 'всех закрытий' in messages[0]
    record.assert_called_once_with(1,'EXCHANGE_FLAT')
    assert any(c.kwargs.get('expected_size')==3 for c in fills.call_args_list)


def test_history_failure_does_not_suppress_confirmed_flat_notification_or_invent_profit():
    messages,record,_,_=run_flat(position(),lookup_error=True)
    record.assert_called_once_with(1,'EXCHANGE_FLAT')
    assert len(messages)==1 and 'позиция закрыта' in messages[0]
    assert 'не подтверждён' in messages[0]
    assert '+0.000%' not in messages[0]


def test_missing_original_fill_size_never_uses_reduced_remainder_as_total():
    messages,_,history,fills=run_flat(position(entry_filled_sz=None,client_order_id=None))
    history.assert_not_called();fills.assert_not_called()
    assert 'не подтверждён' in messages[0]
