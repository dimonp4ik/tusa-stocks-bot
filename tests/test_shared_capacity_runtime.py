import time
from unittest.mock import patch
import pytest
from src import db,okx_trader as okx
from src.shared_capacity_store import CapacityStore,CapacityError
from src.shared_capacity_transport import LocalCapacityTransport,CapacityUnavailable
from src.shared_capacity_runtime import SharedCapacityRuntime


@pytest.fixture
def runtime(tmp_path,monkeypatch):
    monkeypatch.setattr(db,'DB_PATH',str(tmp_path/'intents.sqlite'))
    db.init_db();db.at_add_allowed(7,1)
    clock=[time.time()]
    store=CapacityStore(tmp_path/'shared.sqlite',clock=lambda:clock[0],startup_grace=0)
    transport=LocalCapacityTransport(store,'x'*48)
    store.sync('crypto',db.at_shared_instance_id(),[])
    store.sync('stocks','stocks-instance',[])
    runtime=SharedCapacityRuntime('crypto',transport=transport,clock=lambda:clock[0])
    runtime.test_clock=clock
    assert runtime.prepare(lambda _:None)
    return runtime


def positions():
    return dict(count=0,margin_usd=0.,inst_ids=[],margin_by_inst={})


def account():
    return okx.account_key({'account_uid':'123'})


def intent():
    return db.at_create_entry_intent(1,7,'TEST-X','LONG',1.,100.,3.,99.,'entry-1',account_key=account())


def begin(runtime,pos_id):
    cid=runtime.reserve(pos_id,account(),'TEST-X','entry-1',3.,positions(),100.,runtime.clock())
    runtime.begin(cid,positions(),100.,runtime.clock())
    return cid['claim_id']


def test_lost_begin_response_leaves_claim_until_closed_unsent_reconciliation(runtime):
    pos_id=intent()
    real=runtime.transport.call
    def lost(action,payload):
        result=real(action,payload)
        if action=='begin':
            raise CapacityUnavailable('response lost')
        return result
    runtime.transport.call=lost
    with pytest.raises(CapacityUnavailable):
        begin(runtime,pos_id)
    assert runtime.transport.store.status()['active_claims']==1
    assert db.at_shared_positions()[0]['entry_submit_started_at']==0
    db.at_close_position(pos_id,'ENTRY_SHARED_REFUSED')
    runtime.transport.call=real
    runtime.maintenance(lambda _:None)
    assert runtime.transport.store.status()['active_claims']==0
    assert db.at_shared_positions()==[]


def test_submitted_closed_claim_is_retained_until_terminal_order_and_two_flat_reads(runtime):
    pos_id=intent();begin(runtime,pos_id)
    assert db.at_mark_entry_submitting(pos_id)
    db.at_close_position(pos_id,'EXCHANGE_CLOSED')
    runtime.test_clock[0]+=91
    with (patch.object(okx,'get_account_configuration',return_value=(True,{'account_uid':'123'})),
          patch.object(okx,'get_order_by_client_id',return_value=(True,{'state':'filled'})),
          patch.object(okx,'get_open_position_margin',return_value=(True,positions())),
          patch.object(okx,'get_position_size',side_effect=[(True,0),(True,1)])):
        runtime.maintenance(lambda _:{'test':True})
    assert runtime.transport.store.status()['active_claims']==1
    with (patch.object(okx,'get_account_configuration',return_value=(True,{'account_uid':'123'})),
          patch.object(okx,'get_order_by_client_id',return_value=(True,{'state':'filled'})),
          patch.object(okx,'get_open_position_margin',return_value=(True,positions())),
          patch.object(okx,'get_position_size',return_value=(True,0)) as sizes):
        runtime.maintenance(lambda _:{'test':True})
    assert sizes.call_count==2
    assert runtime.transport.store.status()['active_claims']==0
    assert db.at_shared_positions()==[]


def test_closed_submitted_claim_is_seeded_after_coordinator_database_replacement(runtime,tmp_path):
    pos_id=intent();begin(runtime,pos_id)
    db.at_mark_entry_submitting(pos_id);db.at_close_position(pos_id,'EXCHANGE_CLOSED')
    old_epoch=runtime.epoch
    replacement=CapacityStore(tmp_path/'replacement.sqlite',clock=runtime.clock,startup_grace=0)
    runtime.transport=LocalCapacityTransport(replacement,'x'*48)
    runtime.maintenance(lambda _:None)
    assert runtime.epoch!=old_epoch
    assert replacement.status()['active_claims']==1
    assert not runtime.ready  # the second service has not synchronized
    assert db.at_shared_positions()[0]['shared_epoch']==runtime.epoch


def test_lost_release_response_is_recovered_by_closed_sync_without_resurrection(runtime):
    pos_id=intent();begin(runtime,pos_id)
    db.at_close_position(pos_id,'ENTRY_UNSENT')
    real=runtime.transport.call
    def lost(action,payload):
        result=real(action,payload)
        if action=='release':raise CapacityUnavailable('response lost')
        return result
    runtime.transport.call=lost
    runtime.maintenance(lambda _:None)
    assert runtime.transport.store.status()['active_claims']==0
    assert len(db.at_shared_positions())==1
    runtime.transport.call=real
    runtime.maintenance(lambda _:None)
    assert db.at_shared_positions()==[]


def test_malformed_successful_reservation_cannot_authorize_order_submission(runtime):
    pos_id=intent();real=runtime.transport.call
    runtime.transport.call=lambda action,payload:dict(allowed=True) if action=='reserve' else real(action,payload)
    with pytest.raises(CapacityError):
        begin(runtime,pos_id)
    assert db.at_shared_positions()[0]['shared_claim_id'] is None


def test_local_claim_cannot_be_rebound_to_another_account(runtime):
    pos_id=intent();cid=begin(runtime,pos_id)
    assert not db.at_set_shared_claim(pos_id,cid,runtime.epoch,'0'*64)
    assert db.at_shared_positions()[0]['account_key']==account()


def test_wrong_but_well_formed_claim_id_is_rejected_at_begin(runtime):
    store=runtime.transport.store
    other=store.reserve('crypto',runtime.epoch,dict(account='0'*64,instrument='OTHER-X',intent='other',margin=3.),
        dict(positions(),equity=100.,captured_at=runtime.clock()))
    pos_id=intent();real=runtime.transport.call
    def wrong_id(action,payload):
        answer=real(action,payload)
        return answer|{'claim_id':other['claim_id']} if action=='reserve' else answer
    runtime.transport.call=wrong_id
    with pytest.raises(CapacityError,match='identity'):
        begin(runtime,pos_id)
    assert store.status()['active_claims']==2


def test_invalid_enabled_configuration_never_falls_back_to_independent_entries(monkeypatch):
    monkeypatch.setenv('SHARED_CAPACITY_MODE','invalid')
    runtime=SharedCapacityRuntime('crypto')
    runtime.initialize()
    assert runtime.enabled
    assert not runtime.prepare(lambda _:None)
    assert 'не готова' in runtime.status_text()


def test_definitive_balance_rejection_does_not_leave_a_permanent_claim(runtime):
    pos_id=intent();begin(runtime,pos_id);db.at_mark_entry_submitting(pos_id)
    db.at_close_position(pos_id,'ENTRY_REJECTED',error='Order failed (sCode 51008)')
    runtime.test_clock[0]+=91
    with (patch.object(okx,'get_account_configuration',return_value=(True,{'account_uid':'123'})),
          patch.object(okx,'get_order_by_client_id') as order,
          patch.object(okx,'get_open_position_margin',return_value=(True,positions())),
          patch.object(okx,'get_position_size',return_value=(True,0))):
        runtime.maintenance(lambda _:{'test':True})
    order.assert_not_called()
    assert runtime.transport.store.status()['active_claims']==0
    assert db.at_shared_positions()==[]


def test_nested_okx_balance_error_preserves_rejection_code():
    reply=dict(code='1',msg='All operations failed',data=[dict(sCode='51008',sMsg='Not enough margin',ordId='')])
    with patch.object(okx.requests,'post',return_value=type('Reply',(),{'json':lambda _:reply})()):
        allowed,error=okx.place_market_entry(dict(api_key='test',api_secret='test',passphrase='test'),
            'TEST-X','LONG',1,'test-order')
    assert not allowed and error.endswith('(sCode 51008)')
