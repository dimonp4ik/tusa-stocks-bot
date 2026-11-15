from concurrent.futures import ThreadPoolExecutor
import hashlib
import sqlite3
import threading

import pytest
from src.shared_capacity_store import CapacityStore,CapacityError,snapshot

ACCOUNT=hashlib.sha256(b'test').hexdigest()


@pytest.fixture
def store(tmp_path):
    clock=[1000.]
    obj=CapacityStore(tmp_path/'capacity.sqlite',clock=lambda:clock[0],startup_grace=0)
    obj.test_clock=clock
    for owner in ('crypto','stocks'):
        obj.sync(owner,owner+'-instance',[])
    return obj


def view(store,n=0,margin=0.,equity=100.,ids=None):
    ids=[f'external-{i}' for i in range(n)] if ids is None else ids
    return dict(captured_at=store.test_clock[0],equity=equity,count=n,
        inst_ids=ids,margin_usd=margin,
        margin_by_inst={s:margin/len(ids) for s in ids} if ids else {})


def claim(intent,instrument=None,margin=3.):
    return dict(account=ACCOUNT,instrument=instrument or intent,intent=intent,margin=margin)


def reserve(store,owner,intent,*,n=0,margin=0.,requested=3.,instrument=None):
    return store.reserve(owner,store.status()['epoch'],claim(intent,instrument,requested),view(store,n,margin))


def test_two_services_cannot_repeat_stale_snapshots_into_thirteen_positions(store):
    # Nine manual/other positions are known; peer fills can remain invisible
    # across multiple reads. Writer headroom alone allows both second entries.
    tasks=[('crypto','c1'),('stocks','s1'),('crypto','c2'),('stocks','s2')]
    with ThreadPoolExecutor(max_workers=4) as workers:
        results=list(workers.map(lambda args:reserve(store,*args,n=9,margin=27.),tasks))
    assert sum(r['allowed'] for r in results)==3
    assert store.status()['active_claims']==3


def test_instrument_is_unique_across_bots_and_repeated_intent_is_idempotent(store):
    a=reserve(store,'crypto','first',instrument='BTC-X')
    again=reserve(store,'crypto','first',instrument='BTC-X')
    assert a['allowed'] and again['claim_id']==a['claim_id']
    assert not reserve(store,'stocks','second',instrument='BTC-X')['allowed']
    with pytest.raises(CapacityError,match='payload changed'):
        reserve(store,'crypto','first',instrument='OTHER-X')
    assert store.status()['active_claims']==1


def test_local_quotas_and_global_margin_are_separate(store):
    for i in range(4):
        assert reserve(store,'crypto',f'c{i}')['allowed']
    assert reserve(store,'crypto','overflow')['reason']=='service_position_limit'
    assert reserve(store,'stocks','s1')['allowed']
    # This independent account starts at 82% occupied: an 8% request fills
    # the 90% budget and a concurrent 3% request must be rejected.
    other='0'*64
    first=store.reserve('stocks',store.status()['epoch'],dict(claim('large',margin=8),account=other),
        view(store,1,82.))
    second=store.reserve('crypto',store.status()['epoch'],dict(claim('small'),account=other),view(store,1,82.))
    assert first['allowed']
    assert second['reason']=='margin_limit'


def test_visible_claims_are_not_counted_twice_but_partial_visibility_keeps_full_reserve(store):
    a=reserve(store,'crypto','c1',requested=10.)
    # Same claim now appears at half margin after a partial fill/close.
    # Union count is one; occupied margin still reserves its original $10.
    visible=view(store,1,5.,equity=20.,ids=['c1'])
    b=store.reserve('stocks',store.status()['epoch'],claim('s1',margin=8.),visible)
    assert a['allowed'] and b['allowed']  # 10+8 = 90% of $20.
    denied=store.reserve('stocks',store.status()['epoch'],claim('s2',margin=.01),visible)
    assert denied['reason']=='margin_limit'


def test_submitted_claim_survives_ttl_and_database_restart(store):
    first=reserve(store,'crypto','submitted')
    epoch=store.status()['epoch']
    assert store.begin('crypto',epoch,first['claim_id'],view(store),identity=first['binding'])['allowed']
    store.test_clock[0]+=1000
    restarted=CapacityStore(store.path,clock=store.clock,startup_grace=0)
    assert restarted.status()['epoch']==epoch
    assert restarted.status()['active_claims']==1
    assert not restarted.status()['ready']  # peer heartbeat expired
    for owner in ('crypto','stocks'):
        restarted.sync(owner,owner+'-instance',[])
    assert restarted.status()['active_claims']==1  # absence is not release proof
    assert not restarted.reserve('stocks',epoch,claim('other',instrument='submitted'),view(store))['allowed']


def test_only_unsent_reservations_expire_and_cannot_be_reused(store):
    first=reserve(store,'crypto','expired')
    store.test_clock[0]+=180
    assert store.status()['active_claims']==0
    result=reserve(store,'crypto','expired')
    assert not result['allowed']
    assert result['claim_id']==first['claim_id']


def test_legacy_sync_adopts_positions_but_cannot_seed_an_unsent_order(store):
    legacy=dict(claim('old'),state='OPEN')
    synced=store.sync('crypto','crypto-instance',[legacy])
    assert synced['claims']['old']
    with pytest.raises(CapacityError,match='unsent'):
        store.sync('stocks','stocks-instance',[dict(claim('fake'),state='RESERVED')])
    with pytest.raises(CapacityError,match='another intent'):
        store.sync('stocks','stocks-instance',[dict(claim('duplicate',instrument='old'),state='OPEN')])
    assert store.status()['active_claims']==1


def test_startup_requires_both_peers_and_grace(tmp_path):
    clock=[1000.]
    s=CapacityStore(tmp_path/'new.sqlite',clock=lambda:clock[0])
    s.test_clock=clock
    epoch=s.status()['epoch']
    s.sync('crypto','crypto-instance',[])
    assert not s.reserve('crypto',epoch,claim('a'),view(s))['allowed']
    s.sync('stocks','stocks-instance',[])
    assert not s.reserve('crypto',epoch,claim('a'),view(s))['allowed']
    clock[0]+=180
    assert s.reserve('crypto',epoch,claim('a'),view(s))['allowed']
    clock[0]+=1
    assert not s.reserve('stocks',epoch,claim('b'),view(s))['allowed']


def test_no_second_service_instance_can_replace_fresh_sync(store):
    with pytest.raises(CapacityError,match='instance already active'):
        store.sync('stocks','different-instance',[])


def test_begin_rechecks_snapshot_capacity_after_external_position_arrives(store):
    first=reserve(store,'crypto','a')
    result=store.begin('crypto',store.status()['epoch'],first['claim_id'],view(store,12,36.),identity=first['binding'])
    assert not result['allowed'] and result['reason']=='position_limit'
    visible=view(store,1,3.,ids=['a'])
    assert not store.begin('crypto',store.status()['epoch'],first['claim_id'],visible,identity=first['binding'])['allowed']


def test_release_requires_terminal_order_and_positive_flat_evidence(store):
    a=reserve(store,'crypto','a');epoch=store.status()['epoch']
    assert store.begin('crypto',epoch,a['claim_id'],view(store),identity=a['binding'])['allowed']
    evidence=dict(kind='terminal_flat',local_status='CLOSED',terminal_order_state='filled',
        flat_reads=2,pending_entries_clear=True)
    with pytest.raises(CapacityError,match='flatness'):
        store.release('crypto',epoch,a['claim_id'],evidence,identity=a['binding'])
    store.test_clock[0]+=91
    for changed in ({'flat_reads':1},{'pending_entries_clear':False},{'terminal_order_state':'live'}):
        with pytest.raises(CapacityError,match='flatness'):
            store.release('crypto',epoch,a['claim_id'],evidence|changed,identity=a['binding'])
    assert store.release('crypto',epoch,a['claim_id'],evidence,identity=a['binding'])['released']
    assert store.release('crypto',epoch,a['claim_id'],evidence,identity=a['binding'])['released']
    assert store.status()['active_claims']==0
    assert not reserve(store,'crypto','a')['allowed']  # durable tombstone


def test_closed_unsent_intent_can_release_after_begin_response_was_lost(store):
    a=reserve(store,'crypto','a');epoch=store.status()['epoch']
    store.begin('crypto',epoch,a['claim_id'],view(store),identity=a['binding'])
    assert store.release('crypto',epoch,a['claim_id'],dict(kind='never_submitted',
        local_status='CLOSED',submit_started_at=0),identity=a['binding'])['released']


def test_valid_claim_id_cannot_submit_or_release_a_different_account(store):
    a=reserve(store,'crypto','a');epoch=store.status()['epoch']
    wrong=a['binding']|{'account':'0'*64}
    with pytest.raises(CapacityError,match='identity'):
        store.begin('crypto',epoch,a['claim_id'],view(store),identity=wrong)
    with pytest.raises(CapacityError,match='identity'):
        store.release('crypto',epoch,a['claim_id'],dict(kind='never_submitted',
            local_status='CLOSED',submit_started_at=0),identity=wrong)
    assert store.status()['active_claims']==1


@pytest.mark.parametrize('change',[
    {'count':.5},{'equity':float('nan')},{'margin_usd':float('inf')},
    {'captured_at':900.},{'margin_by_inst':{'missing':1.}},
    {'inst_ids':['a','a']},{'count':1},
])
def test_malformed_and_stale_snapshots_fail_closed(store,change):
    with pytest.raises(CapacityError):
        snapshot(view(store)|change,store.test_clock[0])


def test_snapshot_is_revalidated_after_waiting_for_database_lock(store):
    connection=sqlite3.connect(store.path,isolation_level=None)
    connection.execute('BEGIN IMMEDIATE')
    entered=threading.Event()
    original=store._transaction

    def delayed():
        entered.set()
        return original()

    store._transaction=delayed
    epoch=''
    # Read epoch before taking the held lock, through its existing connection.
    epoch=connection.execute("SELECT value FROM capacity_meta WHERE key='epoch'").fetchone()[0]
    stale=view(store)
    with ThreadPoolExecutor(max_workers=1) as workers:
        request=workers.submit(store.reserve,'crypto',epoch,claim('waiting'),stale)
        assert entered.wait(2)
        store.test_clock[0]+=16
        connection.rollback();connection.close()
        with pytest.raises(CapacityError,match='stale snapshot'):
            request.result(timeout=3)
