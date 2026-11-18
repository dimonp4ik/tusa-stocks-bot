import json
import sqlite3
import pytest
from flask import Flask
from src.structure_observation_api import register_structure_observation_api
from src.structure_observation_store import ObservationStore
from src.stock_structure_observer import PROFILE

TOKEN='q'*48


@pytest.fixture
def api(tmp_path):
    store=ObservationStore(tmp_path/'observer.db');store.initialize()
    f=dict(time=[900,1800],open=[100.,100.],high=[101.,101.],low=[99.,99.],close=[100.,100.],volume=[1.,1.])
    ref=store.put_frame('TEST-X',f)
    store.claim(PROFILE,2700,2720.,'test')
    store.put_observation(PROFILE,2700,'TEST','TEST-X',2721.,True,False,dict(frame=ref,book=dict(bid=99.,ask=101.)))
    store.finish(PROFILE,2700,2722.,'complete',dict(expected=1,quotes=1))
    # A different profile and its data must not appear in the fixed export.
    store.claim('other_profile',3600,3620.,'OTHER_DATA')
    app=Flask(__name__);register_structure_observation_api(app,TOKEN,lambda:store)
    return app,store


def post(app,body,token=TOKEN):
    return app.test_client().post('/internal/structure-observer/v1/export',data=json.dumps(body),content_type='application/json',
        headers={'Authorization':'Bearer '+token})


def test_export_requires_auth_and_returns_verified_closed_frames_from_only_its_archive(api):
    app,store=api
    assert post(app,{},'bad').status_code==401
    response=post(app,{})
    assert response.status_code==200
    payload=response.get_json()['export']
    assert len(payload['scans'])==1 and payload['next_bucket']==2700
    row=payload['scans'][0]['observations'][0]
    assert row['closed_frame']['time']==[900,1800] and row['payload']['book']['ask']==101.
    assert 'OTHER_DATA' not in response.get_data(as_text=True)
    assert post(app,{'after_bucket':2700}).get_json()['export']['scans']==[]
    with sqlite3.connect(store.path) as c:
        assert c.execute('SELECT COUNT(*) FROM scans').fetchone()[0]==2


@pytest.mark.parametrize('body',[{'after_bucket':True},{'after_bucket':1},{'limit':5},
    {'limit':False},{'query':'SELECT * FROM autotrade_users'},{'path':'/data/bot.db'},[],None])
def test_export_rejects_arbitrary_queries_paths_and_unbounded_requests(api,body):
    app,_=api
    assert post(app,body).status_code==400


def test_export_does_not_invent_reconstructed_candles_if_archived_history_is_missing(api):
    app,store=api
    with sqlite3.connect(store.path) as c:c.execute('DELETE FROM candles')
    row=post(app,{}).get_json()['export']['scans'][0]['observations'][0]
    assert row['frame_error']=='archived_history_unavailable' and 'closed_frame' not in row


def test_disabled_endpoint_and_missing_archive_are_not_created_by_a_read(tmp_path):
    app=Flask(__name__);register_structure_observation_api(app,None,lambda:None)
    assert post(app,{}).status_code==404
    missing=tmp_path/'missing.db';store=ObservationStore(missing)
    assert store.export(PROFILE)==dict(scans=[],next_bucket=0)
    assert not missing.exists()


def test_context_export_is_authenticated_keeps_series_distinct_and_cannot_select_other_profiles(api):
    from src.stock_mark_context import PROFILE as CONTEXT_PROFILE
    _,store=api
    original=store.export(PROFILE)['scans'][0]['observations'][0]
    mark=dict(time=[900,1800],open=[200.,200.],high=[201.,201.],low=[199.,199.],close=[200.,200.])
    reference=store.put_frame('TEST-X',mark,kind='mark')
    store.claim(CONTEXT_PROFILE,2700,2720.,'mark')
    store.put_observation(CONTEXT_PROFILE,2700,'TEST','TEST-X',2721.,True,False,
        dict(frame=reference,trade_frame=original['payload']['frame'],series_kind='mark'))
    store.finish(CONTEXT_PROFILE,2700,2722.,'complete',{})
    app=Flask('context_export');register_structure_observation_api(app,TOKEN,lambda:store,lambda:store)
    client=app.test_client();url='/internal/structure-observer/v1/context-export'
    assert client.post(url,json={}).status_code==401
    response=client.post(url,json={},headers={'Authorization':'Bearer '+TOKEN})
    scan=response.get_json()['export']['scans'][0]
    assert scan['profile']==CONTEXT_PROFILE
    assert scan['observations'][0]['closed_frame']['close']==[200.,200.]
    assert scan['observations'][0]['trade_closed_frame']['close']==[100.,100.]
    assert store.export(PROFILE)['scans'][0]['observations'][0]['closed_frame']['close']==[100.,100.]
    assert client.post(url,json={'profile':PROFILE},headers={'Authorization':'Bearer '+TOKEN}).status_code==400
