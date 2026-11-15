from types import SimpleNamespace
import requests
import pytest
from flask import Flask
from src.shared_capacity_transport import (LocalCapacityTransport,RemoteCapacityTransport,
    CapacityUnavailable,CapacityError,register_capacity_api,from_environment,PREFIX,MAX_BODY)
import test_shared_capacity_store as fixtures
claim=fixtures.claim
view=fixtures.view
store=fixtures.store

TOKEN='x'*48


@pytest.fixture
def app(store):
    app=Flask(__name__)
    register_capacity_api(app,LocalCapacityTransport(store,TOKEN))
    return app


def test_capacity_api_requires_auth_and_does_not_disclose_rejected_payload(app):
    with app.test_client() as c:
        assert c.post(PREFIX+'/status',json={}).status_code==401
        assert c.get(PREFIX+'/status').status_code==405
        response=c.post(PREFIX+'/reserve',json={'owner':'not-valid','account':'private-test-value'},
            headers={'Authorization':'Bearer '+TOKEN})
        assert response.status_code==400
        assert 'private-test-value' not in response.get_data(as_text=True)
        assert TOKEN not in response.get_data(as_text=True)


def test_capacity_api_bounds_request_body(app):
    with app.test_client() as c:
        response=c.post(PREFIX+'/status',data=b'x'*(MAX_BODY+1),
            headers={'Authorization':'Bearer '+TOKEN},content_type='application/json')
        assert response.status_code==413


def test_remote_request_is_https_no_redirect_and_round_trips_to_local_store(app,store):
    calls=[]
    class Session:
        def post(self,url,**kwargs):
            calls.append((url,kwargs))
            with app.test_client() as c:
                response=c.post(url.split('https://coordinator.test',1)[1],
                    headers=kwargs['headers'],json=kwargs['json'])
                return SimpleNamespace(status_code=response.status_code,json=response.get_json)
    remote=RemoteCapacityTransport('https://coordinator.test',TOKEN,session=Session())
    result=remote.call('reserve',dict(owner='stocks',epoch=store.status()['epoch'],
        claim=claim('first'),snapshot=view(store)))
    assert result['allowed']
    again=remote.call('reserve',dict(owner='stocks',epoch=store.status()['epoch'],
        claim=claim('first'),snapshot=view(store)))
    assert again['claim_id']==result['claim_id']
    assert calls[0][1]['allow_redirects'] is False
    assert calls[0][1]['timeout']==(2,3)
    assert TOKEN not in calls[0][0]


@pytest.mark.parametrize('url',['http://coordinator.test','https://user:secret@coordinator.test',
    'https://coordinator.test?token=secret','https://coordinator.test/path','https://coordinator.test:1234'])
def test_remote_refuses_endpoint_that_could_expose_auth(url):
    with pytest.raises(CapacityError,match='HTTPS origin'):
        RemoteCapacityTransport(url,TOKEN)


def test_remote_timeout_and_redirect_fail_closed_without_secret_in_error():
    class Session:
        def post(self,*a,**k):
            raise requests.Timeout('sensitive request '+TOKEN)
    remote=RemoteCapacityTransport('https://coordinator.test',TOKEN,session=Session())
    with pytest.raises(CapacityUnavailable) as err:
        remote.call('status',{})
    assert TOKEN not in str(err.value)
    remote.session=SimpleNamespace(post=lambda *a,**k:SimpleNamespace(status_code=302))
    with pytest.raises(CapacityUnavailable):
        remote.call('status',{})


def test_shared_mode_cannot_silently_activate_without_persistent_local_database(tmp_path):
    assert from_environment('crypto',{}) is None
    env=dict(SHARED_CAPACITY_MODE='local',SHARED_CAPACITY_TOKEN=TOKEN)
    with pytest.raises(CapacityError,match='persistent'):
        from_environment('crypto',env)
    env.update(RAILWAY_VOLUME_MOUNT_PATH=str(tmp_path),DB_PATH=str(tmp_path/'signals.db'))
    local=from_environment('crypto',env)
    assert isinstance(local,LocalCapacityTransport)
    assert not local.call('status',{})['ready']
    env['SHARED_CAPACITY_DB_PATH']=str(tmp_path.parent/'outside.sqlite')
    with pytest.raises(CapacityError,match='mounted volume'):
        from_environment('crypto',env)
    env['SHARED_CAPACITY_MODE']='remote'
    with pytest.raises(CapacityError,match='role'):
        from_environment('crypto',env)
