"""Authenticated coordinator transport; never forwards OKX credentials."""
import hmac
import os
from pathlib import Path
import re
from urllib.parse import urlsplit

import requests

from src.shared_capacity_store import CapacityStore,CapacityError,owner_id

PREFIX='/internal/capacity/v1'
MAX_BODY=512*1024


class CapacityUnavailable(RuntimeError):
    pass


def valid_token(token):
    if not isinstance(token,str) or re.fullmatch(r'[A-Za-z0-9_-]{32,256}',token) is None:
        raise CapacityError('shared capacity token must be a random 32+ character value')
    return token


def dispatch(store,action,payload):
    if not isinstance(payload,dict):
        raise CapacityError('invalid request')
    if action=='sync':
        return store.sync(payload.get('owner'),payload.get('instance'),payload.get('positions'))
    if action=='reserve':
        return store.reserve(payload.get('owner'),payload.get('epoch'),payload.get('claim'),payload.get('snapshot'))
    if action=='begin':
        return store.begin(payload.get('owner'),payload.get('epoch'),payload.get('claim_id'),payload.get('snapshot'),
            identity=payload.get('identity'))
    if action=='release':
        return store.release(payload.get('owner'),payload.get('epoch'),payload.get('claim_id'),payload.get('evidence'),
            identity=payload.get('identity'))
    if action=='status':
        return store.status()
    raise CapacityError('unknown operation')


class LocalCapacityTransport:
    def __init__(self,store,token):
        self.store=store
        self._token=valid_token(token)

    def call(self,action,payload):
        # The crypto worker must not make an HTTP request to its own single
        # synchronous gunicorn worker while handling a webhook.
        try:
            return dispatch(self.store,action,payload)
        except CapacityError:
            raise
        except Exception as exc:
            raise CapacityUnavailable('shared capacity storage unavailable') from exc


class RemoteCapacityTransport:
    def __init__(self,url,token,*,session=None):
        try:
            parsed=urlsplit(url)
            if (parsed.scheme!='https' or not parsed.hostname or parsed.username or parsed.password
                    or parsed.query or parsed.fragment or parsed.path not in ('','/')
                    or parsed.port not in (None,443)):
                raise ValueError('invalid endpoint')
        except (TypeError,ValueError) as exc:
            raise CapacityError('shared coordinator requires an HTTPS origin') from exc
        self.url=url.rstrip('/')
        self._token=valid_token(token)
        self.session=session or requests.Session()

    def call(self,action,payload):
        if action not in {'status','sync','reserve','begin','release'}:
            raise CapacityError('unknown operation')
        try:
            response=self.session.post(self.url+PREFIX+'/'+action,
                headers={'Authorization':'Bearer '+self._token},json=payload,
                timeout=(2,3),allow_redirects=False)
            if response.status_code!=200:
                raise CapacityUnavailable('shared coordinator unavailable or refused request')
            body=response.json()
            if not isinstance(body,dict) or body.get('ok') is not True or not isinstance(body.get('result'),dict):
                raise CapacityUnavailable('invalid shared coordinator response')
            return body['result']
        except CapacityUnavailable:
            raise
        except Exception as exc:
            # Exception strings can contain URLs; do not forward response text
            # or credential-bearing request objects to Telegram or logs.
            raise CapacityUnavailable('shared coordinator unavailable') from exc


def register_capacity_api(app,transport):
    if not isinstance(transport,LocalCapacityTransport):
        return
    from flask import Blueprint,request,jsonify
    api=Blueprint('shared_capacity',__name__)

    @api.route('/<action>',methods=['POST'])
    def invoke(action):
        expected='Bearer '+transport._token
        auth=request.headers.get('Authorization','')
        if not isinstance(auth,str) or not auth.isascii() or not hmac.compare_digest(auth,expected):
            return jsonify(ok=False,error='unauthorized'),401
        if request.content_length is None:
            return jsonify(ok=False,error='content_length_required'),411
        if request.content_length>MAX_BODY:
            return jsonify(ok=False,error='request_too_large'),413
        try:
            return jsonify(ok=True,result=transport.call(action,request.get_json(silent=True)))
        except CapacityError:
            return jsonify(ok=False,error='invalid_capacity_request'),400
        except Exception:
            return jsonify(ok=False,error='capacity_store_unavailable'),503

    app.register_blueprint(api,url_prefix=PREFIX)


def from_environment(owner,environ=None):
    """Configuration failure raises: callers must refuse new entries, not
    silently return to independent account checks once shared mode is enabled.
    """
    owner_id(owner)
    env=os.environ if environ is None else environ
    mode=env.get('SHARED_CAPACITY_MODE','off').strip().lower()
    if mode=='off':
        return None
    if (owner,mode) not in {('crypto','local'),('stocks','remote')}:
        raise CapacityError('invalid shared capacity role')
    token=valid_token(env.get('SHARED_CAPACITY_TOKEN'))
    raw_volume=env.get('RAILWAY_VOLUME_MOUNT_PATH','')
    raw_database=env.get('DB_PATH','')
    if not raw_volume or not raw_database:
        raise CapacityError('shared mode requires persistent local intent storage')
    volume=Path(raw_volume)
    database=Path(raw_database)
    if (not volume.is_absolute() or not volume.is_dir() or not database.is_absolute()
            or not database.resolve().is_relative_to(volume.resolve()) or database.resolve()==volume.resolve()):
        raise CapacityError('local intent database must be inside the mounted volume')
    if mode=='remote':
        return RemoteCapacityTransport(env.get('SHARED_CAPACITY_URL',''),token)
    raw_shared=env.get('SHARED_CAPACITY_DB_PATH',str(volume/'shared_capacity.sqlite3'))
    shared=Path(raw_shared)
    if (not shared.is_absolute() or not shared.resolve().is_relative_to(volume.resolve())
            or shared.resolve() in {volume.resolve(),database.resolve()} or not shared.parent.is_dir()):
        raise CapacityError('shared database must be a separate file inside the mounted volume')
    return LocalCapacityTransport(CapacityStore(shared,
        max_positions=int(env.get('AUTOTRADE_MAX_ACCOUNT_POSITIONS','12')),
        max_margin=float(env.get('AUTOTRADE_MAX_ACCOUNT_MARGIN_FRACTION','.90'))),token)
