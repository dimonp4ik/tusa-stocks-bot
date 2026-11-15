from flask import Flask
import pytest
from src import db
from src.trade_audit_api import register_trade_audit_api

TOKEN='x'*48


@pytest.fixture
def api(tmp_path,monkeypatch):
    monkeypatch.setattr(db,'DB_PATH',str(tmp_path/'signals.sqlite'))
    db.init_db();db.at_add_allowed(7,1)
    db.at_set_keys(7,'NEVER_EXPORT_THIS_KEY','NEVER_EXPORT_THIS_SECRET','NEVER_EXPORT_THIS_PASSWORD')
    signal=db.log_signal(dict(symbol='TEST',direction='LONG',current_price=100,
        reason='NEVER_EXPORT_THIS_TEXT',source='stock_xperp_native_shadow'),101,102,99)
    setup=db.log_setup_candidate(dict(symbol='TEST',direction='LONG',current_price=100,
        reason='NEVER_EXPORT_THIS_TEXT',source='stock_xperp_native_shadow'))
    db.link_setup_to_signal(setup,signal)
    db.record_setup_decision_book(setup,spread_bps=2.,depth_usd=10000.,imbalance=-.4)
    app=Flask(__name__);register_trade_audit_api(app,TOKEN)
    return app,signal


def test_bounded_trade_export_contains_levels_not_credentials_or_narrative(api):
    app,signal=api
    with app.test_client() as c:
        assert c.post('/internal/trade-audit/v1/signals',json={'ids':[signal]}).status_code==401
        response=c.post('/internal/trade-audit/v1/signals',json={'ids':[signal]},
            headers={'Authorization':'Bearer '+TOKEN})
    assert response.status_code==200
    rows=response.get_json()['signals']
    assert len(rows)==1 and rows[0]['entry_price']==100 and rows[0]['sl']==99
    book=response.get_json()['setups'][0]
    assert book['book_imbalance']==-.4 and book['book_sample_kind']=='decision_snapshot'
    assert 'NEVER_EXPORT' not in response.get_data(as_text=True)


@pytest.mark.parametrize('body',[{'ids':['1 OR 1=1']},{'ids':[True]},{'ids':[]},
    {'limit':1000},{'table':'autotrade_users'},{'query':'SELECT * FROM autotrade_users'}])
def test_trade_audit_rejects_arbitrary_queries_and_unbounded_requests(api,body):
    app,_=api
    with app.test_client() as c:
        assert c.post('/internal/trade-audit/v1/signals',json=body,
            headers={'Authorization':'Bearer '+TOKEN}).status_code==400


def test_audit_endpoint_is_absent_without_private_token():
    app=Flask(__name__);register_trade_audit_api(app,None)
    assert app.test_client().post('/internal/trade-audit/v1/signals',json={}).status_code==404
