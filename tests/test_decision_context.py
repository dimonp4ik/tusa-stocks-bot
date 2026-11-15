import json
from src import db
from src.decision_context import context_json


def test_evidence_is_bounded_to_known_scalars_and_never_copies_payloads():
    result=json.loads(context_json(dict(cross_z=2.4,model_quality=.33,
        api_key='PRIVATE',reason='PRIVATE',module='native_xperp_quality_reversal',
        btc_regime={'secret':'PRIVATE'},entry=float('nan'),signal_return=True)))
    assert result['decision_evidence']==dict(cross_z=2.4,model_quality=.33,
        module='native_xperp_quality_reversal')
    assert 'PRIVATE' not in json.dumps(result)
    assert context_json(None) is None


def test_first_decision_is_immutable_when_same_bar_is_reconsidered(tmp_path,monkeypatch):
    monkeypatch.setattr(db,'DB_PATH',str(tmp_path/'audit.sqlite'))
    db.init_db()
    analysis=dict(symbol='TEST',direction='LONG',current_price=100,source='audit',
        signal_bar_ts=900,_decision_context=dict(cross_z=-2.4,entry=100))
    setup=db.log_setup_candidate_once(analysis)
    analysis['_decision_context']['entry']=120
    assert db.log_setup_candidate_once(analysis) is None
    db.init_db()  # migration is idempotent and preserves the first evidence
    with db._conn() as c:
        row=c.execute('SELECT decision_context_json FROM setup_log WHERE id=?',(setup,)).fetchone()
    assert json.loads(row[0])['decision_evidence']['entry']==100


def test_legacy_setup_stays_explicitly_without_evidence(tmp_path,monkeypatch):
    monkeypatch.setattr(db,'DB_PATH',str(tmp_path/'audit.sqlite'));db.init_db()
    setup=db.log_setup_candidate(dict(symbol='TEST',direction='LONG',current_price=100))
    with db._conn() as c:
        assert c.execute('SELECT decision_context_json FROM setup_log WHERE id=?',(setup,)).fetchone()[0] is None
