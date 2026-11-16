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


def test_forming_observation_preserves_zero_volume_and_its_actual_time():
    from src.decision_context import forming_entry_evidence
    sample=forming_entry_evidence(dict(time=[900],confirmed=[0],high=[101],low=[100],
        close=[100.5],volume=[0]),911.25)
    saved=json.loads(context_json(sample))['decision_evidence']
    assert saved['reference_observed_at']==911.25 and saved['reference_volume']==0
    assert saved['reference_bar_ts']==900 and saved['reference_sample_kind']=='forming_15m_snapshot'


def test_finalized_or_out_of_time_data_is_not_called_a_forming_observation():
    from src.decision_context import forming_entry_evidence
    sample=dict(time=[900],confirmed=[1],high=[101],low=[100],close=[100.5])
    assert forming_entry_evidence(sample,911)=={}
    sample['confirmed']=[0]
    assert forming_entry_evidence(sample,899)=={}
    assert forming_entry_evidence(sample,1800)=={}
