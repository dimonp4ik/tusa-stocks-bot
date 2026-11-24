import time
from unittest.mock import patch

import pytest
from src import db, entry_execution


@pytest.fixture
def journal(tmp_path, monkeypatch):
    monkeypatch.setattr(db, 'DB_PATH', str(tmp_path/'signals.sqlite3'))
    db.init_db()
    db.at_add_allowed(7, 1)
    signal = db.log_signal(dict(symbol='TEST', direction='LONG', current_price=100.), 101., 102., 99.)
    intent = db.at_create_entry_intent(signal, 7, 'TEST-X', 'LONG', 2., 100., 4., 99., 'entry1')
    return signal, intent


@pytest.mark.parametrize('status,eligible', [('TP1_PARTIAL', 1), ('SL_HIT', 1), ('OPEN', 0)])
def test_signal_must_still_be_live_and_open_when_the_intent_is_claimed(journal,status,eligible):
    signal, intent = journal
    with db._conn() as conn:
        conn.execute('UPDATE signals SET status=?,autotrade_eligible=? WHERE id=?', (status,eligible,signal))
    assert not db.at_mark_entry_submitting(intent, expected_signal_id=signal)
    assert db.at_release_unsent_entry(intent)


def test_missing_or_wrong_signal_cannot_authorize_market_submission(journal):
    signal, intent = journal
    assert not db.at_mark_entry_submitting(intent, expected_signal_id=signal+1)
    assert db.at_mark_entry_submitting(intent, expected_signal_id=signal)
    assert not db.at_release_unsent_entry(intent)


def receipt(now):
    return dict(instId='TEST-X',clOrdId='entry1',ordId='987',ordType='market',tdMode='isolated',
                side='buy',state='filled',accFillSz='2',fillTime=str(round(now*1000)),reduceOnly='false')


@pytest.mark.parametrize('override', [
    {'clOrdId':'other'}, {'instId':'OTHER'}, {'ordType':'limit'}, {'reduceOnly':'true'},
    {'side':'sell'}, {'accFillSz':'0'}, {'state':'partially_filled'}, {'fillTime':'0'},
])
def test_unrelated_unfilled_or_invalid_order_is_not_entry_evidence(override):
    now = 1795513000.
    position = dict(inst_id='TEST-X',client_order_id='entry1',direction='LONG',entry_submit_started_at=now-1)
    assert entry_execution.confirmed_last_fill_ms(receipt(now) | override, position, now) is None


def test_read_only_worker_persists_verified_last_fill_for_an_already_closed_position(journal,caplog):
    signal, intent = journal
    assert db.at_mark_entry_submitting(intent, expected_signal_id=signal)
    assert db.at_activate_entry_intent(intent, 2., 100., 4., 'oco1', 99.)
    db.at_close_position(intent, 'EXCHANGE_FLAT')
    now = time.time()
    with (caplog.at_level('INFO'),
          patch('src.autotrader._creds_of', return_value={'connection':'local'}),
          patch.object(entry_execution.okx, 'get_order_by_client_id',
                       side_effect=[(False,'temporary'), (True,receipt(now))]) as fetch,
          patch.object(entry_execution.okx, 'place_market_entry') as market):
        entry_execution.audit_entry_fills()
        # Allow the next scheduled retry without advancing the test clock.
        with db._conn() as conn:
            conn.execute('UPDATE autotrade_positions SET entry_fill_checked_at=? WHERE id=?',(now-121,intent))
        entry_execution.audit_entry_fills()
        entry_execution.audit_entry_fills()
    assert fetch.call_count == 2
    market.assert_not_called()
    with db._conn() as conn:
        stamp, status = conn.execute('SELECT entry_last_fill_ms,status FROM autotrade_positions WHERE id=?',
                                     (intent,)).fetchone()
    assert (stamp, status) == (round(now*1000), 'CLOSED')
    assert 'entry fill confirmed' in caplog.text and 'entry1' not in caplog.text
