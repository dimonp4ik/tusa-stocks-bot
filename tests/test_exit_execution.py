import time
from unittest.mock import Mock, patch

import pytest

from src import db, exit_execution as audit, okx_trader


def entry_order(now, direction='LONG'):
    return dict(ordId='91', instId='TEST-X', clOrdId='entry1', ordType='market', tdMode='isolated',
                side='buy' if direction == 'LONG' else 'sell', state='filled', reduceOnly='false',
                avgPx='100', accFillSz='3', fillTime=str(round((now - 10) * 1000)))


def close_order(order_id, now, *, direction='LONG', size='2', price='99', pnl='-2'):
    return dict(ordId=order_id, instId='TEST-X', ordType='market', tdMode='isolated',
                side='sell' if direction == 'LONG' else 'buy', state='filled', reduceOnly='true',
                avgPx=price, accFillSz=size, pnl=pnl, feeCcy='USDC', ccy='USDC',
                fillTime=str(round((now - 1) * 1000)))


def position(now, direction='LONG'):
    return dict(inst_id='TEST-X', client_order_id='entry1', direction=direction, entry_filled_sz=3.,
                entry_submit_started_at=now-11, entry_last_fill_ms=round((now-10)*1000),
                closed_at=now, sl_algo_id='main', tp1_algo_id='partial', tp1_sz=1.)


def broker_at(now, direction='LONG'):
    broker = Mock()
    broker.get_order_by_client_id.return_value = True, entry_order(now, direction)
    broker.get_protection_order_ids.side_effect = lambda creds, inst, algo, **kw: (
        ('92', '94') if algo == 'main' else ('93',))
    children = {
        '92': close_order('92', now, direction=direction),
        '93': close_order('93', now, direction=direction, size='1', price='101', pnl='1'),
        '94': dict(close_order('94', now, direction=direction), accFillSz='0', state='canceled', fillTime=''),
    }
    broker.get_order_by_exchange_id.side_effect = lambda creds, inst, order, **kw: (True, children[order])
    return broker, children


@pytest.mark.parametrize('direction', ['LONG', 'SHORT'])
def test_protective_receipt_uses_confirmed_partial_and_full_runner(direction):
    now = 1795583400.
    broker, _ = broker_at(now, direction)
    receipt = audit.get_exit_receipt(position(now, direction), {}, broker, now)
    assert receipt['contracts'] == 3 and receipt['tp1_contracts'] == 1
    assert receipt['tp1_exit_px'] == 101 and receipt['runner_exit_px'] == 99
    assert receipt['exit_px'] == pytest.approx(99 + 2/3)
    assert receipt['pnl_usdc'] == -1


@pytest.mark.parametrize('override', [
    {'ordId': 'wrong'}, {'instId': 'OTHER'}, {'feeCcy': 'USDT'}, {'fillTime': '0'},
    {'pnl': ''}, {'pnl': 'NaN'}, {'reduceOnly': 'false'}, {'tdMode': 'cross'},
    {'state': 'partially_filled'}, {'accFillSz': '1'},
])
def test_unrelated_unpriced_or_incomplete_receipts_are_never_counted(override):
    now = 1795583400.
    broker, children = broker_at(now)
    children['92'].update(override)
    assert audit.get_exit_receipt(position(now), {}, broker, now) is None


def test_exchange_order_lookup_is_read_only_and_verifies_identity():
    details = close_order('92', 1795583400.)
    with patch.object(okx_trader, '_request', return_value=(True, [details])) as request:
        assert okx_trader.get_order_by_exchange_id({}, 'TEST-X', '92') == (True, details)
        request.assert_called_once_with({}, 'GET', '/api/v5/trade/order',
                                      params={'instId': 'TEST-X', 'ordId': '92'}, timeout=2)
        assert okx_trader.get_order_by_exchange_id({}, 'TEST-X', '93')[0] is False


def test_worker_backfills_a_closed_position_once_and_reports_only_to_its_owner(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(db, 'DB_PATH', str(tmp_path / 'positions.sqlite3'))
    db.init_db()
    db.at_add_allowed(7, 1)
    s = db.log_signal(dict(symbol='TEST', direction='LONG', current_price=100.), 101., 102., 99.)
    pos = db.at_create_entry_intent(s, 7, 'TEST-X', 'LONG', 3., 100., 4., 99., 'entry1')
    assert db.at_activate_entry_intent(pos, 3., 100., 4., 'main', 99., 'partial', 1., entry_filled_sz=3.)
    now = time.time()
    with db._conn() as conn:
        conn.execute('UPDATE autotrade_positions SET entry_submit_started_at=?, entry_last_fill_ms=? WHERE id=?',
                     (now-11, round((now-10)*1000), pos))
    assert db.at_exit_fill_checks(now) == []
    db.at_close_position(pos, 'EXCHANGE_FLAT')
    assert 'сверено 0 из 1' in audit.cash_status(7, now)
    broker, children = broker_at(now)
    prior = children['92']['pnl']
    children['92']['pnl'] = ''
    with (caplog.at_level('INFO'), patch.object(audit, 'okx', broker),
          patch('src.autotrader._creds_of', return_value={'opaque': 'connection'})):
        audit.audit_exit_fills()
        assert 'сверено 0 из 1' in audit.cash_status(7, now)
        with db._conn() as conn:
            conn.execute('UPDATE autotrade_positions SET exit_receipt_checked_at=? WHERE id=?', (now-181, pos))
        children['92']['pnl'] = prior
        audit.audit_exit_fills()
        audit.audit_exit_fills()
    assert broker.get_order_by_client_id.call_count == 2
    assert 'сверено 1 из 1' in audit.cash_status(7, now)
    assert '-1.00 USDC' in audit.cash_status(7, now)
    assert audit.cash_status(8, now) == ''
    assert 'exit receipt verified' in caplog.text
    assert 'entry1' not in caplog.text and '-1.00' not in caplog.text
