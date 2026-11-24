from unittest.mock import patch

from src import autotrader as at
from src import db
from src.exit_summary import confirmed_partial_tp1


POSITION = dict(id=1, user_id=7, signal_id=2, inst_id='TEST-X', direction='LONG',
                status='OPEN', opened_at=1., sz=3., entry_filled_sz=3., tp1_sz=1.,
                sl_algo_id='stop', tp1_algo_id='tp1')


def test_poll_retries_delayed_tp1_and_announces_it_only_once():
    with (patch.object(at, 'AUTOTRADE_ENABLED', True),
          patch.object(at, 'at_all_open_positions', return_value=[POSITION]),
          patch.object(at, 'get_signal_by_id', return_value={'status': 'TP1_PARTIAL'}),
          patch.object(at, 'at_get', return_value={'user_id': 7}),
          patch.object(at, '_creds_of', return_value={'test': True}),
          patch.object(at.okx, 'get_position_size', side_effect=[(True, 3.)] + [(True, 2.)]*3),
          patch.object(at.okx, 'get_protection_order_ids', side_effect=[None, ('11',), ('11',)]) as orders,
          patch.object(at.okx, 'get_last_fill_px', return_value=101.) as fills,
          patch.object(at, 'at_reduce_position_sz', side_effect=[True, False]) as reduce,
          patch.object(at, '_dm') as dm):
        for _ in range(4):
            at.poll_exchange_closes()

    assert orders.call_count == 3
    orders.assert_called_with({'test': True}, 'TEST-X', 'tp1', ord_type='conditional')
    fills.assert_called_with({'test': True}, 'TEST-X', since_ts=1., side='sell',
                             order_ids=('11',), expected_size=1.)
    reduce.assert_called_with(1, 2., expected_sz=3.)
    dm.assert_called_once()
    assert '33.3%' in dm.call_args.args[1]


def test_decrease_without_exact_tp1_fill_is_not_attributed_to_tp1():
    with (patch.object(at.okx, 'get_protection_order_ids', return_value=('11',)),
          patch.object(at.okx, 'get_last_fill_px', return_value=None) as fills):
        assert confirmed_partial_tp1(POSITION, 2., {}, at.okx) is None
        assert confirmed_partial_tp1(POSITION, 1., {}, at.okx) is None
    fills.assert_called_once()


def test_size_update_compares_previous_size_and_requires_an_open_position(tmp_path, monkeypatch):
    monkeypatch.setattr(db, 'DB_PATH', str(tmp_path/'positions.sqlite3'))
    db.init_db()
    pos_id = db.at_log_position(2, 7, 'TEST-X', 'LONG', 3., 100., 3., 'stop', 99., 'tp1', 1.)
    assert db.at_reduce_position_sz(pos_id, 2., expected_sz=3.)
    assert not db.at_reduce_position_sz(pos_id, 1., expected_sz=3.)
    assert not db.at_reduce_position_sz(pos_id, 0., expected_sz=2.)
    assert db.at_all_open_positions()[0]['sz'] == 2.
    db.at_close_position(pos_id, 'TEST')
    assert not db.at_reduce_position_sz(pos_id, 1., expected_sz=2.)
