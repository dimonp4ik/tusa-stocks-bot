import os
from unittest.mock import patch

os.environ['BOT_STARTUP_ENABLED'] = '0'

import pytest

import main
from src import db, telegram_notifier


@pytest.mark.parametrize('direction,gap_open', [('LONG', 99.4), ('SHORT', 100.6)])
def test_fixed_runner_does_not_book_entry_price_after_a_known_gap(direction, gap_open):
    start = 2_000_000_100.
    sign = 1 if direction == 'LONG' else -1
    frame = dict(time=[start, start + 900], confirmed=[True, False],
        open=[100 + sign * .3, gap_open],
        high=[100.6 if sign > 0 else 99.8, 100.2 if sign > 0 else 100.7],
        low=[100.2 if sign > 0 else 99.4, 99.3 if sign > 0 else 99.8],
        close=[100 + sign * .4, 100 - sign * .2])
    signal = dict(id=71, symbol='BTCUSDT', direction=direction, entry_price=100.,
        sl=100 - sign, tp1=100 + sign * .5, tp2=100 + sign * 2,
        atr=1., opened_at=start - 60, tp1_hit_at=start + 30,
        status='TP1_PARTIAL', strategy_tp1_close_frac=.05, strategy_runner_mode='fixed_be',
        strategy_bar_seconds=900, strategy_entry_bar_ts=start,
        runner_activation_bar_ts=start, strategy_stop_on_close=0, strategy_max_bars=48)
    with (patch.object(main, 'get_open_signals', return_value=[signal]),
          patch.object(main.time, 'time', return_value=start + 960),
          patch.object(main, 'get_klines_xperp', return_value=frame),
          patch.object(main, 'update_signal_status') as update,
          patch.object(main, 'send_signal_update'),
          patch.object(main.autotrader, 'mirror_transition')):
        main._check_open_signals()
    assert update.call_args.args[:3] == (71, 'BREAKEVEN', gap_open)
    assert update.call_args.kwargs['realized_r'] == pytest.approx(-.545)


def test_negative_be_is_a_loss_and_positive_expiry_a_win_in_streak_and_weekly(tmp_path, monkeypatch):
    monkeypatch.setattr(db, 'DB_PATH', str(tmp_path / 'audit.sqlite'))
    db.init_db()
    for i, (status, value) in enumerate((('BREAKEVEN', -.545), ('EXPIRED', .2), ('BREAKEVEN', 0.))):
        sid = db.log_signal(dict(symbol='TEST', direction='LONG', current_price=100., trend_1h='up'),
                            100.5, 102., 99.)
        db.update_signal_status(sid, status, 100., realized_r=value)
        with db._conn() as c:
            c.execute('UPDATE signals SET opened_at=? WHERE id=?', (db.time_mod.time() - 3 + i, sid))
    stats = db.get_stats(since_ts=0.)
    assert stats['streak'] == ['➖', '✅', '❌']
    weekly = db.get_weekly_stats()
    assert weekly['n_win'] == 1 and weekly['wr'] == pytest.approx(33.3)
    assert weekly['top3'] == [('TEST', 1, 1)]
    assert weekly['n_loss'] == 1 and weekly['n_flat'] == 1


def test_gap_notification_reports_the_model_loss_instead_of_exact_breakeven():
    sig = dict(symbol='TEST', direction='LONG', entry_price=100., sl=99.,
               tp1=100.5, tp2=102., strategy_tp1_close_frac=.05, strategy_runner_mode='fixed_be')
    with patch.object(telegram_notifier, '_send_message', return_value=True) as send:
        assert telegram_notifier.send_signal_update(sig, 'BREAKEVEN', 99.4)
    text = send.call_args.args[0]
    assert 'ГЭП' in text
    assert '-0.5450R' in text
    assert 'остаток закрыт по входу' not in text
