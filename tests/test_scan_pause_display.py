import os
os.environ['BOT_STARTUP_ENABLED']='0'
os.environ['PYTHON_DOTENV_DISABLED']='1'
import json
from unittest.mock import patch
import main


def status(row,timing):
    values={'native_scan_audit':json.dumps(row),'native_scan_last_timing':json.dumps(timing)}
    with (patch.object(main.time,'time',return_value=1_800_000_000),
          patch.object(main,'get_bot_state',side_effect=lambda k:values.get(k)),
          patch.object(main,'get_all_setups_since',return_value=[]),
          patch.object(main,'get_open_signals',return_value=[]),
          patch.object(main,'get_recent_signals',return_value=[])):
        return main._native_scan_status_text()


def test_daily_pause_separates_launch_check_from_previous_market_scan():
    text=status(dict(ts=1_800_000_000-480,result='daily_loss_pause'),
        dict(load_seconds=0,setup_seconds=.08,setup_ready_at=1_800_000_000-1133*60))
    assert 'Проверка запуска: 8 мин назад' in text
    assert 'Данные не запрашивались' in text and '0/0' not in text
    assert 'Последний полный скан: 1133 мин назад' in text


def test_complete_scan_uses_current_counts_and_timing():
    timing=dict(load_seconds=.5,setup_seconds=.08,setup_ready_at=1_800_000_000-30)
    text=status(dict(ts=1_800_000_000-10,result='no_candidate_passed',fetched=44,targets=44,
        strategy={'timing':timing}),{})
    assert 'Данные: 44/44' in text
    assert 'Время скана · загрузка 0.50с' in text
    assert 'Последний полный скан' not in text


def test_pause_audit_without_timing_does_not_log_an_exception(caplog):
    with patch.object(main,'set_bot_state') as write:
        main._persist_native_scan_audit(dict(ts=1_800_000_000,result='daily_loss_pause'))
    write.assert_called_once()
    assert 'Native scan audit persistence failed' not in caplog.text
