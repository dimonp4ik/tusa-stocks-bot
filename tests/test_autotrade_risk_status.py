import copy
import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock, patch

import main
from src import autotrader as at
from src.autotrade_risk_status import risk_status_text


NOW = datetime(2026, 11, 17, 12, tzinfo=timezone.utc).timestamp()


def test_old_latched_pause_does_not_invent_original_equity_or_expire():
    state = dict(paused=True, day='2026-11-16', peak=130, daily_paused=True)
    original = copy.deepcopy(state)
    text = risk_status_text(state, now=NOW)
    assert 'Новые входы остановлены: лимит общей просадки' in text
    assert 'после 3 свежих проверок за минимум 2 минуты' in text
    assert 'исходного убытка' in text
    assert 'Сохранённый максимум счёта: $130.00' in text
    assert 'Последняя проверка' not in text
    assert state == original


def test_daily_pause_only_applies_on_its_utc_date():
    state = dict(paused=False, daily_paused=True, day='2026-11-17')
    assert 'дневной лимит потерь' in risk_status_text(state, now=NOW)
    state['day'] = '2026-11-16'
    text = risk_status_text(state, now=NOW)
    assert 'сохранённой действующей паузы нет' in text
    assert 'заново проверяет' in text


def test_recovered_balance_still_shows_latched_pause_and_last_check():
    text = risk_status_text(dict(paused=True, peak=130, equity=131,
        checked_at=NOW-120, max_drawdown=.03), now=NOW)
    assert 'Новые входы остановлены' in text
    assert '2 мин назад' in text
    assert 'просадка 0.00%; порог 3%' in text


def test_invalid_saved_data_is_not_presented_as_confirmed_equity():
    assert 'недоступны' in risk_status_text([], now=NOW)
    for bad in (float('nan'), float('inf'), -1, None):
        text = risk_status_text(dict(paused=True, peak=130, equity=bad,
            checked_at=NOW), now=NOW)
        assert 'Последняя проверка' not in text


def test_old_peak_and_current_threshold_are_visible_without_inventing_trigger_balance():
    text = risk_status_text(dict(paused=True, peak=130), now=NOW, max_drawdown=.03)
    assert 'Сохранённый максимум счёта: $130.00' in text
    assert 'при текущем лимите 3%: $126.10' in text
    assert 'сумму исходного убытка' in text
    assert 'Последняя проверка' not in text


def test_menu_distinguishes_enabled_connection_from_latched_account_pause():
    records = {'risk_guard:7': json.dumps(dict(paused=True, peak=130))}
    with (patch.object(main, 'get_bot_state', side_effect=lambda k: records.get(k)),
          patch.object(main, '_execution_state_text', return_value='PAPER: биржевые ордера выключены'),
          patch.object(main, '_at_reply_kb') as reply):
        main._at_show_menu(7, dict(active=1, size_value=3, last_balance=125, tp1_close_pct=5))
    text = reply.call_args.args[1]
    assert 'Подключение: 🟢 включено' in text
    assert 'PAPER: биржевые ордера выключены' in text
    assert 'Новые входы остановлены: лимит общей просадки' in text
    assert 'после 3 свежих проверок за минимум 2 минуты' in text


def test_menu_reports_corrupt_guard_without_resetting_it():
    with (patch.object(main, 'get_bot_state', return_value='{broken'),
          patch.object(main, 'set_bot_state') as save,
          patch.object(main, '_execution_state_text', return_value='LIVE'),
          patch.object(main, '_at_reply_kb') as reply):
        main._at_show_menu(7, dict(active=1, size_value=3, last_balance=125))
    assert 'сохранённые данные недоступны' in reply.call_args.args[1]
    save.assert_not_called()


def test_latched_entry_records_current_snapshot_without_sending_order():
    previous = dict(paused=True, peak=130, day='2026-11-16', daily_start=130)
    with (patch.object(at, '_shared_capacity', SimpleNamespace(prepare=Mock(return_value=True), enabled=True)),
          patch.object(at.okx, 'get_balance', return_value=(True, 131)),
          patch.object(at, 'at_set_balance'),
          patch.object(at, 'get_bot_state', return_value=json.dumps(previous)),
          patch.object(at, 'set_bot_state') as save,
          patch.object(at, '_entry_skip') as skip,
          patch.object(at.okx, 'place_market_entry') as order):
        at._open_for_account_locked(dict(user_id=7), dict(id=1), 'TEST-X', 'TEST', {}, 'a'*64)
    state = json.loads(save.call_args.args[1])
    assert state['equity'] == 131
    assert state['peak'] == 131
    assert state['checked_at'] > 0
    assert state['max_drawdown'] == at.AUTOTRADE_MAX_DRAWDOWN
    assert state['paused'] is True
    assert 'drawdown limit' in skip.call_args.args[3]
    order.assert_not_called()
