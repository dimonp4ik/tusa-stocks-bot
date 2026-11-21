from unittest.mock import patch

import pytest

import main


def message(text):
    return {'message': {'chat': {'id': 42}, 'text': text}}


@pytest.mark.parametrize(('text', 'expected'), [
    ('/start', 'Меню готово'), ('🛡 Режим', 'РЕЖИМ ИСПОЛНЕНИЯ'),
    ('📋 Открытые сделки', 'Нет активных позиций'), ('📈 Результаты', 'PAPER-итог'),
    ('❓ Помощь', 'Как пользоваться ботом'),
])
def test_menu_responds_via_the_inbound_connection_without_telegram_egress(text, expected):
    with (patch.object(main._requests, 'post', side_effect=AssertionError('Telegram egress unavailable')),
          patch.object(main, '_is_admin', return_value=False),
          patch.object(main, '_execution_state_text', return_value='PAPER'),
          patch.object(main, '_native_scan_status_text', return_value='Рынок проверен'),
          patch.object(main, 'get_open_signals', return_value=[]),
          patch.object(main, '_paper_results_text', return_value='*PAPER-итог*')):
        response = main.app.test_client().post('/webhook', json=message(text))
    assert response.status_code == 200
    body = response.get_json()
    assert body['method'] == 'sendMessage' and body['chat_id'] == 42
    assert expected in body['text'] and 'parse_mode' not in body
    if text == '/start':
        assert body['reply_markup']['keyboard']


def test_tp1_callback_returns_the_interactive_panel_instead_of_just_acknowledging():
    update = {'callback_query': {'id': 'test-id', 'from': {'id': 42}, 'data': 'at_tp1pct',
                                 'message': {'chat': {'id': 42}, 'message_id': 11}}}
    with (patch.object(main, 'DEPLOYMENT_MODE', 'live'),
          patch.object(main, 'at_get', return_value={'allowed': True}),
          patch.dict(main._at_onboarding, {}, clear=True),
          patch.object(main._requests, 'post', side_effect=AssertionError('Telegram egress unavailable'))):
        response = main.app.test_client().post('/webhook', json=update)
        assert main._at_onboarding[42]['step'] == 'tp1pct_change'
    body = response.get_json()
    assert body['method'] == 'sendMessage'
    assert body['reply_markup']['inline_keyboard'][0][0]['callback_data'] == 'at_tp1auto50'


def test_non_webhook_replies_still_use_the_regular_telegram_transport():
    with patch.object(main._requests, 'post') as transport:
        transport.return_value.status_code = 200
        main._reply(42, 'scheduled message')
    assert transport.call_args.kwargs['json']['text'] == 'scheduled message'
