from unittest.mock import patch

from src import okx_connection_check as check


CREDS = dict(api_key='dummy-key', api_secret='dummy-secret', passphrase='dummy-passphrase')
KEY_ERROR = dict(reason='config_unavailable', detail="API key doesn't exist (code 50119)")


def test_50119_probes_only_official_regional_config_and_uses_a_sanitized_result():
    config = dict(account_uid='00099', position_mode='net_mode')
    with (patch.object(check.okx, '_base_url', return_value='https://my.okx.com'),
          patch.object(check.okx, 'validate_autotrade_account', return_value=(False, KEY_ERROR)),
          patch.object(check.okx, 'get_account_configuration', return_value=(True, config)) as get_config,
          patch.object(check.okx, 'get_balance') as balance):
        text = check.check_connection(CREDS)

    get_config.assert_called_once_with(CREDS, base_url='https://eea.okx.com')
    balance.assert_not_called()
    assert 'европейском API' in text and '50119' in text
    assert '00099' not in text and not any(v in text for v in CREDS.values())


def test_primary_failure_does_not_claim_a_regional_failure_without_a_matching_code():
    with (patch.object(check.okx, '_base_url', return_value='https://my.okx.com'),
          patch.object(check.okx, 'validate_autotrade_account', return_value=(False, KEY_ERROR)),
          patch.object(check.okx, 'get_account_configuration', return_value=(False, 'request failed')) as regional):
        assert 'на текущем адресе' in check.check_connection(CREDS)
        regional.return_value = False, KEY_ERROR['detail']
        assert 'на обоих адресах' in check.check_connection(CREDS)


def test_compatible_connection_checks_balance_without_displaying_equity():
    with (patch.object(check.okx, 'validate_autotrade_account', return_value=(True, {})),
          patch.object(check.okx, 'get_balance', return_value=(True, 123.45)) as balance,
          patch.object(check.okx, 'get_account_configuration') as regional):
        text = check.check_connection(CREDS)
    balance.assert_called_once_with(CREDS)
    regional.assert_not_called()
    assert 'подтвердил торговые права' in text and '123.45' not in text


def test_callback_accepts_only_private_chat_for_the_saved_connection():
    import main

    connection = dict(allowed=True, api_key_enc='encrypted')
    with (patch.object(main, 'at_get', return_value=connection),
          patch.object(main, '_answer_callback') as answer,
          patch.object(check, 'start_user_check', return_value=True) as start):
        assert main._at_handle_callback('cb', -123, 42, 'at_check_connection')
        start.assert_not_called()
        assert main._at_handle_callback('cb', 42, 42, 'at_check_connection')
        start.assert_called_once()
        assert start.call_args.args[:2] == (42, connection)
        answer.assert_called_with('cb', 'Проверяю подключение OKX.')
