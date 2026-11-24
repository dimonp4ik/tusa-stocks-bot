"""Read-only checks for a saved OKX autotrading connection."""

import logging
import re
import threading

from src import okx_trader as okx


_LOG = logging.getLogger(__name__)
_ACTIVE = set()
_LOCK = threading.Lock()
_FIELDS = ('api_key_enc', 'api_secret_enc', 'passphrase_enc')
_REGIONAL_HOST = 'https://eea.okx.com'


def exchange_error_code(detail):
    match = re.search(r'\(code ([0-9]{5})(?:,|\))', str(detail))
    return match.group(1) if match else None


def account_failure_label(result):
    if (result.get('reason') == 'config_unavailable'
            and exchange_error_code(result.get('detail')) == '50119'):
        return 'OKX не нашёл сохранённый API-ключ (50119)'
    return None


def check_connection(credentials):
    account_ok, result = okx.validate_autotrade_account(credentials)
    if not account_ok:
        if account_failure_label(result):
            if okx._base_url() == 'https://my.okx.com':
                found, regional = okx.get_account_configuration(credentials, base_url=_REGIONAL_HOST)
                if found and okx.account_key(regional):
                    return ('⚠️ OKX распознал сохранённый ключ на европейском API, '
                            'а текущий адрес бота вернул 50119. Передай этот результат администратору.')
                if exchange_error_code(regional) == '50119':
                    return ('❌ OKX не нашёл сохранённый ключ на обоих адресах (50119). '
                            'Проверь ключ в разделе API своего аккаунта OKX. '
                            'При необходимости нажми «Заменить API-ключи».')
            return ('❌ OKX не нашёл сохранённый ключ на текущем адресе бота (50119). '
                    'Проверь ключ в разделе API своего аккаунта OKX.')
        reason = result.get('reason')
        return {
            'long_short_mode': '⚠️ На счёте включён режим Long/Short. Для входов боту нужен режим Net.',
            'trade_permission_missing': '⚠️ В сохранённом ключе отсутствует право Trade.',
            'account_identity_missing': '⚠️ OKX не подтвердил идентификатор счёта для общего лимита позиций.',
        }.get(reason, '⚠️ OKX пока не подтвердил настройки сохранённого подключения. Повтори проверку позже.')

    balance_ok, _ = okx.get_balance(credentials)
    if not balance_ok:
        return '⚠️ Ключ принят и торговые права подтверждены, но OKX не вернул баланс.'
    return '✅ OKX принял сохранённый ключ и подтвердил торговые права, режим Net и доступ к балансу.'


def start_user_check(user_id, connection, deliver):
    """Answer asynchronously and discard results for a replaced connection."""
    with _LOCK:
        if user_id in _ACTIVE:
            return False
        _ACTIVE.add(user_id)

    def run():
        from src.autotrader import _creds_of
        from src.db import at_get

        try:
            try:
                credentials = _creds_of(connection)
                message = (check_connection(credentials) if credentials else
                           '⚠️ Сохранённое подключение не удалось прочитать. Открой «Заменить API-ключи».')
            except Exception as exc:
                _LOG.warning('OKX connection check unavailable (%s)', type(exc).__name__)
                message = '⚠️ Проверка подключения сейчас недоступна. Попробуй позже.'
            current = at_get(user_id) or {}
            if current.get('allowed') and all(current.get(k) == connection.get(k) for k in _FIELDS):
                deliver(message)
        except Exception as exc:
            _LOG.warning('OKX connection check response unavailable (%s)', type(exc).__name__)
        finally:
            with _LOCK:
                _ACTIVE.discard(user_id)

    try:
        threading.Thread(target=run, name='okx-saved-connection', daemon=True).start()
    except RuntimeError:
        with _LOCK:
            _ACTIVE.discard(user_id)
        return False
    return True
