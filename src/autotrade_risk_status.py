"""Read-only presentation of the last account-equity guard decision."""
import math
from datetime import datetime, timezone


def _positive(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def risk_status_text(state, *, now, max_drawdown=None):
    """Never reevaluate/reset a guard or promise that an entry can be sent."""
    if not isinstance(state, dict):
        return "Проверка лимитов счёта: сохранённые данные недоступны."
    day = datetime.fromtimestamp(now, timezone.utc).date().isoformat()
    if state.get('paused'):
        lines = ["⏸ Новые входы остановлены: лимит общей просадки.",
                 "Входы возобновятся после 3 свежих проверок за минимум 2 минуты "
                 "в пределах прежнего лимита и без дневной паузы."]
        status=state.get('recovery_status')
        if status=='checking':
            lines.append(f"Подтверждений восстановления: {int(state.get('recovery_checks') or 0)}/3.")
        elif status=='daily_pause':
            lines.append('Сначала должна закончиться дневная пауза: следующий день UTC.')
        elif status=='balance_unavailable':
            lines.append('Свежий баланс пока недоступен; восстановление не подтверждено.')
        threshold=_positive(state.get('recovery_equity_threshold'))
        if threshold is not None:
            lines.append(f'Для восстановления нужен баланс выше ${threshold:.2f} с запасом к лимиту.')
    elif state.get('daily_paused') and state.get('day') == day:
        lines = ["⏸ Новые входы остановлены: дневной лимит потерь.",
                 "Дневная пауза действует до следующего дня UTC."]
    else:
        lines = ["Лимиты счёта: сохранённой действующей паузы нет.",
                 "Перед входом бот заново проверяет счёт и ограничения."]
    peak = _positive(state.get('peak'))
    equity = _positive(state.get('equity'))
    checked = _positive(state.get('checked_at'))
    limit = _positive(state.get('max_drawdown'))
    if peak is not None:
        current_limit = _positive(max_drawdown)
        threshold = (f" Порог от этого максимума при текущем лимите "
                     f"{current_limit*100:g}%: ${peak*(1-current_limit):.2f}."
                     if current_limit is not None and current_limit < 1 else "")
        lines.append(f"Сохранённый максимум счёта: ${peak:.2f}.{threshold}")
    # Legacy records contain peak/day but no timestamped equity. Do not pair
    # their peak with an unrelated last_balance and invent the trigger loss.
    if peak is not None and equity is not None and checked is not None and checked <= now + 1:
        age = max(0, int((now - checked) / 60))
        drawdown = max(0., (peak - equity) / peak * 100.)
        suffix = f"; порог {limit * 100:g}%" if limit is not None and limit < 1 else ""
        lines.append(f"Последняя проверка ({age} мин назад): баланс ${equity:.2f}, "
                     f"максимум ${peak:.2f}, просадка {drawdown:.2f}%{suffix}.")
    elif state.get('paused') or (state.get('daily_paused') and state.get('day') == day):
        lines.append("В старой записи нет снимка баланса и времени срабатывания; "
                     "сумму исходного убытка по ней подтвердить нельзя.")
    return '\n'.join(lines)
