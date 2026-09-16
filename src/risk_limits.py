"""Pure risk calculations. Percent settings are fractions of account equity."""
import math


def finite_positive(value, name):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f'{name} must be finite and positive')
    return number


def bounded_margin(*, requested, equity, leverage, entry, stop, direction,
                   open_risk, trade_fraction, portfolio_fraction, cost_fraction):
    """Cap loss to the actual exchange backstop plus estimated costs.

    Gaps/slippage beyond the cost reserve can exceed this budget. Do not round
    UP to an exchange minimum; an unaffordable minimum means skipping the trade.
    """
    requested = finite_positive(requested,'requested margin')
    equity = finite_positive(equity,'equity')
    leverage = finite_positive(leverage,'leverage')
    entry = finite_positive(entry,'entry')
    stop = finite_positive(stop,'stop')
    if direction not in ('LONG','SHORT'):
        raise ValueError('Unknown direction')
    if (direction == 'LONG' and stop >= entry) or (direction == 'SHORT' and stop <= entry):
        raise ValueError('Stop must be on the losing side of entry')
    for value in (trade_fraction, portfolio_fraction, cost_fraction):
        if not math.isfinite(value) or not 0 < value < 1:
            raise ValueError('Risk fractions must lie between 0 and 1')
    if not math.isfinite(open_risk) or open_risk < 0:
        raise ValueError('Open risk must be finite and nonnegative')
    budget = min(equity*trade_fraction, max(0.,equity*portfolio_fraction-open_risk))
    return min(requested, equity, budget/(leverage*(abs(entry-stop)/entry+cost_fraction)))


def position_risk(position, contract_value, cost_fraction):
    entry = finite_positive(position['entry_px'],'position entry')
    stop = finite_positive(position['sl_px'],'position stop')
    sz = finite_positive(position['sz'],'position contracts')
    cv = finite_positive(contract_value,'contract value')
    direction = position['direction']
    if direction not in ('LONG','SHORT'):
        raise ValueError('Unknown position direction')
    distance = max(0., entry-stop if direction == 'LONG' else stop-entry)
    return sz*cv*(distance+entry*cost_fraction)


def equity_guard(state, *, equity, day, max_daily_loss, max_drawdown, peak_floor=None):
    equity = finite_positive(equity,'equity')
    for limit in (max_daily_loss,max_drawdown):
        if not math.isfinite(limit) or not 0 < limit < 1:
            raise ValueError('Drawdown limits must lie between 0 and 1')
    state = dict(state or {})
    floor = equity
    if peak_floor not in (None, 0, 0.0, ''):
        floor = finite_positive(peak_floor,'peak floor')
    peak = max(equity, floor, finite_positive(state.get('peak',equity),'peak'))
    daily_start = finite_positive(state.get('daily_start',equity),'daily start')
    same_day = state.get('day') == day
    if not same_day:
        daily_start = equity
    # Drawdown pause is latched; recovering equity must not silently restart it.
    paused = bool(state.get('paused')) or equity <= peak*(1-max_drawdown)
    daily_paused = (same_day and bool(state.get('daily_paused'))) or equity <= daily_start*(1-max_daily_loss)
    state.update(peak=peak,day=day,daily_start=daily_start,paused=paused,daily_paused=daily_paused)
    return state, 'drawdown limit' if paused else ('daily loss limit' if daily_paused else None)
