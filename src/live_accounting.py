"""Quote-based estimates for bot signals, never a substitute for exchange fills."""
import math


def trailing_observation(candles, *, direction, entry, atr, multiple, quote):
    """Compute a stop from known extrema; only the current quote can trigger it.

    A candle's old low may have preceded its high. It cannot prove that a stop
    computed from that high ever executed, or award that stop price as a fill.
    """
    values = (entry, atr, multiple, quote)
    if not all(math.isfinite(float(v)) for v in values) or entry <= 0 or quote <= 0 or atr < 0 or multiple < 0:
        raise ValueError('Invalid trailing observation')
    if direction == 'LONG':
        best = max([entry,quote]+list(candles.get('high') or []))
        stop = max(entry,best-atr*multiple)
        return stop, quote <= stop
    if direction == 'SHORT':
        best = min([entry,quote]+list(candles.get('low') or []))
        stop = min(entry,best+atr*multiple)
        return stop, quote >= stop
    raise ValueError('Unknown direction')


def marked_r(*, direction, entry, exit_price, sl, tp1, tp1_fraction=0., reached_tp1=False):
    risk = abs(entry-sl)
    if risk <= 0:
        raise ValueError('Positive stop distance required')
    if direction not in ('LONG','SHORT'):
        raise ValueError('Unknown direction')
    sign = 1 if direction == 'LONG' else -1
    fraction = max(0.,min(1.,tp1_fraction)) if reached_tp1 else 0.
    return fraction*sign*(tp1-entry)/risk+(1-fraction)*sign*(exit_price-entry)/risk
