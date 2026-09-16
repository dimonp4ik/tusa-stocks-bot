"""Causal candle alignment shared by historical research and regression tests."""
from bisect import bisect_right
from dataclasses import dataclass


def closed_snapshot(candles, decision_time, lookback, fallback_end, interval_sec):
    """Timestamps are candle OPEN times; values become available at open + interval.

    Return no bars when the first bar has not closed. Never infer a timeframe
    from adjacent timestamps: weekends and exchange gaps change that spacing.
    """
    if interval_sec <= 0 or lookback <= 0:
        raise ValueError('Positive interval and lookback required')
    if not candles or not candles.get('close'):
        return {}
    if decision_time is not None and candles.get('time'):
        end = bisect_right(candles['time'], decision_time - interval_sec)
    else:
        end = fallback_end
    end = max(0, min(end, len(candles['close'])))
    return {k: v[max(0, end - lookback):end] for k, v in candles.items()}


@dataclass(frozen=True)
class ExitResult:
    outcome: str
    bar: int
    price: float
    gross_r: float
    mae_r: float
    mfe_tp1: float
    reached_tp1: bool


def simulate_exit(candles, indices, *, direction, entry, sl, tp1, tp2, atr,
                  tp1_fraction, trail, trail_mult, stop_on_close, backstop_r,
                  choose_trail, intrabar_entry=False):
    """Conservative OHLC replay with previous-bar stops and gap-aware fills.

    Stop orders already resting on the exchange take priority on ambiguous bars.
    A zone touch cannot claim a favourable excursion from before its entry.
    Newly computed trailing levels become active on the NEXT candle only.
    Funding and intrabar path/latency still require exchange data.
    """
    sign = 1 if direction == 'LONG' else -1
    risk = abs(entry-sl)
    if risk <= 0 or entry <= 0:
        raise ValueError('Positive entry and stop distance required')
    indices = list(indices)
    if not indices:
        raise ValueError('No execution bars')
    fraction = max(0., min(1., tp1_fraction))
    reached = False
    best = entry
    stop = entry-sign*risk*max(1., backstop_r) if stop_on_close else sl
    mae = mfe = 0.
    banked = 0.

    def done(outcome, j, price):
        remaining = 1-fraction if reached else 1.
        return ExitResult(outcome, j, price, banked+remaining*sign*(price-entry)/risk, mae, mfe, reached)

    for j in indices:
        o,h,l,c = (float(candles[k][j]) for k in ('open','high','low','close'))
        adverse = entry-l if sign == 1 else h-entry
        favourable = h-entry if sign == 1 else entry-l
        mae = max(mae, adverse/risk)
        mfe = max(mfe, favourable/max(abs(tp1-entry), 1e-12))
        stop_hit = l <= stop if sign == 1 else h >= stop
        if stop_hit:
            px = min(o, stop) if sign == 1 else max(o, stop)
            # Opening price precedes a watched intrabar entry.
            if intrabar_entry and j == indices[0]:
                px = stop
            return done('TRAIL' if reached and trail else ('TP1' if reached else 'SL'), j, px)
        if not reached and stop_on_close and sign*(c-sl) <= 0:
            return done('SL',j,c)
        if intrabar_entry and j == indices[0]:
            # No intrabar ordering is available; only the close is known to be
            # after the zone touch. Do not bank an earlier high/low as a target.
            h = max(entry,c)
            l = min(entry,c)
        if sign*((h if sign == 1 else l)-tp2) >= 0:
            if not reached:
                banked = fraction*sign*(tp1-entry)/risk
                reached = True
            return done('TP2',j,tp2)
        if not reached and sign*((h if sign == 1 else l)-tp1) >= 0:
            reached = True
            banked = fraction*sign*(tp1-entry)/risk
            if fraction == 1:
                return done('TP1',j,tp1)
            trail_mult = max(0., choose_trail(h,l,c))
            # The move back through breakeven may have followed TP1. Count the
            # adverse ordering; never award a profit from a candle's range alone.
            if (l <= entry if sign == 1 else h >= entry):
                return done('TP1',j,min(entry,c) if sign == 1 else max(entry,c))
        if reached:
            best = max(best,h) if sign == 1 else min(best,l)
            candidate = best-sign*max(0.,atr)*trail_mult if trail else entry
            stop = max(stop,entry,candidate) if sign == 1 else min(stop,entry,candidate)
    return done('EXPIRED',indices[-1],float(candles['close'][indices[-1]]))
