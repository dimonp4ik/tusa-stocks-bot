"""Closed-bar trade activity diagnostics; no order or admission side effects."""
import math


def closed_activity(candle,entry_index):
    if entry_index<8 or entry_index>len(candle['time']):
        return None
    times=candle['time'][entry_index-8:entry_index]
    if any(b-a!=900 for a,b in zip(times,times[1:])):
        return None
    values=[float(v) for v in candle['volume'][entry_index-8:entry_index]]
    if len(values)!=8 or not all(math.isfinite(v) and v>=0 for v in values):
        raise ValueError('invalid closed trade activity')
    return dict(last_bar_active=values[-1]>0,last_two_active=all(v>0 for v in values[-2:]),
                active_eight=sum(v>0 for v in values))
