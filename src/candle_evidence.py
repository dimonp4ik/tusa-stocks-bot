"""Exclude explicitly empty/invalid-volume OHLC from trade-touch evidence."""
import math


def bar_has_trade(candles,index):
    # Legacy feeds without volume retain their prior contract. Native OKX
    # candles carry volume: a zero bar repeats an older price, not a new trade.
    if 'volume' not in candles:return True
    try:
        volume=float(candles['volume'][index])
        return math.isfinite(volume) and volume>0
    except (IndexError,KeyError,TypeError,ValueError):
        return False


def traded_bars(candles):
    indices=[i for i in range(len(candles.get('close') or [])) if bar_has_trade(candles,i)]
    return {key:[values[i] for i in indices] for key,values in candles.items()}
