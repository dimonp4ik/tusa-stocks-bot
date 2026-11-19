"""Exclude explicitly empty/invalid-volume OHLC from trade-touch evidence."""
import math


def bar_has_trade(candles,index):
    # OKX base volume can round to zero even when fractional contracts traded.
    key = 'volume_contracts' if 'volume_contracts' in candles else 'volume'
    if key not in candles:
        return True
    try:
        volume = float(candles[key][index])
        return math.isfinite(volume) and volume > 0
    except (IndexError,KeyError,TypeError,ValueError):
        return False


def traded_bars(candles):
    indices=[i for i in range(len(candles.get('close') or [])) if bar_has_trade(candles,i)]
    return {key:[values[i] for i in indices] for key,values in candles.items()}
