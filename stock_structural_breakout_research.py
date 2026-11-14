"""Closed-bar breakout prototype. No credentials, notifications or orders."""
import numpy as np
import pandas as pd


def entries(candle, symbol, *, lookback=20, minimum_volume_ratio=1.5, target_r=2.):
    c = pd.DataFrame({k: candle[k] for k in ("time","open","high","low","close","volume")})
    prior_close = c.close.shift(1)
    tr = pd.concat([c.high-c.low,(c.high-prior_close).abs(),(c.low-prior_close).abs()],axis=1).max(axis=1)
    atr = tr.rolling(14,min_periods=14).mean()
    top = c.high.shift(1).rolling(lookback,min_periods=lookback).max()
    bottom = c.low.shift(1).rolling(lookback,min_periods=lookback).min()
    fast, slow = c.close.rolling(20,min_periods=20).mean(), c.close.rolling(96,min_periods=96).mean()
    volume_base = c.volume.shift(1).rolling(20,min_periods=20).median().replace(0,np.nan)
    volume_ratio = c.volume/volume_base
    active = c.volume.gt(0).rolling(8,min_periods=8).sum().ge(6)
    location = (c.close-c.low)/(c.high-c.low).replace(0,np.nan)
    long = (c.close.gt(top) & prior_close.le(top.shift(1)) & fast.gt(slow) & location.ge(.75))
    short = (c.close.lt(bottom) & prior_close.ge(bottom.shift(1)) & fast.lt(slow) & location.le(.25))
    long_stop = c.low.rolling(5,min_periods=5).min()-.1*atr
    short_stop = c.high.rolling(5,min_periods=5).max()+.1*atr
    output = []
    for i in np.flatnonzero(((long|short)&active&volume_ratio.ge(minimum_volume_ratio)).to_numpy()):
        if i+1 >= len(c) or c.time.iloc[i+1]-c.time.iloc[i] != 900:
            continue
        direction = 1 if long.iloc[i] else -1
        opened = float(c.open.iloc[i+1])
        stop = float(long_stop.iloc[i] if direction>0 else short_stop.iloc[i])
        risk = direction*(opened-stop)/opened if opened>0 else np.nan
        if not np.isfinite(risk) or not .005 <= risk*target_r <= .02 or risk>.01:
            continue
        output.append(dict(symbol=symbol,entry_index=int(i+1),entry_time=int(c.time.iloc[i+1]),
            direction_n=direction,structural_stop=stop,risk_pct=risk,
            volume_ratio=float(volume_ratio.iloc[i]),signal_time=int(c.time.iloc[i])))
    return output
