"""Research-only cross-sectional shocks and closed-bar directional context."""
import math
import numpy as np
import pandas as pd

SCORES=('raw_z','normalized_z','robust_normalized_z')
PHASES=('reversal','continuation','closed_phase')


def feature_pool(candles):
    rows=[]
    for symbol,candle in sorted(candles.items()):
        t=np.asarray(candle['time'],dtype=np.int64)
        o,h,l,c,v=[np.asarray(candle[k],dtype=float) for k in ('open','high','low','close','volume')]
        if not all(len(a)==len(t) for a in (o,h,l,c,v)) or len(t)<65:
            raise ValueError('incomplete source arrays')
        if (not all(np.isfinite(a).all() for a in (o,h,l,c,v))
                or (np.minimum.reduce((o,h,l,c))<=0).any() or (v<0).any()
                or not ((l<=np.minimum(o,c))&(np.maximum(o,c)<=h)).all()
                or (np.diff(t)<=0).any()):raise ValueError('invalid source arrays')
        close=pd.Series(c);prev=close.shift(1).to_numpy()
        tr=np.maximum.reduce((h-l,np.abs(h-prev),np.abs(l-prev)))
        atr=pd.Series(tr).rolling(14,min_periods=14).mean()
        move=close/close.shift(2)-1
        pre_atr=atr.shift(2)/close.shift(2)
        span=h-l
        with np.errstate(divide='ignore',invalid='ignore'):
            location=np.divide(c-l,span,out=np.full(len(c),.5),where=span>0)
            upper=np.divide(h-np.maximum(o,c),span,out=np.zeros(len(c)),where=span>0)
            lower=np.divide(np.minimum(o,c)-l,span,out=np.zeros(len(c)),where=span>0)
        gaps=pd.Series(np.r_[0,np.diff(t)!=900])
        history_ok=(gaps.rolling(63,min_periods=63).sum().eq(0)&(np.arange(len(t))>=63))
        frame=pd.DataFrame(dict(symbol=symbol,entry_index=np.arange(len(t))+1,entry_time=t+900,
            signal_return=move,normalized_move=move/pre_atr.replace(0,np.nan),
            atr14_fraction=atr/close,body_fraction=(c-o)/c,close_location=location,
            upper_wick=upper,lower_wick=lower,own_trend_16=close/close.shift(16)-1,
            active_bars_8=pd.Series(v>0).rolling(8,min_periods=8).sum(),
            last_volume=v,history_ok=history_ok))
        # Two-bar shocks require all their own preceding intervals. No next
        # candle timestamp/open/range participates in any entry feature.
        aligned=np.r_[False,False,t[2:]-t[:-2]==1800]
        rows.append(frame.loc[aligned & move.notna()].copy())
    all_rows=pd.concat(rows,ignore_index=True)
    by=all_rows.groupby('entry_time',sort=False)
    all_rows['universe_n']=by.signal_return.transform('size')
    all_rows['cross_mean']=by.signal_return.transform('mean')
    all_rows['raw_z']=(all_rows.signal_return-all_rows.cross_mean)/by.signal_return.transform('std').replace(0,np.nan)
    normal_mean=by.normalized_move.transform('mean')
    normal_std=by.normalized_move.transform('std').replace(0,np.nan)
    all_rows['normalized_z']=(all_rows.normalized_move-normal_mean)/normal_std
    median=by.normalized_move.transform('median')
    all_rows['_normal_deviation']=(all_rows.normalized_move-median).abs()
    mad=all_rows.groupby('entry_time',sort=False)._normal_deviation.transform('median').replace(0,np.nan)
    all_rows['robust_normalized_z']=(all_rows.normalized_move-median)/(1.4826*mad)
    all_rows['normalized_universe_n']=by.normalized_move.transform('count')
    return all_rows.drop(columns=['_normal_deviation'])


def phase_direction(row,phase):
    if phase not in PHASES:raise ValueError('unknown phase')
    move=float(row['signal_return'])
    if not math.isfinite(move) or move==0:return None
    impulse=1 if move>0 else -1
    if phase=='reversal':return -impulse
    if phase=='continuation':return impulse
    body=float(row['body_fraction']);location=float(row['close_location'])
    own=float(row['own_trend_16']);basket=float(row['basket_trend_16'])
    upper=float(row['upper_wick']);lower=float(row['lower_wick'])
    active=float(row['active_bars_8']);volume=float(row['last_volume'])
    if not all(math.isfinite(x) for x in (body,location,own,basket,upper,lower,active,volume)):
        return None
    favorable_location=location if impulse>0 else 1-location
    rejection=upper if impulse>0 else lower
    if (impulse*body>0 and favorable_location>=.7 and impulse*own>0
            and impulse*basket>0 and active>=6 and volume>0):
        return impulse
    if impulse*body<0 and rejection>=.4:return -impulse
    return None


def prepare_entries(pool,context,score,phase):
    if score not in SCORES:raise ValueError('unknown shock score')
    required=['basket_trend_16','basket_trend_32','basket_trend_64','basket_trend_96']
    fields=['entry_time',*required,'premarket_paused']
    f=pool.merge(context[fields],on='entry_time',validate='many_to_one').dropna(subset=required)
    keep=(f.history_ok & f.universe_n.ge(10) & f.atr14_fraction.le(.01)
          & f.signal_return.abs().ge(.004) & f[score].abs().ge(2)
          & ~f.premarket_paused)
    if score!='raw_z':keep &= f.normalized_universe_n.ge(10)
    f=f[keep].copy()
    f['direction_n']=[phase_direction(row,phase) for row in f.to_dict('records')]
    f=f.dropna(subset=['direction_n']);f['direction_n']=f.direction_n.astype(int)
    # Preserve existing reversal-specific guards for fades. Continuations
    # deliberately use their own phase context instead of a reversal guard.
    reversal=f.direction_n*np.sign(f.signal_return)<0
    reversal_ok=((f.direction_n.gt(0)&f.basket_trend_96.ge(-.005626823134606253)&f.basket_trend_64.ge(-.04))
                 |(f.direction_n.lt(0)&f.basket_trend_16.gt(-.0075)))
    f=f[~reversal | reversal_ok].copy()
    f['cross_z']=f[score]
    f['audited_target_pct']=np.where(f.basket_trend_32.le(-.04),2.,
        np.where((f.entry_time//3600%24).between(6,11),3.,2.5))
    return f
