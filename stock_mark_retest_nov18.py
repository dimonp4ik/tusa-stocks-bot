"""Confirm a closed retest after a mark breakout; never place a limit entry."""
from datetime import datetime,timezone
import math
import numpy as np

from src.market_hours import is_market_open
from stock_mark_hypotheses_nov18 import finite_ema,HORIZON,LOOKBACK


def retest_signals(mark):
    t=np.asarray(mark['time'],dtype=np.int64)
    o,h,l,c=[np.asarray(mark[k],dtype=float) for k in ('open','high','low','close')]
    fast,slow=finite_ema(c,16),finite_ema(c,64)
    prev=np.r_[c[0],c[:-1]]
    tr=np.maximum.reduce((h-l,abs(h-prev),abs(l-prev)))
    atr=np.r_[np.full(13,np.nan),np.convolve(tr,np.ones(14)/14,mode='valid')]
    gap=np.r_[0,np.cumsum(np.diff(t)!=900)]
    pending=None;out=[]
    for j in range(LOOKBACK,len(t)):
        if gap[j]!=gap[j-LOOKBACK]:pending=None;continue
        if pending is not None:
            sign,level,reference_atr,opened=pending
            if j-opened>4 or sign*(c[j]-level)<-.5*reference_atr or sign*(fast[j]-slow[j])<=0:
                pending=None;continue
            touch=(l[j]<=level+.1*reference_atr if sign>0 else h[j]>=level-.1*reference_atr)
            position=(c[j]-l[j] if sign>0 else h[j]-c[j])/max(h[j]-l[j],1e-15)
            if (touch and 0<sign*(c[j]-level)<=.75*reference_atr and sign*(c[j]-o[j])>0 and position>=.6):
                out.append(dict(index=j,direction_n=sign,level=level,breakout_at=int(t[opened])+900,
                    known_at=int(t[j])+900,atr=float(atr[j]),strength=abs(c[j]-o[j])/atr[j]))
                pending=None
            continue
        if not .0005<=atr[j]/c[j]<=.02:continue
        high=float(max(h[j-16:j]));low=float(min(l[j-16:j]))
        if c[j]>high+.1*atr[j] and fast[j]>slow[j] and c[j]>o[j]:pending=(1,high,atr[j],j)
        elif c[j]<low-.1*atr[j] and fast[j]<slow[j] and c[j]<o[j]:pending=(-1,low,atr[j],j)
    return out


def candidates(mark,trade,symbol,*,kind='retest16',target_fraction,regular_only,entry_stress=0.,trigger_kind='trade'):
    if kind!='retest16' or target_fraction not in (.01,.015,.02) or entry_stress not in (0.,.0005) or trigger_kind not in ('trade','mark'):
        raise ValueError('unknown retest hypothesis')
    lookup={int(t):i for i,t in enumerate(trade['time'])};rows=[]
    for setup in retest_signals(mark):
        j=setup['index'];stamp=mark['time'][j];entry_time=stamp+900;sign=setup['direction_n']
        if regular_only and not is_market_open(datetime.fromtimestamp(entry_time,timezone.utc)):continue
        i=lookup.get(stamp)
        if i is None or i<1 or list(trade['time'][i-1:i+1])!=[stamp-900,stamp]:continue
        if not math.isfinite(float(trade['volume'][i])) or trade['volume'][i]<=0:continue
        reference=float(trade['close'][i]);mark_reference=float(mark['close'][j])
        if not math.isfinite(reference) or reference<=0 or abs(reference/mark_reference-1)>.001:continue
        entry=reference*(1+sign*entry_stress)
        source=mark if trigger_kind=='mark' else trade;index=j if trigger_kind=='mark' else i
        stop=(min(source['low'][index-1:index+1])-.25*setup['atr'] if sign>0
              else max(source['high'][index-1:index+1])+.25*setup['atr'])
        risk=max(.003,sign*(entry-stop)/entry)
        if not math.isfinite(risk) or risk>.015 or target_fraction/risk<1.5:continue
        target=entry*(1+sign*target_fraction)
        rows.append(dict(symbol=symbol,entry_time=entry_time,entry_index=i+1,
            direction='LONG' if sign>0 else 'SHORT',direction_n=sign,entry=entry,
            sl=entry*(1-sign*risk),tp1=target,risk_pct=risk,target_fraction=target_fraction,
            target_pct=100*target_fraction,stop_pct=100*risk,signal_strength=setup['strength'],
            mark_reference=mark_reference,trade_reference=reference,known_at=entry_time,
            breakout_at=setup['breakout_at'],retest_level=setup['level'],trigger_kind=trigger_kind,
            module='mark_retest16',module_priority=0,hold_bars=HORIZON))
    return rows
