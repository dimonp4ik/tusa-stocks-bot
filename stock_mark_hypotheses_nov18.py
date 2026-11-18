"""Fixed causal mark features, trade-price levels and trade-trigger proxies."""
from datetime import datetime,timezone
import math
import numpy as np

from src.market_hours import is_market_open

KINDS=('break16','ema_resume','reclaim16')
LOOKBACK=128
HORIZON=32


def finite_ema(values,span):
    x=np.asarray(values,dtype=float);out=np.full(len(x),np.nan)
    if len(x)<LOOKBACK:return out
    alpha=2/(span+1)
    weights=np.r_[alpha*(1-alpha)**np.arange(LOOKBACK-1),(1-alpha)**(LOOKBACK-1)]
    out[LOOKBACK-1:]=np.convolve(x,weights,mode='valid')
    return out


def candidates(mark,trade,symbol,*,kind,target_fraction,regular_only,entry_stress=0.,trigger_kind='trade'):
    if kind not in KINDS or target_fraction not in (.01,.015,.02) or entry_stress not in (0.,.0005):
        raise ValueError('unknown mark hypothesis')
    if trigger_kind not in ('trade','mark'):raise ValueError('unknown trigger-price source')
    mt=np.asarray(mark['time'],dtype=np.int64)
    o,h,l,c=[np.asarray(mark[k],dtype=float) for k in ('open','high','low','close')]
    fast,slow=finite_ema(c,16),finite_ema(c,64)
    lookup={int(t):i for i,t in enumerate(trade['time'])}
    records=[]
    for j in range(LOOKBACK,len(mt)):
        stamp=int(mt[j]);entry_time=stamp+900
        if regular_only and not is_market_open(datetime.fromtimestamp(entry_time,timezone.utc)):continue
        if list(mt[j-LOOKBACK:j+1])!=list(range(stamp-LOOKBACK*900,stamp+900,900)):continue
        i=lookup.get(stamp)
        if i is None or i<3 or list(trade['time'][i-3:i+1])!=list(range(stamp-2700,stamp+900,900)):continue
        if not math.isfinite(float(trade['volume'][i])) or trade['volume'][i]<=0:continue
        trade_reference=float(trade['close'][i])
        if not math.isfinite(trade_reference) or trade_reference<=0 or abs(trade_reference/c[j]-1)>.001:continue
        atr=float(np.mean(np.maximum.reduce((h[j-13:j+1]-l[j-13:j+1],
            np.abs(h[j-13:j+1]-c[j-14:j]),np.abs(l[j-13:j+1]-c[j-14:j])))))
        if not .0005<=atr/c[j]<=.02:continue
        prior_high=float(max(h[j-16:j]));prior_low=float(min(l[j-16:j]))
        sign=None
        if kind=='break16':
            if c[j]>prior_high and fast[j]>slow[j] and c[j]>o[j]:sign=1
            elif c[j]<prior_low and fast[j]<slow[j] and c[j]<o[j]:sign=-1
        elif kind=='ema_resume':
            if fast[j]>slow[j] and c[j-1]<=fast[j-1] and c[j]>fast[j] and c[j]>o[j]:sign=1
            elif fast[j]<slow[j] and c[j-1]>=fast[j-1] and c[j]<fast[j] and c[j]<o[j]:sign=-1
        else:
            if l[j]<prior_low and c[j]>prior_low and c[j]>o[j]:sign=1
            elif h[j]>prior_high and c[j]<prior_high and c[j]<o[j]:sign=-1
        if sign is None:continue
        entry=trade_reference*(1+sign*entry_stress)
        levels=mark if trigger_kind=='mark' else trade;level_index=j if trigger_kind=='mark' else i
        lows=[float(v) for v in levels['low'][level_index-3:level_index+1]]
        highs=[float(v) for v in levels['high'][level_index-3:level_index+1]]
        stop=min(lows)-.25*atr if sign>0 else max(highs)+.25*atr
        risk=max(.003,sign*(entry-stop)/entry)
        if not math.isfinite(risk) or risk>.015 or target_fraction/risk<1.5:continue
        stop=entry*(1-sign*risk);target=entry*(1+sign*target_fraction)
        records.append(dict(symbol=symbol,entry_time=entry_time,entry_index=i+1,
            direction='LONG' if sign>0 else 'SHORT',direction_n=sign,
            entry=entry,sl=stop,tp1=target,risk_pct=risk,signal_strength=abs(c[j]-o[j])/atr,
            target_fraction=target_fraction,target_pct=100*target_fraction,stop_pct=100*risk,
            mark_reference=float(c[j]),trade_reference=trade_reference,known_at=entry_time,
            trigger_kind=trigger_kind,
            module='mark_'+kind,module_priority=0,hold_bars=HORIZON))
    return records


def resolve(row,trade,*,mark=None,exit_stress=0.):
    if exit_stress not in (0.,.0005):raise ValueError('unknown exit-price stress')
    trigger_kind=row.get('trigger_kind','trade')
    if trigger_kind=='mark' and mark is None:raise ValueError('mark_trigger_history_required')
    source=mark if trigger_kind=='mark' else trade
    i=int(np.searchsorted(source['time'],row['entry_time']));end=i+HORIZON
    expected=list(range(int(row['entry_time']),int(row['entry_time'])+HORIZON*900,900))
    if list(source['time'][i:end])!=expected:
        return dict(row,outcome='UNKNOWN_HISTORY',outcome_known=False)
    sign=int(row['direction_n']);entry,stop,target=(float(row[k]) for k in ('entry','sl','tp1'))
    last_price=None
    for j in range(i,end):
        o,h,l,c=[float(source[k][j]) for k in ('open','high','low','close')]
        if not all(math.isfinite(x) for x in (o,h,l,c)) or min(o,h,l,c)<=0 or not l<=min(o,c)<=max(o,c)<=h:
            return dict(row,outcome='UNKNOWN_PRICES',outcome_known=False)
        if trigger_kind=='trade':
            volume=float(source['volume'][j])
            if not math.isfinite(volume) or volume<0:return dict(row,outcome='UNKNOWN_VOLUME',outcome_known=False)
            if volume==0:continue
        last_price=c;code=None;price=None
        if sign*(o-stop)<=0:price,code=o,-1
        elif sign*(o-target)>=0:price,code=target,1
        elif (l<=stop if sign>0 else h>=stop):price,code=stop,-1
        elif (h>=target if sign>0 else l<=target):price,code=target,1
        if code is not None:break
    else:
        if last_price is None:return dict(row,outcome='UNKNOWN_NO_POST_ENTRY_TRADES',outcome_known=False)
        price,code=last_price,0;j=end-1
    fill=price*(1-sign*exit_stress)
    signed=sign*(fill/entry-1)
    return dict(row,outcome_known=True,exit_time=int(source['time'][j])+900,exit_reference=fill,
        exit_price_kind='hypothetical_'+trigger_kind+'_trigger_market_fill',
        signed_return=signed,net_r=signed/row['risk_pct'],outcome_code=code,
        outcome='TP' if code==1 else 'SL' if code==-1 else 'TIME')
