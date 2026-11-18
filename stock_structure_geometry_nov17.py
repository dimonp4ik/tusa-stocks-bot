"""Frozen, causal single-target stock brackets; research only."""
import math


POLICIES=('baseline','swing2_original_target','swing8_original_target',
          'swing2_impulse_origin','swing8_impulse_origin','swing2_two_r','swing8_two_r',
          'confirmed_swing2_impulse_origin','confirmed_swing2_two_r')


def bracket(row,candles,policy):
    if policy not in POLICIES:raise ValueError('unknown structural policy')
    index=int(row['entry_index'])
    if index<8:raise ValueError('insufficient closed structural history')
    entry_time=int(row['entry_time'])
    if list(candles['time'][index-8:index])!=list(range(entry_time-8*900,entry_time,900)):
        raise ValueError('closed structural history is not aligned')
    entry=float(candles['close'][index-1]);sign=int(row['direction_n'])
    if sign not in (-1,1) or not math.isfinite(entry) or entry<=0:
        raise ValueError('invalid structural entry')
    if policy.startswith('confirmed_'):
        opening=float(candles['open'][index-1])
        if not math.isfinite(opening) or opening<=0:raise ValueError('invalid signal open')
        if sign*(entry-opening)<=0:return None
        policy=policy.removeprefix('confirmed_')
    if policy=='baseline':
        risk=.01;target_fraction=float(row['audited_target_pct'])/100
    else:
        lookback=2 if policy.startswith('swing2_') else 8
        highs=[float(v) for v in candles['high'][index-lookback:index]]
        lows=[float(v) for v in candles['low'][index-lookback:index]]
        if not all(math.isfinite(v) and v>0 for v in highs+lows):
            raise ValueError('invalid closed structural prices')
        atr=float(row['atr14_fraction'])*entry
        if not math.isfinite(atr) or atr<0:raise ValueError('invalid closed ATR')
        extreme=min(lows)-.25*atr if sign>0 else max(highs)+.25*atr
        risk=max(.003,sign*(entry-extreme)/entry)
        if risk>.02:return None
        if policy.endswith('original_target'):
            target_fraction=float(row['audited_target_pct'])/100
        elif policy.endswith('impulse_origin'):
            origin=float(candles['close'][index-3])
            move=sign*(origin/entry-1)
            if not math.isfinite(move) or move<=0:return None
            target_fraction=max(.005,min(.03,move))
        else:
            target_fraction=max(.005,min(.03,2*risk))
    if not .005<=target_fraction<=.03 or not math.isfinite(risk):
        raise ValueError('invalid target or stop fraction')
    return dict(entry=entry,stop=entry*(1-sign*risk),target=entry*(1+sign*target_fraction),
                risk_fraction=risk,target_fraction=target_fraction,direction_n=sign)


def resolve(row,candles,geometry):
    start=int(row['entry_index']);last=start+63
    if last>=len(candles['time']):raise ValueError('incomplete outcome horizon')
    if list(candles['time'][start:last+1])!=list(range(int(row['entry_time']),int(row['entry_time'])+64*900,900)):
        raise ValueError('noncontiguous outcome horizon')
    entry,stop,target=(geometry[k] for k in ('entry','stop','target'))
    sign=geometry['direction_n'];price=float(candles['close'][last]);code=0;end=last
    for i in range(start,last+1):
        o,h,l,c=(float(candles[k][i]) for k in ('open','high','low','close'))
        if not all(math.isfinite(v) and v>0 for v in (o,h,l,c)) or not l<=min(o,c)<=max(o,c)<=h:
            raise ValueError('invalid outcome candle')
        if (o<=stop if sign>0 else o>=stop):price,code,end=o,-1,i;break
        if (o>=target if sign>0 else o<=target):price,code,end=target,1,i;break
        if (l<=stop if sign>0 else h>=stop):price,code,end=stop,-1,i;break
        if (h>=target if sign>0 else l<=target):price,code,end=target,1,i;break
    signed=sign*(price/entry-1)
    return dict(row,entry=entry,sl=stop,tp1=target,risk_pct=geometry['risk_fraction'],
        target_pct=100*geometry['target_fraction'],stop_pct=100*geometry['risk_fraction'],
        net_r=signed/geometry['risk_fraction'],signed_return=signed,
        exit_time=int(candles['time'][end])+900,outcome_code=code,
        outcome='TP' if code==1 else 'SL' if code==-1 else 'TIME',
        direction='LONG' if sign>0 else 'SHORT',hold_bars=64,
        module='stock_structural_bracket_research',module_priority=0)
