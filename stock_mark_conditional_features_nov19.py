"""Explicit entry-time-only features; no outcome columns enter the matrix."""
from datetime import datetime,timezone
import math
from zoneinfo import ZoneInfo

import numpy as np
from stock_mark_hypotheses_nov18 import finite_ema,HORIZON,LOOKBACK
from src.market_hours import is_market_open

FAMILIES=('mark_break16','mark_ema_resume','mark_reclaim16','mark_retest16')
HORIZONS=(1,2,4,16,64)
FEATURES=('atr_fraction','body_atr','close_location','favorable_wick_atr','adverse_wick_atr',
    'ema_gap_atr','distance_high16_atr','distance_low16_atr','range16_atr','range4_atr',
    'volume_ratio16','zero_volume_share16','basis_now','basis_abs_mean16','trade_atr_ratio',
    'risk_fraction','target_fraction','reward_risk','signal_strength','time_sin','time_cos',
    'weekday_sin','weekday_cos','regular','long',
    *(f'own_return_{n}' for n in HORIZONS),
    *(f'{symbol}_return_{n}' for symbol in ('QQQ','SPY') for n in (2,16,64)),
    'basket_return_2','basket_return_16','own_minus_qqq16','qqq_minus_spy16',
    *(f'family_{name}' for name in FAMILIES))
ET=ZoneInfo('America/New_York')


def atr_at(arrays,index):
    h,l,c=(arrays[k] for k in ('high','low','close'))
    return float(np.mean(np.maximum.reduce((h[index-13:index+1]-l[index-13:index+1],
        abs(h[index-13:index+1]-c[index-14:index]),abs(l[index-13:index+1]-c[index-14:index])))))


class FeatureContext:
    def __init__(self,datasets):
        self.data={}
        for symbol,(mark,trade) in datasets.items():
            m={k:np.asarray(v,dtype=float if k!='time' else np.int64) for k,v in mark.items()}
            t={k:np.asarray(v,dtype=float if k!='time' else np.int64) for k,v in trade.items()}
            if (np.diff(m['time'])<=0).any() or (np.diff(t['time'])<=0).any():raise ValueError('source_time_order')
            self.data[symbol]=dict(mark=m,trade=t,mark_lookup={int(x):i for i,x in enumerate(m['time'])},
                trade_lookup={int(x):i for i,x in enumerate(t['time'])},fast=finite_ema(m['close'],16),slow=finite_ema(m['close'],64))

    def _mark_returns(self,symbol,stamp):
        item=self.data[symbol];m=item['mark'];j=item['mark_lookup'].get(stamp)
        if j is None or j<64 or list(m['time'][j-64:j+1])!=list(range(stamp-64*900,stamp+900,900)):
            raise ValueError('missing_prior_context')
        return {n:float(m['close'][j]/m['close'][j-n]-1) for n in HORIZONS}

    def row(self,entry):
        symbol=entry['symbol'];stamp=int(entry['entry_time'])-900;sign=int(entry['direction_n'])
        if sign not in (-1,1) or entry['module'] not in FAMILIES:raise ValueError('invalid_entry_schema')
        item=self.data[symbol];m=item['mark'];t=item['trade']
        j=item['mark_lookup'].get(stamp);i=item['trade_lookup'].get(stamp)
        if (j is None or i is None or j<LOOKBACK or i<16
                or list(m['time'][j-LOOKBACK:j+1])!=list(range(stamp-LOOKBACK*900,stamp+900,900))
                or list(t['time'][i-15:i+1])!=list(range(stamp-15*900,stamp+900,900))):
            raise ValueError('missing_closed_feature_history')
        atr=atr_at(m,j);trade_atr=atr_at(t,i)
        if not math.isfinite(atr) or atr<=0:raise ValueError('invalid_feature_volatility')
        o,h,l,c=[float(m[k][j]) for k in ('open','high','low','close')]
        span=h-l
        volume=t['volume'][i-15:i+1];mean_volume=float(np.mean(volume))
        if not np.isfinite(volume).all() or (volume<0).any() or mean_volume<=0:raise ValueError('missing_trade_activity')
        basis=[]
        for cursor in range(j-15,j+1):
            own_time=int(m['time'][cursor]);trade_cursor=item['trade_lookup'].get(own_time)
            if trade_cursor is None:raise ValueError('missing_prior_basis')
            basis.append(float(t['close'][trade_cursor]/m['close'][cursor]-1))
        now=datetime.fromtimestamp(entry['entry_time'],timezone.utc);local=now.astimezone(ET)
        minute=local.hour*60+local.minute
        own=self._mark_returns(symbol,stamp)
        qqq=self._mark_returns('QQQUSDT',stamp);spy=self._mark_returns('SPYUSDT',stamp)
        peers=[self._mark_returns(peer,stamp) for peer in self.data if peer!=symbol]
        if not peers:raise ValueError('missing_basket_context')
        upper=h-max(o,c);lower=min(o,c)-l
        f=dict(atr_fraction=atr/c,body_atr=sign*(c-o)/atr,
            close_location=((c-l if sign>0 else h-c)/span if span>0 else .5),
            favorable_wick_atr=(upper if sign>0 else lower)/atr,
            adverse_wick_atr=(lower if sign>0 else upper)/atr,
            ema_gap_atr=sign*(item['fast'][j]-item['slow'][j])/atr,
            distance_high16_atr=(max(m['high'][j-16:j])-c)/atr,
            distance_low16_atr=(c-min(m['low'][j-16:j]))/atr,
            range16_atr=(max(m['high'][j-15:j+1])-min(m['low'][j-15:j+1]))/atr,
            range4_atr=(max(m['high'][j-3:j+1])-min(m['low'][j-3:j+1]))/atr,
            volume_ratio16=float(volume[-1]/mean_volume),zero_volume_share16=float(np.mean(volume==0)),
            basis_now=sign*basis[-1],basis_abs_mean16=float(np.mean(np.abs(basis))),trade_atr_ratio=trade_atr/atr,
            risk_fraction=float(entry['risk_pct']),target_fraction=float(entry['target_fraction']),
            reward_risk=float(entry['target_fraction']/entry['risk_pct']),signal_strength=float(entry['signal_strength']),
            time_sin=math.sin(2*math.pi*minute/1440),time_cos=math.cos(2*math.pi*minute/1440),
            weekday_sin=math.sin(2*math.pi*local.weekday()/7),weekday_cos=math.cos(2*math.pi*local.weekday()/7),
            regular=float(is_market_open(now)),long=float(sign>0))
        f.update({f'own_return_{n}':sign*own[n] for n in HORIZONS})
        for name,value in (('QQQ',qqq),('SPY',spy)):
            f.update({f'{name}_return_{n}':sign*value[n] for n in (2,16,64)})
        f.update(basket_return_2=sign*float(np.mean([p[2] for p in peers])),
            basket_return_16=sign*float(np.mean([p[16] for p in peers])),
            own_minus_qqq16=sign*(own[16]-qqq[16]),qqq_minus_spy16=sign*(qqq[16]-spy[16]))
        f.update({f'family_{family}':float(entry['module']==family) for family in FAMILIES})
        if set(f)!=set(FEATURES) or not all(math.isfinite(v) for v in f.values()):raise ValueError('invalid_feature_vector')
        return [float(f[name]) for name in FEATURES]


def mature_before(frame,cutoff):
    # A fixed full horizon purges quick known losers near the boundary together
    # with hypothetical winners whose remaining outcome is not yet available.
    return (frame.entry_time+HORIZON*900<cutoff)&frame.exit_time.lt(cutoff)&frame.outcome_known.eq(True)
