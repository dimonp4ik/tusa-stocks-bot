"""Independent entry hypotheses on hashed caches. Offline, no live bot imports."""
import argparse
import csv
import hashlib
import json
import math
import pickle
from pathlib import Path
from filter_lab import gate, metrics, timestamp
from src.backtest_integrity import simulate_exit

FAMILIES=('range_reversion','trend_pullback','breakout')
TARGETS=(.25,.5,1.,2.)

def signal_at(c,i,family):
    # i is next execution bar; every feature ends at i-1.
    if i<100:return None
    closes=c['close'];last=closes[i-1]
    recent=closes[i-20:i];mean=sum(recent)/20
    std=math.sqrt(sum((x-mean)**2 for x in recent)/20)
    tr=[max(c['high'][j]-c['low'][j],abs(c['high'][j]-closes[j-1]),abs(c['low'][j]-closes[j-1])) for j in range(i-14,i)]
    atr=sum(tr)/14
    if atr<=0 or std<=0 or last<=0:return None
    path=sum(abs(closes[j]-closes[j-1]) for j in range(i-19,i))
    eff=abs(last-closes[i-20])/path if path else 0
    slow=sum(closes[i-96:i])/96
    z=(last-mean)/std
    direction=0
    if family=='range_reversion' and eff<.3:
        if z<=-2:direction=1
        elif z>=2:direction=-1
    elif family=='trend_pullback':
        if mean>slow and last<mean-.5*atr and last>slow and last>closes[i-2]:direction=1
        elif mean<slow and last>mean+.5*atr and last<slow and last<closes[i-2]:direction=-1
    elif family=='breakout':
        if last>max(c['high'][i-21:i-1]) and last>slow:direction=1
        elif last<min(c['low'][i-21:i-1]) and last<slow:direction=-1
    elif family=='sweep_reclaim':
        prior_low=min(c['low'][i-21:i-1]);prior_high=max(c['high'][i-21:i-1])
        o=c['open'][i-1];h=c['high'][i-1];l=c['low'][i-1]
        span=h-l
        if span>0:
            if l<prior_low and last>prior_low and last>o and (last-l)/span>=.7:direction=1
            elif h>prior_high and last<prior_high and last<o and (h-last)/span>=.7:direction=-1
    elif family=='exhaustion_reversal':
        # Closed reversal after two directional candles outside the local mean.
        if closes[i-4]>closes[i-3]>closes[i-2] and last>closes[i-2] and last<mean-std:direction=1
        elif closes[i-4]<closes[i-3]<closes[i-2] and last<closes[i-2] and last>mean+std:direction=-1
    if not direction:return None
    return direction,atr,eff,z

def cost_allows(entry,risk,target,max_fraction,roundtrip_cost=.0012):
    if not math.isfinite(roundtrip_cost) or roundtrip_cost<0:
        return False
    if max_fraction is None:return True
    if not all(math.isfinite(x) and x>0 for x in (entry,risk,target,max_fraction)):
        return False
    return roundtrip_cost*entry <= max_fraction*risk*target


def replay(c,symbol,family,target,max_cost_fraction=None,roundtrip_cost=.0012):
    rows=[]
    for i in range(100,len(c['time'])-48):
        signal=signal_at(c,i,family)
        if signal is None:continue
        if c['time'][i]-c['time'][i-1]!=900:continue
        direction,atr,eff,z=signal
        entry=float(c['open'][i]);risk=2*atr
        if entry<=0 or risk>=entry:continue
        if not cost_allows(entry,risk,target,max_cost_fraction,roundtrip_cost):continue
        sl=entry-direction*risk;tp=entry+direction*risk*target
        if tp<=0:continue
        side='LONG' if direction==1 else 'SHORT'
        exit=simulate_exit(c,range(i,i+48),direction=side,entry=entry,sl=sl,tp1=tp,tp2=tp,
            atr=atr,tp1_fraction=0,trail=False,trail_mult=0,stop_on_close=False,
            backstop_r=1,choose_trail=lambda *a:0)
        rows.append(dict(symbol=symbol,direction=side,entry_time=c['time'][i],
            exit_time=c['time'][exit.bar]+900,entry=entry,sl=sl,tp1=tp,
            outcome=exit.outcome,net_r=exit.gross_r-roundtrip_cost*entry/risk,
            risk_pct=risk/entry,size_mult=1,eff_ratio=eff,z=z))
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',default='filter_universe');p.add_argument('--cap',type=int,required=True)
    p.add_argument('--families',default=','.join(FAMILIES))
    p.add_argument('--max-cost-target-fraction',type=float)
    p.add_argument('--label',required=True)
    p.add_argument('--roundtrip-cost',type=float,default=.0012)
    p.add_argument('--cache-dir',type=Path,default=Path('backtest_cache'))
    a=p.parse_args()
    if not math.isfinite(a.roundtrip_cost) or a.roundtrip_cost<0:
        p.error("Round-trip cost must be finite and nonnegative")
    if a.max_cost_target_fraction is not None and not 0<a.max_cost_target_fraction<1:
        p.error('Cost/target fraction must be between zero and one')
    families=tuple(a.families.split(','))
    if not set(families)<=set(FAMILIES+('sweep_reclaim','exhaustion_reversal')):
        p.error('Unknown entry family')
    tag=a.manifest+('_'+a.label if a.label else '')
    if a.max_cost_target_fraction is not None and not a.label:
        p.error('Filtered experiments need a distinct label')
    folder=Path('reports/audit_2026_09_08')
    manifest=json.loads((folder/(a.manifest+'.json')).read_text())
    caches=[]
    for name,info in manifest['cache_manifest'].items():
        if info['interval_sec']!=900:continue
        raw=(a.cache_dir/name).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=info['sha256']:raise ValueError('Cache changed: '+name)
        caches.append((name.split('_')[0],pickle.loads(raw)))
    plan=dict(max_cost_target_fraction=a.max_cost_target_fraction,families=families,targets=TARGETS,stop_atr=2,timeout_bars=48,cost_roundtrip=a.roundtrip_cost,
        manifest_sha256=hashlib.sha256((folder/(a.manifest+'.json')).read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        status='EXPLORATORY_ONLY',notes=['Full final target exit, no partial TP1; alternative to trailing.',
        'Equal initial R per trade; do not compare totals directly to legacy size-weighted SMC.',
        'Previously inspected SWAP history, funding excluded, not independent live proof.',
        'Universe survivorship and historical gaps remain. Portfolio gates reapplied per variant.'])
    (folder/('entry_hypotheses_'+tag+'_plan.json')).write_text(json.dumps(plan,indent=2),encoding='utf-8')
    results=[]
    for family in families:
        for target in TARGETS:
            rows=[]
            for symbol,c in caches:rows.extend(replay(c,symbol,family,target,a.max_cost_target_fraction,a.roundtrip_cost))
            accepted=gate(sorted(rows,key=lambda x:(x['entry_time'],x['symbol'])),a.cap)
            train_end=timestamp('2026-06-01');val_end=timestamp('2026-07-01')
            record=dict(family=family,target_r=target,raw_n=len(rows),all=metrics(accepted),
                early=metrics([r for r in accepted if r['exit_time']<train_end]),
                june=metrics([r for r in accepted if train_end<=r['entry_time'] and r['exit_time']<val_end]),
                july_august=metrics([r for r in accepted if r['entry_time']>=val_end]))
            results.append(record)
            (folder/('entry_hypotheses_'+tag+'.json')).write_text(json.dumps({'plan':plan,'results':results},indent=2),encoding='utf-8')
            print(json.dumps({k:record[k] for k in ('family','target_r','all')}),flush=True)
            if accepted:
                with (folder/f'entry_{tag}_{family}_{target}.csv').open('w',newline='',encoding='utf-8') as f:
                    writer=csv.DictWriter(f,fieldnames=list(accepted[0]));writer.writeheader();writer.writerows(accepted)

if __name__=='__main__':main()
