"""Causal relative-strength overlay research; no runtime bot imports or network."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path
from filter_lab import load,gate,metrics,timestamp

RULES=('baseline','relative_1d','relative_4h','market_1d','relative_and_market',
       'relative_1d_contrarian','quiet_market','strong_market')

def closed_returns(c,decision):
    # Exact close timestamps prevent stale data and candles open at decision.
    lookup=c['closed_lookup']
    now=lookup.get(decision)
    before4=lookup.get(decision-14400);before24=lookup.get(decision-86400)
    if now is None or before4 is None or before24 is None:return None
    if min(now,before4,before24)<=0:return None
    return now/before4-1,now/before24-1

def allow(rule,row,features):
    if rule=='baseline':return True
    if features is None:return False
    own4,own24,market4,market24=features
    side=1 if row['direction']=='LONG' else -1
    relative=side*(own24-market24)>0
    if rule=='relative_1d':return relative
    if rule=='relative_4h':return side*(own4-market4)>0
    if rule=='market_1d':return side*market24>0
    if rule=='relative_and_market':return relative and side*market24>0
    if rule=='relative_1d_contrarian':return side*(own24-market24)<0
    if rule=='quiet_market':return abs(market24)<.01
    if rule=='strong_market':return abs(market24)>=.01 and side*market24>0
    raise ValueError(rule)

def main():
    p=argparse.ArgumentParser();p.add_argument('--benchmark',required=True);p.add_argument('--cap',required=True,type=int)
    p.add_argument('--manifest',default='filter_universe');a=p.parse_args()
    folder=Path('reports/audit_2026_09_08');mp=folder/(a.manifest+'.json');rp=folder/(a.manifest+'_raw.csv')
    manifest=json.loads(mp.read_text());caches={}
    for name,info in manifest['cache_manifest'].items():
        if info['interval_sec']!=900:continue
        raw=(Path('backtest_cache')/name).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=info['sha256']:raise ValueError(name)
        data=pickle.loads(raw)
        caches[name.split('_')[0]]={'closed_lookup':{t+900:v for t,v in zip(data['time'],data['close'])}}
    benchmark=caches[a.benchmark];rows=load(rp);features={}
    for i,row in enumerate(rows):
        own=closed_returns(caches[row['symbol']],row['entry_time']);market=closed_returns(benchmark,row['entry_time'])
        features[i]=(*own,*market) if own and market else None
    t=timestamp('2026-06-01');v=timestamp('2026-07-01');results=[]
    for rule in RULES:
        accepted=gate([r for i,r in enumerate(rows) if allow(rule,r,features[i])],a.cap)
        results.append(dict(rule=rule,early=metrics([r for r in accepted if r['exit_time']<t]),
            june=metrics([r for r in accepted if t<=r['entry_time'] and r['exit_time']<v]),
            july_august=metrics([r for r in accepted if r['entry_time']>=v]),all=metrics(accepted)))
    # Predeclared eligibility on early + June only; evaluation does not rescue failures.
    eligible=[x for x in results if x['rule']!='baseline' and x['early']['n']>=60 and x['june']['n']>=20
              and x['early']['stress_net_r']>0 and x['june']['stress_net_r']>0]
    chosen=max(eligible,key=lambda x:x['june']['daily_lcb'],default=None)
    report=dict(status='RESEARCH_ONLY',benchmark=a.benchmark,missing_context=sum(x is None for x in features.values()),
        candidates=len(RULES)-1,selected_on_early_and_june=chosen,results=results,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        raw_sha256=hashlib.sha256(rp.read_bytes()).hexdigest(),manifest_sha256=hashlib.sha256(mp.read_bytes()).hexdigest(),
        limitations=['Previously inspected history, not untouched holdout. Funding excluded.',
        'SWAP is not actual X-Perp execution venue. Relative strength is unadjusted for beta.',
        'Exact close timestamps required at decision, 4h and 24h ago; missing context skips filtered entries.',
        'Underlying SMC features and legacy risk weights retained; R totals are not account returns.'])
    (folder/('relative_strength_'+a.manifest+'.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
