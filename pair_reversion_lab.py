"""Fixed-pair, equal-notional research; never imports or sends live orders."""
import argparse, hashlib, json, math, pickle, statistics
from pathlib import Path

def net_return(a0,b0,a1,b1,side,cost=.0006):
    # Unit initial gross notional: half in each leg, fixed quantities.
    return .5*side*(a1/a0-b1/b0)-.5*cost*(2+a1/a0+b1/b0)

def replay(a,b,threshold):
    ia={t:i for i,t in enumerate(a['time'])};ib={t:i for i,t in enumerate(b['time'])}
    ts=sorted(set(ia)&set(ib));rows=[];i=168
    while i<len(ts)-49:
        window=ts[i-168:i]
        if ts[i]-window[0]!=168*3600:i+=1;continue
        ratios=[math.log(a['close'][ia[t]]/b['close'][ib[t]]) for t in window]
        mean=statistics.mean(ratios);sd=statistics.pstdev(ratios)
        if sd<=0:i+=1;continue
        z=(ratios[-1]-mean)/sd
        if not threshold<=abs(z)<5:i+=1;continue
        # Entire entry/holding window must be contiguous for both legs.
        if ts[i+48]-ts[i]!=48*3600:i+=1;continue
        side=-1 if z>0 else 1
        a0=a['open'][ia[ts[i]]];b0=b['open'][ib[ts[i]]]
        # Gross convergence approximation must cover base costs at least four times.
        if .5*(abs(z)-.5)*sd < 4*.0012:i+=1;continue
        worst=0.;outcome='TIMEOUT';k=i+47
        for j in range(i,i+48):
            ac=a['close'][ia[ts[j]]];bc=b['close'][ib[ts[j]]]
            worst=min(worst,net_return(a0,b0,ac,bc,side))
            live_z=(math.log(ac/bc)-mean)/sd
            if side*live_z>=-.5:outcome='CONVERGED';k=j;break
            if side*live_z<=-5:outcome='DIVERGED';k=j;break
        # Both legs exit next hour's open; no same-close execution assumption.
        a1=a['open'][ia[ts[k+1]]];b1=b['open'][ib[ts[k+1]]]
        net=net_return(a0,b0,a1,b1,side)
        rows.append(dict(entry_time=ts[i],exit_time=ts[k+1],net=net,
                         stress=net_return(a0,b0,a1,b1,side,.0011),
                         worst_hour_close=min(worst,net),outcome=outcome))
        i=k+2
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--pairs',required=True);args=p.parse_args()
    folder=Path('reports/audit_2026_09_08');mp=folder/(args.manifest+'.json');d=json.loads(mp.read_text());cache={}
    for name,m in d['cache_manifest'].items():
        if m['interval_sec']!=3600:continue
        raw=(Path('backtest_cache')/name).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=m['sha256']:raise ValueError(name)
        cache[name.split('_')[0]]=pickle.loads(raw)
    plan=dict(pairs=args.pairs,thresholds=[2,3],formation_hours=168,exit_z=.5,stop_z=5,
              max_hold_hours=48,entry_notional_per_leg=.5,fee_and_slippage_per_side=.0006,
              source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              manifest_sha256=hashlib.sha256(mp.read_bytes()).hexdigest())
    out=dict(plan=plan,status='RESEARCH_ONLY',results=[],limitations=[
        'Prespecified economic pairs, not a replication of Gatev distance selection.',
        'Previously inspected SWAP caches, not actual X-Perp executions; funding excluded.',
        'Two-leg non-atomic execution and intrahour loss excursions not modeled.',
        'No cointegration claim; equal notional does not remove all market exposure.',
        'Totals are additive returns on unit gross notional per pair, not compounded account PnL.'])
    for pair in args.pairs.split(','):
        x,y=pair.split(':')
        if x not in cache or y not in cache:raise ValueError('Missing pair '+pair)
        for threshold in (2,3):
            rows=replay(cache[x],cache[y],threshold);wins=sum(r['net']>0 for r in rows)
            equity=peak=dd=0
            for r in rows:
                dd=max(dd,peak-equity-r['worst_hour_close']);equity+=r['net'];peak=max(peak,equity)
            result=dict(pair=pair,threshold=threshold,n=len(rows),wr=100*wins/len(rows) if rows else 0,
                        net_pct=100*equity,stress_pct=100*sum(r['stress'] for r in rows),
                        hour_marked_dd_pct=100*dd,trades=rows)
            out['results'].append(result);print(json.dumps({k:v for k,v in result.items() if k!='trades'}),flush=True)
    (folder/('pair_reversion_'+args.manifest+'.json')).write_text(json.dumps(out,indent=2),encoding='utf-8')
if __name__=='__main__':main()
