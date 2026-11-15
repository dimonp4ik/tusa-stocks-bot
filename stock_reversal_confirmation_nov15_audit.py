"""Small causal diagnostic grid; inspected historical months are not a holdout."""
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import stock_xperp_nov7_cluster_diagnostic as route
from stock_module_cluster_audit import portfolio_gate
from stock_momentum_phase_nov14_audit import metrics,window
from config import NATIVE_XPERP_DAILY_LOSS_R_CAP

ROOT=Path('reports/audit_2026_09_08')


def main():
    source=ROOT/'stock_xperp_nov14_preportfolio.pkl'
    manifest=ROOT/'stock_xperp_forward_nov14_merged/manifest.json'
    meta=json.loads(manifest.read_text())
    candles={}
    for symbol,item in meta['symbols'].items():
        if not item.get('path'):continue
        raw=Path(item['path']).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==item['sha256']
        candles[symbol]=pickle.loads(raw)
    frame=pd.read_pickle(source)
    latest=min(v['end'] for v in meta['symbols'].values() if v.get('end'))
    frame=frame[frame.entry_time.le(latest-63*900)].copy()
    frame=route.long_guard.apply_guard(route.premarket.apply_guard(frame))
    values=[]
    for row in frame.itertuples():
        c=candles[row.symbol];i=int(row.entry_index)-1
        assert int(c['time'][i])+900==int(row.entry_time)
        o,h,l,close=(float(c[k][i]) for k in ('open','high','low','close'))
        atr=row.atr14_fraction*close
        values.append(dict(confirm_body=row.direction_n*(close-o)>0,
            reject_extreme=(close-l)/(h-l)<=.8 if h>l and row.direction_n<0 else
                (close-l)/(h-l)>=.2 if h>l else False,
            body_at_most_atr=abs(close-o)<=atr,
            own_liquid=sum(float(v)>0 for v in c['volume'][i-7:i+1])>=6 and float(c['volume'][i])>0))
    extra=pd.DataFrame(values,index=frame.index)
    policies={
        'baseline':np.ones(len(frame),dtype=bool),
        'confirm_body':extra.confirm_body,
        'reject_extreme':extra.reject_extreme,
        'body_at_most_atr':extra.body_at_most_atr,
        'backing_within_stop':frame.backing_level_distance.le(.01),
        'own_liquid':extra.own_liquid,
        'confirm_body_and_liquid':extra.confirm_body & extra.own_liquid,
        'reject_extreme_and_liquid':extra.reject_extreme & extra.own_liquid,
    }
    report=dict(status='EXPLORATORY_NOT_LIVE_APPROVAL',objective_verified=False,
        limitations=['Months have already been inspected; these are not independent holdouts.',
            'Nominal standalone 3% margin/10x entry cohorts, not actual account PnL.',
            'No intraminute fills, spread reconstruction or contract rounding.',
            'Native contracts do not supply 2020-2025 history; July coverage is incomplete.'],results={})
    for name,mask in policies.items():
        part=frame[mask]
        rows=portfolio_gate(part.to_dict('records'),None,rolling_hour_cap=4,
            rolling_direction_hour_cap=2,rank_key=route.quality.ranker('z_high'),
            loss_streak_cap=2,direction_cap=4,daily_loss_r_cap=NATIVE_XPERP_DAILY_LOSS_R_CAP)
        result=pd.DataFrame(rows,columns=frame.columns)
        if name=='baseline':
            expected=pd.read_pickle(ROOT/'stock_factor_nov14_baseline.pkl')
            key=lambda d:set(zip(d.symbol,d.entry_time,d.direction))
            assert key(result)==key(expected),'baseline admission parity failed'
        report['results'][name]={m:metrics(window(result,a,b)) for m,a,b in (
            ('july_partial','2026-07-01','2026-08-01'),('august','2026-08-01','2026-09-01'),
            ('september','2026-09-01','2026-10-01'),('october','2026-10-01','2026-11-01'),
            ('november_partial','2026-11-01','2026-12-01'))}
        print(name,{m:round(v['return_pct'],2) for m,v in report['results'][name].items()},flush=True)
    report['sources']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (
        source,manifest,Path(__file__),Path('stock_module_cluster_audit.py'))}
    (ROOT/'stock_reversal_confirmation_nov15.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':main()
