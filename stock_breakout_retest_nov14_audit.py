"""First-retest market-entry study; geometry frozen before later intervals."""
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
from config import NATIVE_XPERP_DAILY_LOSS_R_CAP
from stock_breakout_retest_research import retest_entries
from stock_momentum_phase_nov14_audit import ROOT,window,metrics
from stock_module_cluster_audit import portfolio_gate
import stock_xperp_native_growth_forward_monitor as market


def main():
    manifest=ROOT/'stock_xperp_forward_nov14_merged/manifest.json'
    candles,sources,_=market.load_manifest(manifest)
    arrays={s:{k:np.asarray(v) for k,v in c.items()} for s,c in candles.items()}
    last=min(c['time'][-1] for c in candles.values())
    frames,variants={},[]
    for wait,volume in itertools.product((2,4,8),(1.,1.5)):
        pool=[]
        for symbol,candle in arrays.items():
            pool.extend(retest_entries(candle,symbol,maximum_wait=wait,minimum_volume_ratio=volume))
        pool=[r for r in pool if r['entry_time']<=last-63*900]
        resolved=[]
        for row in pool:
            result=market.resolve_row(row,arrays[row['symbol']],row['risk_pct']*200,row['risk_pct']*100,64)
            result['module']='structural_breakout_retest_research'
            resolved.append(result)
        admitted=portfolio_gate(resolved,None,rolling_hour_cap=4,rolling_direction_hour_cap=2,
            rank_key=lambda r:-r['volume_ratio'],loss_streak_cap=2,direction_cap=4,
            daily_loss_r_cap=NATIVE_XPERP_DAILY_LOSS_R_CAP)
        frame=pd.DataFrame(admitted)
        if frame.empty:
            continue
        key=f'wait_{wait}_volume_{volume}'
        frames[key]=frame
        selection={m:metrics(window(frame,a,b)) for m,a,b in (
            ('july','2026-07-01','2026-08-01'),('august','2026-08-01','2026-09-01'))}
        variants.append(dict(key=key,maximum_wait=wait,minimum_volume_ratio=volume,
            candidates=len(pool),selection=selection))
        print(key,flush=True)
    eligible=[v for v in variants if all(x['resolved']>=30 for x in v['selection'].values())]
    best=max(eligible,key=lambda v:(min(x['return_pct'] for x in v['selection'].values()),
        -max(x['dd_pct'] for x in v['selection'].values()))) if eligible else None
    report=dict(status='RESEARCH_ONLY_NO_LIVE_CHANGES',objective_verified=False,
        selected=best,variants=variants,margin_fraction=.03,leverage=10,target_r=2,
        daily_loss_r_cap=NATIVE_XPERP_DAILY_LOSS_R_CAP,
        limitations=['Eligible immediate-breakout anchors only; not every possible level retest',
            'Historical periods were examined previously; November motivated the hypothesis',
            'Purged entry cohorts, not full marked account-month returns',
            'November 11 cohort starts at 00:00 UTC and overlaps prior snapshot hours',
            'Standalone nominal stock sizing; shared account, lot rounding and execution spread not simulated',
            'July is partial coverage; no native-contract data before mid-2026'],sources=sources)
    if best:
        frame=frames[best['key']]
        report['later']={name:metrics(window(frame,a,b)) for name,a,b in (
            ('september','2026-09-01','2026-10-01'),('october','2026-10-01','2026-11-01'),
            ('november_partial','2026-11-01','2026-12-01'),('since_nov11','2026-11-11','2026-12-01'))}
        frame.to_pickle(ROOT/'stock_breakout_retest_nov14_selected.pkl')
    report['sources'].update({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (
        Path(__file__),Path('stock_breakout_retest_research.py'),
        Path('stock_structural_breakout_research.py'),Path('stock_module_cluster_audit.py'))})
    (ROOT/'stock_breakout_retest_nov14.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('variants','sources')},indent=2))


if __name__=='__main__':
    main()
