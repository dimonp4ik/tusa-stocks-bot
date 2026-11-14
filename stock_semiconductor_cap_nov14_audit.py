"""Reroute current stock candidates with an optional same-benchmark ETF cap."""
import hashlib
import json
from pathlib import Path
import pandas as pd
import stock_xperp_nov7_cluster_diagnostic as route
from stock_module_cluster_audit import portfolio_gate
from stock_factor_exposure_research import semiconductor_proxy_key,SOURCE
from stock_momentum_phase_nov14_audit import metrics,window
from config import NATIVE_XPERP_DAILY_LOSS_R_CAP


def main():
    root=route.ROOT
    path=root/'stock_xperp_nov14_preportfolio.pkl'
    raw=pd.read_pickle(path)
    old=pd.read_pickle(root/'stock_xperp_nov11_preportfolio.pkl')
    old_meta=json.loads((root/'stock_xperp_forward_nov11_merged/manifest.json').read_text())
    old_end=min(v['end'] for v in old_meta['symbols'].values() if 'end' in v)
    parity=route.parity(old,raw,old_end-64*900)
    meta=json.loads((root/'stock_xperp_forward_nov14_merged/manifest.json').read_text())
    end=min(v['end'] for v in meta['symbols'].values() if 'end' in v)
    raw=raw[raw.entry_time.le(end-63*900)].copy()
    guarded=route.long_guard.apply_guard(route.premarket.apply_guard(raw))
    outputs={}
    report=dict(status='RESEARCH_ONLY_NOT_LIVE_POLICY',objective_verified=False,
        reference=SOURCE,historical_prefix_parity=parity,
        note='SOXL/SOXS mapping uses daily benchmark direction, not a guaranteed intraday hedge or beta estimate',
        limitations=['Purged nominal standalone stock cohorts, not shared account PnL',
            'Hypothesis uses previously inspected eras; no pristine forward approval',
            'Historical order-book spread, exact publication latency and lot rounding not simulated'],results={})
    for name,cap in (('baseline',None),('etf_same_benchmark_cap1',1)):
        values=portfolio_gate(guarded.to_dict('records'),None,rolling_hour_cap=4,
            rolling_direction_hour_cap=2,rank_key=route.quality.ranker('z_high'),
            loss_streak_cap=2,direction_cap=4,daily_loss_r_cap=NATIVE_XPERP_DAILY_LOSS_R_CAP,
            exposure_key=semiconductor_proxy_key,exposure_cap=cap)
        frame=pd.DataFrame(values,columns=guarded.columns)
        outputs[name]=frame
        frame.to_pickle(root/f'stock_factor_nov14_{name}.pkl')
        report['results'][name]={m:metrics(window(frame,a,b)) for m,a,b in (
            ('july','2026-07-01','2026-08-01'),('august','2026-08-01','2026-09-01'),
            ('september','2026-09-01','2026-10-01'),('october','2026-10-01','2026-11-01'),
            ('november_partial','2026-11-01','2026-12-01'))}
    def ids(frame):
        return set(zip(frame.symbol,frame.entry_time,frame.direction))
    a,b=[ids(outputs[k]) for k in ('baseline','etf_same_benchmark_cap1')]
    report['removed']=len(a-b);report['added_by_replacement']=len(b-a)
    report['sources']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (
        path,Path(__file__),Path('stock_factor_exposure_research.py'),Path('stock_module_cluster_audit.py'))}
    (root/'stock_semiconductor_cap_nov14.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='sources'},indent=2))


if __name__=='__main__':
    main()
