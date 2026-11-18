"""Predeclared mark signals, matched triggers and recalculated portfolio gates."""
import collections
from datetime import datetime,timezone
import hashlib
import itertools
import json
from pathlib import Path
import pickle

import pandas as pd

from stock_mark_hypotheses_nov18 import candidates,resolve,KINDS,HORIZON
from stock_module_cluster_audit import portfolio_gate
from stock_structure_geometry_nov17_audit import readiness_masks
from config import NATIVE_XPERP_DAILY_LOSS_R_CAP

ROOT=Path('reports/audit_2026_09_08')
SOURCE=ROOT/'stock_mark_matched_nov18/manifest.json'


def protocol():
    return dict(protocol='matched_mark_hypotheses_nov18_v1',declared_at=datetime.now(timezone.utc).isoformat(),
        kinds=list(KINDS),targets=[.01,.015,.02],regular_only=[True,False],triggers=['trade','mark'],
        execution_price_stresses=[0.,.0005],horizon_bars=HORIZON,ema_closed_window=128,
        minimum_risk=.003,maximum_risk=.015,minimum_target_risk=1.5,mark_trade_agreement=.001,
        sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (SOURCE,Path('stock_mark_hypotheses_nov18.py'))},
        objective_verified=False,live_activation_supported=False)


def gate(rows):
    return portfolio_gate(rows,None,rolling_hour_cap=4,rolling_direction_hour_cap=2,
        rank_key=lambda r:-r['signal_strength'],loss_streak_cap=2,direction_cap=4,
        daily_loss_r_cap=NATIVE_XPERP_DAILY_LOSS_R_CAP)


def main():
    manifest=json.loads(SOURCE.read_text(encoding='utf-8'));datasets={};sources={}
    for symbol,meta in manifest['symbols'].items():
        if meta['status']!='OK':continue
        values=[]
        for field,digest_field in (('path','sha256'),('trade_path','trade_sha256')):
            p=Path(meta[field]);raw=p.read_bytes()
            if hashlib.sha256(raw).hexdigest()!=meta[digest_field]:raise ValueError('matched price source changed')
            sources[str(p)]=meta[digest_field];values.append(pickle.loads(raw))
        datasets[symbol]=values
    if len(datasets)!=11:raise ValueError('the declared eleven-contract source is incomplete')
    cutoff=min(min(m['time'][-1],t['time'][-1])+900 for m,t in datasets.values())-HORIZON*900
    declared_path=ROOT/'stock_mark_hypotheses_nov18_protocol.json'
    declaration=protocol()
    if declared_path.exists():
        existing=json.loads(declared_path.read_text(encoding='utf-8'))
        if existing['sources']!=declaration['sources']:raise ValueError('declared hypothesis source changed')
        declaration=existing
    else:declared_path.write_text(json.dumps(declaration,indent=2),encoding='utf-8')
    report=dict(status='EXPLORATORY_MARK_SIGNAL_AND_TRIGGER_HYPOTHESES',objective_verified=False,
        live_activation_supported=False,declared_protocol=declaration,mature_cohort_last_entry=cutoff,
        results={},limitations=[
            'Previously inspected underlying market history; new mark sources do not create an untouched holdout.',
            '36 fixed policies, two execution-price scenarios, no refit after their historical scores.',
            'Mark is an indicator/trigger reference, not an executable market price.',
            'Entry reference is a known last-trade close within 10bp of mark; no historical contemporaneous bid/ask.',
            'Mark-trigger scenarios are a different hypothetical execution contract from the current last-trigger LIVE bot.',
            'Price stresses are scenarios, not calibrated estimates or guaranteed execution bounds.',
            'Unknown outcome history or no subsequent trades makes the policy unscorable, never a free flat payoff.',
            'Stop-first ranges, gap-aware loss and target-price cap; no ordered ticks or publication latency.',
            'Time exits use a last public-price proxy, not a confirmed market fill.',
            'Portfolio and readiness are recomputed for each policy; lot-rounded shared cash is separate.',
            'July and November partial; no permission to enable orders and no promise of monthly profit.'])
    for kind,target,regular,trigger in itertools.product(KINDS,(.01,.015,.02),(True,False),('trade','mark')):
        for stress in (0.,.0005):
            name=f'{kind}_target{target}_regular{int(regular)}_{trigger}_stress{stress}'
            rows=[];unknown=[]
            for symbol,(mark,trade) in datasets.items():
                events=candidates(mark,trade,symbol,kind=kind,target_fraction=target,regular_only=regular,
                    entry_stress=stress,trigger_kind=trigger)
                for r in events:
                    if r['entry_time']>cutoff:continue
                    out=resolve(r,trade,mark=mark,exit_stress=stress)
                    if out['outcome_known']:rows.append(out)
                    else:unknown.append(dict(symbol=symbol,entry_time=r['entry_time'],reason=out['outcome']))
            if unknown:
                report['results'][name]=dict(status='UNSCORABLE_UNKNOWN_OUTCOMES',unknown=unknown)
                print(name,'unscorable',len(unknown),flush=True);continue
            accepted=gate(rows)
            f=pd.DataFrame(accepted)
            if not len(f):
                report['results'][name]=dict(status='NO_ADMITTED_SIGNALS',candidates=len(rows),signals=0)
                print(name,'no_signals',flush=True);continue
            masks=readiness_masks(f)
            f=f.assign(forward_r_ready=masks.r_ready,forward_price_ready=masks.price_ready)
            output=ROOT/f'stock_mark_hypotheses_nov18_{name}.pkl';f.to_pickle(output)
            months={}
            for month in (7,8,9,10,11):
                start=datetime(2026,month,1,tzinfo=timezone.utc).timestamp()
                end=datetime(2026,month+1,1,tzinfo=timezone.utc).timestamp()
                selected=f[f.entry_time.ge(start)&f.exit_time.lt(end)]
                months[f'2026-{month:02}']=dict(signals=len(selected),sum_price_return=float(selected.signed_return.sum()),
                    profitable_exit_pct=100*float(selected.signed_return.gt(0).mean()) if len(selected) else None,
                    entries_per_calendar_day=len(selected)/((end-start)/86400))
            report['results'][name]=dict(status='RESOLVED_REFERENCE_PATHS',candidates=len(rows),signals=len(f),
                r_and_price_ready=int((masks.r_ready&masks.price_ready).sum()),monthly_signal_diagnostics=months,
                trades_path=str(output),trades_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
            print(name,dict(signals=len(f),ready=int((masks.r_ready&masks.price_ready).sum()),
                price_sums={k:round(v['sum_price_return'],4) for k,v in months.items()}),flush=True)
    for p in (SOURCE,Path(__file__),Path('stock_mark_hypotheses_nov18.py'),Path('stock_module_cluster_audit.py'),
        Path('stock_structure_geometry_nov17_audit.py'),Path('src/market_hours.py')):
        sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    report['sources']=sources
    (ROOT/'stock_mark_hypotheses_nov18.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')


if __name__=='__main__':main()
