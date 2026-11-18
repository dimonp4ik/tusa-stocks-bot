"""Shared research runner for fixed, source-pinned native mark hypotheses."""
from datetime import datetime,timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import pickle
import re

import pandas as pd
from stock_mark_hypotheses_nov18 import resolve,HORIZON
from stock_mark_hypotheses_nov18_audit import gate
from stock_structure_geometry_nov17_audit import readiness_masks

ROOT=Path('reports/audit_2026_09_08')
SOURCE=ROOT/'stock_mark_matched_nov18/manifest.json'


def run_study(builder,*,kind,prefix,parameters,code_paths):
    if re.fullmatch('[a-z0-9_]+',prefix) is None:raise ValueError('invalid research prefix')
    manifest=json.loads(SOURCE.read_text(encoding='utf-8'));data={};hashes={}
    for symbol,meta in manifest['symbols'].items():
        if meta['status']!='OK':raise ValueError('declared native source is incomplete')
        values=[]
        for path_key,sha_key in (('path','sha256'),('trade_path','trade_sha256')):
            p=Path(meta[path_key]);raw=p.read_bytes()
            if hashlib.sha256(raw).hexdigest()!=meta[sha_key]:raise ValueError('matched source changed')
            hashes[str(p)]=meta[sha_key];values.append(pickle.loads(raw))
        data[symbol]=values
    cutoff=min(min(m['time'][-1],t['time'][-1])+900 for m,t in data.values())-HORIZON*900
    protocol_path=ROOT/f'{prefix}_protocol.json'
    source_hashes={str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [SOURCE,*code_paths,Path(__file__)]}
    now=datetime.now(timezone.utc).timestamp()
    protocol=dict(declared_at=datetime.fromtimestamp(now,timezone.utc).isoformat(),
        first_eligible_observer_bucket=math.ceil(now/900)*900,parameters=parameters,sources=source_hashes)
    if protocol_path.exists():
        old=json.loads(protocol_path.read_text(encoding='utf-8'))
        if old['sources']!=protocol['sources'] or old['parameters']!=parameters:raise ValueError('declared study changed')
        protocol=old
    else:protocol_path.write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    report=dict(status='EXPLORATORY_MATCHED_MARK_RETEST_PATHS',objective_verified=False,
        live_activation_supported=False,protocol=protocol,mature_cohort_last_entry=cutoff,results={},limitations=[
            'Exploratory repair after inspecting the initial breakout family; no pristine historical validation.',
            'Only fixed declared parameters; confirmed closed retest, market-reference entry and no limit entry.',
            'Known last-trade/mark agreement is not executable historical bid/ask; latency is absent.',
            'Mark-trigger scenarios are a different hypothetical order contract from current LIVE last-trigger orders.',
            'Unknown outcome history is unscorable; no missing outcomes are credited as flat or winning.',
            'Stop-first and opening-gap proxies; market fill stress is an assumption, not an execution bound.',
            'Portfolio and R/price readiness are recomputed; shared cash is a separate counterfactual.',
            'Previously inspected market eras, partial July/November, no monthly profit forecast.'])
    for target,regular,trigger,stress in itertools.product((.01,.015,.02),(True,False),('trade','mark'),(0.,.0005)):
        name=f'{kind}_target{target}_regular{int(regular)}_{trigger}_stress{stress}'
        rows=[];unknown=[]
        for symbol,(mark,trade) in data.items():
            events=builder(mark,trade,symbol,kind=kind,target_fraction=target,regular_only=regular,
                trigger_kind=trigger,entry_stress=stress)
            for r in events:
                if r['entry_time']>cutoff:continue
                result=resolve(r,trade,mark=mark,exit_stress=stress)
                if result['outcome_known']:rows.append(result)
                else:unknown.append(dict(symbol=symbol,entry_time=r['entry_time'],reason=result['outcome']))
        if unknown:
            report['results'][name]=dict(status='UNSCORABLE_UNKNOWN_OUTCOMES',unknown=unknown)
            print(name,'unscorable',len(unknown),flush=True);continue
        admitted=gate(rows);f=pd.DataFrame(admitted)
        if not len(f):
            report['results'][name]=dict(status='NO_ADMITTED_SIGNALS',signals=0,candidates=len(rows));continue
        masks=readiness_masks(f);f=f.assign(forward_r_ready=masks.r_ready,forward_price_ready=masks.price_ready)
        path=ROOT/f'{prefix}_{name}.pkl';f.to_pickle(path)
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
            trades_path=str(path),trades_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        print(name,dict(signals=len(f),ready=int((masks.r_ready&masks.price_ready).sum()),
            price_sums={k:round(v['sum_price_return'],4) for k,v in months.items()}),flush=True)
    report['sources']=hashes|source_hashes
    (ROOT/f'{prefix}.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
