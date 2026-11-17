"""Apply the frozen hypothesis to first-seen archived future observations."""
import hashlib
import json
import math
from pathlib import Path

import stock_close_anchor_nov16_audit as anchor
from stock_structure_geometry_nov17 import bracket
from stock_volatility_phase_nov18 import feature_pool,prepare_entries
from src.structure_observation_store import frame_hash
from src.stock_structure_observer import NATIVE_SOURCE_SHA

PROTOCOL=Path('STOCK_NORMALIZED_SHOCK_WATCH_V1.json')
FOLDER=Path('reports/structure_observer_prospective_nov18')


def inspect_snapshot(scan,protocol):
    bucket=scan['bucket']
    if bucket<protocol['first_eligible_bucket']:return dict(bucket=bucket,status='before_protocol')
    summary=scan.get('summary') or {};rows=scan['observations']
    if (scan['status']!='complete' or not summary.get('history_complete') or
            summary.get('native_source_sha256')!=NATIVE_SOURCE_SHA):
        return dict(bucket=bucket,status='incomplete_or_different_protocol')
    frames={};observed={}
    for row in rows:
        f=row.get('closed_frame');ref=row['payload'].get('frame')
        if not f or not ref or frame_hash(f)!=ref['sha256']:
            return dict(bucket=bucket,status='missing_or_changed_archived_history')
        t=f['time']
        if len(t)<64 or t[-1]!=bucket-900 or any(b-a!=900 for a,b in zip(t,t[1:])):
            return dict(bucket=bucket,status='unaligned_archived_history')
        frames[row['symbol']]=f;observed[row['symbol']]=row
    if len(frames)!=summary['expected']:return dict(bucket=bucket,status='incomplete_universe')
    native=anchor.causal_candidates(frames)
    native=native[native.entry_time.eq(bucket)]
    pool=feature_pool(frames);context=anchor.context_table(frames)
    normalized=prepare_entries(pool,context,'normalized_z','reversal')
    normalized=normalized[normalized.entry_time.eq(bucket)]
    strategies={}
    for name,candidates in (('native_swing2_original_target',native),
        ('normalized_z_reversal_swing2_original_target',normalized)):
        values=[]
        for r in candidates.to_dict('records'):
            g=bracket(r,frames[r['symbol']],'swing2_original_target')
            if g is None:continue
            observation=observed[r['symbol']];payload=observation['payload'];book=payload['book']
            quote=None;between=False;risk=None
            if observation['quote_ok'] and book['ok']:
                quote=book['ask'] if g['direction_n']>0 else book['bid']
                risk=g['direction_n']*(quote-g['stop'])/quote
                between=risk>0 and g['direction_n']*(g['target']-quote)>0
            age=payload['received_at']-bucket
            values.append(dict(symbol=r['symbol'],instrument=observation['instrument'],
                direction='LONG' if g['direction_n']>0 else 'SHORT',geometry=g,
                decision_at=summary['decision_at'],quote_received_at=payload['received_at'],
                reference_quote=quote,quote_risk_fraction=risk,quote_between_levels=between,
                quote_age_from_signal=age,quote_within_signal_age=0<=age<=180,
                portfolio_admission_unverified=True,private_fill_unverified=True))
        strategies[name]=values
    return dict(bucket=bucket,status='observed',instruments=len(frames),
        valid_books=sum(bool(r['quote_ok']) for r in rows),strategies=strategies)


def main():
    protocol=json.loads(PROTOCOL.read_text(encoding='utf-8'))
    for name,digest in protocol['code'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:
            raise ValueError('frozen_hypothesis_source_changed: '+name)
    output=[];sources={}
    for p in sorted(FOLDER.glob('bucket_*.json')):
        raw=p.read_bytes();scan=json.loads(raw)
        output.append(inspect_snapshot(scan,protocol));sources[str(p)]=hashlib.sha256(raw).hexdigest()
    report=dict(status='FROZEN_PROSPECTIVE_CANDIDATE_OBSERVATIONS_NOT_PNL',objective_verified=False,
        live_activation_supported=False,first_eligible_bucket=protocol['first_eligible_bucket'],rows=output,
        observed_windows=sum(r['status']=='observed' for r in output),sources=sources,
        limitations=['Candidate and quote evidence only, no virtual portfolio admission or outcomes yet.',
            'Structural bracket remains based on known close; observed quote reference is recorded separately.',
            'One/few windows do not establish winrate, monthly profit, liquidity or execution feasibility.'])
    target=FOLDER/'frozen_hypothesis_observations.json'
    target.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(observed_windows=report['observed_windows'],windows=[dict(bucket=r['bucket'],
        status=r['status'],candidates={k:len(v) for k,v in r.get('strategies',{}).items()}) for r in output])),flush=True)


if __name__=='__main__':main()
