"""All candidate outcomes before portfolio selection, with entry-only features."""
import hashlib
import json
from pathlib import Path
import pickle

import pandas as pd

from stock_mark_hypotheses_nov18 import candidates,resolve,KINDS,HORIZON
from stock_mark_retest_nov18 import candidates as retest_candidates
from stock_mark_conditional_features_nov19 import FeatureContext,FEATURES

ROOT=Path('reports/audit_2026_09_08')
SOURCE=ROOT/'stock_mark_matched_nov18/manifest.json'
CACHE=ROOT/'stock_mark_conditional_nov19_pool.pkl'
META=ROOT/'stock_mark_conditional_nov19_pool_sources.json'


def load_pool():
    code=[Path(__file__),Path('stock_mark_conditional_features_nov19.py'),
        Path('stock_mark_hypotheses_nov18.py'),Path('stock_mark_retest_nov18.py')]
    provenance={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [SOURCE,*code]}
    if CACHE.exists() or META.exists():
        if not CACHE.exists() or not META.exists():raise ValueError('incomplete_conditional_cache')
        meta=json.loads(META.read_text(encoding='utf-8'))
        if meta['sources']!=provenance or hashlib.sha256(CACHE.read_bytes()).hexdigest()!=meta['sha256']:
            raise ValueError('conditional_cache_source_changed')
        return pd.read_pickle(CACHE),meta
    manifest=json.loads(SOURCE.read_text(encoding='utf-8'));datasets={};data_hashes={}
    for symbol,item in manifest['symbols'].items():
        if item['status']!='OK':raise ValueError('declared_contract_source_incomplete')
        pair=[]
        for key,sha_key in (('path','sha256'),('trade_path','trade_sha256')):
            path=Path(item[key]);raw=path.read_bytes()
            if hashlib.sha256(raw).hexdigest()!=item[sha_key]:raise ValueError('conditional_price_source_changed')
            data_hashes[str(path)]=item[sha_key];pair.append(pickle.loads(raw))
        datasets[symbol]=pair
    context=FeatureContext(datasets)
    cutoff=min(min(m['time'][-1],t['time'][-1])+900 for m,t in datasets.values())-HORIZON*900
    records=[];feature_cache={}
    target_index=FEATURES.index('target_fraction');rr_index=FEATURES.index('reward_risk')
    for symbol,(mark,trade) in sorted(datasets.items()):
        before=len(records)
        for family in (*KINDS,'retest16'):
            builder=retest_candidates if family=='retest16' else candidates
            for target in (.01,.015,.02):
                raw=builder(mark,trade,symbol,kind=family,target_fraction=target,regular_only=False,
                    trigger_kind='mark',entry_stress=.0005)
                for r in raw:
                    if r['entry_time']>cutoff:continue
                    out=resolve(r,trade,mark=mark,exit_stress=.0005)
                    if not out['outcome_known']:raise ValueError('unknown_outcome_in_conditional_pool')
                    key=(symbol,r['entry_time'],r['direction'],r['module'],r['risk_pct'],r['signal_strength'])
                    if key not in feature_cache:feature_cache[key]=context.row(r)
                    features=list(feature_cache[key]);features[target_index]=target;features[rr_index]=target/r['risk_pct']
                    records.append(dict(out,**{'feature__'+name:v for name,v in zip(FEATURES,features)}))
        print('conditional_pool',symbol,'rows',len(records)-before,flush=True)
    frame=pd.DataFrame(records)
    if frame.duplicated(['symbol','entry_time','direction','module','target_fraction']).any():
        raise ValueError('duplicate_conditional_candidate')
    frame.to_pickle(CACHE)
    metadata=dict(sources=provenance,data_hashes=data_hashes,sha256=hashlib.sha256(CACHE.read_bytes()).hexdigest(),
        candidates=len(frame),feature_names=list(FEATURES),mature_cohort_last_entry=cutoff,
        trigger_kind='mark',entry_price_stress=.0005,exit_price_stress=.0005,
        basis='all hypothetical candidates, before any portfolio or future prediction filtering')
    META.write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    return frame,metadata


if __name__=='__main__':
    frame,meta=load_pool()
    print('conditional_pool_complete',len(frame),len(FEATURES),flush=True)
