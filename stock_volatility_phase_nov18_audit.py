"""Causal shock normalization, phase direction and structural brackets."""
import hashlib
import itertools
import json
import pickle
from pathlib import Path

import pandas as pd

import stock_close_anchor_nov16_audit as anchor
from stock_structure_geometry_nov17 import bracket,resolve
from stock_structure_geometry_nov17_audit import readiness_masks
from stock_volatility_phase_nov18 import feature_pool,prepare_entries,SCORES,PHASES
from src.stock_xperp_native_shadow import AUDITED_SYMBOLS

ROOT=Path('reports/audit_2026_09_08')


def main():
    manifest=ROOT/'stock_xperp_forward_nov14_merged/manifest.json'
    meta=json.loads(manifest.read_text(encoding='utf-8'));candles={};sources={}
    for symbol,item in meta['symbols'].items():
        if symbol not in AUDITED_SYMBOLS or not item.get('path'):continue
        p=Path(item['path']);raw=p.read_bytes();digest=hashlib.sha256(raw).hexdigest()
        if digest!=item['sha256']:raise ValueError('native source hash mismatch')
        candles[symbol]=pickle.loads(raw);sources[str(p)]=digest
    last=min(c['time'][-1] for c in candles.values())
    original=anchor.causal_candidates(candles)
    original=original[original.entry_time.le(last-63*900)].copy()
    print('native_candidates',len(original),flush=True)
    pool=feature_pool(candles)
    context=anchor.context_table(candles)
    print('normalized_feature_rows',len(pool),flush=True)
    expected_path=ROOT/'stock_close_anchor_nov16_causal_candidates_known_close.pkl'
    expected=pd.read_pickle(expected_path)
    stock_structure_report=ROOT/'stock_structure_geometry_nov17.json'
    old_structure=json.loads(stock_structure_report.read_text(encoding='utf-8'))
    cases=[('native_baseline',original,'baseline'),('native_swing2_original_target',original,'swing2_original_target')]
    for score,phase in itertools.product(SCORES,PHASES):
        entries=prepare_entries(pool,context,score,phase)
        entries=entries[entries.entry_time.le(last-63*900)].copy()
        for geometry in ('swing2_original_target','swing2_two_r'):
            cases.append((f'{score}_{phase}_{geometry}',entries,geometry))
    report=dict(status='EXPLORATORY_NATIVE_NORMALIZED_SHOCK_AND_PHASE_GRID',objective_verified=False,
        live_activation_supported=False,native_candidates=len(original),feature_rows=len(pool),
        policy_count=len(cases),results={},limitations=[
            'Previously inspected native history; 18 new hypotheses and two parity controls, no untouched validation.',
            'Current 44-contract universe and specifications are assumed across history; listings and warmup vary.',
            'Shock normalization uses ATR ending before the two impulse bars; it cannot absorb the shock being measured.',
            'Cross-sectional mean/std and median/MAD use contemporaneously known closed bars; zero dispersion is unknown.',
            'New families require 64 contiguous closed bars; baseline retains its original warmup for exact parity.',
            'Reversal retains native directional stress guards; continuation is a distinct hypothetical family.',
            'Closed-phase continuation requires aligned body/trends, strong close and active volume; fades need wick rejection.',
            'Known closed prices are references, not executable market quotes; 64-bar stop-first paths are proxies.',
            'Stop beyond two closed bars plus .25 ATR, .3%-2% distance; original targets or 2R clipped .5%-3%.',
            'Portfolio and 20-outcome R/price readiness are recalculated separately for every hypothesis.',
            'Standalone gross 3% margin/10x; joint contract-rounded cash is a separate audit.',
            'No private fills, historical bid/ask, financing, floating equity or actual customer account PnL.',
            'July and November partial; no claim of live safety or monthly 30% profitability.'])
    for name,entries,geometry in cases:
        rows=[]
        for row in entries.to_dict('records'):
            c=candles[row['symbol']];g=bracket(row,c,geometry)
            if g is not None:rows.append(resolve(row,c,g))
        values=pd.DataFrame(rows)
        gated=anchor.gate(values) if len(values) else values
        if not len(gated):gated=values.iloc[:0].copy()
        if name in ('native_baseline','native_swing2_original_target'):
            reference=expected if name=='native_baseline' else pd.read_pickle(old_structure['results']['swing2_original_target']['trades_path'])
            fields=['symbol','entry_time','direction','net_r','exit_time']
            pd.testing.assert_frame_equal(gated[fields].sort_values(fields[:3]).reset_index(drop=True),
                reference[fields].sort_values(fields[:3]).reset_index(drop=True),check_dtype=False,atol=1e-9,rtol=0)
        masks=readiness_masks(gated)
        gated=gated.assign(forward_r_ready=masks.r_ready,forward_price_ready=masks.price_ready)
        path=ROOT/f'stock_volatility_phase_nov18_{name}.pkl';gated.to_pickle(path)
        result=dict(candidates=len(entries),bracket_candidates=len(values),signals=len(gated),
            monthly=anchor.monthly(gated),r_and_price_ready=int((masks.r_ready&masks.price_ready).sum()),
            trades_path=str(path),trades_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        report['results'][name]=result
        print(name,'signals',len(gated),'months',{m:round(v['return_pct'],3) for m,v in result['monthly'].items()},flush=True)
    for p in (manifest,expected_path,stock_structure_report,Path(__file__),Path('stock_volatility_phase_nov18.py'),
        Path('stock_structure_geometry_nov17.py'),Path('stock_structure_geometry_nov17_audit.py'),
        Path(anchor.__file__),Path('stock_module_cluster_audit.py')):
        sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    report['sources']=sources
    (ROOT/'stock_volatility_phase_nov18.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')


if __name__=='__main__':main()
