"""Fixed activity guards, rerouting the full native candidate pool causally."""
import hashlib
import json
import pickle
from pathlib import Path
import pandas as pd

import stock_close_anchor_nov16_audit as anchor
from stock_closed_activity_nov17 import closed_activity
from src.stock_xperp_native_shadow import AUDITED_SYMBOLS

ROOT=Path('reports/audit_2026_09_08')


def main():
    manifest=ROOT/'stock_xperp_forward_nov14_merged/manifest.json'
    meta=json.loads(manifest.read_text(encoding='utf-8'))
    candles={};sources={}
    for symbol,item in meta['symbols'].items():
        if symbol not in AUDITED_SYMBOLS or not item.get('path'):continue
        p=Path(item['path']);raw=p.read_bytes();digest=hashlib.sha256(raw).hexdigest()
        if digest!=item['sha256']:raise ValueError('native candle hash mismatch')
        candles[symbol]=pickle.loads(raw);sources[str(p)]=digest
    last=min(c['time'][-1] for c in candles.values())
    candidates=anchor.causal_candidates(candles)
    candidates=candidates[candidates.entry_time.le(last-63*900)].copy()
    rows=[]
    for row in candidates.to_dict('records'):
        activity=closed_activity(candles[row['symbol']],int(row['entry_index']))
        rows.append(dict(anchor.resolve(row,candles[row['symbol']],
            reference='known_signal_close',cap_target_gap=True),activity=activity))
    resolved=pd.DataFrame(rows)
    baseline=anchor.gate(resolved)
    expected_path=ROOT/'stock_close_anchor_nov16_causal_candidates_known_close.pkl'
    expected=pd.read_pickle(expected_path)
    fields=['symbol','entry_time','direction','net_r','exit_time']
    pd.testing.assert_frame_equal(baseline[fields].sort_values(fields[:3]).reset_index(drop=True),
        expected[fields].sort_values(fields[:3]).reset_index(drop=True),check_dtype=False,atol=1e-10,rtol=0)
    report=dict(status='EXPLORATORY_NATIVE_ACTIVITY_GUARDS',objective_verified=False,
        live_activation_supported=False,candidates=len(rows),baseline_parity_rows=len(baseline),results={},
        limitations=['Previously inspected native data; no untouched holdout or native pre-2026 history.',
            'Only closed positive-volume bars count; activity is not quoted spread/depth or executable liquidity.',
            'Known signal close and 15m stop-first fills are price scenarios, not actual market fills.',
            'Standalone gross 3% margin/10x closed entry cohorts; no shared account or real equity guard here.',
            'Model pauses and quality sorting replay after the new guard, not filtering a finished winning ledger.',
            'No financing,fees,intrabar equity or calendar account-PnL reconstruction.'])
    for policy in ('baseline','last_bar_active','last_two_active','six_of_eight'):
        if policy=='baseline':mask=[True]*len(rows)
        else:mask=[bool(r['activity'] and (r['activity']['active_eight']>=6
            if policy=='six_of_eight' else r['activity'][policy])) for r in rows]
        frame=anchor.gate(resolved.loc[mask])
        result=dict(candidates_kept=sum(mask),monthly=anchor.monthly(frame))
        frame.to_pickle(ROOT/f'stock_closed_activity_nov17_{policy}.pkl')
        report['results'][policy]=result
        print(policy,'kept',sum(mask),'monthly',[(m,round(v['return_pct'],3),round(v['dd_pct'],3))
            for m,v in result['monthly'].items()],flush=True)
    report['baseline_activity']=dict(missing=sum(r['activity'] is None for r in rows),
        zero_last_bar=sum(r['activity'] is not None and not r['activity']['last_bar_active'] for r in rows),
        published_zero_last_bar=sum(r['activity'] is not None and not r['activity']['last_bar_active']
            for r in baseline.to_dict('records')))
    for p in (manifest,expected_path,Path(__file__),Path(anchor.__file__),Path('stock_closed_activity_nov17.py'),
              Path('src/stock_xperp_native_shadow.py'),Path('stock_module_cluster_audit.py'),
              Path('stock_momentum_phase_nov14_audit.py')):
        sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    report['sources']=sources
    (ROOT/'stock_closed_activity_nov17.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':main()
