"""Nine causal bracket scenarios and explicit R-versus-price readiness checks."""
import bisect
import hashlib
import json
import math
import pickle
from pathlib import Path

import pandas as pd

import stock_close_anchor_nov16_audit as anchor
from stock_structure_geometry_nov17 import POLICIES,bracket,resolve
from src.stock_xperp_native_shadow import AUDITED_SYMBOLS

ROOT=Path('reports/audit_2026_09_08')


def readiness_masks(frame):
    closed=frame.sort_values(['exit_time','entry_time','symbol'])
    times=closed.exit_time.tolist();r=closed.net_r.tolist();price=closed.signed_return.tolist()
    rows=[]
    for row in frame.itertuples():
        stop=bisect.bisect_left(times,row.entry_time)
        rows.append(dict(symbol=row.symbol,entry_time=row.entry_time,
            r_ready=stop>=20 and math.fsum(r[stop-20:stop])>1e-10,
            price_ready=stop>=20 and math.fsum(price[stop-20:stop])>1e-12))
    return pd.DataFrame(rows,index=frame.index)


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
    original=candidates.to_dict('records')
    expected_path=ROOT/'stock_close_anchor_nov16_causal_candidates_known_close.pkl'
    expected=pd.read_pickle(expected_path)
    report=dict(status='STRUCTURAL_STOPS_AND_TARGETS_RESEARCH',objective_verified=False,
        live_activation_supported=False,initial_candidates=len(original),policies=list(POLICIES),results={},
        limitations=[
            'All historical months were inspected previously; no new independent holdout.',
            'Known closed signal price is observable, not a historical executable market quote.',
            'Stop beyond 2/8 closed bars plus .25 closed ATR; stop distance floored .3%, rejected above 2%.',
            'Targets are original 2/2.5/3%,pre-impulse close or twice stop risk; alternate targets clipped .5%-3%.',
            'Closed-body confirmation is evaluated before entry, never from the following candle.',
            '15m 64-bar paths remain stop-first; no historical ticks,spread or private fills.',
            'Standalone gross nominal 3% margin/10x; shared cash/account guard is assessed separately.',
            'Variable risk makes sum of R different from sum of price returns; both readiness masks are recorded.',
            'No hypothetical R readiness is used to lift any real account pause.'])
    for policy in POLICIES:
        rows=[]
        for row in original:
            c=candles[row['symbol']];g=bracket(row,c,policy)
            if g is not None:rows.append(resolve(row,c,g))
        values=pd.DataFrame(rows)
        gated=anchor.gate(values)
        if gated.empty:gated=values.iloc[:0].copy()
        if policy=='baseline':
            keys=['symbol','entry_time','direction','net_r','exit_time']
            pd.testing.assert_frame_equal(gated[keys].sort_values(keys[:3]).reset_index(drop=True),
                expected[keys].sort_values(keys[:3]).reset_index(drop=True),check_dtype=False,atol=1e-9,rtol=0)
            report['baseline_parity_rows']=len(gated)
        masks=readiness_masks(gated)
        gated=gated.assign(forward_r_ready=masks.r_ready,forward_price_ready=masks.price_ready)
        path=ROOT/f'stock_structure_geometry_nov17_{policy}.pkl'
        gated.to_pickle(path)
        result=dict(candidates_kept=len(values),signals=len(gated),monthly=anchor.monthly(gated),
            readiness=dict(r_ready=int(masks.r_ready.sum()),price_ready=int(masks.price_ready.sum()),
                r_ready_without_positive_price_sum=int((masks.r_ready & ~masks.price_ready).sum())),
            trades_path=str(path),trades_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        report['results'][policy]=result
        print(policy,'signals',len(gated),'monthly',[(m,round(v['return_pct'],3),round(v['dd_pct'],3))
            for m,v in result['monthly'].items()],'readiness',result['readiness'],flush=True)
    for p in (manifest,expected_path,Path(__file__),Path('stock_structure_geometry_nov17.py'),
              Path(anchor.__file__),Path('stock_module_cluster_audit.py'),
              Path('stock_xperp_native_economics_oct18_audit.py'),Path('src/stock_xperp_native_shadow.py')):
        sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    report['sources']=sources
    (ROOT/'stock_structure_geometry_nov17.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':main()
