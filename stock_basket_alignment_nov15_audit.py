"""Bounded archive parity after timestamp-based basket alignment; no orders."""
from bisect import bisect_right
import hashlib
import json
from pathlib import Path
import subprocess
import time
import types

import numpy as np
import pandas as pd
import stock_xperp_native_growth_forward_monitor as monitor
from src import stock_xperp_native_shadow as current

ROOT=Path('reports/audit_2026_09_08')
OLD_COMMIT='be6a7d1'


def main():
    source=subprocess.check_output(['git','show',f'{OLD_COMMIT}:src/stock_xperp_native_shadow.py'])
    old=types.ModuleType('stock_native_before_alignment')
    exec(compile(source,'stock_native_before_alignment.py','exec'),old.__dict__)
    manifest=ROOT/'stock_xperp_forward_nov14_merged/manifest.json'
    preportfolio=ROOT/'stock_xperp_nov14_preportfolio.pkl'
    candles,sources,_=monitor.load_manifest(manifest)
    entries=sorted(pd.read_pickle(preportfolio).entry_time.unique())
    eligible=[]
    for entry in entries:
        stamp=entry-900;count=0
        for symbol,c in candles.items():
            if symbol not in current.AUDITED_SYMBOLS:
                continue
            end=bisect_right(c['time'],stamp)
            count+=end>=current.REQUIRED_HISTORY_BARS and c['time'][end-1]==stamp
        if count>=current.MINIMUM_UNIVERSE:
            eligible.append(stamp)
    selected=[eligible[int(i)] for i in np.linspace(0,len(eligible)-1,24)]
    rows=[];old_seconds=new_seconds=0.
    for stamp in selected:
        sliced={}
        for symbol,c in candles.items():
            if symbol not in current.AUDITED_SYMBOLS:
                continue
            end=bisect_right(c['time'],stamp)
            if end<current.REQUIRED_HISTORY_BARS or c['time'][end-1]!=stamp:
                continue
            start=max(0,end-500)
            sliced[symbol]={k:v[start:end] for k,v in c.items()}
        if len(sliced)<current.MINIMUM_UNIVERSE:
            raise ValueError('archive sample lacks aligned universe')
        t=time.perf_counter()
        old_trend=old.basket_trends(sliced,stamp,(16,32,64,96))
        before=old.native_shadow_setups(sliced,now=stamp+960)
        old_seconds+=time.perf_counter()-t
        t=time.perf_counter()
        new_trend=current.basket_trends(sliced,stamp,(16,32,64,96))
        after=current.native_shadow_setups(sliced,now=stamp+960)
        new_seconds+=time.perf_counter()-t
        assert old_trend is not None and new_trend is not None
        assert np.allclose(list(old_trend.values()),list(new_trend.values()),rtol=0,atol=1e-12)
        assert len(before)==len(after)
        for a,b in zip(before,after):
            assert a.keys()==b.keys()
            for k,value in a.items():
                if isinstance(value,(float,int)) and not isinstance(value,bool):
                    assert np.isclose(value,b[k],rtol=0,atol=1e-12),(stamp,k,value,b[k])
                else:
                    assert value==b[k],(stamp,k,value,b[k])
        rows.append(dict(signal_time=int(stamp),aligned_symbols=len(sliced),setups=len(after),
            trends=new_trend,setup_sha256=hashlib.sha256(json.dumps(after,sort_keys=True).encode()).hexdigest()))
        print(json.dumps(rows[-1]),flush=True)
    sources.update({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (
        manifest,preportfolio,Path(__file__),Path('src/stock_xperp_native_shadow.py'))})
    report=dict(status='ALIGNED_ARCHIVE_PARITY_PASSED',old_commit=OLD_COMMIT,
        old_source_sha256=hashlib.sha256(source).hexdigest(),samples=len(rows),
        total_setups=sum(r['setups'] for r in rows),old_seconds=old_seconds,new_seconds=new_seconds,
        cold_start_timestamps_excluded=len(entries)-len(eligible),
        earliest_eligible_signal_time=int(eligible[0]),
        selection='24 evenly spaced preportfolio entry timestamps with >=10 aligned histories of required length, chosen without outcomes; at most 500 preceding bars',
        limitations=['Only aligned archive prefixes; mixed-age feed histories covered separately by unit tests',
            'No profit uplift or causal attribution of client losses established'],results=rows,sources=sources)
    (ROOT/'stock_basket_alignment_nov15.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
