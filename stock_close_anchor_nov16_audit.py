"""Execution-reference sensitivity, with causal candidate gates and no rule search."""
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from stock_module_cluster_audit import portfolio_gate
from stock_momentum_phase_nov14_audit import metrics,window
import stock_xperp_native_cross_section_oct18_audit as features
import stock_xperp_native_volatility_guard_oct18_audit as volatility
import stock_xperp_nov7_cluster_diagnostic as old_route
from src import stock_xperp_native_shadow as native
from config import NATIVE_XPERP_DAILY_LOSS_R_CAP

ROOT=Path('reports/audit_2026_09_08')


def resolve(row,candle,*,reference,cap_target_gap):
    if reference not in ('known_signal_close','historical_first_open'):
        raise ValueError('unknown entry reference')
    index=int(row['entry_index']);last=index+63
    if index<1 or last>=len(candle['time']):raise ValueError('incomplete outcome horizon')
    if any(b-a!=900 for a,b in zip(candle['time'][index:last+1],candle['time'][index+1:last+1])):
        raise ValueError('noncontiguous outcome horizon')
    sign=int(row['direction_n'])
    entry=float(candle['close'][index-1] if reference=='known_signal_close' else candle['open'][index])
    stop_fraction=.01;target_fraction=float(row['audited_target_pct'])/100
    stop=entry*(1-sign*stop_fraction);target=entry*(1+sign*target_fraction)
    code=0;price=float(candle['close'][last]);end_index=last
    for i in range(index,last+1):
        o,h,l=(float(candle[k][i]) for k in ('open','high','low'))
        if (o<=stop if sign>0 else o>=stop):
            price,code,end_index=o,-1,i;break
        if (o>=target if sign>0 else o<=target):
            price,code,end_index=(target if cap_target_gap else o),1,i;break
        if (l<=stop if sign>0 else h>=stop):
            price,code,end_index=stop,-1,i;break
        if (h>=target if sign>0 else l<=target):
            price,code,end_index=target,1,i;break
    signed=sign*(price/entry-1)
    return dict(row,entry=entry,signed_return=signed,net_r=signed/.01,
        exit_time=int(candle['time'][end_index])+900,outcome_code=code,
        outcome='TP' if code==1 else 'SL' if code==-1 else 'TIME',
        direction='LONG' if sign>0 else 'SHORT',risk_pct=.01,
        stop_pct=1.,target_pct=100*target_fraction,hold_bars=64,
        module='close_anchor_integrity_audit',module_priority=0)


def context_table(candles):
    # Basket means remain valid when cross-sectional deviation is zero.
    # The candidate feature table drops those rows because z is undefined;
    # using that table here would invent holes in an otherwise quiet market.
    moves=pd.concat([pd.Series(np.asarray(c['close'][2:])/np.asarray(c['close'][:-2])-1,
        index=np.asarray(c['time'][2:])+900,dtype=float) for c in candles.values()],axis=1)
    grid=np.arange(int(moves.index.min()),int(moves.index.max())+900,900)
    moves=moves.reindex(grid)
    context=pd.DataFrame(dict(cross_mean=moves.mean(axis=1),universe_n=moves.notna().sum(axis=1)))
    context.loc[context.universe_n.lt(native.MINIMUM_UNIVERSE),'cross_mean']=np.nan
    for bars in (16,32,64,96):
        context[f'basket_trend_{bars}']=context.cross_mean.rolling(bars,min_periods=bars).sum()
    liquid=[]
    for c in candles.values():
        volume=pd.Series(c['volume'],index=np.asarray(c['time'])+900,dtype=float)
        counts=volume.gt(0).rolling(8,min_periods=8).sum().reindex(grid)
        liquid.append(counts)
    counts=pd.concat(liquid,axis=1)
    context['liquid_universe_ratio_8x6']=counts.ge(6).sum(axis=1)/counts.notna().sum(axis=1).replace(0,np.nan)
    local=pd.to_datetime(context.index,unit='s',utc=True).tz_convert('America/New_York')
    minutes=local.hour*60+local.minute
    trigger=((minutes>=195)&(minutes<240)&context.basket_trend_64.le(-.06)
        &context.liquid_universe_ratio_8x6.le(.55))
    context['premarket_paused']=trigger.groupby(local.date).cummax()
    context.index.name='entry_time'
    return context.reset_index()


def causal_candidates(candles):
    candles={s:c for s,c in candles.items() if s in native.AUDITED_SYMBOLS}
    all_features=features.feature_rows(candles,2)
    context=context_table(candles)
    entries=features.select_entries(all_features,2.,.004)
    entries=volatility.add_atr14(entries,candles)
    entries=entries[entries.atr14_fraction.le(.01)].merge(
        context.drop(columns=['cross_mean','universe_n']),on='entry_time',validate='many_to_one')
    required=[f'basket_trend_{b}' for b in (16,32,64,96)]
    entries=entries.dropna(subset=required)
    keep=((entries.direction_n.gt(0)&entries.basket_trend_96.ge(native.LONG_MIN_BASKET_TREND)
        &entries.basket_trend_64.ge(-.04))
        |(entries.direction_n.lt(0)&entries.basket_trend_16.gt(-.0075)))
    entries=entries[keep & ~entries.premarket_paused].copy()
    # A prior close belongs inside its own 32-bar high/low. This uses no
    # next-bar first transaction to decide whether the setup exists.
    entries=entries[entries.entry_index.ge(32)].copy()
    entries['audited_target_pct']=np.where(entries.basket_trend_32.le(-.04),2.,
        np.where((entries.entry_time//3600%24).between(6,11),3.,2.5))
    return entries


def runtime_prefix_parity(candles,frame):
    times=np.unique(np.concatenate([np.asarray(c['time'])+900 for c in candles.values()]))
    # Fixed, outcome-independent coverage across the archive after 200 bars.
    eligible=times[times>=times[0]+200*900]
    samples=eligible[np.linspace(0,len(eligible)-1,24,dtype=int)]
    checked=0
    for stamp in samples:
        prefixes={}
        for symbol,c in candles.items():
            end=int(np.searchsorted(c['time'],stamp-900,side='right'))
            if end:prefixes[symbol]={k:v[max(0,end-200):end] for k,v in c.items()}
        expected=native.native_shadow_setups(prefixes,now=float(stamp)+15)
        actual=frame[frame.entry_time.eq(stamp)]
        assert set(actual.symbol)=={r['symbol'] for r in expected},f'candidate parity failed at {stamp}'
        for row in expected:
            observed=actual[actual.symbol.eq(row['symbol'])].iloc[0]
            for field in ('cross_z','signal_return','atr14_fraction',
                    'basket_trend_16','basket_trend_32','basket_trend_64','basket_trend_96'):
                assert abs(float(observed[field])-row[field])<1e-10,(stamp,field)
            assert abs(observed['liquid_universe_ratio_8x6']-row['liquid_universe_ratio'])<1e-10
        checked+=len(expected)
    return dict(timestamps=len(samples),matched_candidates=checked)


def gate(frame):
    return pd.DataFrame(portfolio_gate(frame.to_dict('records'),None,
        rolling_hour_cap=4,rolling_direction_hour_cap=2,
        rank_key=old_route.quality.ranker('z_high'),loss_streak_cap=2,
        direction_cap=4,daily_loss_r_cap=NATIVE_XPERP_DAILY_LOSS_R_CAP))


def monthly(frame):
    return {m:metrics(window(frame,a,b)) for m,a,b in (
        ('july_partial','2026-07-01','2026-08-01'),('august','2026-08-01','2026-09-01'),
        ('september','2026-09-01','2026-10-01'),('october','2026-10-01','2026-11-01'),
        ('november_partial','2026-11-01','2026-12-01'))}


def main():
    manifest=ROOT/'stock_xperp_forward_nov14_merged/manifest.json'
    metadata=json.loads(manifest.read_text());candles={};sources={}
    for symbol,item in metadata['symbols'].items():
        if not item.get('path') or symbol not in native.AUDITED_SYMBOLS:continue
        p=Path(item['path']);raw=p.read_bytes();digest=hashlib.sha256(raw).hexdigest()
        assert digest==item['sha256'];sources[str(p)]=digest
        candles[symbol]=pickle.loads(raw)
    last=min(c['time'][-1] for c in candles.values())
    source=ROOT/'stock_xperp_nov14_preportfolio.pkl';old=pd.read_pickle(source)
    old=old[old.entry_time.le(last-63*900)].copy()
    original=old_route.long_guard.apply_guard(old_route.premarket.apply_guard(old))
    baseline=gate(original)
    expected=pd.read_pickle(ROOT/'stock_factor_nov14_baseline.pkl')
    keys=['symbol','entry_time','direction','net_r','exit_time']
    pd.testing.assert_frame_equal(baseline[keys].sort_values(keys[:3]).reset_index(drop=True),
        expected[keys].sort_values(keys[:3]).reset_index(drop=True),check_dtype=False,atol=1e-10,rtol=0)
    new=causal_candidates(candles)
    parity=runtime_prefix_parity(candles,new)
    new=new[new.entry_time.le(last-63*900)].copy()
    outputs={'legacy_reference':baseline};candidate_counts={'legacy_reference':len(original)}
    for name,pool,reference,cap in (
        ('same_candidates_known_close',original,'known_signal_close',True),
        ('causal_candidates_next_open',new,'historical_first_open',False),
        ('causal_candidates_known_close',new,'known_signal_close',True)):
        resolved=pd.DataFrame([resolve(row,candles[row['symbol']],reference=reference,cap_target_gap=cap)
            for row in pool.to_dict('records')])
        output=gate(resolved);outputs[name]=output;candidate_counts[name]=len(pool)
        output.to_pickle(ROOT/f'stock_close_anchor_nov16_{name}.pkl')
        print(name,{m:round(v['return_pct'],2) for m,v in monthly(output).items()},flush=True)
    report=dict(status='ENTRY_REFERENCE_INTEGRITY_DIAGNOSTIC',objective_verified=False,
        baseline_parity_rows=len(baseline),candidate_counts=candidate_counts,
        runtime_prefix_parity=parity,
        results={name:monthly(frame) for name,frame in outputs.items()},
        limitations=['Known signal close is an observable reference, not a certified executable market fill.',
            '15m exit paths retain unknown intrabar ordering, stop first, delayed exit recognition.',
            'Causal version observes premarket triggers on all scans, not only qualifying candidates.',
            'No historical bid/ask, contract rounding, funding or actual client ledger.',
            'Standalone 3% margin/10x closed entry cohorts, partial July and November coverage.',
            'Previously inspected history; no new strategy selection or live approval.'],sources=sources)
    report['sources'].update({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (source,manifest,Path(__file__))})
    (ROOT/'stock_close_anchor_nov16.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('sources','results')},indent=2))


if __name__=='__main__':main()
