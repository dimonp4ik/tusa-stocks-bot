"""Select small entry filters on train/validation, then evaluate a held-out period.

This is research on historical development data, not proof of live profitability.
Filters run BEFORE portfolio gates. No outcome/future excursion is an input.
"""
from __future__ import annotations
import argparse
import collections
import csv
import hashlib
import heapq
import itertools
import json
import math
from pathlib import Path
import statistics
from datetime import datetime, timezone


def timestamp(value):
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc).timestamp()


def load(path):
    rows=list(csv.DictReader(path.open(encoding='utf-8-sig',newline='')))
    numeric=['entry_time','exit_time','entry','sl','tp1','net_r','cost_r','size_mult',
             'mtf_score','volume_ratio','rsi','eff_ratio','vol_atr_pct','vol_ratio_regime',
             'quality_score','extension_atr','bos_extension_atr','room_atr','premium']
    for row in rows:
        for k in numeric:
            try:row[k]=float(row[k])
            except (KeyError,TypeError,ValueError):row[k]=None
        if row['entry_time'] is None or row['exit_time'] is None or row['net_r'] is None:
            raise ValueError('Trade timestamps and realized outcome required')
        direction=1 if row['direction']=='LONG' else -1
        row['aligned_1h']=int(row['trend_1h']==('bullish' if direction==1 else 'bearish'))
        row['aligned_4h']=int(row['trend_4h']==('bullish' if direction==1 else 'bearish'))
        row['extension']=row.get('extension_atr') if row.get('extension_atr') is not None else row.get('bos_extension_atr')
        row['rsi_directional']=row['rsi'] if direction==1 else 100-row['rsi']
        row['risk_pct']=abs(row['entry']-row['sl'])/row['entry']
        row['tp1_r']=direction*(row['tp1']-row['entry'])/abs(row['entry']-row['sl'])
    return sorted(rows,key=lambda r:(r['entry_time'],r['symbol']))


def gate(rows, cap):
    """Deterministic event replay. Only already closed outcomes inform the pause."""
    pending=[];opened={};last={};per_scan=collections.Counter();out=[]
    day=None;streak=0;paused=False
    for row in rows:
        now=row['entry_time'];new_day=int(now//86400)
        if new_day!=day:day=new_day;streak=0;paused=False
        while pending and pending[0][0] < now:
            end,serial,done=heapq.heappop(pending)
            opened.pop(done['symbol'],None)
            if int(end//86400)==day:
                streak=streak+1 if done['outcome']=='SL' else 0
                if streak>=3:paused=True
        if paused or row['symbol'] in opened:continue
        key=(row['symbol'],row['direction'])
        if now-last.get(key,-1e20)<10800:continue
        scan=int(now//900)
        if per_scan[scan]>=3:continue
        if sum(r['direction']==row['direction'] for r in opened.values())>=cap:continue
        opened[row['symbol']]=row;last[key]=now;per_scan[scan]+=1
        out.append(row);heapq.heappush(pending,(row['exit_time'],len(out),row))
    return out


def metrics(rows):
    rs=[r['net_r'] for r in rows];n=len(rs)
    gains=sum(r for r in rs if r>0);losses=-sum(r for r in rs if r<0)
    wins=sum(r>0 for r in rs);p=wins/n if n else 0
    z=1.96
    lower=(p+z*z/(2*n)-z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)))/(1+z*z/n) if n else 0
    peak=equity=dd=0.;daily=collections.defaultdict(float)
    for row in sorted(rows,key=lambda r:(r['exit_time'],r['symbol'])):
        equity+=row['net_r'];peak=max(peak,equity);dd=max(dd,peak-equity)
        daily[int(row['exit_time']//86400)]+=row['net_r']
    values=list(daily.values())
    # Cluster by exit day, preserving cross-asset loss clusters in selection.
    lcb=statistics.mean(values)-1.96*statistics.stdev(values)/math.sqrt(len(values)) if len(values)>1 else -1e9
    return dict(n=n,wr=100*p,wr_lower_95=100*lower,net_r=sum(rs),mean_r=sum(rs)/n if n else 0,
                pf=gains/losses if losses else None,dd_r=dd,active_days=len(daily),daily_lcb=lcb,
                symbols=len({r['symbol'] for r in rows}),
                stress_net_r=sum(r['net_r']-(.001/r['risk_pct'])*(r.get('size_mult') or 1) for r in rows))


def matches(row,rule):
    for field,op,value in rule:
        actual=row.get(field)
        if actual is None:return False
        if op=='ge' and not actual>=value:return False
        if op=='le' and not actual<=value:return False
        if op=='eq' and actual!=value:return False
        if op=='ne' and actual==value:return False
    return True


def candidates(rows):
    bounds={
        'volume_ratio':[1.5,2,2.5,3,4,6], 'eff_ratio':[.2,.3,.4,.5,.6],
        'rsi_directional':[40,45,50,55,60,65,70], 'mtf_score':[13,14,15,16,17,18],
        'vol_ratio_regime':[.7,1,1.5,2,3], 'extension':[.5,1,1.5,2,3],
        'risk_pct':[.004,.006,.01,.015,.02,.03], 'tp1_r':[.5,.75,1,1.5,2],
        'vol_atr_pct':[.002,.004,.006,.01,.015], 'quality_score':[75,80,85,90]}
    result=[]
    for field,levels in bounds.items():
        for level in levels:
            for op in ('ge','le'):result.append(((field,op,level),))
    for field in ('direction','entry_source','session','trend_1h','trend_4h','aligned_1h','aligned_4h'):
        for value in sorted({r[field] for r in rows}):
            for op in ('eq','ne'):result.append(((field,op,value),))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('raw',type=Path);p.add_argument('--cap',type=int,required=True)
    p.add_argument('--train-end',default='2026-06-01');p.add_argument('--validation-end',default='2026-07-01')
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();rows=load(a.raw);train_end=timestamp(a.train_end);val_end=timestamp(a.validation_end)
    train=[r for r in rows if r['entry_time']<train_end and r['exit_time']<train_end]
    # Candidate generation sees training features only. Holdout is never consulted
    # until the selected ruleset has been frozen using train and validation.
    atoms=candidates(train);evaluated=[];dedup=set()
    min_train=max(60,int(len(gate(train,a.cap))*.2))
    def evaluate(rule):
        selected=[r for r in train if matches(r,rule)]
        mask=tuple((r['symbol'],r['entry_time']) for r in selected)
        if mask in dedup:return None
        dedup.add(mask)
        m=metrics(gate(selected,a.cap))
        if m['n']<min_train or m['symbols']<4 or m['active_days']<25:return None
        item=dict(rule=rule,train=m);evaluated.append(item);return item
    singles=[r for rule in atoms if (r:=evaluate(rule)) is not None]
    singles.sort(key=lambda x:(x['train']['daily_lcb'],x['train']['net_r']),reverse=True)
    for first in singles[:20]:
        for second in atoms:
            if first['rule'][0][0]==second[0][0]:continue
            evaluate(first['rule']+second)
    evaluated.sort(key=lambda x:(x['train']['daily_lcb'],x['train']['net_r']),reverse=True)
    shortlist=evaluated[:20]
    for item in shortlist:
        prefix=gate([r for r in rows if r['entry_time']<val_end and matches(r,item['rule'])],a.cap)
        val=[r for r in prefix if train_end<=r['entry_time']<val_end and r['exit_time']<val_end]
        item['validation']=metrics(val)
    eligible=[x for x in shortlist if x['validation']['n']>=20 and x['validation']['net_r']>0
              and x['train']['net_r']>0]
    chosen=max(eligible,key=lambda x:(x['validation']['daily_lcb'],x['train']['daily_lcb']),default=None)
    if chosen:
        all_selected=gate([r for r in rows if matches(r,chosen['rule'])],a.cap)
        chosen['test']=metrics([r for r in all_selected if r['entry_time']>=val_end])
        chosen['months']={month:metrics([r for r in all_selected if datetime.fromtimestamp(r['entry_time'],timezone.utc).strftime('%Y-%m')==month])
            for month in sorted({datetime.fromtimestamp(r['entry_time'],timezone.utc).strftime('%Y-%m') for r in all_selected})}
    base=gate(rows,a.cap)
    result=dict(raw_sha256=hashlib.sha256(a.raw.read_bytes()).hexdigest(),arguments={**vars(a),'raw':str(a.raw),'out':str(a.out)},
        candidate_count=len(evaluated),minimum_train_trades=min_train,chosen=chosen,shortlist=shortlist,
        baseline=dict(train=metrics([r for r in base if r['exit_time']<train_end]),
                      validation=metrics([r for r in base if train_end<=r['entry_time']<val_end and r['exit_time']<val_end]),
                      test=metrics([r for r in base if r['entry_time']>=val_end])),
        status='RESEARCH_ONLY',limitations=['Holdout separated for this filter search; underlying rules were already developed on historical data.',
          'SWAP candles do not prove execution on X-Perp. Funding is excluded.',
          'Many candidates tested; Wilson interval does not correct for selection. Independent confirmation required.',
          'Baseline size weights retained; dollar risk limits are not modeled in these R figures.'])
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['candidate_count','chosen','baseline']}),flush=True)


if __name__=='__main__':main()
