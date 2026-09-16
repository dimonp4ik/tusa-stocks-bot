"""Same signal callbacks, paired venue prices. Diagnostic, not exact exchange fills."""
import argparse,csv,json,pickle,hashlib,os
from pathlib import Path
os.environ['PYTHON_DOTENV_DISABLED']='1'
from config import STOP_CLOSE_CONFIRM, STOP_EXCHANGE_BACKSTOP_R
from filter_lab import gate,metrics


def replay_callback(row,c):
    """Market entry at next bar open, exchange protection then signal callback close."""
    lookup={t:i for i,t in enumerate(c['time'])}
    t0=int(float(row['entry_time']));end=int(float(row['exit_time']))-900
    if t0 not in lookup or end not in lookup:raise ValueError('missing execution timestamps')
    first=lookup[t0];last=lookup[end]
    if last<first:raise ValueError('exit before entry')
    if c['time'][first:last+1]!=list(range(t0,end+900,900)):raise ValueError('execution gap')
    direction=row['direction'];sign=1 if direction=='LONG' else -1
    entry=float(c['open'][first]);sl=float(row['sl']);tp1=float(row['tp1']);tp2=float(row['tp2'])
    planned=float(row['entry']);risk=abs(planned-sl)
    if risk<=0:raise ValueError('invalid risk')
    if not (sl<entry<tp1 if sign==1 else tp1<entry<sl):return None
    stop=entry-sign*risk*max(1.,STOP_EXCHANGE_BACKSTOP_R) if STOP_CLOSE_CONFIRM else sl
    outcome=row['outcome'];price=float(c['close'][last]);bar=last
    for j in range(first,last+1):
        o,h,l=(float(c[k][j]) for k in ('open','high','low'))
        if (l<=stop if sign==1 else h>=stop):
            price=min(o,stop) if sign==1 else max(o,stop);outcome='SL';bar=j;break
        if (h>=tp2 if sign==1 else l<=tp2):
            price=tp2;outcome='TP2';bar=j;break
    weight=float(row['size_mult']);net=(sign*(price-entry)-.0002*(entry+price))/risk*weight
    return dict(symbol=row['symbol'],direction=direction,entry_time=t0,exit_time=c['time'][bar]+900,
      entry=entry,exit_price=price,net_r=net,outcome=outcome,risk_pct=risk/entry,size_mult=weight,
      source_outcome=row['outcome'])


def main():
    folder=Path('reports/audit_2026_09_08');path=folder/'venue_swap_tagged_raw.csv'
    with path.open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
    market=[r for r in rows if r['entry_is_intrabar']=='False']
    results={};paired=[];inputs={}
    for venue in ('swap_matched_xperp','xperp_cache'):
        allrows=[];rejected=0;errors=[];candles={}
        for symbol in sorted({r['symbol'] for r in market}):
            p=folder/venue/(symbol+'_15min_18000.pkl');raw=p.read_bytes();inputs[str(p)]=hashlib.sha256(raw).hexdigest();candles[symbol]=pickle.loads(raw)
        for r in market:
            try:actual=replay_callback(r,candles[r['symbol']])
            except ValueError as exc:errors.append(str(exc));continue
            if actual is None:rejected+=1;continue
            allrows.append(actual)
        accepted=gate(sorted(allrows,key=lambda r:(r['entry_time'],r['symbol'])),4 if Path.cwd().name=='tusa stocks' else 3)
        results[venue]={'raw':metrics(allrows),'gated':metrics(accepted),'bracket_rejections':rejected,'errors':errors}
        if accepted:
            with (folder/('same_callbacks_'+venue+'.csv')).open('w',newline='',encoding='utf-8') as f:
                writer=csv.DictWriter(f,fieldnames=list(accepted[0]));writer.writeheader();writer.writerows(accepted)
    report={'status':'EXECUTION_DIAGNOSTIC','market_entries':len(market),'excluded_intrabar_entries':len(rows)-len(market),
      'results':results,'input_hashes':inputs,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      'limitations':['Original signal callback times fixed; market fills approximated by 15m open/close, not actual bid/ask.',
       'Exchange TP2/backstop can terminate earlier. Stop wins on ambiguous OHLC bars.',
       'Signal callbacks originate from conservative historical simulator, not production polling logs.',
       'No partial TP1; no historical depth/funding. New order-book guard cannot be reproduced from OHLC.',
       'All intrabar zone entries excluded; crypto requires minute-level trigger reconstruction.',
       'R normalized to original signal risk; gates reapplied using modeled execution outcomes.']}
    (folder/'same_signal_execution_review.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report))
if __name__=='__main__':main()
