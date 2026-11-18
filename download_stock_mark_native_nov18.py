"""Matched historical public mark candles; no surrogate contracts or volume."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import pickle
import re
import time

from src.binance_client import _okx_get
from src.stock_mark_context import MarkContextSource,SYMBOLS

ENDPOINT='/api/v5/market/history-mark-price-candles'


def parse_history(rows,start,end):
    unique={};conflicts=0
    for row in rows:
        if not isinstance(row,(list,tuple)) or len(row)!=6 or str(row[5]) not in ('0','1'):
            raise ValueError('invalid_mark_schema')
        stamp=int(row[0])
        if stamp%900000:raise ValueError('unaligned_mark_history')
        if str(row[5])!='1' or not start<=stamp//1000<end:continue
        prices=[float(x) for x in row[1:5]]
        if not all(math.isfinite(x) and x>0 for x in prices):raise ValueError('invalid_mark_prices')
        o,h,l,c=prices
        if not l<=min(o,c)<=max(o,c)<=h:raise ValueError('invalid_mark_ohlc')
        if stamp in unique and unique[stamp]!=prices:conflicts+=1
        unique.setdefault(stamp,prices)
    times=sorted(unique)
    frame=dict(time=[t//1000 for t in times],**{key:[unique[t][i] for t in times]
        for i,key in enumerate(('open','high','low','close'))})
    gaps=[(a,b) for a,b in zip(frame['time'],frame['time'][1:]) if b-a!=900]
    return frame,dict(conflicting_duplicates=conflicts,gaps=gaps)


def fetch_symbol(symbol,meta,source,specs,mapping,folder,start_floor):
    trade_path=Path(meta['path']);raw=trade_path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=meta['sha256']:raise ValueError('trade_source_hash_mismatch')
    trade=pickle.loads(raw);instrument=meta['instrument']
    if not re.fullmatch(r'[A-Z0-9_-]{3,96}',instrument):raise ValueError('invalid_instrument_id')
    current=mapping.get(symbol[:-4])
    if current!=instrument:
        return dict(status='CONTRACT_MISMATCH',instrument=instrument,current_instrument=current)
    start=max(start_floor,int(trade['time'][0]));end=int(trade['time'][-1])+900
    cache=folder/'pages'/symbol;cache.mkdir(parents=True,exist_ok=True)
    after=end*1000;pages=[];rows=[]
    for _ in range(400):
        page_path=cache/f'{after}.json'
        params=dict(instId=instrument,bar='15m',limit=100,after=str(after))
        if page_path.exists():
            page_raw=page_path.read_bytes();stored=json.loads(page_raw)
            if stored['params']!=params or stored['endpoint']!=ENDPOINT:raise ValueError('page_request_mismatch')
            body=stored['response']
        else:
            source._pace();body=_okx_get(ENDPOINT,params,timeout=8)
            stored=dict(endpoint=ENDPOINT,params=params,received_at=time.time(),response=body)
            page_raw=json.dumps(stored,sort_keys=True,separators=(',',':')).encode()
            page_path.write_bytes(page_raw)
        if str(body.get('code','0'))!='0':raise ValueError('cached_exchange_error')
        page=body.get('data',[])
        pages.append(dict(path=str(page_path),sha256=hashlib.sha256(page_raw).hexdigest()))
        if not page:break
        oldest=min(int(r[0]) for r in page)
        if oldest>=after:raise ValueError('historical_pagination_not_advancing')
        if max(int(r[0]) for r in page)>end*1000:raise ValueError('mark_request_range_ignored')
        rows.extend(page);after=oldest
        if oldest<=start*1000 or len(page)<100:break
    else:raise ValueError('historical_page_bound_exceeded')
    frame,quality=parse_history(rows,start,end)
    if not frame['time']:raise ValueError('empty_mark_history')
    path=folder/f'{symbol}_mark_15m.pkl';payload=pickle.dumps(frame,protocol=pickle.HIGHEST_PROTOCOL)
    if path.exists() and path.read_bytes()!=payload:raise ValueError('existing_mark_snapshot_disagrees')
    if not path.exists():path.write_bytes(payload)
    status=('OK' if frame['time'][0]==start and frame['time'][-1]==end-900 and not quality['gaps']
        else 'SHORT_OR_GAPPED_HISTORY')
    return dict(status=status,instrument=instrument,path=str(path),sha256=hashlib.sha256(payload).hexdigest(),
        trade_path=str(trade_path),trade_sha256=meta['sha256'],requested_start=start,requested_end=end,
        start=frame['time'][0],end=frame['time'][-1]+900,bars=len(frame['time']),
        kind='mark',volume_available=False,spec=specs.get(instrument),pages=pages,**quality)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trade-manifest',type=Path,default=Path('reports/audit_2026_09_08/stock_xperp_forward_nov14_merged/manifest.json'))
    p.add_argument('--output',type=Path,default=Path('reports/audit_2026_09_08/stock_mark_matched_nov18'))
    p.add_argument('--symbols',nargs='+',default=list(SYMBOLS))
    args=p.parse_args();meta=json.loads(args.trade_manifest.read_text(encoding='utf-8'))
    if any(s not in SYMBOLS for s in args.symbols):raise ValueError('symbol_outside_declared_universe')
    args.output.mkdir(parents=True,exist_ok=True)
    source=MarkContextSource();mapping,specs=source.inventory()
    report=dict(status='MATCHED_NATIVE_MARK_PRICE_SOURCE_NOT_EXECUTION',created_at=time.time(),symbols={},
        source_manifest=str(args.trade_manifest),source_manifest_sha256=hashlib.sha256(args.trade_manifest.read_bytes()).hexdigest(),
        endpoint=ENDPOINT,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitations=['Public historical mark is a feature source, not executable bid/ask or private fill history.',
            'Only exact instruments matching the archived trade data and current contract are accepted.',
            'Previously inspected market periods; no untouched validation or profitability claim.'])
    start=datetime(2026,7,1,tzinfo=timezone.utc).timestamp()
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs={}
        for symbol in args.symbols:
            item=meta['symbols'].get(symbol,{})
            if not item.get('path'):
                report['symbols'][symbol]=dict(status='NO_MATCHING_TRADE_HISTORY');continue
            jobs[pool.submit(fetch_symbol,symbol,item,source,specs,mapping,args.output,start)]=symbol
        for future in as_completed(jobs):
            symbol=jobs[future]
            try:value=future.result()
            except Exception as exc:value=dict(status='UNRESOLVED',error_type=type(exc).__name__)
            report['symbols'][symbol]=value
            (args.output/'manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
            print(json.dumps(dict(symbol=symbol,status=value['status'],bars=value.get('bars'),pages=len(value.get('pages',[])))),flush=True)
    print('finished',len(report['symbols']),flush=True)


if __name__=='__main__':main()
