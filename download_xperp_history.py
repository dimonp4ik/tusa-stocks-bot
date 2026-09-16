"""Download public X-Perp candles matching frozen SWAP cache windows. No keys/orders."""
import argparse,hashlib,json,pickle,time
from pathlib import Path
import requests

BARS={900:'15m',3600:'1H',14400:'4H',86400:'1Dutc'}

def main():
 p=argparse.ArgumentParser();p.add_argument('--symbols',required=True);a=p.parse_args()
 folder=Path('reports/audit_2026_09_08');source=json.loads((folder/'filter_universe.json').read_text())
 instruments=json.loads((folder/'public_xperp_instruments.json').read_text())['data']
 mapping={x['instId'].split('-')[0]+'USDT':x['instId'] for x in instruments if '_UM_XPERP-' in x.get('instId','') and x.get('state')=='live'}
 out=folder/'xperp_cache';out.mkdir(exist_ok=True);results=[]
 symbols=a.symbols.split(',')
 for name,meta in source['cache_manifest'].items():
  symbol=name.split('_')[0]
  if symbol not in symbols:continue
  inst=mapping.get(symbol)
  if not inst:raise ValueError('No X-Perp instrument for '+symbol)
  dest=out/name
  if dest.exists():
   raw=dest.read_bytes();data=pickle.loads(raw)
  else:
   after=int(meta['end']*1000)+1;allrows={}
   for page in range(100):
    response=requests.get('https://www.okx.com/api/v5/market/history-candles',params={'instId':inst,'bar':BARS[meta['interval_sec']],'after':after,'limit':300},timeout=25)
    response.raise_for_status();body=response.json()
    if body.get('code')!='0':raise RuntimeError(str(body))
    batch=body.get('data',[])
    if not batch:break
    for row in batch:
     t=int(row[0])//1000
     if meta['start']<=t<=meta['end'] and row[-1]=='1':allrows[t]=row
    oldest=min(int(row[0]) for row in batch)
    if oldest<=meta['start']*1000:break
    if oldest>=after:raise RuntimeError('Pagination stalled')
    after=oldest;time.sleep(.16)
   times=sorted(allrows)
   data={'time':times,**{key:[float(allrows[t][i]) for t in times] for i,key in enumerate(('open','high','low','close','volume'),1)}}
   raw=pickle.dumps(data);dest.write_bytes(raw)
  original=pickle.loads((Path('backtest_cache')/name).read_bytes())
  matched=len(set(data['time'])&set(original['time']))
  item=dict(file=name,instrument=inst,bars=len(data['time']),original_bars=meta['bars'],matched_timestamps=matched,
    exact_timestamps=data['time']==original['time'],sha256=hashlib.sha256(raw).hexdigest())
  results.append(item);print(json.dumps(item),flush=True)
  (out/('manifest_'+ '_'.join(symbols)+'.json')).write_text(json.dumps({'status':'PUBLIC_VENUE_DATA','results':results},indent=2),encoding='utf-8')

if __name__=='__main__':main()
