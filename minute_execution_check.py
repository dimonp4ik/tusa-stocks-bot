"""Read-only minute-resolution diagnostic for preselected fixed-target entries."""
import argparse,csv,json,hashlib,time
from pathlib import Path
import requests
from src.backtest_integrity import simulate_exit

def main():
 p=argparse.ArgumentParser();p.add_argument('--symbols',required=True);a=p.parse_args()
 folder=Path('reports/audit_2026_09_08');minute_dir=folder/'minute_cache';minute_dir.mkdir(exist_ok=True)
 path=folder/'entry_filter_universe_cost10_trend_pullback_0.25.csv'
 with path.open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
 selected=[]
 for symbol in a.symbols.split(','):
  selected.extend([r for r in rows if r['symbol']==symbol and float(r['entry_time'])>=1785542400][:5])
 results=[]
 for row in selected:
  start=int(float(row['entry_time']));end=start+48*900
  symbol=row['symbol'];inst=symbol[:-4]+'-USDT-SWAP';cache=minute_dir/f'{symbol}_{start}_{end}.json'
  if cache.exists():data=json.loads(cache.read_text())
  else:
   data=[];after=end*1000
   for page in range(4):
    response=requests.get('https://www.okx.com/api/v5/market/history-candles',params={'instId':inst,'bar':'1m','after':after,'limit':300},timeout=20)
    response.raise_for_status();body=response.json()
    if body.get('code')!='0':raise RuntimeError(str(body))
    batch=body.get('data',[])
    if not batch:break
    data.extend(batch);oldest=min(int(x[0]) for x in batch)
    if oldest<=start*1000:break
    if oldest>=after:raise ValueError('Nonadvancing pagination')
    after=oldest;time.sleep(.15)
   cache.write_text(json.dumps(data),encoding='utf-8')
  minute={int(x[0])//1000:x for x in data if start<=int(x[0])//1000<end and x[-1]=='1'}
  expected=list(range(start,end,60))
  if sorted(minute)!=expected:
   results.append({'symbol':symbol,'entry_time':start,'error':'Incomplete confirmed minute coverage','bars':len(minute)})
   continue
  c={k:[float(minute[t][j]) for t in expected] for j,k in enumerate(['time','open','high','low','close'])}
  entry=float(row['entry']);sl=float(row['sl']);target=float(row['tp1']);risk=abs(entry-sl)
  actual=simulate_exit(c,range(len(expected)),direction=row['direction'],entry=entry,sl=sl,tp1=target,tp2=target,
       atr=risk/2,tp1_fraction=0,trail=False,trail_mult=0,stop_on_close=False,backstop_r=1,choose_trail=lambda *a:0)
  net=actual.gross_r-.0004*entry/risk
  result={'symbol':symbol,'entry_time':start,'minute_net_r':net,'fifteen_minute_net_r':float(row['net_r']),
          'difference_r':net-float(row['net_r']),'minute_outcome':actual.outcome,'original_outcome':row['outcome'],
          'cache_sha256':hashlib.sha256(cache.read_bytes()).hexdigest()}
  results.append(result);print(json.dumps(result),flush=True)
 report={'status':'EXECUTION_DIAGNOSTIC_ONLY','selection':'First five accepted cost10 trend pullback 0.25R entries per named symbol after 2026-08-01.',
         'limitations':['Subset of already accepted trades; portfolio gates not replayed, no strategy return claim.',
                        'Fixed final exit diagnostic, not production trailing validation. SWAP candles, not X-Perp.',
                        'Minute OHLC still has intraminute uncertainty. Same fee/slippage estimate; funding excluded.'],
         'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'results':results}
 (folder/'minute_execution_diagnostic.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
if __name__=='__main__':main()
