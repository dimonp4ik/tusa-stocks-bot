"""Create matched SWAP windows for paired X-Perp source comparisons."""
import argparse,json,pickle,hashlib
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--symbols',required=True);p.add_argument('--only-1hour',action='store_true');a=p.parse_args()
 root=Path('reports/audit_2026_09_08');venue=root/'xperp_cache';out=root/'swap_matched_xperp';out.mkdir(exist_ok=True)
 manifest=json.loads((venue/('manifest_'+a.symbols.replace(',','_')+'.json')).read_text())
 if len(manifest['results'])!=len(a.symbols.split(','))*4:raise ValueError('Download incomplete')
 results=[]
 for info in manifest['results']:
  name=info['file']
  if a.only_1hour and '_1hour_' not in name:continue
  raw=(venue/name).read_bytes()
  if hashlib.sha256(raw).hexdigest()!=info['sha256']:raise ValueError(name)
  x=pickle.loads(raw);s=pickle.loads((Path('backtest_cache')/name).read_bytes());lookup={t:i for i,t in enumerate(s['time'])}
  if not x['time']:raise ValueError('Empty venue history')
  if any(t not in lookup for t in x['time']):raise ValueError('Unmatched timestamps')
  matched={key:[values[lookup[t]] for t in x['time']] for key,values in s.items()}
  data=pickle.dumps(matched);(out/name).write_bytes(data)
  results.append({'file':name,'bars':len(x['time']),'start':x['time'][0],'end':x['time'][-1],
                  'sha256':hashlib.sha256(data).hexdigest(),'xperp_sha256':info['sha256']})
 (out/('manifest_'+a.symbols.replace(',','_')+'.json')).write_text(json.dumps({'results':results,
  'purpose':'Same timestamps and warmup on both venues; source portability test, not actual account PnL.'},indent=2),encoding='utf-8')
 print(json.dumps(results))
if __name__=='__main__':main()
