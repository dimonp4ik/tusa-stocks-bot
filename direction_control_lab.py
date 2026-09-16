"""Frozen SMC candidates: compare direction with identical fixed exits."""
import argparse,csv,json,pickle,hashlib
from pathlib import Path
from filter_lab import gate,metrics,timestamp
from src.backtest_integrity import simulate_exit
p=argparse.ArgumentParser();p.add_argument('--label',default='all_structure_timing');p.add_argument('--cap',type=int,required=True);p.add_argument('--cache-dir',type=Path,default=Path('backtest_cache'));args=p.parse_args()
f=Path('reports/audit_2026_09_08');mp=f/(args.label+'.json');manifest=json.loads(mp.read_text(encoding='utf-8'));cache={}
for name,meta in manifest['cache_manifest'].items():
 if meta['interval_sec']!=900:continue
 raw=(args.cache_dir/name).read_bytes()
 if hashlib.sha256(raw).hexdigest()!=meta['sha256']:raise ValueError(name)
 c=pickle.loads(raw);cache[name.split('_')[0]]=(c,{t:i for i,t in enumerate(c['time'])})
source=f/(args.label+'_raw.csv');rows=list(csv.DictReader(source.open(encoding='utf-8')))
report={'status':'RESEARCH_ONLY','source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
 'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 'plan':{'directions':['original','reversed'],'targets':[.25,1.0],'timeout_bars':192,'roundtrip_cost':.0012,'max_cost_target_fraction':.25,'cache_dir':str(args.cache_dir)},'results':[],
 'limitations':['Previously inspected SWAP history; no fresh holdout or actual X-Perp execution.',
 'Funding excluded. Intrabar entries use conservative OHLC ordering.',
 'Entry candidate generation unchanged; no claim of complete independent opportunity set.',
 'Equal risk units replace legacy size weighting. No production change.']}
for mode in ('original','reversed'):
 for target in (.25,1.0):
  candidates=[]
  for row in rows:
   c,lookup=cache[row['symbol']];ts=float(row['entry_time']);j=lookup.get(ts)
   if j is None or j+192>len(c['time']):continue
   if c['time'][j+191]-c['time'][j]!=191*900:continue
   entry=float(row['entry']);risk=abs(entry-float(row['sl']))
   if risk<=0 or .0012*entry>.25*risk*target:continue
   sign=(1 if row['direction']=='LONG' else -1)*(1 if mode=='original' else -1)
   sl=entry-sign*risk;tp=entry+sign*risk*target
   if min(sl,tp)<=0:continue
   direction='LONG' if sign==1 else 'SHORT'
   r=simulate_exit(c,range(j,j+192),direction=direction,entry=entry,sl=sl,tp1=tp,tp2=tp,
    atr=float(row['atr_at_signal']),tp1_fraction=0,trail=False,trail_mult=0,stop_on_close=False,
    backstop_r=1,choose_trail=lambda *a:0,intrabar_entry=row['entry_is_intrabar']=='True')
   candidates.append(dict(symbol=row['symbol'],direction=direction,entry_time=ts,exit_time=c['time'][r.bar]+900,
    net_r=r.gross_r-.0012*entry/risk,risk_pct=risk/entry,size_mult=1,outcome=r.outcome))
  accepted=gate(sorted(candidates,key=lambda x:(x['entry_time'],x['symbol'])),args.cap)
  split=timestamp('2026-07-01')
  result={'mode':mode,'target':target,'all':metrics(accepted),'july_august':metrics([r for r in accepted if r['entry_time']>=split])}
  report['results'].append(result);print(json.dumps(result),flush=True)
(f/('direction_control_'+args.label+'.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')
