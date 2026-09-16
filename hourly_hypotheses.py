"""Closed-hour hypotheses with 15-minute execution and 48h holding cap."""
import argparse,json,pickle,hashlib
from pathlib import Path
from entry_hypotheses import signal_at,FAMILIES,TARGETS
from src.backtest_integrity import simulate_exit
from filter_lab import gate,metrics,timestamp


def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',default='filter_universe');p.add_argument('--cap',type=int,required=True);a=p.parse_args()
 folder=Path('reports/audit_2026_09_08');mp=folder/(a.manifest+'.json');manifest=json.loads(mp.read_text());candles={}
 for name,meta in manifest['cache_manifest'].items():
  if meta['interval_sec'] not in (900,3600):continue
  raw=(Path('backtest_cache')/name).read_bytes()
  if hashlib.sha256(raw).hexdigest()!=meta['sha256']:raise ValueError(name)
  candles.setdefault(name.split('_')[0],{})[meta['interval_sec']]=pickle.loads(raw)
 plan={'families':FAMILIES,'targets':TARGETS,'stop_atr':2,'max_hold_hours':48,'roundtrip_cost':.0012,
       'max_cost_target_fraction':.25,'manifest_sha256':hashlib.sha256(mp.read_bytes()).hexdigest(),
       'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 (folder/('hourly_'+a.manifest+'_plan.json')).write_text(json.dumps(plan,indent=2),encoding='utf-8')
 results=[]
 for family in FAMILIES:
  for target in TARGETS:
   rows=[];skipped_cost=0
   for symbol,pair in candles.items():
    h=pair[3600];c=pair[900];lookup={t:i for i,t in enumerate(c['time'])}
    for i in range(100,len(h['time'])+1):
     decision=h['time'][i-1]+3600
     j=lookup.get(decision)
     if j is None or j+192>len(c['time']):continue
     signal=signal_at(h,i,family)
     if signal is None:continue
     direction,atr,eff,z=signal;entry=float(c['open'][j]);risk=2*atr
     if not 0<risk<entry:continue
     if .0012*entry>.25*risk*target:skipped_cost+=1;continue
     side='LONG' if direction==1 else 'SHORT';sl=entry-direction*risk;tp=entry+direction*risk*target
     if tp<=0:continue
     end=j+192
     # Reject holding windows with missing candles; do not bridge gaps silently.
     if c['time'][end-1]-c['time'][j]!=191*900:continue
     r=simulate_exit(c,range(j,end),direction=side,entry=entry,sl=sl,tp1=tp,tp2=tp,
       atr=atr,tp1_fraction=0,trail=False,trail_mult=0,stop_on_close=False,backstop_r=1,choose_trail=lambda *a:0)
     rows.append(dict(symbol=symbol,direction=side,entry_time=decision,exit_time=c['time'][r.bar]+900,
       outcome=r.outcome,net_r=r.gross_r-.0012*entry/risk,risk_pct=risk/entry,size_mult=1))
   accepted=gate(sorted(rows,key=lambda r:(r['entry_time'],r['symbol'])),a.cap)
   t=timestamp('2026-06-01');v=timestamp('2026-07-01')
   result={'family':family,'target':target,'skipped_cost':skipped_cost,'all':metrics(accepted),
     'early':metrics([r for r in accepted if r['exit_time']<t]),
     'june':metrics([r for r in accepted if t<=r['entry_time'] and r['exit_time']<v]),
     'july_august':metrics([r for r in accepted if r['entry_time']>=v])}
   results.append(result)
   (folder/('hourly_'+a.manifest+'_review.json')).write_text(json.dumps({'status':'RESEARCH_ONLY','plan':plan,'results':results,
    'limitations':['Previously inspected SWAP history; funding and executable spread omitted.',
     'Full final target exit, zero partial TP1; different from current live trailing.',
     'Equal R sizing, not legacy risk weights. No live policy selected from this sweep.']},indent=2),encoding='utf-8')
   print(json.dumps({'family':family,'target':target,'all':result['all']}),flush=True)
if __name__=='__main__':main()
