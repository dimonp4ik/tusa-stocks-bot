"""Chronological regularized logistic entry model, research only. No runtime imports."""
import json,math,statistics,hashlib,argparse
from pathlib import Path
from filter_lab import load,gate,metrics,timestamp

NUMERIC=('volume_ratio','eff_ratio','rsi_directional','vol_atr_pct','vol_ratio_regime','extension','risk_pct','tp1_r','mtf_score')
CATEGORICAL=('direction','entry_source','trend_1h','trend_4h','session')

def schema(train):
    numeric={}
    for key in NUMERIC:
        vals=[r[key] for r in train if r.get(key) is not None and math.isfinite(r[key])]
        numeric[key]=(statistics.mean(vals),max(statistics.pstdev(vals),1e-9)) if vals else (0,1)
    cats={key:sorted({r.get(key,'') for r in train}) for key in CATEGORICAL}
    return numeric,cats

def vector(row,numeric,cats):
    values=[1.]
    for key,(mean,sd) in numeric.items():
        x=row.get(key);x=mean if x is None or not math.isfinite(x) else x
        values.append(max(-5,min(5,(x-mean)/sd)))
    for key,levels in cats.items():values.extend(float(row.get(key,'')==v) for v in levels)
    return values

def probability(x,w):
    z=max(-30,min(30,sum(a*b for a,b in zip(x,w))))
    return 1/(1+math.exp(-z))

def fit(rows,design):
    xs=[vector(r,*design) for r in rows];ys=[float(r['net_r']>0) for r in rows]
    w=[0.]*len(xs[0]);rate=.1;penalty=.01
    for epoch in range(300):
        gradient=[0.]*len(w)
        for x,y in zip(xs,ys):
            error=probability(x,w)-y
            for j,value in enumerate(x):gradient[j]+=error*value
        for j in range(len(w)):w[j]-=rate*(gradient[j]/len(rows)+(penalty*w[j] if j else 0))
    return w

def calibration(rows,probs):
    bins=[]
    for lower,upper in [(0,.5),(.5,.6),(.6,.7),(.7,.8),(.8,.85),(.85,.9),(.9,1.01)]:
        indices=[i for i,p in enumerate(probs) if lower<=p<upper]
        bins.append({'range':[lower,upper],'n':len(indices),
           'predicted':statistics.mean(probs[i] for i in indices) if indices else None,
           'observed':statistics.mean(rows[i]['net_r']>0 for i in indices) if indices else None})
    return {'brier':statistics.mean((p-float(r['net_r']>0))**2 for r,p in zip(rows,probs)) if rows else None,'bins':bins}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',default='multiyear_taker_raw.csv')
    parser.add_argument('--train-end',default='2025-01-01')
    parser.add_argument('--validation-end',default='2026-01-01')
    parser.add_argument('--cap',type=int,default=3)
    args=parser.parse_args()
    folder=Path('reports/audit_2026_09_08');p=folder/args.input;rows=load(p)
    train_end=timestamp(args.train_end);validation_end=timestamp(args.validation_end)
    train=[r for r in rows if r['exit_time']<train_end]
    design=schema(train);weights=fit(train,design)
    probs=[probability(vector(r,*design),weights) for r in rows]
    validation=[(r,q) for r,q in zip(rows,probs) if train_end<=r['entry_time'] and r['exit_time']<validation_end]
    thresholds=[]
    for cutoff in (.5,.55,.6,.65,.7,.75,.8,.85,.9):
        prefix=gate([r for r,q in zip(rows,probs) if r['entry_time']<validation_end and q>=cutoff],args.cap)
        val=metrics([r for r in prefix if train_end<=r['entry_time'] and r['exit_time']<validation_end])
        thresholds.append({'cutoff':cutoff,'validation':val})
    eligible=[x for x in thresholds if x['validation']['n']>=60 and x['validation']['stress_net_r']>0 and x['validation']['daily_lcb']>0]
    chosen=max(eligible,key=lambda x:x['validation']['daily_lcb'],default=None)
    if chosen:
        selected=gate([r for r,q in zip(rows,probs) if q>=chosen['cutoff']],args.cap)
        chosen={**chosen,'test':metrics([r for r in selected if r['entry_time']>=validation_end])}
    test=[(r,q) for r,q in zip(rows,probs) if r['entry_time']>=validation_end]
    report={'arguments':vars(args),'status':'RESEARCH_ONLY','training_rows':len(train),'schema':design,'weights':weights,
      'thresholds':thresholds,'chosen':chosen,
      'validation_calibration':calibration([r for r,q in validation],[q for r,q in validation]),
      'test_calibration':calibration([r for r,q in test],[q for r,q in test]),
      'input_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      'limitations':['Previously inspected historical SWAP candidates, no untouched live evidence. Funding omitted.',
       'No hyperparameter sweep. Standardization and category vocabulary fitted on training only.',
       'Calibration describes raw candidates, while selected returns use portfolio gates.',
       'High predicted probabilities are not guarantees; reject models lacking sufficient validation evidence.']}
    (folder/'chronological_logistic_review.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'chosen':chosen,'thresholds':thresholds,'test_calibration':report['test_calibration']}))
if __name__=='__main__':main()
