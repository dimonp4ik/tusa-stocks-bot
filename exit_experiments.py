"""Finite offline exit experiment; all variants recorded, never selects a live policy."""
import json
import subprocess
import sys
from pathlib import Path

VARIANTS = [
    ('early_tight', .25, .15, 'hard_stop'),
    ('early_wide', .25, .5, 'hard_stop'),
    ('later_wide', 1., .5, 'hard_stop'),
    ('wide_close', .6, .5, 'current'),
]

def main():
    root=Path(__file__).resolve().parent
    is_stocks=root.name=='tusa stocks'
    symbols=('AAPLUSDT,AMZNUSDT,METAUSDT,NVDAUSDT,TSLAUSDT,SPYUSDT,XAUUSDT,XAGUSDT' if is_stocks else
             'BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,LINKUSDT,SUIUSDT,ADAUSDT,AVAXUSDT')
    folder=root/'reports/audit_2026_09_08'
    plan={'variants':VARIANTS,'symbols':symbols,'status':'EXPLORATORY_ONLY',
          'constraints':['TP1 partial fraction unchanged at configured 0%',
                         'Previously inspected history; no untouched holdout claim',
                         'All variants retained including failures; no live selection',
                         'Next validation must include unseen windows and actual venue costs']}
    (folder/'exit_experiment_plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    for label,activation,trail,profile in VARIANTS:
        cmd=[sys.executable,'audit_replay.py','--label','exit_'+label,'--symbols',symbols,
             '--activation-r',str(activation),'--trail-distance-atr',str(trail),'--profile',profile]
        with (folder/('exit_'+label+'.log')).open('w',encoding='utf-8') as log:
            result=subprocess.run(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'{label} failed; see experiment log')
        report=json.loads((folder/('exit_'+label+'.json')).read_text())
        print(json.dumps({'variant':label,**report['gated']}),flush=True)

if __name__=='__main__':main()
