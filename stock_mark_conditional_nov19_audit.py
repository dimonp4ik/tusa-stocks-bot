"""One fixed price-return regressor; September selection, October onward replay."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from stock_mark_conditional_features_nov19 import FEATURES, mature_before
from stock_mark_conditional_pool_nov19 import load_pool, ROOT, META
from stock_module_cluster_audit import portfolio_gate
from stock_structure_geometry_nov17_audit import readiness_masks
from config import NATIVE_XPERP_DAILY_LOSS_R_CAP


def stamp(value):
    return pd.Timestamp(value, tz='UTC').timestamp()


TRAIN_END=stamp('2026-09-01')
SELECT_MID=stamp('2026-09-16')
SELECT_END=stamp('2026-10-01')
THRESHOLDS=(0., .0005, .001, .002, .004, .006)
MODEL=dict(max_leaf_nodes=7, min_samples_leaf=100, max_iter=150,
    learning_rate=.05, l2_regularization=5., early_stopping=False,
    random_state=20261119)
COLUMNS=['feature__'+name for name in FEATURES]


def train_model(frame, model=None):
    training=frame[mature_before(frame, TRAIN_END)].copy()
    if training.empty or not np.isfinite(training[COLUMNS+['signed_return']].to_numpy()).all():
        raise ValueError('invalid_training_matrix')
    if model is None:
        from sklearn.ensemble import HistGradientBoostingRegressor
        model=HistGradientBoostingRegressor(**MODEL)
    # All contemporaneous assets/targets/families share one timestamp's weight.
    weights=1./training.groupby('entry_time').entry_time.transform('size').to_numpy()
    weights*=len(weights)/weights.sum()
    with threadpool_limits(limits=2):
        model.fit(training[COLUMNS].to_numpy(), training.signed_return.to_numpy(), sample_weight=weights)
    return model, dict(rows=len(training), first_entry=float(training.entry_time.min()),
        last_entry=float(training.entry_time.max()), last_exit=float(training.exit_time.max()),
        full_horizon_purged=True, target='signed_price_return', timestamp_balanced=True)


def rank_events(frame, threshold):
    chosen=frame[frame.predicted_price_return.ge(threshold)].copy()
    if not np.isfinite(frame.predicted_price_return).all():raise ValueError('invalid_prediction')
    return chosen.sort_values(['entry_time','symbol','predicted_price_return','module','direction','target_fraction'],
        ascending=[True,True,False,True,True,True],kind='stable').drop_duplicates(['symbol','entry_time'])


def gate(frame, threshold):
    ranked=rank_events(frame, threshold)
    records=portfolio_gate(ranked.to_dict('records'),None,rolling_hour_cap=4,rolling_direction_hour_cap=2,
        rank_key=lambda r:(-r['predicted_price_return'],r['module'],r['direction'],r['target_fraction']),
        loss_streak_cap=2,direction_cap=4,daily_loss_r_cap=NATIVE_XPERP_DAILY_LOSS_R_CAP)
    return pd.DataFrame(records, columns=frame.columns)


def metrics(frame):
    values=frame.sort_values(['exit_time','entry_time','symbol']).signed_return.to_numpy(dtype=float)
    cumulative=np.r_[0.,np.cumsum(values)]
    return dict(signals=len(frame), sum_price_return=float(values.sum()),
        maximum_additive_price_drawdown=float(np.max(np.maximum.accumulate(cumulative)-cumulative)),
        profitable_exit_pct=100*float(np.mean(values>0)) if len(values) else None,
        sum_r=float(frame.net_r.sum()))


def select_threshold(frame):
    # Neither labels nor candidates from October may enter this selection.
    halves=[frame[frame.entry_time.ge(lo)&mature_before(frame,hi)]
        for lo,hi in ((TRAIN_END,SELECT_MID),(SELECT_MID,SELECT_END))]
    diagnostics={};qualified=[]
    for threshold in THRESHOLDS:
        parts=[metrics(gate(half,threshold)) for half in halves]
        valid=all(p['signals']>=15 and p['sum_price_return']>0 for p in parts)
        score=min(p['sum_price_return']-p['maximum_additive_price_drawdown'] for p in parts)
        diagnostics[str(threshold)]=dict(halves=parts,qualified=valid,selection_score=score)
        if valid:qualified.append((score,sum(p['sum_price_return'] for p in parts),threshold))
    chosen=max(qualified)[2] if qualified else None
    return chosen,diagnostics


def main():
    declaration=ROOT/'stock_mark_conditional_nov19_protocol.json'
    sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        (Path(__file__),Path('stock_mark_conditional_features_nov19.py'),
         Path('stock_mark_conditional_pool_nov19.py'),Path('stock_module_cluster_audit.py'),
         Path('stock_structure_geometry_nov17_audit.py'))}
    protocol=dict(declared_at=datetime.now(timezone.utc).isoformat(),sources=sources,model=MODEL,
        thresholds=THRESHOLDS,train_end=TRAIN_END,selection_end=SELECT_END,
        selection_minimum_signals_per_half=15,selection_requires_both_positive=True,
        selection_score='minimum half (sum price returns minus additive price drawdown)',
        later_start=SELECT_END,refit_after_selection=False)
    if declaration.exists():
        prior=json.loads(declaration.read_text(encoding='utf-8'))
        if prior['sources']!=sources:raise ValueError('conditional_declared_source_changed')
        protocol=prior
    else:declaration.write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    frame,meta=load_pool()
    model,training=train_model(frame)
    model_path=ROOT/'stock_mark_conditional_nov19_model.pkl'
    model_path.write_bytes(pickle.dumps(dict(model=model,features=FEATURES,protocol=protocol),protocol=5))
    print('conditional_fit_complete',training,flush=True)
    with threadpool_limits(limits=2):
        # Do not even score the later cohort until selection has been locked.
        selection=frame[frame.entry_time.ge(TRAIN_END)&frame.entry_time.lt(SELECT_END)].copy()
        selection['predicted_price_return']=model.predict(selection[COLUMNS].to_numpy())
        threshold,selection_report=select_threshold(selection)
        print('conditional_selection',threshold,selection_report,flush=True)
        results={};later_metrics=None
        if threshold is not None:
            later=frame[frame.entry_time.ge(SELECT_END)].copy()
            later['predicted_price_return']=model.predict(later[COLUMNS].to_numpy())
            accepted=gate(later,threshold)
            masks=readiness_masks(accepted) if len(accepted) else None
            accepted['forward_r_ready']=masks.r_ready if masks is not None else False
            accepted['forward_price_ready']=masks.price_ready if masks is not None else False
            path=ROOT/'stock_mark_conditional_nov19_later.pkl';accepted.to_pickle(path)
            months={}
            for month in ('2026-10','2026-11'):
                subset=accepted[pd.to_datetime(accepted.exit_time,unit='s',utc=True).dt.strftime('%Y-%m').eq(month)]
                months[month]=metrics(subset)
            results['fixed_price_regressor']=dict(status='RESOLVED_REFERENCE_PATHS',
                trades_path=str(path),trades_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                monthly_signal_diagnostics=months,signals=len(accepted))
            later_metrics=dict(all=metrics(accepted),by_exit_month=months,
                fresh_forward_ready=int((accepted.forward_r_ready&accepted.forward_price_ready).sum()))
    report=dict(status='EXPLORATORY_CONDITIONAL_PRICE_RETURN',objective_verified=False,
        live_activation_supported=False,protocol=protocol,training=training,
        pool_sha256=meta['sha256'],feature_names=FEATURES,chosen_threshold=threshold,
        selection=selection_report,later=later_metrics,results=results,
        sources=dict(sources,**{str(META):hashlib.sha256(META.read_bytes()).hexdigest(),
            str(model_path):hashlib.sha256(model_path.read_bytes()).hexdigest()}),
        limitations=['All market eras were inspected in earlier studies; this is no pristine independent holdout.',
            'Mark triggers and 5bp entry/exit price stresses are hypothetical, not confirmed bid/ask fills.',
            'July-August model, September selection only; no later-period refit or threshold adjustment.',
            'October portfolio and last20 PAPER readiness start fresh without September outcomes.',
            'Eleven currently matched contracts; historical contract specs and data vintages not independently verified.',
            'Additive price-return diagnostics are not deposit returns; shared lot-rounded cash is separate.',
            'November is partial; no live activation, risk-setting change or monthly-profit promise.'])
    (ROOT/'stock_mark_conditional_nov19.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print('conditional_later',later_metrics,flush=True)


if __name__=='__main__':main()
