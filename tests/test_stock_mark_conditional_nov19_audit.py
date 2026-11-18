import numpy as np
import pandas as pd
from stock_mark_conditional_nov19_audit import (
    COLUMNS, TRAIN_END, SELECT_END, rank_events, select_threshold, train_model)


class CaptureModel:
    def fit(self,x,y,sample_weight):
        self.x=x.copy();self.y=y.copy();self.weights=sample_weight.copy()


def test_fit_excludes_unmatured_and_all_selection_and_later_labels():
    rows=[]
    for entry,exit_time,value in [(TRAIN_END-40*900,TRAIN_END-30*900,.01),
            (TRAIN_END-900,TRAIN_END-1,-.02),(TRAIN_END+900,TRAIN_END+1800,.03),
            (SELECT_END+900,SELECT_END+1800,.04)]:
        rows.append(dict(entry_time=entry,exit_time=exit_time,signed_return=value,
            outcome_known=True,**{name:1. for name in COLUMNS}))
    frame=pd.DataFrame(rows);first=CaptureModel();train_model(frame,first)
    changed=frame.copy();changed.loc[1:,'signed_return']=9999.
    second=CaptureModel();train_model(changed,second)
    np.testing.assert_array_equal(first.y,[.01]);np.testing.assert_array_equal(first.y,second.y)
    np.testing.assert_array_equal(first.x,second.x)


def test_ranking_chooses_only_prediction_and_has_deterministic_target_ties():
    frame=pd.DataFrame([dict(symbol='A',entry_time=1,module='a',direction='LONG',
        target_fraction=t,predicted_price_return=p,signed_return=result)
        for t,p,result in [(.01,.002,-1),(.02,.002,100),(.015,.003,-10)]])
    selected=rank_events(frame,.001)
    assert selected.target_fraction.tolist()==[.015]
    frame['signed_return']=frame.signed_return*-100
    assert rank_events(frame,.001).target_fraction.tolist()==[.015]


def test_october_labels_never_participate_in_september_threshold_choice():
    frame=pd.DataFrame([dict(symbol='A',entry_time=SELECT_END+900,exit_time=SELECT_END+1800,
        module='a',direction='LONG',target_fraction=.01,predicted_price_return=.003,
        signed_return=100.,net_r=100.,outcome_known=True)])
    chosen,diagnostics=select_threshold(frame)
    assert chosen is None
    assert all(p['signals']==0 for d in diagnostics.values() for p in d['halves'])
