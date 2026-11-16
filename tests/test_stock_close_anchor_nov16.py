import copy
import pandas as pd
import pytest
from stock_close_anchor_nov16_audit import resolve,causal_candidates
from src.stock_xperp_native_shadow import AUDITED_SYMBOLS


def outcome_candles(opening):
    return dict(time=[i*900 for i in range(65)],open=[100]+[opening]*64,
        high=[100]+[opening]*64,low=[100]+[opening]*64,close=[100]+[opening]*64)


def test_known_reference_does_not_receive_a_future_open_price_as_its_fill():
    row=dict(entry_index=1,direction_n=1,audited_target_pct=2.5)
    result=resolve(row,outcome_candles(120),reference='known_signal_close',cap_target_gap=True)
    assert result['entry']==100 and result['net_r']==pytest.approx(2.5)
    original=resolve(row,outcome_candles(120),reference='historical_first_open',cap_target_gap=False)
    assert original['entry']==120 and original['net_r']==0


def test_observed_stop_gap_is_not_clamped_to_one_r():
    row=dict(entry_index=1,direction_n=1,audited_target_pct=2.5)
    result=resolve(row,outcome_candles(95),reference='known_signal_close',cap_target_gap=True)
    assert result['net_r']==pytest.approx(-5)


def test_incomplete_horizon_cannot_be_called_a_resolved_time_exit():
    row=dict(entry_index=2,direction_n=1,audited_target_pct=2.5)
    with pytest.raises(ValueError):
        resolve(row,outcome_candles(100),reference='known_signal_close',cap_target_gap=True)


def test_future_prices_and_volume_cannot_change_past_candidate_decisions():
    universe={}
    for n,symbol in enumerate(sorted(AUDITED_SYMBOLS)[:10]):
        closes=[100.]*400
        if n==9:closes[250]=100.7
        universe[symbol]=dict(time=[i*900 for i in range(400)],open=[100.]*400,
            high=[max(100.,v)+.02 for v in closes],low=[min(100.,v)-.02 for v in closes],
            close=closes,volume=[1.]*400)
    stamp=251*900
    expected=causal_candidates(universe)
    expected=expected[expected.entry_time.le(stamp)]
    assert (expected.entry_time==stamp).any()
    changed=copy.deepcopy(universe)
    for data in changed.values():
        for field in ('open','high','low','close','volume'):
            data[field][251:]=[v*9 for v in data[field][251:]]
    observed=causal_candidates(changed)
    observed=observed[observed.entry_time.le(stamp)]
    pd.testing.assert_frame_equal(expected,observed)
