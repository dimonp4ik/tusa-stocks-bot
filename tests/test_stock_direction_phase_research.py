from stock_direction_phase_research import direction_for


def test_momentum_and_reversal_are_distinct_without_future_outcomes():
    assert direction_for(signal_return=.01,basket_trend_64=.05,policy="momentum") == 1
    assert direction_for(signal_return=.01,basket_trend_64=.05,policy="reversal") == -1
    assert direction_for(signal_return=-.01,basket_trend_64=-.05,policy="momentum") == -1


def test_aligned_momentum_refuses_counter_basket_entry():
    assert direction_for(signal_return=.01,basket_trend_64=-.05,policy="momentum_aligned") is None


def test_mixed_policy_boundary_is_fixed():
    assert direction_for(signal_return=.01,basket_trend_64=.0399,policy="mixed_strong_trend") == -1
    assert direction_for(signal_return=.01,basket_trend_64=.04,policy="mixed_strong_trend") == 1


def test_nonfinite_feature_cannot_emit_direction():
    assert direction_for(signal_return=.01,basket_trend_64=float("nan"),policy="momentum") is None
