import numpy as np
import pandas as pd

from stock_native_residual_close_nov16_audit import factor_features, policy_masks


def test_future_prices_cannot_change_a_closed_factor_decision():
    rng = np.random.default_rng(1611)
    time = np.arange(400) * 900
    factor = 100 * np.exp(np.cumsum(rng.normal(0, .002, 400)))
    stock = 50 * np.exp(1.5 * np.log(factor / 100) + np.cumsum(rng.normal(0, .002, 400)))
    benchmark = dict(time=time.tolist(), close=factor.tolist())
    candles = {'TEST': dict(time=time.tolist(), close=stock.tolist())}
    before = factor_features(candles, benchmark)
    entry_time = int(time[250])
    # The signal closes at entry_time; both the entry bar and later bars are future.
    benchmark['close'][250:] = (factor[250:] * 1.4).tolist()
    candles['TEST']['close'][250:] = (stock[250:] * .6).tolist()
    after = factor_features(candles, benchmark)
    a = before[before.entry_time.le(entry_time)].reset_index(drop=True)
    b = after[after.entry_time.le(entry_time)].reset_index(drop=True)
    assert len(a) > 100
    pd.testing.assert_frame_equal(a, b)
    prefix = factor_features({'TEST': {k: v[:250] for k, v in candles['TEST'].items()}},
                             {k: v[:250] for k, v in benchmark.items()})
    pd.testing.assert_frame_equal(a, prefix.reset_index(drop=True))


def test_unknown_index_evidence_cannot_authorize_a_filtered_trade():
    frame = pd.DataFrame([dict(direction_n=1, beta=np.nan, factor_residual=np.nan,
                               residual_z=np.nan, factor_trend_32=np.nan)])
    assert all(not bool(mask.iloc[0]) for mask in policy_masks(frame).values())


def test_unused_missing_residual_z_does_not_block_a_known_index_guard():
    frame = pd.DataFrame([dict(direction_n=1, beta=1., factor_residual=-.003,
                               residual_z=np.nan, factor_trend_32=.01)])
    masks = policy_masks(frame)
    assert masks['avoid_index_countertrend'].iloc[0]
    assert masks['residual_move'].iloc[0]
    assert not masks['residual_move_and_z'].iloc[0]


def test_mature_factor_values_match_the_independent_older_cash_calculation():
    import stock_factor_residual_shock_oct26_audit as earlier
    rng = np.random.default_rng(1612)
    time = np.arange(400) * 900
    factor = 100 * np.exp(np.cumsum(rng.normal(0, .002, 400)))
    stock = 50 * np.exp(np.cumsum(rng.normal(0, .002, 400)))
    benchmark = dict(time=time.tolist(), close=factor.tolist())
    candles = {'TEST': dict(time=time.tolist(), close=stock.tolist())}
    a = earlier.feature_rows(candles, benchmark, 2, 96)
    b = factor_features(candles, benchmark)
    combined = a.merge(b, on=['symbol', 'entry_time'], validate='one_to_one', suffixes=('_old', '_new'))
    assert len(combined) > 150
    for field in ('beta', 'factor_return', 'factor_residual', 'residual_z'):
        np.testing.assert_allclose(combined[field + '_old'], combined[field + '_new'], atol=1e-12, rtol=1e-12)
