"""Native index-factor diagnostics with the causal known-close entry benchmark.

No model selection or live changes. Historical months have already been seen.
"""
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import stock_close_anchor_nov16_audit as anchor
import stock_factor_residual_shock_oct26_audit as residual
from src.stock_xperp_native_shadow import AUDITED_SYMBOLS

ROOT = Path('reports/audit_2026_09_08')
FACTORS = ('QQQUSDT', 'SPYUSDT')


def factor_features(candles, benchmark):
    times = np.asarray(benchmark['time'])
    close = pd.Series(benchmark['close'], index=times, dtype=float)
    trend = (close / close.shift(32) - 1).where(pd.Series(times, index=times).diff(32).eq(32 * 900))
    trend.index = trend.index + 900
    pieces = []
    for symbol, candle in candles.items():
        t = np.asarray(candle['time'])
        frame = pd.DataFrame(dict(time=t, stock=np.log(candle['close']),
            factor=np.log(close.reindex(t).to_numpy())))
        contiguous = frame.time.diff().eq(900)
        stock_one = frame.stock.diff().where(contiguous)
        factor_one = frame.factor.diff().where(contiguous)
        covariance = stock_one.rolling(96, min_periods=72).cov(factor_one)
        variance = factor_one.rolling(96, min_periods=72).var(ddof=1)
        beta = (covariance / variance.replace(0, np.nan)).shift(1)
        continuous_move = frame.time.diff(2).eq(1800)
        factor_move = frame.factor.diff(2).where(continuous_move)
        shock = frame.stock.diff(2).where(continuous_move) - beta * factor_move
        scale = shock.rolling(96, min_periods=72).std(ddof=1).shift(1)
        entry_time = t + 900
        pieces.append(pd.DataFrame(dict(symbol=symbol, entry_time=entry_time,
            beta=beta, factor_return=np.expm1(factor_move), factor_residual=np.expm1(shock),
            residual_z=shock / scale.replace(0, np.nan),
            factor_trend_32=pd.Series(entry_time).map(trend))))
    # Preserve unavailable fields independently. An index-only guard must not
    # silently require the residual-z warm-up or nonzero residual variance.
    return pd.concat(pieces, ignore_index=True).replace([np.inf, -np.inf], np.nan)


def policy_masks(frame):
    # Preserve a stock-specific shock in the original reversal direction.
    # The cut is half the production minimum raw two-bar move, fixed before
    # looking at this run's outcomes. Unavailable evidence does not pass.
    own = (frame.direction_n * frame.factor_residual <= -.002).fillna(False)
    against = (frame.direction_n * frame.beta * frame.factor_trend_32 >= -.005).fillna(False)
    return dict(residual_move=own,
                residual_move_and_z=own & frame.residual_z.abs().ge(1.),
                avoid_index_countertrend=against,
                residual_move_and_index=own & against)


def main():
    manifest = ROOT / 'stock_xperp_forward_nov14_merged/manifest.json'
    meta = json.loads(manifest.read_text(encoding='utf-8'))
    candles = {}
    sources = {}
    for symbol, item in meta['symbols'].items():
        if not item.get('path') or symbol not in AUDITED_SYMBOLS | set(FACTORS):
            continue
        path = Path(item['path'])
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != item['sha256']:
            raise ValueError('native source hash mismatch')
        candles[symbol] = pickle.loads(raw)
        sources[str(path)] = digest
    stocks = {s: c for s, c in candles.items() if s in AUDITED_SYMBOLS}
    last = min(c['time'][-1] for c in stocks.values())
    candidates = anchor.causal_candidates(stocks)
    candidates = candidates[candidates.entry_time.le(last - 63 * 900)].copy()
    resolved = pd.DataFrame([anchor.resolve(row, stocks[row['symbol']],
        reference='known_signal_close', cap_target_gap=True) for row in candidates.to_dict('records')])
    baseline = anchor.gate(resolved)
    expected = pd.read_pickle(ROOT / 'stock_close_anchor_nov16_causal_candidates_known_close.pkl')
    keys = ['symbol', 'entry_time', 'direction', 'net_r', 'exit_time']
    pd.testing.assert_frame_equal(baseline[keys].sort_values(keys[:3]).reset_index(drop=True),
        expected[keys].sort_values(keys[:3]).reset_index(drop=True), check_dtype=False, atol=1e-10, rtol=0)
    report = dict(status='EXPLORATORY_CAUSAL_REFERENCE_NOT_LIVE_APPROVAL', objective_verified=False,
        baseline_rows=len(baseline), candidate_rows=len(resolved),
        protocol=dict(beta_bars=96, signal_bars=2, index_trend_bars=32,
            minimum_residual_fraction=.002, maximum_countertrend_fraction=.005,
            minimum_residual_z=1., model_selection=False),
        coverage={}, results={'baseline': anchor.monthly(baseline)}, limitations=[
            'Already inspected 2026 native history; this is not an independent holdout.',
            'Known signal close is observable, not an executable bid/ask fill.',
            '15m stop-first paths, gross standalone 3% margin/10x closed entry cohorts.',
            'No shared-account fills, contract rounding, funding or intrabar equity.',
            'Initial listings and November coverage are partial; no native 2020-2025 evidence.'])
    print('baseline', {m: v['return_pct'] for m, v in report['results']['baseline'].items()}, flush=True)
    fields = ['symbol', 'entry_time', 'factor_return', 'beta', 'factor_residual', 'residual_z', 'factor_trend_32']
    for symbol in FACTORS:
        features = factor_features(stocks, candles[symbol])
        frame = resolved.merge(features[fields], on=['symbol', 'entry_time'], how='left', validate='one_to_one')
        report['coverage'][symbol] = dict(available=int(frame.beta.notna().sum()), total=len(frame))
        for name, mask in policy_masks(frame).items():
            admitted = anchor.gate(frame[mask])
            if admitted.empty:
                admitted = frame.iloc[:0].copy()
            key = f'{symbol}_{name}'
            report['results'][key] = anchor.monthly(admitted)
            print(key, {m: v['return_pct'] for m, v in report['results'][key].items()}, flush=True)
    complete_months = ('august', 'september', 'october')
    report['historical_checks'] = {
        name: dict(
            each_complete_month_at_least_30=all(values[m]['return_pct'] >= 30 for m in complete_months),
            no_complete_month_drawdown_increase=all(
                values[m]['dd_pct'] <= report['results']['baseline'][m]['dd_pct'] for m in complete_months))
        for name, values in report['results'].items()}
    report['live_change_approved'] = False
    sources.update({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
        manifest, Path(__file__), Path(anchor.__file__), Path(residual.__file__),
        Path('src/stock_xperp_native_shadow.py'), Path('stock_module_cluster_audit.py'), Path('config.py'),
        ROOT / 'stock_close_anchor_nov16_causal_candidates_known_close.pkl')})
    report['sources'] = sources
    out = ROOT / 'stock_native_residual_close_nov16.json'
    out.write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
