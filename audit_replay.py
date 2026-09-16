"""Offline replay with immutable cache inputs; never starts the bot or calls OKX.

Example: python audit_replay.py --label before --symbols BTCUSDT,ETHUSDT
Use the same cache manifest and arguments for before/after comparisons.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
from datetime import datetime, timezone


def summarize(trades):
    ordered = sorted(trades, key=lambda t: (t.exit_time or 0, t.symbol))
    rs = [t.net_r for t in ordered]
    gain = sum(r for r in rs if r > 0)
    loss = -sum(r for r in rs if r < 0)
    equity = peak = dd = 0.0
    for r in rs:
        equity += r
        peak = max(peak, equity)
        dd = max(dd, peak - equity)
    return dict(trades=len(rs), win_rate=100 * sum(r > 0 for r in rs) / len(rs) if rs else None,
                net_r=sum(rs), profit_factor=gain / loss if loss else None,
                closed_equity_drawdown_r=dd, worst_trade_r=min(rs) if rs else None)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--label', required=True)
    p.add_argument('--cache-dir', type=Path, help='Explicit alternate venue cache; same frozen filenames')
    p.add_argument('--symbols', required=True)
    p.add_argument('--candles', type=int, default=18000)
    p.add_argument('--end-date')
    p.add_argument('--stride', type=int, default=1)
    p.add_argument('--fee-rate', type=float, default=0.0005)
    p.add_argument('--slippage-rate', type=float, default=0.0001)
    p.add_argument('--profile', choices=['current','full_tp1','hard_stop','hard_stop_full_tp1','quote_levels'], default='current')
    p.add_argument('--activation-r', type=float, help='Research-only TP1 activation distance; partial fraction remains unchanged')
    p.add_argument('--trail-distance-atr', type=float, help='Research-only constant trailing distance, bypasses context profile')
    args = p.parse_args()
    if args.activation_r is not None and args.activation_r <= 0:
        p.error('--activation-r must be positive')
    if args.trail_distance_atr is not None and args.trail_distance_atr < 0:
        p.error('--trail-distance-atr must be nonnegative')
    # Prevent dotenv secrets/runtime overrides from contaminating this offline run.
    os.environ['PYTHON_DOTENV_DISABLED'] = '1'
    os.environ['AUTOTRADE_ENABLED'] = '0'
    if args.profile in ('full_tp1','hard_stop_full_tp1'):
        os.environ['TP1_CLOSE_FRAC'] = '1'
    if args.profile in ('hard_stop','hard_stop_full_tp1'):
        os.environ['STOP_CLOSE_CONFIRM'] = '0'
    if args.profile == 'quote_levels':
        os.environ['LEVELS_FROM_STRUCTURE'] = '0'
    os.environ['BOT_STARTUP_ENABLED'] = '0'
    if args.activation_r is not None:
        os.environ['TP1_R_MULT'] = str(args.activation_r)
    if args.trail_distance_atr is not None:
        os.environ['TRAIL_ATR_MULT'] = str(args.trail_distance_atr)
        os.environ['EXIT_PROFILE'] = 'constant_research'
    import backtest as bt
    import requests

    def no_network(*a, **kw):
        raise RuntimeError('Network disabled in audit replay')
    requests.sessions.Session.request = no_network
    manifest = {}
    missing_inputs = []
    end_ms = int(datetime.fromisoformat(args.end_date).replace(tzinfo=timezone.utc).timestamp() * 1000) if args.end_date else None

    def cached(symbol, interval, interval_sec, count, **kw):
        path = bt.cache_path(symbol, interval, count, kw.get('end_date_ms'))
        if args.cache_dir is not None:
            path = args.cache_dir / path.name
        if not path.is_file():
            missing_inputs.append(path.name)
            raise FileNotFoundError(f'Required frozen cache missing: {path.name}')
        raw = path.read_bytes()
        data = bt._normalize_cached_candles(pickle.loads(raw))
        if not data or not data.get('time'):
            raise ValueError(f'Invalid cached candles: {path.name}')
        times = data['time']
        if any(a >= b for a, b in zip(times, times[1:])):
            raise ValueError(f'Non-monotonic timestamps: {path.name}')
        if len({len(v) for v in data.values()}) != 1:
            raise ValueError(f'Mismatched candle arrays: {path.name}')
        manifest[path.name] = dict(sha256=hashlib.sha256(raw).hexdigest(), bars=len(times),
                                   start=times[0], end=times[-1], interval_sec=interval_sec)
        return data
    bt.fetch_history = cached
    # The disabled KNN overlay has no effect on decisions and is costly to recompute.
    if not getattr(bt, 'KNN_RISK_OVERLAY', False):
        bt.knn_direction_score = lambda *a, **kw: None
    results = []
    for symbol in args.symbols.split(','):
        r = bt.backtest_symbol(symbol, candles=args.candles, tp_window=192, warmup=300,
            stride=args.stride, window_15m=300, window_1h=90, window_4h=50,
            use_prefilter=True, refresh_cache=False, fee_rate=args.fee_rate,
            slippage_rate=args.slippage_rate, execution_delay_bars=0,
            adverse_entry_bps=0, exit_policy='trail', trail_atr_mult=bt.TRAIL_ATR_MULT,
            end_date_ms=end_ms)
        results.append(r)
        print(json.dumps(dict(symbol=symbol, error=r.error, seconds=round(r.elapsed_sec, 2),
                              **summarize(r.trade_records))), flush=True)
    raw = [t for r in results for t in r.trade_records]
    gated = bt.apply_live_gates(raw)
    folder = Path('reports') / 'audit_2026_09_08'
    folder.mkdir(parents=True, exist_ok=True)
    bt.write_trades_csv(str(folder / f'{args.label}.csv'), gated)
    bt.write_trades_csv(str(folder / f'{args.label}_raw.csv'), raw)
    report = dict(arguments={**vars(args), "cache_dir": str(args.cache_dir) if args.cache_dir else None}, git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  source_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in [Path('backtest.py'),Path('config.py'),*sorted(Path('src').glob('*.py'))]},
                  missing_inputs=missing_inputs,
                  errors={r.symbol: r.error for r in results if r.error},
                  raw=summarize(raw), gated=summarize(gated), cache_manifest=manifest,
                  limitations=['Historical development data; not untouched out-of-sample evidence.',
                               'Estimated execution costs; funding and account fills are not included.',
                               'Drawdown uses closed trades, excludes unrealized intratrade losses.',
                               'Live LLM/news gates and exchange microstructure are not reproduced.'])
    (folder / f'{args.label}.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'label': args.label, 'errors': report['errors'], 'gated': report['gated']}), flush=True)
    return 1 if report['errors'] or missing_inputs else 0


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    raise SystemExit(main())
