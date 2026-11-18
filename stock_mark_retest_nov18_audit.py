"""Fixed retest repair, using the same source and exit conventions as baseline."""
from pathlib import Path
from mark_native_study_runner_nov18 import run_study
from stock_mark_retest_nov18 import candidates


if __name__=='__main__':
    run_study(candidates,kind='retest16',prefix='stock_mark_retest_nov18',parameters=dict(
        breakout_bars=16,pending_bars=4,breakout_extra_atr=.1,touch_tolerance_atr=.1,
        maximum_retest_extension_atr=.75,invalid_close_atr=.5,minimum_close_location=.6,
        stop_structure_bars=2,stop_atr_buffer=.25,minimum_risk=.003,maximum_risk=.015,
        minimum_target_risk=1.5,targets=[.01,.015,.02],sessions=['regular','all'],
        trigger_kinds=['trade','mark'],execution_price_stresses=[0.,.0005],horizon_bars=32),
        code_paths=[Path(__file__),Path('stock_mark_retest_nov18.py'),Path('stock_mark_hypotheses_nov18.py')])
