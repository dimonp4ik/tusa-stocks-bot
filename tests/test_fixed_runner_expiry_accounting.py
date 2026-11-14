import os
os.environ['BOT_STARTUP_ENABLED']='0'
import pytest
from unittest.mock import patch
import main


@pytest.mark.parametrize('direction',['LONG','SHORT'])
@pytest.mark.parametrize('activation_bar',[0,10,47])
@pytest.mark.parametrize('stored_entry',[True,False])
def test_expiry_stays_anchored_to_entry_and_marks_remaining_position(direction,activation_bar,stored_entry):
    start=1_800_000_000
    bars=[]
    for i in range(49):
        if i<activation_bar:
            bar=(100.1,100.5,99.5,100.2)
        elif i==activation_bar:
            bar=(100.1,101.2,99.5,99.6 if activation_bar==47 else 101.)
        else:
            bar=(101.5,103.,100.1,102.5)
        if i==48:
            bar=(102.5,110.,100.1,108.)  # A target after the deadline is irrelevant.
        if direction=='SHORT':
            o,h,l,c=bar;bar=(200-o,200-l,200-h,200-c)
        bars.append(bar)
    frame=dict(time=[start+900*i for i in range(49)],confirmed=[True]*49,
        **{k:[b[j] for b in bars] for j,k in enumerate(('open','high','low','close'))})
    signal=dict(id=731,symbol='BTCUSDT',direction=direction,entry_price=100.,
        sl=98. if direction=='LONG' else 102.,tp1=101. if direction=='LONG' else 99.,
        tp2=104. if direction=='LONG' else 96.,atr=1.,opened_at=start,status='TP1_PARTIAL',
        tp1_hit_at=start+activation_bar*900+1,runner_activation_bar_ts=start+activation_bar*900,
        strategy_tp1_close_frac=.05,strategy_runner_mode='fixed_be',strategy_bar_seconds=900,
        strategy_entry_bar_ts=start,strategy_stop_on_close=0,strategy_max_bars=48)
    if not stored_entry:
        signal.pop('strategy_entry_bar_ts')
    with patch.object(main,'get_open_signals',return_value=[signal]), \
         patch.object(main.time,'time',return_value=start+49*900), \
         patch.object(main,'get_klines_xperp',return_value=frame), \
         patch.object(main,'update_signal_status') as update, \
         patch.object(main,'send_signal_update'), \
         patch.object(main.autotrader,'mirror_transition'):
        main._check_open_signals()
    assert update.call_args.args[1]=='TP1_EXPIRED'
    expected=bars[47][3]
    assert update.call_args.args[2]==expected
    sign=1 if direction=='LONG' else -1
    assert update.call_args.kwargs['realized_r']==pytest.approx(.025+.95*sign*(expected-100)/2)
