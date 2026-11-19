import pytest
from unittest.mock import patch
import main
from test_monitor_intrabar_lifecycle import signal,run_monitor


@pytest.mark.parametrize('direction',['LONG','SHORT'])
@pytest.mark.parametrize('mode',['fixed_be','single_target'])
@pytest.mark.parametrize('price',[95.,105.])
def test_explicit_empty_native_bar_cannot_confirm_target_or_stop(direction,mode,price):
    row=signal(direction=direction,strategy_runner_mode=mode)
    if direction=='SHORT':row.update(sl=101.,tp1=99.,tp2=98.)
    if mode=='single_target':row.update(tp1=row['tp2'],strategy_tp1_close_frac=1.)
    frame=dict(time=[1000.],open=[price],high=[price],low=[price],close=[price],volume=[0.],confirmed=[True])
    update,notify,mirror=run_monitor(row,frame)
    update.assert_not_called();notify.assert_not_called();mirror.assert_not_called()


def test_zero_current_bar_cannot_trigger_or_amend_a_new_trailing_stop():
    row=signal(status='TP1_PARTIAL',strategy_runner_mode='post_tp1_v2',tp1_hit_at=900.,runner_trail_atr_mult=1.)
    frame=dict(time=[1000.],open=[99.],high=[99.],low=[99.],close=[99.],volume=[0.],confirmed=[True])
    with patch.object(main,'TRAIL_RUNNER_ENABLED',True),patch.object(main.autotrader,'update_trailing') as amend:
        update,notify,mirror=run_monitor(row,frame)
    update.assert_not_called();notify.assert_not_called();mirror.assert_not_called();amend.assert_not_called()


def test_real_earlier_target_remains_valid_when_followed_by_an_empty_bar():
    row=signal(status='TP1_PARTIAL',strategy_runner_mode='post_tp1_v2',tp1_hit_at=900.,runner_trail_atr_mult=1.)
    frame=dict(time=[1000.,1300.],open=[101.,102.2],high=[102.4,102.2],low=[101.,102.2],
        close=[102.2,102.2],volume=[1.,0.],confirmed=[True,False])
    with patch.object(main,'TRAIL_RUNNER_ENABLED',True),patch.object(main.autotrader,'update_trailing'):
        update,_,_=run_monitor(row,frame)
    assert update.call_args.args[1]=='TP2_HIT'


@pytest.mark.parametrize('price',[95.,105.])
def test_unsent_counterfactual_stays_unknown_without_new_trades(price):
    setup=dict(symbol='TESTUSDT',direction='LONG',entry_price=100.,sl=98.,tp1=101.,tp2=104.,atr=1.,
        ts=1801.,signal_bar_ts=900.,strategy_bar_seconds=900,strategy_entry_bar_ts=1800.,
        strategy_runner_mode='fixed_be',strategy_tp1_close_frac=.05,strategy_max_bars=1,strategy_expire_at=0.)
    frame=dict(time=[1800.],open=[price],high=[price],low=[price],close=[price],volume=[0.],confirmed=[True])
    assert main._resolve_unsent_venue_setup(setup,frame,now=2700.) is None


def test_volume_filter_handles_invalid_values_and_preserves_legacy_missing_volume():
    from src.candle_evidence import bar_has_trade,traded_bars
    frame=dict(close=[1.]*6,volume=[0.,-1.,float('nan'),float('inf'),None,.001])
    assert [i for i in range(6) if bar_has_trade(frame,i)]==[5]
    assert traded_bars(frame)==dict(close=[1.],volume=[.001])
    assert bar_has_trade(dict(close=[1.]),0)
