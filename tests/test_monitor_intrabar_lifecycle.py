from unittest.mock import patch
from types import SimpleNamespace

import main
from src import telegram_notifier as notifier


def signal(**overrides):
    row = {
        "id": 77,
        "symbol": "TESTUSDT",
        "direction": "LONG",
        "entry_price": 100.0,
        "tp1": 101.0,
        "tp2": 102.0,
        "sl": 99.0,
        "opened_at": 1000.0,
        "status": "OPEN",
        "atr": 1.0,
        "strategy_bar_seconds": 300,
        "strategy_entry_bar_ts": 1000.0,
        "strategy_runner_mode": "fixed_be",
        "strategy_tp1_close_frac": .7,
        "strategy_stop_on_close": 0,
        "strategy_max_bars": 0,
        "strategy_expire_at": 0,
    }
    row.update(overrides)
    return row


def candle(*, high, low, close, confirmed=False):
    return {
        "time": [1000.0],
        "open": [100.0],
        "high": [high],
        "low": [low],
        "close": [close],
        "confirmed": [confirmed],
    }


def run_monitor(row, bars, *, deep_breached=True):
    with patch.object(main, "get_open_signals", return_value=[row]), \
         patch.object(main, "get_klines_xperp", return_value=bars), \
         patch.object(main, "update_signal_status") as update, \
         patch.object(main, "send_signal_update", return_value=True) as notify, \
         patch.object(main.autotrader, "mirror_transition") as mirror, \
         patch.object(main, "_deep_feed_breached", return_value=deep_breached), \
         patch.object(main, "time", SimpleNamespace(time=lambda: 2000.0)):
        main._check_open_signals()
    return update, notify, mirror


def test_one_poll_crossing_both_targets_records_and_notifies_both():
    update, notify, mirror = run_monitor(
        signal(), candle(high=102.4, low=99.5, close=102.2))

    assert [item.args[1] for item in update.call_args_list] == [
        "TP1_PARTIAL", "TP2_HIT"]
    assert [item.args[1] for item in notify.call_args_list] == [
        "TP1_PARTIAL", "TP2_HIT"]
    assert [item.args[1] for item in mirror.call_args_list] == [
        "TP1_PARTIAL", "TP2_HIT"]


def test_exchange_backstop_wins_over_tp2_inside_same_candle():
    # Even an optional deep-feed disagreement cannot undo a real exchange
    # backstop fill on the traded X-Perp venue.
    with patch.object(main, "STOP_EXCHANGE_BACKSTOP_R", 1.5), \
         patch("config.STOP_REQUIRE_GLOBAL_CONFIRM", True):
        update, notify, _ = run_monitor(
            signal(strategy_stop_on_close=1),
            candle(high=102.4, low=98.4, close=101.5),
            deep_breached=False)

    assert [item.args[1] for item in update.call_args_list] == ["SL_HIT"]
    assert update.call_args.args[2] == 98.5
    assert [item.args[1] for item in notify.call_args_list] == ["SL_HIT"]


def test_single_target_does_not_invent_intermediate_tp1_transition():
    update, notify, mirror = run_monitor(
        signal(tp1=102.0, tp2=102.0, strategy_runner_mode="single_target",
               strategy_tp1_close_frac=1.0),
        candle(high=102.4, low=99.5, close=102.2))

    assert [item.args[1] for item in update.call_args_list] == ["TP2_HIT"]
    assert [item.args[1] for item in notify.call_args_list] == ["TP2_HIT"]
    assert [item.args[1] for item in mirror.call_args_list] == ["TP2_HIT"]


def test_single_target_update_uses_single_target_wording():
    sent = []
    row = signal(tp1=102.0, tp2=102.0,
                 strategy_runner_mode="single_target",
                 strategy_tp1_close_frac=1.0)
    with patch.object(notifier, "_send_message",
                      side_effect=lambda message: sent.append(message) or True):
        assert notifier.send_signal_update(row, "TP2_HIT", 102.0)

    assert "ЦЕЛЬ ДОСТИГНУТА" in sent[0]
    assert "TP2 ДОСТИГНУТ" not in sent[0]
