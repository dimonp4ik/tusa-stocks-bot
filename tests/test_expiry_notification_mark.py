from unittest.mock import patch
import pytest
from src import telegram_notifier as notifier


@pytest.mark.parametrize('direction,exit_price,expected',[
    ('LONG',102.5,'+1.2125R'),('SHORT',97.5,'+1.2125R'),
    ('LONG',99.6,'-0.1650R'),('SHORT',100.4,'-0.1650R')])
def test_expiry_message_reports_signed_model_mark_not_fictitious_breakeven(direction,exit_price,expected):
    sig=dict(symbol='BTCUSDT',direction=direction,entry_price=100,
        sl=98 if direction=='LONG' else 102,tp1=101 if direction=='LONG' else 99,
        tp2=104 if direction=='LONG' else 96,strategy_tp1_close_frac=.05,
        strategy_runner_mode='fixed_be')
    with patch.object(notifier,'_send_message',return_value=True) as send:
        assert notifier.send_signal_update(sig,'TP1_EXPIRED',exit_price)
    text=send.call_args.args[0]
    assert expected in text
    assert 'Расчётная цена' in text
    assert 'Исполнение на вашем счёте' in text
