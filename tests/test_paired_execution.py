import unittest
from unittest.mock import patch
import paired_callback_execution as model

class PairedExecutionTests(unittest.TestCase):
    def row(self):return dict(symbol='TEST',entry_time=0,exit_time=1800,direction='LONG',entry=100,sl=90,tp1=106,tp2=120,size_mult=1,outcome='SL')
    def bars(self):return dict(time=[0,900],open=[100,99],high=[102,100],low=[98,89],close=[99,89])

    def test_callback_preserves_stop_outcome_for_portfolio_pause(self):
        with patch.object(model,'STOP_CLOSE_CONFIRM',True),patch.object(model,'STOP_EXCHANGE_BACKSTOP_R',2):
            trade=model.replay_callback(self.row(),self.bars())
        self.assertEqual(trade['outcome'],'SL')
        self.assertEqual(trade['exit_price'],89)

    def test_actual_exchange_backstop_precedes_callback(self):
        bars=self.bars();bars['low'][0]=70
        with patch.object(model,'STOP_CLOSE_CONFIRM',True),patch.object(model,'STOP_EXCHANGE_BACKSTOP_R',2):
            trade=model.replay_callback(self.row(),bars)
        self.assertEqual(trade['exit_time'],900)
        self.assertEqual(trade['exit_price'],80)

    def test_market_entry_beyond_target_is_skipped(self):
        bars=self.bars();bars['open'][0]=107
        self.assertIsNone(model.replay_callback(self.row(),bars))

    def test_missing_execution_bar_rejected(self):
        bars=self.bars();bars['time'][1]=1800
        with self.assertRaises(ValueError):model.replay_callback(self.row(),bars)
