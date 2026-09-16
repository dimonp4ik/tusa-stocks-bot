import unittest
from src.indicators import detect_bos,find_swing_points

class StructureTimingTests(unittest.TestCase):
    def test_latest_closed_candle_is_eligible(self):
        self.assertEqual(detect_bos([100]*11+[111],[(1,110)],[(1,90)],swing_lookback=2),('bullish',11,110))
    def test_old_break_uses_level_known_at_that_time(self):
        closes=[100]*15;closes[7]=111
        self.assertEqual(detect_bos(closes,[(1,110),(9,120)],[(1,90)],swing_lookback=2),('bullish',7,110))
    def test_future_confirmed_level_cannot_create_old_break(self):
        closes=[100]*15;closes[6]=115
        self.assertEqual(detect_bos(closes,[(1,120),(9,110)],[(1,90)],swing_lookback=2),(None,None,None))
    def test_pivot_requires_right_hand_confirmation(self):
        high=[1,2,5,2,1];low=[0]*5
        self.assertNotIn((2,5),find_swing_points(high[:4],low[:4],2)[0])
        self.assertIn((2,5),find_swing_points(high,low,2)[0])

    def test_sweep_cannot_use_not_yet_confirmed_level(self):
        from src.indicators import detect_liquidity_sweep
        highs=[105]*12;lows=[101]*12;closes=[103]*12
        lows[8]=99
        result=detect_liquidity_sweep(highs,lows,closes,[(1,110)],[(7,100)],swing_lookback=2)
        self.assertFalse(result['bullish'])
        lows[11]=99
        self.assertTrue(detect_liquidity_sweep(highs,lows,closes,[(1,110)],[(7,100)],swing_lookback=2)['bullish'])

    def test_micro_break_uses_latest_closed_bar(self):
        from unittest.mock import patch
        from src.indicators import detect_choch
        closes=[100]*20+[111]
        with patch('src.indicators.find_swing_points',return_value=([(2,110)],[(2,90)])):
            self.assertEqual(detect_choch(closes,closes,closes),'bullish')

    def test_micro_break_does_not_backdate_new_pivot(self):
        from unittest.mock import patch
        from src.indicators import detect_choch
        closes=[100]*21;closes[14]=115
        with patch('src.indicators.find_swing_points',return_value=([(2,120),(16,110)],[(2,90)])):
            self.assertIsNone(detect_choch(closes,closes,closes))
