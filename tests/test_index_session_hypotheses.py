import unittest
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from index_session_hypotheses import atr_at, candidates


NY = ZoneInfo("America/New_York")


class IndexSessionHypothesisTests(unittest.TestCase):
    def test_opening_breakout_enters_after_closed_break_bar(self):
        opening = datetime(2026, 7, 6, 9, 30, tzinfo=NY)
        times = [int((opening + timedelta(minutes=15 * i)).timestamp()) for i in range(4)]
        candles = {
            "time": times,
            "open": [100, 100, 100, 102],
            "high": [101, 101, 103, 103],
            "low": [99, 99, 100, 101],
            "close": [100, 100, 102, 102],
        }
        found = candidates(candles, "opening_breakout")
        self.assertEqual(found, [(1, 2, 3, 3)])

    def test_atr_does_not_read_entry_or_future_bars(self):
        candles = {
            "close": [100 + i for i in range(20)],
            "high": [101 + i for i in range(20)],
            "low": [99 + i for i in range(20)],
        }
        before = atr_at(candles, 16)
        candles["high"][16:] = [999] * 4
        candles["low"][16:] = [-999] * 4
        candles["close"][16:] = [500] * 4
        self.assertEqual(before, atr_at(candles, 16))
