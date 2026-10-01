import os
import unittest
from unittest.mock import patch

os.environ["BOT_STARTUP_ENABLED"] = "0"

import main


class VenueShadowTrackerTests(unittest.TestCase):
    def setup(self):
        return {
            "id": 7,
            "symbol": "TESTUSDT",
            "direction": "LONG",
            "entry_price": 100.0,
            "tp1": 101.0,
            "tp2": 101.0,
            "sl": 99.0,
            "atr": 1.0,
            "ts": 1000.0,
            "signal_bar_ts": 0.0,
        }

    @staticmethod
    def candles(*, high, low, close):
        return {
            "time": [900], "open": [100.0], "high": [high],
            "low": [low], "close": [close], "volume": [1.0],
        }

    def test_ambiguous_intrabar_market_entry_is_stop_first(self):
        value = main._resolve_unsent_venue_setup(
            self.setup(), self.candles(high=101.2, low=98.8, close=98.9), now=1900)
        self.assertIsNotNone(value)
        self.assertEqual(value[:3], ("SL", 0, 0))
        self.assertLessEqual(value[3], -1.0)

    def test_target_resolves_with_exact_r(self):
        value = main._resolve_unsent_venue_setup(
            self.setup(), self.candles(high=101.2, low=99.5, close=101.0), now=1900)
        self.assertEqual(value[:3], ("TP2", 1, 1))
        self.assertAlmostEqual(value[3], 1.0)

    def test_open_setup_stays_unresolved(self):
        value = main._resolve_unsent_venue_setup(
            self.setup(), self.candles(high=100.5, low=99.5, close=100.2), now=1900)
        self.assertIsNone(value)

    def test_calendar_expiry_records_mark_to_market_r(self):
        setup = self.setup()
        value = main._resolve_unsent_venue_setup(
            setup, self.candles(high=100.5, low=99.5, close=100.2),
            now=setup["ts"] + (main.SIGNAL_EXPIRY_MAX_DAYS + 1) * 86400,
        )
        self.assertEqual(value[:3], ("EXPIRED", 0, 0))
        self.assertAlmostEqual(value[3], .2)

    def test_worker_only_persists_outcome(self):
        rows = [self.setup()]
        candles = self.candles(high=101.2, low=99.5, close=101.0)
        with patch.object(main, "get_unresolved_venue_setups", return_value=rows), \
                patch.object(main, "get_klines_xperp", return_value=candles), \
                patch.object(main, "mark_setup_resolved") as resolved, \
                patch.object(main, "send_signal") as sent, \
                patch.object(main.time, "time", return_value=1900):
            main._track_unsent_venue_setups()
        resolved.assert_called_once_with(7, "TP2", 1, 1, net_r=1.0)
        sent.assert_not_called()


if __name__ == "__main__":
    unittest.main()
