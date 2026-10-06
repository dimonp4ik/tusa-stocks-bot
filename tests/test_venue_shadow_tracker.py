import os
import unittest
from unittest.mock import patch

os.environ["BOT_STARTUP_ENABLED"] = "0"

import main


class VenueShadowTrackerTests(unittest.TestCase):
    @staticmethod
    def variant():
        return {
            "variant_id": 17, "symbol": "TESTUSDT", "direction": "LONG",
            "entry_price": 100.0, "tp1": 100.25, "sl": 99.0,
            "policy": {
                "first_fraction": .5, "mid_fraction": .2,
                "far_fraction": .3, "mid_target_r": .5,
                "far_target_r": 4.0, "entry_bar_ts": 900.0,
                "expire_at": 1800.0,
            },
        }

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

    def test_five_minute_shadow_uses_next_5m_bar_and_touch_stop(self):
        setup = {
            **self.setup(), "source": "stock_venue_shadow_5m",
            "ts": 1230.0, "signal_bar_ts": 900.0,
        }
        candles = {
            "time": [900, 1200], "open": [100.0, 100.0],
            "high": [101.5, 100.5], "low": [99.5, 98.8],
            "close": [101.0, 100.2], "volume": [1.0, 1.0],
        }
        with patch.object(main, "STOP_CLOSE_CONFIRM", True):
            value = main._resolve_unsent_venue_setup(setup, candles, now=1600)
        self.assertEqual(value[:3], ("SL", 0, 0))
        self.assertAlmostEqual(value[3], -1.0)

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

    def test_worker_fetches_five_minute_history_for_five_minute_source(self):
        setup = {
            **self.setup(), "source": "stock_venue_shadow_5m",
            "ts": 1230.0, "signal_bar_ts": 900.0,
        }
        candles = {
            "time": [1200], "open": [100.0], "high": [101.2],
            "low": [99.5], "close": [101.0], "volume": [1.0],
        }
        with patch.object(main, "get_unresolved_venue_setups",
                          return_value=[setup]), \
                patch.object(main, "get_klines_xperp",
                             return_value=candles) as fetch, \
                patch.object(main, "mark_setup_resolved") as resolved, \
                patch.object(main.time, "time", return_value=1600):
            main._track_unsent_venue_setups()
        self.assertEqual(fetch.call_args.kwargs["interval"], "5m")
        resolved.assert_called_once_with(7, "TP2", 1, 1, net_r=1.0)

    def test_ladder_variant_is_stop_first_on_ambiguous_bar(self):
        candles = {
            "time": [900], "open": [100.0], "high": [100.3],
            "low": [98.9], "close": [100.1], "confirmed": [True],
        }
        value = main._resolve_shadow_exit_variant(
            self.variant(), candles, now=1300)
        self.assertEqual(value, ("SL", -1.0))

    def test_ladder_variant_breakeven_starts_on_next_bar(self):
        candles = {
            "time": [900, 1200], "open": [100.0, 100.25],
            "high": [100.3, 100.6], "low": [99.5, 100.0],
            "close": [100.25, 100.0], "confirmed": [True, True],
        }
        value = main._resolve_shadow_exit_variant(
            self.variant(), candles, now=1600)
        self.assertEqual(value[0], "TP_NEAR_BE")
        self.assertAlmostEqual(value[1], .125)

    def test_ladder_variant_far_target_uses_all_three_fractions(self):
        candles = {
            "time": [900], "open": [100.0], "high": [104.1],
            "low": [99.5], "close": [104.0], "confirmed": [True],
        }
        value = main._resolve_shadow_exit_variant(
            self.variant(), candles, now=1300)
        self.assertEqual(value[0], "TP_FAR")
        self.assertAlmostEqual(value[1], 1.425)

    def test_ladder_variant_waits_for_final_session_bar_before_expiry_close(self):
        candles = {
            "time": [900, 1200], "open": [100.0, 100.1],
            "high": [100.2, 100.2], "low": [99.5, 99.5],
            "close": [100.1, 100.1], "confirmed": [True, True],
        }
        self.assertIsNone(main._resolve_shadow_exit_variant(
            self.variant(), candles, now=1900))

    def test_ladder_variant_worker_only_persists_paper_outcome(self):
        candles = {
            "time": [900], "open": [100.0], "high": [104.1],
            "low": [99.5], "close": [104.0], "confirmed": [True],
        }
        with patch.object(main, "get_unresolved_shadow_exit_variants",
                          return_value=[self.variant()]), \
                patch.object(main, "get_klines_xperp", return_value=candles), \
                patch.object(main, "mark_shadow_exit_variant_resolved") as saved, \
                patch.object(main, "send_signal") as sent, \
                patch.object(main.time, "time", return_value=1300):
            main._track_shadow_exit_variants()
        saved.assert_called_once_with(17, "TP_FAR", 1.425)
        sent.assert_not_called()


if __name__ == "__main__":
    unittest.main()
