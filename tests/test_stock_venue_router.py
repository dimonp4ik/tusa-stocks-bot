import unittest
from unittest.mock import patch

from src import binance_client
from src.stock_venue_router import (
    BALANCED_MODULES, FREQUENCY_MODULES, LOW_DRAWDOWN_MODULES,
    PRECISION_MODULES, PROFIT_MODULES, ROBUST_FREQUENCY_EXCLUDED_SYMBOLS,
    _entry_filter_matches, _target_for, stock_venue_setups,
)


class StockVenueRouterTests(unittest.TestCase):
    def test_profit_profile_produces_market_bracket_without_size(self):
        row = {
            "family": "opening_drive", "direction": "SHORT",
            "qqq_regime": "bear", "session_bucket": "00_30",
            "trend_5_20_dir_atr": 1.2,
        }
        with patch("src.stock_venue_router._latest_family_rows", return_value=[row]), \
                patch("src.stock_venue_router.attach_market_context"), \
                patch("src.stock_venue_router.attach_intraday_market_context"), \
                patch("src.stock_venue_router.market_context_by_day", return_value={}), \
                patch("src.stock_venue_router.market_intraday_context_by_time",
                      return_value={}), \
                patch("src.stock_venue_router.atr_at", return_value=2.0):
            setups = stock_venue_setups({"time": []}, {}, {}, market_price=100.0,
                                        profile="profit")
        self.assertEqual(len(PROFIT_MODULES), 6)
        self.assertEqual(len(setups), 1)
        self.assertEqual(setups[0]["module"], "drive_short_bear_open")
        self.assertEqual(setups[0]["entry_source"], "MARKET")
        self.assertEqual((setups[0]["entry"], setups[0]["sl"], setups[0]["tp"]),
                         (100.0, 102.0, 98.0))
        self.assertNotIn("size", setups[0])
        self.assertNotIn("size_mult", setups[0])

    def test_precision_profile_only_contains_frozen_precision_module(self):
        self.assertEqual(len(PRECISION_MODULES), 1)
        self.assertEqual(PRECISION_MODULES[0].name, "drive_short_bear_open")
        row = {
            "family": "opening_reclaim", "direction": "LONG",
            "qqq_regime": "bear", "session_bucket": "45_90",
            "spy_opening_range_atr": 2.0, "signal_body_dir_atr": .2,
        }
        with patch("src.stock_venue_router._latest_family_rows", return_value=[row]), \
                patch("src.stock_venue_router.attach_market_context"), \
                patch("src.stock_venue_router.attach_intraday_market_context"), \
                patch("src.stock_venue_router.market_context_by_day", return_value={}), \
                patch("src.stock_venue_router.market_intraday_context_by_time",
                      return_value={}), \
                patch("src.stock_venue_router.atr_at", return_value=2.0):
            self.assertEqual(stock_venue_setups({"time": []}, {}, {}, market_price=100.0,
                                                profile="precision"), [])

    def test_balanced_profile_has_frozen_targets_and_no_gap_fade(self):
        self.assertEqual(len(BALANCED_MODULES), 5)
        self.assertNotIn("gap_fade", {module.family for module in BALANCED_MODULES})
        self.assertEqual(
            [(module.name, module.target_r) for module in BALANCED_MODULES],
            [
                ("drive_short_bear_open", 1.0),
                ("reclaim_long_bear_45_90", .25),
                ("breakout_long_bear_45_90", .25),
                ("breakout_long_bear_open", .5),
                ("drive_long_bear_open", .25),
            ],
        )
        row = {
            "family": "opening_reclaim", "direction": "LONG",
            "qqq_regime": "bear", "session_bucket": "45_90",
            "spy_opening_range_atr": 2.0, "signal_body_dir_atr": .2,
        }
        with patch("src.stock_venue_router._latest_family_rows", return_value=[row]), \
                patch("src.stock_venue_router.attach_market_context"), \
                patch("src.stock_venue_router.attach_intraday_market_context"), \
                patch("src.stock_venue_router.market_context_by_day", return_value={}), \
                patch("src.stock_venue_router.market_intraday_context_by_time",
                      return_value={}), \
                patch("src.stock_venue_router.atr_at", return_value=2.0):
            setups = stock_venue_setups({"time": []}, {}, {}, market_price=100.0,
                                        profile="balanced")
        self.assertEqual(len(setups), 1)
        self.assertEqual(setups[0]["entry_source"], "MARKET")
        self.assertEqual((setups[0]["entry"], setups[0]["sl"], setups[0]["tp"]),
                         (100.0, 98.0, 100.5))
        self.assertNotIn("size", setups[0])

    def test_frequency_and_low_drawdown_profiles_are_frozen(self):
        self.assertEqual(
            [(module.name, module.target_r) for module in FREQUENCY_MODULES],
            [
                ("drive_short_bear_open", 1.0),
                ("reclaim_long_bear_45_90", .25),
                ("gap_fade_short_bear_open", 1.0),
                ("breakout_long_bear_45_90", .25),
                ("breakout_long_bear_open", .5),
                ("drive_long_bear_open", .25),
            ],
        )

    def test_robust_frequency_requires_symbol_and_blocks_calibration_losers(self):
        self.assertEqual(
            ROBUST_FREQUENCY_EXCLUDED_SYMBOLS,
            {"GOOGLUSDT", "INTCUSDT", "SPYUSDT", "TSLAUSDT"},
        )
        for symbol in (None, "GOOGLUSDT", "INTCUSDT", "SPYUSDT", "TSLAUSDT"):
            self.assertEqual(
                stock_venue_setups(
                    {"time": []}, {}, {}, market_price=100.0,
                    profile="robust_frequency", symbol=symbol,
                ),
                [],
            )
        row = {
            "family": "opening_drive", "direction": "SHORT",
            "qqq_regime": "bear", "session_bucket": "00_30",
            "trend_5_20_dir_atr": 1.2,
        }
        with patch("src.stock_venue_router._latest_family_rows", return_value=[row]), \
                patch("src.stock_venue_router.attach_market_context"), \
                patch("src.stock_venue_router.attach_intraday_market_context"), \
                patch("src.stock_venue_router.market_context_by_day", return_value={}), \
                patch("src.stock_venue_router.market_intraday_context_by_time",
                      return_value={}), \
                patch("src.stock_venue_router.atr_at", return_value=2.0):
            allowed = stock_venue_setups(
                {"time": []}, {}, {}, market_price=100.0,
                profile="robust_frequency", symbol="AAPLUSDT",
            )
        self.assertEqual(len(allowed), 1)
        self.assertEqual(
            [(module.name, module.target_r) for module in LOW_DRAWDOWN_MODULES],
            [
                ("drive_short_bear_open", 1.0),
                ("breakout_long_bear_45_90", .75),
                ("breakout_long_bear_open", .5),
            ],
        )

    def test_robust_dynamic_upgrades_target_only_when_frozen_gate_matches(self):
        base = {
            "family": "opening_breakout", "direction": "LONG",
            "qqq_regime": "bear", "session_bucket": "45_90",
            "gap_dir_atr": -1.25,
        }
        patches = (
            patch("src.stock_venue_router._latest_family_rows", return_value=[base]),
            patch("src.stock_venue_router.attach_market_context"),
            patch("src.stock_venue_router.attach_intraday_market_context"),
            patch("src.stock_venue_router.market_context_by_day", return_value={}),
            patch("src.stock_venue_router.market_intraday_context_by_time",
                  return_value={}),
            patch("src.stock_venue_router.atr_at", return_value=2.0),
        )
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
            static = stock_venue_setups(
                {"time": []}, {}, {}, market_price=100.0,
                profile="robust_frequency", symbol="AAPLUSDT",
            )
            dynamic = stock_venue_setups(
                {"time": []}, {}, {}, market_price=100.0,
                profile="robust_dynamic", symbol="AAPLUSDT",
            )
        self.assertEqual(static[0]["target_r"], .25)
        self.assertEqual(dynamic[0]["target_r"], 1.0)
        self.assertEqual(dynamic[0]["tp"], 102.0)

    def test_all_dynamic_target_boundaries_fail_closed(self):
        breakout_late = FREQUENCY_MODULES[3]
        breakout_open = FREQUENCY_MODULES[4]
        drive_open = FREQUENCY_MODULES[5]
        self.assertEqual(_target_for(
            breakout_late, {"gap_dir_atr": -1.0}, "robust_dynamic"), 1.0)
        self.assertEqual(_target_for(
            breakout_late, {"gap_dir_atr": -.999}, "robust_dynamic"), .25)
        self.assertEqual(_target_for(
            breakout_late, {"gap_dir_atr": float("nan")}, "robust_dynamic"), .25)
        self.assertEqual(_target_for(
            breakout_open, {"signal_range_atr": 2.0}, "robust_dynamic"), 1.0)
        self.assertEqual(_target_for(
            breakout_open, {"signal_range_atr": 1.999}, "robust_dynamic"), .5)
        self.assertEqual(_target_for(
            drive_open, {"prior_day_dir_atr": 1.0}, "robust_dynamic"), 1.0)
        self.assertEqual(_target_for(
            drive_open, {"prior_day_dir_atr": 1.001}, "robust_dynamic"), .25)
        self.assertEqual(_target_for(
            drive_open, {"prior_day_dir_atr": 0.0}, "robust_frequency"), .25)

    def test_entry_filtered_profile_boundaries_fail_closed(self):
        gap_fade = FREQUENCY_MODULES[2]
        breakout_late = FREQUENCY_MODULES[3]
        drive_open = FREQUENCY_MODULES[5]
        profile = "robust_dynamic_entry_filtered"
        self.assertTrue(_entry_filter_matches(
            gap_fade, {"qqq_gap_dir_atr": -1.0}, profile))
        self.assertFalse(_entry_filter_matches(
            gap_fade, {"qqq_gap_dir_atr": -.999}, profile))
        self.assertFalse(_entry_filter_matches(
            gap_fade, {"qqq_gap_dir_atr": float("nan")}, profile))
        self.assertTrue(_entry_filter_matches(
            breakout_late, {"opening_range_atr": 2.0}, profile))
        self.assertFalse(_entry_filter_matches(
            breakout_late, {"opening_range_atr": 1.999}, profile))
        self.assertFalse(_entry_filter_matches(breakout_late, {}, profile))
        self.assertTrue(_entry_filter_matches(
            drive_open, {"prior_range_ratio": 1.5}, profile))
        self.assertFalse(_entry_filter_matches(
            drive_open, {"prior_range_ratio": 1.501}, profile))
        self.assertTrue(_entry_filter_matches(
            gap_fade, {"qqq_gap_dir_atr": 10.0}, "robust_dynamic"))

    def test_xperp_fetch_uses_history_endpoint_after_first_page(self):
        def row(timestamp):
            value = float(timestamp)
            return [str(timestamp * 1000), str(value), str(value + 1),
                    str(value - 1), str(value), "0", "1", "0", "1"]

        first = [row(timestamp) for timestamp in range(500, 200, -1)]
        second = [row(timestamp) for timestamp in range(200, 98, -1)]
        with binance_client._kl_lock:
            binance_client._kl_cache.clear()
        with patch.object(binance_client, "get_xperp_instruments",
                          return_value={"AAPL": "AAPL_USD_UM_XPERP-15SEP26"}), \
                patch.object(binance_client, "_okx_get",
                             side_effect=[{"data": first}, {"data": second}]) as request:
            candles = binance_client.get_klines_xperp("AAPLUSDT", limit=400)
        self.assertEqual(request.call_count, 2)
        self.assertEqual(request.call_args_list[0].args[0], "/api/v5/market/candles")
        self.assertEqual(request.call_args_list[1].args[0],
                         "/api/v5/market/history-candles")
        self.assertEqual(len(candles["time"]), 400)
        self.assertEqual(candles["time"], sorted(candles["time"]))
        self.assertEqual(request.call_args_list[1].args[1]["after"], first[-1][0])

    def test_xperp_signal_fetch_excludes_forming_candle(self):
        def row(timestamp, confirmed):
            value = float(timestamp)
            return [str(timestamp * 1000), str(value), str(value + 1),
                    str(value - 1), str(value), "0", "1", "0", confirmed]

        page = [row(5, "0"), row(4, "1"), row(3, "1"), row(2, "1"), row(1, "1")]
        with binance_client._kl_lock:
            binance_client._kl_cache.clear()
        with patch.object(binance_client, "get_xperp_instruments",
                          return_value={"AAPL": "AAPL_USD_UM_XPERP-15SEP26"}), \
                patch.object(binance_client, "_okx_get", return_value={"data": page}):
            closed = binance_client.get_klines_xperp("AAPLUSDT", limit=3)
            with_forming = binance_client.get_klines_xperp(
                "AAPLUSDT", limit=3, include_forming=True)
        self.assertEqual(closed["time"], [2, 3, 4])
        self.assertEqual(closed["confirmed"], [1, 1, 1])
        self.assertEqual(with_forming["time"], [2, 3, 4, 5])
        self.assertEqual(with_forming["confirmed"], [1, 1, 1, 0])


if __name__ == "__main__":
    unittest.main()
