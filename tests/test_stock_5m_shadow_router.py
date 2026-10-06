import unittest
from datetime import datetime, timezone

from src.stock_5m_shadow_router import (
    PULLBACK_LADDER_TREND_MIN, PULLBACK_LADDER_VARIANT,
    PULLBACK_MODULE, RANGE_MODULE, candidate_context, shadow_setup,
)


def candles(values, end_time, volumes=None):
    count = len(values)
    return {
        "time": [end_time - 300 * (count - 1 - index) for index in range(count)],
        "open": [value for value in values],
        "high": [value + .6 for value in values],
        "low": [value - .6 for value in values],
        "close": list(values),
        "volume": list(volumes or [1.0] * count),
    }


class StockFiveMinuteShadowRouterTests(unittest.TestCase):
    def setUp(self):
        self.end = int(datetime(2026, 9, 10, 15, 0,
                                tzinfo=timezone.utc).timestamp())
        self.qqq = candles([100.0] * 110, self.end)

    def test_range_reversion_short_route_is_frozen_shadow_market_setup(self):
        values = [100.0] * 90 + [100.0 if index % 2 == 0 else 101.0
                                 for index in range(20)]
        values[-1] = 102.0
        stock = candles(values, self.end)
        context = candidate_context(stock, self.qqq, "AAPLUSDT")
        self.assertEqual(context["module"], RANGE_MODULE)
        setup = shadow_setup(
            stock, self.qqq, entry_time=self.end + 300,
            market_price=102.0, symbol="AAPLUSDT")
        self.assertEqual(setup["direction"], "SHORT")
        self.assertEqual(setup["target_r"], .5)
        self.assertGreaterEqual(setup["first_target_move_pct"], .5)
        self.assertTrue(setup["_shadow_experiment"])
        self.assertTrue(setup["_five_minute_shadow"])

    def test_trend_pullback_short_route_uses_two_atr_risk(self):
        values = [220 - .3 * index for index in range(110)]
        values[-2], values[-1] = 194.0, 193.0
        volumes = [1.0] * 110
        volumes[-1] = .5
        stock = candles(values, self.end, volumes)
        stock["high"] = [value + .8 for value in values]
        stock["low"] = [value - .8 for value in values]
        context = candidate_context(stock, self.qqq, "NVDAUSDT")
        self.assertEqual(context["module"], PULLBACK_MODULE)
        setup = shadow_setup(
            stock, self.qqq, entry_time=self.end + 300,
            market_price=193.0, symbol="NVDAUSDT")
        self.assertEqual(setup["target_r"], .25)
        self.assertAlmostEqual(setup["atr"], 2 * context["source_atr"])
        self.assertGreaterEqual(setup["first_target_move_pct"], .5)
        self.assertGreaterEqual(setup["trend_dir_atr"],
                                PULLBACK_LADDER_TREND_MIN)
        self.assertEqual(setup["_shadow_exit_variant"]["name"],
                         PULLBACK_LADDER_VARIANT)
        self.assertEqual(setup["_shadow_exit_variant"]["mid_target_r"], .5)
        self.assertEqual(setup["_shadow_exit_variant"]["far_target_r"], 4.0)

    def test_missing_alignment_and_unapproved_symbol_fail_closed(self):
        values = [100.0] * 90 + [100.0 if index % 2 == 0 else 101.0
                                 for index in range(20)]
        values[-1] = 102.0
        stock = candles(values, self.end)
        shifted = dict(self.qqq)
        shifted["time"] = [value + 60 for value in self.qqq["time"]]
        self.assertIsNone(candidate_context(stock, shifted, "AAPLUSDT"))
        self.assertIsNone(candidate_context(stock, self.qqq, "TSLAUSDT"))

    def test_micro_target_is_not_published_even_in_shadow(self):
        values = [100.0] * 90 + [100.0 if index % 2 == 0 else 100.05
                                 for index in range(20)]
        values[-1] = 100.2
        stock = candles(values, self.end)
        stock["high"] = [value + .01 for value in values]
        stock["low"] = [value - .01 for value in values]
        self.assertIsNone(shadow_setup(
            stock, self.qqq, entry_time=self.end + 300,
            market_price=100.2, symbol="AAPLUSDT"))

    def test_outside_regular_session_fails_closed(self):
        after_hours = int(datetime(2026, 9, 10, 21, 0,
                                   tzinfo=timezone.utc).timestamp())
        values = [100.0] * 90 + [100.0 if index % 2 == 0 else 101.0
                                 for index in range(20)]
        values[-1] = 102.0
        stock = candles(values, after_hours)
        qqq = candles([100.0] * 110, after_hours)
        self.assertIsNone(candidate_context(stock, qqq, "AAPLUSDT"))


if __name__ == "__main__":
    unittest.main()
