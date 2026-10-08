import unittest

from src.research_profit_gate import (
    economic_gate,
    minimum_binary_hit_rate_pct,
    projected_monthly_return_pct,
    required_signed_price_return_pct,
)


class ResearchProfitGateTests(unittest.TestCase):
    def test_required_edge_uses_client_margin_and_fixed_leverage(self):
        required = required_signed_price_return_pct(
            monthly_target=.30, margin_fraction=.03, leverage=10,
            trades_per_day=3, market_days=22)
        self.assertAlmostEqual(required, 1.32770926, places=6)

    def test_micro_target_cannot_reach_monthly_objective(self):
        self.assertIsNone(minimum_binary_hit_rate_pct(
            target_pct=.01, stop_pct=1, monthly_target=.30,
            margin_fraction=.03, leverage=10,
            trades_per_day=3, market_days=22))

    def test_two_percent_target_needs_about_eighty_percent_hits(self):
        required = minimum_binary_hit_rate_pct(
            target_pct=2, stop_pct=1, monthly_target=.30,
            margin_fraction=.03, leverage=10,
            trades_per_day=3, market_days=22)
        self.assertIsNotNone(required)
        self.assertGreater(required, 77)
        self.assertLess(required, 80)

    def test_projection_uses_geometric_trade_growth(self):
        projected = projected_monthly_return_pct(
            [.02]*8+[-.01]*2, trades_per_day=3,
            margin_fraction=.03, leverage=10, market_days=22)
        self.assertGreater(projected, 30)

    def test_positive_but_too_small_system_fails(self):
        result = economic_gate(
            [.005]*6+[-.01]*4, observed_trades_per_day=3,
            monthly_target=.30, required_trades_per_day=3,
            margin_fraction=.03, leverage=10, market_days=22)
        self.assertTrue(result["frequency_pass"])
        self.assertFalse(result["return_pass"])
        self.assertFalse(result["passes"])


if __name__ == "__main__":
    unittest.main()
