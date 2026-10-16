import unittest

import stock_profit_first_literal_runner_oct17_audit as audit


class StockProfitFirstLiteralRunnerAuditTest(unittest.TestCase):
    def test_objective_ignores_win_rate(self):
        value = {"months": 2, "dd_pct": 9,
                 "trades_per_calendar_day": 3,
                 "months_ge_30pct": 2, "literal_target_hit_rate": 35}
        self.assertTrue(audit.objective_ok(value))

    def test_selection_requires_profit_frequency_and_bounded_drawdown(self):
        value = {"return_pct": 1, "dd_pct": 15,
                 "trades_per_calendar_day": 3}
        self.assertTrue(audit.selection_ok(value))
        self.assertFalse(audit.selection_ok({**value, "return_pct": 0}))
        self.assertFalse(audit.selection_ok({**value, "dd_pct": 15.01}))


if __name__ == "__main__":
    unittest.main()
