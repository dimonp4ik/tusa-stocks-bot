import unittest

import pandas as pd

import stock_profit_first_multisession_oct17_audit as audit


class StockProfitFirstMultisessionAuditTest(unittest.TestCase):
    def test_selection_is_profit_first(self):
        value = {"return_pct": 1, "dd_pct": 15,
                 "trades_per_calendar_day": 3,
                 "literal_target_hit_rate": 25}
        self.assertTrue(audit.selection_ok(value))

    def test_objective_ignores_win_rate(self):
        value = {"months": 1, "dd_pct": 9,
                 "trades_per_calendar_day": 3,
                 "months_ge_30pct": 1,
                 "literal_target_hit_rate": 25}
        self.assertTrue(audit.objective_ok(value))


if __name__ == "__main__":
    unittest.main()
