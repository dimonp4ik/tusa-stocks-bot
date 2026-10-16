import unittest

import pandas as pd

import stock_breadth_transition_oct16_audit as audit


def feature(**updates):
    row = {
        "symbol": "AAPLUSDT",
        "session_date": "2023-01-03",
        "slot": "middle",
        "ret5": .01,
        "ret20": .02,
        "gap": 0.0,
        "intraday": .003,
        "last_hour": .002,
        "cross_z5": .5,
        "qqq_ret20": .02,
        "qqq_intraday": .006,
        "qqq_last_hour": .002,
        "breadth_intra": .875,
        "prior_breadth": .5,
        "prior_qqq_intraday": .002,
        "breadth_delta": .375,
        "qqq_delta": .004,
    }
    row.update(updates)
    return row


class BreadthTransitionAuditTests(unittest.TestCase):
    def test_transition_uses_only_prior_checkpoint_in_same_session(self):
        frame = pd.DataFrame([
            feature(symbol="AAPLUSDT", session_date="2023-01-03",
                    slot="early", breadth_intra=.25, qqq_intraday=-.004),
            feature(symbol="AAPLUSDT", session_date="2023-01-03",
                    slot="middle", breadth_intra=.625, qqq_intraday=.002),
            feature(symbol="AAPLUSDT", session_date="2023-01-04",
                    slot="early", breadth_intra=.875, qqq_intraday=.009),
            feature(symbol="MSFTUSDT", session_date="2023-01-03",
                    slot="early", breadth_intra=.25, qqq_intraday=-.004),
            feature(symbol="MSFTUSDT", session_date="2023-01-03",
                    slot="middle", breadth_intra=.5, qqq_intraday=.001),
        ])

        result = audit.transitions(frame)

        self.assertEqual(len(result), 2)
        aapl = result[result.symbol.eq("AAPLUSDT")].iloc[0]
        msft = result[result.symbol.eq("MSFTUSDT")].iloc[0]
        self.assertAlmostEqual(aapl.prior_breadth, .25)
        self.assertAlmostEqual(aapl.breadth_delta, .375)
        self.assertAlmostEqual(aapl.qqq_delta, .006)
        self.assertAlmostEqual(msft.prior_breadth, .25)
        self.assertAlmostEqual(msft.breadth_delta, .25)

    def test_acceleration_is_symmetric_for_long_and_short(self):
        long_row = feature()
        short_row = feature(
            symbol="MSFTUSDT", ret5=-.01, ret20=-.02,
            intraday=-.003, last_hour=-.002, cross_z5=-.5,
            qqq_ret20=-.02, qqq_intraday=-.006,
            qqq_last_hour=-.002, breadth_intra=.125,
            prior_breadth=.5, prior_qqq_intraday=-.002,
            breadth_delta=-.375, qqq_delta=-.004)

        result = audit.apply_spec(
            pd.DataFrame([long_row, short_row]), audit.SPECS[1])

        self.assertEqual(result.symbol.tolist(), ["AAPLUSDT", "MSFTUSDT"])
        self.assertEqual(result.direction_n.tolist(), [1, -1])
        self.assertTrue((result.breadth_delta_dir > 0).all())

    def test_reversal_direction_follows_completed_breadth_flip(self):
        row = feature(
            intraday=-.001, last_hour=-.002,
            qqq_intraday=.006, qqq_last_hour=-.002,
            breadth_intra=.25, prior_breadth=.75,
            breadth_delta=-.5, qqq_delta=.004)

        result = audit.apply_spec(pd.DataFrame([row]), audit.SPECS[3])

        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0].direction_n, -1)
        self.assertGreater(result.iloc[0].breadth_delta_dir, 0)


if __name__ == "__main__":
    unittest.main()
