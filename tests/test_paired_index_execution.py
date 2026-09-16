import unittest

from paired_index_execution import replay


class PairedIndexExecutionTests(unittest.TestCase):
    def candles(self):
        return {
            "time": [0, 900, 1800],
            "open": [200, 200, 200],
            "high": [201, 202, 201],
            "low": [199, 199, 199],
            "close": [200, 201, 200],
        }

    def row(self):
        return {
            "entry_time": "0", "exit_time": "2700", "entry": "100",
            "sl": "98", "tp1": "102", "tp2": "104", "direction": "LONG",
            "outcome": "EXPIRED", "size_mult": "1", "symbol": "TEST",
        }

    def test_levels_scale_into_execution_price_space(self):
        result = replay(self.row(), self.candles())
        self.assertEqual(result["basis_at_entry"], 2)
        self.assertEqual(result["outcome"], "EXPIRED")
        expected = (0 - 0.0006 * (200 + 200)) / 4
        self.assertAlmostEqual(result["net_r"], expected)

    def test_execution_gap_is_rejected(self):
        candles = self.candles()
        candles["time"][1] = 1200
        with self.assertRaisesRegex(ValueError, "execution gap"):
            replay(self.row(), candles)
