import unittest

import stock_vwap_structure_oct15_audit as audit


def proposal(entry, exit_, symbol="AAPLUSDT"):
    return {
        "symbol": symbol,
        "family": "vwap_reclaim",
        "direction": "LONG",
        "direction_n": 1,
        "entry_time": entry,
        "exit_time": exit_,
        "target_pct": .5,
        "stop_pct": 1.0,
        "deviation_atr": .3,
        "rvol": 1.2,
        "body_atr": .2,
        "market_move_dir_atr": .1,
        "market_vwap_side_dir": .1,
        "score": 1.0,
    }


class VwapStructureAuditTests(unittest.TestCase):
    def test_symbol_is_available_after_prior_position_exits(self):
        config = {"family": "vwap_reclaim", "deviation": .15,
                  "rvol": .8, "market": False}
        rows = [proposal(100, 200), proposal(150, 250), proposal(300, 400)]
        accepted = audit.select(rows, config)
        self.assertEqual([row["entry_time"] for row in accepted], [100, 300])

    def test_ambiguous_bar_counts_stop_before_target(self):
        candles = {
            "time": [100], "open": [100.0], "high": [101.0],
            "low": [99.0], "close": [100.2],
        }
        row = proposal(100, 100)
        row.update({"entry_index": 0, "session_end": 0,
                    "_execution_1": (0,), "_execution_2": (0,)})
        result = audit.resolve(row, candles, target_pct=.5, stop_pct=.5)
        self.assertEqual(result["outcome"], "SL")
        self.assertAlmostEqual(result["signed_return"], -.005)


if __name__ == "__main__":
    unittest.main()
