import unittest
from unittest.mock import patch

import pandas as pd

import stock_volume_pressure_oct16_audit as audit


def feature(**updates):
    row = {
        "symbol": "AAPLUSDT",
        "intraday": .004,
        "last_hour": .002,
        "session_pressure": .25,
        "last_hour_pressure": .2,
        "qqq_intraday": .003,
        "qqq_session_pressure": .3,
        "qqq_last_hour_pressure": .25,
        "flow_breadth": .875,
        "last_flow_breadth": .875,
        "flow_breadth_delta": .25,
        "rvol": 1.2,
        "qqq_rvol": 1.1,
    }
    row.update(updates)
    return row


class VolumePressureAuditTests(unittest.TestCase):
    def test_weighted_pressure_respects_volume_and_close_location(self):
        value = audit.weighted_pressure(
            high=pd.Series([11.0, 11.0]).to_numpy(),
            low=pd.Series([9.0, 9.0]).to_numpy(),
            close=pd.Series([11.0, 9.0]).to_numpy(),
            volume=pd.Series([3.0, 1.0]).to_numpy(),
            indices=pd.Series([0, 1], dtype="int64").to_numpy())
        self.assertAlmostEqual(value, .5)

    def test_checkpoint_pressure_cannot_read_entry_or_future_bars(self):
        frame = pd.DataFrame([{
            "symbol": "AAPLUSDT", "session_date": "2023-01-03",
            "slot": "early", "entry_index": 3,
        }])
        candle = {
            "high": [11.0] * 5, "low": [9.0] * 5,
            "close": [11.0, 11.0, 11.0, 9.0, 9.0],
            "volume": [1.0] * 5,
        }
        changed_future = {**candle, "close": [11.0] * 5,
                          "volume": [1.0, 1.0, 1.0, 1000.0, 1000.0]}
        table = (frame, 5, pd.Series(range(5), dtype="int64").to_numpy())
        groups = {"2023-01-03": [0, 1, 2, 3, 4]}

        with patch.object(audit.base, "session_table", return_value=table), \
                patch.object(audit, "regular_groups", return_value=groups):
            original = audit.checkpoint_volume("AAPLUSDT", candle)
            changed = audit.checkpoint_volume("AAPLUSDT", changed_future)

        self.assertAlmostEqual(original.iloc[0].session_pressure, 1.0)
        self.assertAlmostEqual(
            original.iloc[0].session_pressure,
            changed.iloc[0].session_pressure)
        self.assertAlmostEqual(
            original.iloc[0].last_hour_pressure,
            changed.iloc[0].last_hour_pressure)

    def test_continuation_is_symmetric_for_long_and_short(self):
        short = feature(
            symbol="MSFTUSDT", intraday=-.004, last_hour=-.002,
            session_pressure=-.25, last_hour_pressure=-.2,
            qqq_intraday=-.003, qqq_session_pressure=-.3,
            qqq_last_hour_pressure=-.25, flow_breadth=.125,
            last_flow_breadth=.125, flow_breadth_delta=-.25)

        result = audit.apply_spec(
            pd.DataFrame([feature(), short]), audit.SPECS[0])

        self.assertEqual(result.symbol.tolist(), ["AAPLUSDT", "MSFTUSDT"])
        self.assertEqual(result.direction_n.tolist(), [1, -1])
        self.assertTrue((result.session_pressure_dir > 0).all())


if __name__ == "__main__":
    unittest.main()
