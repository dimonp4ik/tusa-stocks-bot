import unittest

import pandas as pd

from stock_forward_monitor import frozen_rows, validate_family_coverage
from src.stock_venue_router import MAX_INDEX_INTRADAY_MOVE_ATR


class StockForwardMonitorParityTests(unittest.TestCase):
    @staticmethod
    def _row(**updates):
        row = {
            "symbol": "AAPLUSDT",
            "entry_time": 1_790_000_000,
            "family": "opening_drive",
            "direction": "SHORT",
            "target_r": 1.0,
            "signal_minutes": 30,
            "qqq_regime": "bear",
            "trend_5_20_dir_atr": 1.1,
            "qqq_intraday_move_atr": 1.0,
            "spy_intraday_move_atr": -1.0,
        }
        row.update(updates)
        return row

    def test_strict_profile_rejects_abnormal_index_context(self):
        frame = pd.DataFrame([self._row(
            qqq_intraday_move_atr=MAX_INDEX_INTRADAY_MOVE_ATR + .01,
        )])
        self.assertEqual(frozen_rows(
            frame, "robust_dynamic_entry_filtered"), [])

    def test_unfiltered_profile_preserves_its_frozen_behavior(self):
        frame = pd.DataFrame([self._row(
            qqq_intraday_move_atr=MAX_INDEX_INTRADAY_MOVE_ATR + .01,
        )])
        rows = frozen_rows(frame, "robust_dynamic")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["module"], "drive_short_bear_open")

    def test_strict_profile_keeps_sane_context(self):
        frame = pd.DataFrame([self._row()])
        rows = frozen_rows(frame, "robust_dynamic_entry_filtered")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["module"], "drive_short_bear_open")

    def test_truncated_warmup_is_rejected_instead_of_reporting_zero(self):
        frame = pd.DataFrame([self._row(entry_time=2_000)])
        with self.assertRaisesRegex(ValueError, "longer warm-up window"):
            validate_family_coverage(frame, previous_max=1_000)

    def test_family_dataset_must_extend_beyond_reference(self):
        frame = pd.DataFrame([self._row(entry_time=1_000)])
        with self.assertRaisesRegex(ValueError, "no interval after"):
            validate_family_coverage(frame, previous_max=1_000)


if __name__ == "__main__":
    unittest.main()
