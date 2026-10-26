import hashlib
import json
from pathlib import Path
import unittest

import pandas as pd

import stock_xperp_trend_pullback_router_oct26_audit as audit


ROOT = Path("reports/audit_2026_09_08")
REPORT = ROOT / "stock_xperp_trend_pullback_router_oct26.json"


class StockXperpTrendPullbackRouterTests(unittest.TestCase):
    def test_path_feature_excludes_two_bar_extreme_being_traded(self):
        frame = pd.DataFrame([{
            "symbol": "TESTUSDT", "entry_index": 100,
            "direction_n": 1,
        }])
        close = [100.0 + index for index in range(101)]
        candles = {"TESTUSDT": {"close": close}}
        first = audit.add_path_features(frame, candles)
        changed = list(close)
        changed[98] = 10_000.0
        changed[99] = 1.0
        second = audit.add_path_features(
            frame, {"TESTUSDT": {"close": changed}})
        for horizon in audit.HORIZONS:
            self.assertEqual(
                first.loc[0, f"directed_prior_return_{horizon}"],
                second.loc[0, f"directed_prior_return_{horizon}"])
            self.assertEqual(
                first.loc[0, f"prior_efficiency_{horizon}"],
                second.loc[0, f"prior_efficiency_{horizon}"])

    def test_report_rejects_route_that_misses_economic_objective(self):
        report = json.loads(REPORT.read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "REJECTED_OBJECTIVE")
        self.assertFalse(report["objective_pass"])
        self.assertFalse(report["ready_for_live_orders"])
        self.assertFalse(report["orders_possible"])
        self.assertTrue(report["confirmation_pass"])
        self.assertLess(
            report["fresh"]["margin_3pct"]["projected_22_session_pct"], 30)

    def test_report_sources_and_overlap_are_integral(self):
        report = json.loads(REPORT.read_text(encoding="utf-8"))
        self.assertEqual(report["overlap_verification"]["mismatches"], 0)
        self.assertGreater(report["overlap_verification"]["overlap_bars"], 0)
        expected = next(value for name, value in report["sources"].items()
                        if name.endswith(
                            "stock_xperp_trend_pullback_router_oct26_audit.py"))
        path = Path(audit.__file__)
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected)
        for path in (audit.PROFILE, audit.QUALITY, audit.VOLATILITY,
                     audit.HOUR_CAP):
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(),
                report["sources"][str(path)])


if __name__ == "__main__":
    unittest.main()
