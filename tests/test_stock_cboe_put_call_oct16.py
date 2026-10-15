import json
import unittest

import pandas as pd

from download_stock_cboe_put_call_history import parse_page
from stock_cboe_put_call_material_oct16_audit import attach_flow, required_hit_rate


class CboePutCallSourceTest(unittest.TestCase):
    @staticmethod
    def _page(day: str) -> str:
        ratios = [
            {"name": "TOTAL PUT/CALL RATIO", "value": "1.01"},
            {"name": "INDEX PUT/CALL RATIO", "value": "1.20"},
            {"name": "EXCHANGE TRADED PRODUCTS PUT/CALL RATIO", "value": "0.90"},
            {"name": "EQUITY PUT/CALL RATIO", "value": "0.70"},
            {"name": "CBOE VOLATILITY INDEX (VIX) PUT/CALL RATIO", "value": "0.40"},
            {"name": "SPX + SPXW PUT/CALL RATIO", "value": "1.30"},
        ]
        metric_rows = [
            {"name": "VOLUME", "call": 10, "put": 9, "total": 19},
            {"name": "OPEN INTEREST", "call": 20, "put": 18, "total": 38},
        ]
        payload = {"ratios": ratios}
        for name in (
                "SUM OF ALL PRODUCTS", "INDEX OPTIONS",
                "EXCHANGE TRADED PRODUCTS", "EQUITY OPTIONS",
                "CBOE VOLATILITY INDEX (VIX)", "SPX + SPXW"):
            payload[name] = metric_rows
        embedded = (
            'optionsData\\":' + json.dumps(payload, separators=(",", ":"))
            + ',\\"selectedDate\\":\\"' + day + '\\"')
        return embedded

    def test_parse_valid_and_closed_pages(self):
        day = "2024-01-03"
        row = parse_page(self._page(day), day)
        self.assertEqual(row["date"], day)
        self.assertEqual(row["equity_pc"], 0.7)
        self.assertEqual(row["total_volume_total"], 19)
        self.assertIsNone(parse_page('optionsData\\":null', "2024-01-01"))

    def test_attach_flow_never_uses_same_session_close(self):
        features = pd.DataFrame({
            "session_date": ["2024-01-02", "2024-01-03"],
            "symbol": ["AAPL", "MSFT"],
        })
        flow = pd.DataFrame({
            "date_epoch": [
                int(pd.Timestamp("2024-01-01", tz="UTC").timestamp()),
                int(pd.Timestamp("2024-01-02", tz="UTC").timestamp()),
            ],
            "source_day": ["prior", "same_for_first"],
        })
        joined = attach_flow(features, flow).sort_values("session_date")
        self.assertEqual(joined.source_day.tolist(), ["prior", "same_for_first"])
        self.assertTrue((joined.date_epoch < joined.session_epoch).all())

    def test_required_hit_rate_preserves_impossible_none(self):
        self.assertIsNone(required_hit_rate(0.1, 0.75))


if __name__ == "__main__":
    unittest.main()
