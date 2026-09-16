import unittest
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

from src.regime_filters import stock_opening_breakout_short
from src.telegram_notifier import bracket_for_analysis


NY = ZoneInfo("America/New_York")


def cash_day(day: datetime, closes: list[float]) -> dict:
    start = day.replace(hour=9, minute=30, tzinfo=NY)
    times = [int((start + timedelta(minutes=15 * i)).timestamp()) for i in range(len(closes))]
    return {
        "time": times,
        "open": list(closes),
        "high": [value + .5 for value in closes],
        "low": [value - .5 for value in closes],
        "close": list(closes),
    }


def combine(*parts):
    return {key: sum((part[key] for part in parts), []) for key in parts[0]}


class StockRegimeFilterTests(unittest.TestCase):
    def test_regime_paper_row_is_logged_once_with_exact_bracket(self):
        from src import db
        with tempfile.TemporaryDirectory() as tmp, patch.object(
                db, "DB_PATH", str(Path(tmp) / "test.db")):
            db.init_db()
            analysis = {"symbol": "TEST", "direction": "SHORT", "decision": "SHORT",
                        "current_price": 100, "atr": 2, "fixed_stop_atr": 1,
                        "fixed_target_r": .5, "source": "regime", "signal_bar_ts": 123}
            first = db.log_setup_candidate_once(analysis)
            second = db.log_setup_candidate_once(analysis)
            self.assertIsNotNone(first)
            self.assertIsNone(second)
            with db._conn() as connection:
                row = connection.execute("SELECT tp1,tp2,sl FROM setup_log").fetchone()
            self.assertEqual(tuple(row), (99.0, 99.0, 102.0))

    def test_fixed_family_bracket_closes_full_trade_at_target(self):
        analysis = {"direction": "SHORT", "atr": 2,
                    "fixed_stop_atr": 1, "fixed_target_r": .5}
        tp1, tp2, sl = bracket_for_analysis(analysis, 100)
        self.assertEqual((tp1, tp2, sl), (99.0, 99.0, 102.0))

    def test_latest_first_opening_range_break_is_a_market_short(self):
        previous = cash_day(datetime(2026, 8, 3), [100.0] * 26)
        current = cash_day(datetime(2026, 8, 4), [100.0, 100.0, 100.0, 98.0])
        setup = stock_opening_breakout_short(
            combine(previous, current), market_price=97.9, target_r=.5)
        self.assertIsNotNone(setup)
        self.assertEqual(setup["direction"], "SHORT")
        self.assertEqual(setup["entry"], 97.9)
        self.assertAlmostEqual(setup["entry"] - setup["tp"],
                               .5 * (setup["sl"] - setup["entry"]))

    def test_rejects_exhausted_prior_day(self):
        previous = cash_day(datetime(2026, 8, 3),
                            [105 - 5 * i / 25 for i in range(26)])
        current = cash_day(datetime(2026, 8, 4), [100.0, 100.0, 100.0, 98.0])
        self.assertIsNone(stock_opening_breakout_short(combine(previous, current)))


if __name__ == "__main__":
    unittest.main()
