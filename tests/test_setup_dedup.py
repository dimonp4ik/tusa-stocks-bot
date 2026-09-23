import concurrent.futures
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ["BOT_STARTUP_ENABLED"] = "0"

from src import db


class SetupDedupTests(unittest.TestCase):
    def test_concurrent_scans_claim_closed_bar_once(self):
        analysis = {
            "symbol": "TESTUSDT", "direction": "SHORT", "decision": "SHORT",
            "current_price": 100.0, "atr": 1.0, "fixed_stop_atr": 1.0,
            "fixed_target_r": 1.0, "signal_bar_ts": 1_700_000_000.0,
            "source": "stock_venue_regime",
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "dedup.db")
            with patch.object(db, "DB_PATH", path), \
                    patch("src.telegram_notifier.bracket_for_analysis",
                          return_value=(99.0, 98.0, 101.0)):
                db.init_db()
                with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
                    ids = list(pool.map(
                        lambda _: db.log_setup_candidate_once(analysis), range(40)))
                with db._conn() as conn:
                    rows = conn.execute(
                        "SELECT COUNT(*) FROM setup_log WHERE symbol='TESTUSDT'"
                    ).fetchone()[0]

        self.assertEqual(sum(item is not None for item in ids), 1)
        self.assertEqual(rows, 1)


if __name__ == "__main__":
    unittest.main()
