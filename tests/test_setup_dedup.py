import concurrent.futures
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ["BOT_STARTUP_ENABLED"] = "0"

from src import db


class SetupDedupTests(unittest.TestCase):
    def test_shadow_exit_variant_is_attached_once_to_deduplicated_setup(self):
        analysis = {
            "symbol": "TESTUSDT", "direction": "SHORT", "decision": "SHORT",
            "current_price": 100.0, "atr": 1.0, "fixed_stop_atr": 1.0,
            "fixed_target_r": .25, "signal_bar_ts": 1_700_000_000.0,
            "source": "stock_venue_shadow_5m",
            "shadow_exit_variant": {
                "name": "candidate_v1", "first_fraction": .5,
                "mid_fraction": .2, "far_fraction": .3,
                "mid_target_r": .5, "far_target_r": 4.0,
                "entry_bar_ts": 1_700_000_300.0,
                "expire_at": 1_700_010_000.0,
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "variant.db")
            with patch.object(db, "DB_PATH", path), \
                    patch("src.telegram_notifier.bracket_for_analysis",
                          return_value=(99.75, 92.0, 102.0)):
                db.init_db()
                first = db.log_setup_candidate_once(analysis)
                second = db.log_setup_candidate_once(analysis)
                rows = db.get_unresolved_shadow_exit_variants()

        self.assertIsNotNone(first)
        self.assertIsNone(second)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["setup_log_id"], first)
        self.assertEqual(rows[0]["variant"], "candidate_v1")
        self.assertEqual(rows[0]["policy"]["far_target_r"], 4.0)

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

    def test_unresolved_venue_query_keeps_both_active_sources_and_excludes_others(self):
        base = {
            "symbol": "TESTUSDT", "direction": "SHORT", "decision": "SHORT",
            "current_price": 100.0, "atr": 1.0, "fixed_stop_atr": 1.0,
            "fixed_target_r": 1.0,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "venue.db")
            with patch.object(db, "DB_PATH", path), \
                    patch("src.telegram_notifier.bracket_for_analysis",
                          return_value=(99.0, 99.0, 101.0)):
                db.init_db()
                venue_id = db.log_setup_candidate_once({
                    **base, "signal_bar_ts": 1_700_000_000.0,
                    "source": "stock_venue_regime",
                })
                five_minute_id = db.log_setup_candidate_once({
                    **base, "signal_bar_ts": 1_700_000_300.0,
                    "source": "stock_venue_shadow_5m",
                })
                native_id = db.log_setup_candidate_once({
                    **base, "signal_bar_ts": 1_700_000_900.0,
                    "source": "stock_xperp_native_shadow",
                })
                db.log_setup_candidate_once({
                    **base, "signal_bar_ts": 1_700_001_800.0,
                    "source": "retired_strategy",
                })
                with db._conn() as conn:
                    conn.execute("UPDATE setup_log SET ts=?",
                                 (db.time_mod.time() - 10 * 86400,))
                rows = db.get_unresolved_venue_setups()

        self.assertCountEqual([row["id"] for row in rows],
                              [venue_id, five_minute_id, native_id])

    def test_five_minute_row_is_eligible_before_fifteen_minute_row(self):
        base = {
            "symbol": "TESTUSDT", "direction": "SHORT", "decision": "SHORT",
            "current_price": 100.0, "atr": 1.0, "fixed_stop_atr": 1.0,
            "fixed_target_r": 1.0,
        }
        now = 2_000_000_000.0
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "venue_age.db")
            with patch.object(db, "DB_PATH", path), \
                    patch("src.telegram_notifier.bracket_for_analysis",
                          return_value=(99.0, 99.0, 101.0)):
                db.init_db()
                venue_id = db.log_setup_candidate_once({
                    **base, "signal_bar_ts": now - 900,
                    "source": "stock_venue_regime",
                })
                five_minute_id = db.log_setup_candidate_once({
                    **base, "signal_bar_ts": now - 600,
                    "source": "stock_venue_shadow_5m",
                })
                with db._conn() as conn:
                    conn.execute("UPDATE setup_log SET ts=? WHERE id IN (?, ?)",
                                 (now - 600, venue_id, five_minute_id))
                with patch.object(db.time_mod, "time", return_value=now):
                    rows = db.get_unresolved_venue_setups()

        self.assertEqual([row["id"] for row in rows], [five_minute_id])

    def test_cap_impact_uses_exact_realised_r_for_native_routes(self):
        base = {
            "symbol": "TESTUSDT", "direction": "SHORT", "decision": "SHORT",
            "current_price": 100.0, "atr": 1.0, "fixed_stop_atr": 1.0,
            "fixed_target_r": 2.5, "source": "stock_xperp_native_shadow",
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "cap_impact.db")
            with patch.object(db, "DB_PATH", path), \
                    patch("src.telegram_notifier.bracket_for_analysis",
                          return_value=(97.5, 97.5, 101.0)):
                db.init_db()
                winner = db.log_setup_candidate_once({
                    **base, "signal_bar_ts": 1_700_000_000.0,
                })
                loser = db.log_setup_candidate_once({
                    **base, "signal_bar_ts": 1_700_000_900.0,
                })
                db.mark_setup_blocked(winner, "earnings_blackout")
                db.mark_setup_blocked(loser, "earnings_blackout")
                db.mark_setup_resolved(winner, "TP2", 1, 1, net_r=2.5)
                db.mark_setup_resolved(loser, "SL", 0, 0, net_r=-1.0)
                stats = db.get_cap_impact_stats(0)["earnings_blackout"]

        self.assertEqual(stats["n"], 2)
        self.assertEqual(stats["reached_tp1"], 1)
        self.assertEqual(stats["sl"], 1)
        self.assertAlmostEqual(stats["saved_r"], -1.5)


if __name__ == "__main__":
    unittest.main()
