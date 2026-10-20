import os
import subprocess
import sys
import unittest
from unittest.mock import patch

os.environ["BOT_STARTUP_ENABLED"] = "0"

import config
import main
from src import telegram_notifier as notifier


class ShadowDeploymentTests(unittest.TestCase):
    def test_real_orders_fail_closed(self):
        self.assertEqual(config.DEPLOYMENT_MODE, "shadow")
        self.assertFalse(config.AUTOTRADE_ENABLED)
        self.assertTrue(config.NATIVE_XPERP_SHADOW_ENABLED)
        # The legacy strategy is not a flag any more - it was deleted on
        # 2026-09-16, so there is nothing left to switch on.
        self.assertFalse(hasattr(config, "LEGACY_STRATEGY_ENABLED"))
        self.assertEqual(config.STOCK_VENUE_FILTER_MODE, "paper")

    def test_native_candidate_cannot_be_enabled_in_live_mode(self):
        env = {**os.environ, "DEPLOYMENT_MODE": "live",
               "NATIVE_XPERP_SHADOW_ENABLED": "1"}
        value = subprocess.check_output(
            [sys.executable, "-c",
             "import config; print(config.NATIVE_XPERP_SHADOW_ENABLED)"],
            env=env, text=True,
        ).strip()
        self.assertEqual(value, "False")

    def test_scan_dispatches_only_native_shadow_router(self):
        calls = []
        with patch.object(main, "_check_open_signals",
                          side_effect=lambda: calls.append("reconcile")), \
             patch.object(main, "_run_native_xperp_shadow_scan",
                          side_effect=lambda: calls.append("scan")) as scan:
            main.run_scan()
        scan.assert_called_once_with()
        self.assertEqual(calls, ["reconcile", "scan"])

    def test_scan_fails_closed_when_book_cannot_be_reconciled(self):
        with patch.object(main, "_check_open_signals",
                          side_effect=RuntimeError("feed unavailable")), \
             patch.object(main, "_run_native_xperp_shadow_scan") as scan:
            self.assertIsNone(main.run_scan())
        scan.assert_not_called()

    def test_telegram_paper_signal_is_explicit_and_untradeable(self):
        analysis = {
            "symbol": "AAPLUSDT", "direction": "SHORT", "decision": "SHORT",
            "current_price": 100.0, "atr": 1.0, "fixed_stop_atr": 1.0,
            "fixed_target_r": 1.0, "signals": ["Venue module frozen_test"],
            "_shadow_only": True,
        }
        sent = []
        with patch.object(notifier, "_send_message", side_effect=lambda text: sent.append(text) or True),              patch("src.db.log_signal", return_value=42):
            self.assertTrue(notifier.send_signal(analysis))
        self.assertEqual(analysis["_signal_id"], 42)
        self.assertIn("ОРДЕР НЕ ОТКРЫТ", sent[0])
        self.assertNotIn("Плечо", sent[0])

    def test_telegram_barbell_signal_shows_both_market_exit_levels(self):
        analysis = {
            "symbol": "AAPLUSDT", "direction": "SHORT", "decision": "SHORT",
            "current_price": 100.0, "atr": 1.0, "fixed_stop_atr": 1.0,
            "fixed_target_r": .5, "fixed_runner_target_r": 1.0,
            "strategy_tp1_close_frac": .70,
            "signals": ["Venue module shadow_stock_5m_range_reversion_short"],
            "_shadow_only": True,
        }
        sent = []
        with patch.object(notifier, "_send_message",
                          side_effect=lambda text: sent.append(text) or True), \
             patch("src.db.log_signal", return_value=43):
            self.assertTrue(notifier.send_signal(analysis))
        self.assertIn("TP1, закрыть 70%", sent[0])
        self.assertIn("TP2, остаток 30%", sent[0])
        self.assertIn("`1.00R`", sent[0])

    def test_session_helpers_exist(self):
        """The venue scan calls all three on a closed-market tick.

        They were missing from the working tree on 2026-09-16 - backtest.py still
        documents _in_extended_window as its original - so run_scan raised
        NameError every five minutes outside regular hours, which is most of the
        day. Calling them here is the cheapest way to keep that from returning.
        """
        self.assertIn(main._in_extended_window(), (True, False))
        self.assertIn(main._off_session_enabled(), (True, False))
        self.assertIsInstance(main._session_windows_text(), str)

    def test_closed_market_scan_returns_quietly(self):
        with patch("src.market_hours.is_market_open", return_value=False),              patch.object(main, "_off_session_enabled", return_value=False),              patch.object(main, "_in_extended_window", return_value=False),              patch.object(main, "get_xperp_instruments") as instruments:
            main._run_stock_venue_strategy_scan()
        instruments.assert_not_called()

    def test_publish_path_never_opens_real_positions(self):
        """No autotrader call may reappear in the scan without a deliberate change."""
        import inspect
        src = inspect.getsource(main._run_stock_venue_strategy_scan)
        # Comments are allowed to name it - the comment there explains why the
        # call is absent. Only executable lines are checked.
        code = chr(10).join(ln.split("#", 1)[0] for ln in src.splitlines())
        self.assertNotIn("open_positions_for_signal", code)

    def test_every_button_has_a_handler(self):
        """A button with no branch does nothing when pressed - adm_health was one."""
        import re
        from pathlib import Path
        src = Path("main.py").read_text(encoding="utf-8")
        buttons = sorted(set(re.findall(r'"callback_data":\s*"([^"{]+)"', src)))
        self.assertTrue(buttons)
        for cb in buttons:
            handled = (re.search(r'==\s*"%s"' % re.escape(cb), src)
                       or re.search(r'data in \([^)]*"%s"' % re.escape(cb), src)
                       or any(re.search(r'startswith\(\s*\(?"%s' % re.escape(cb[:k]), src)
                              for k in range(4, len(cb) + 1)))
            self.assertTrue(handled, "no handler for button %s" % cb)

    def test_health_panel_reports_without_network(self):
        shown = []
        with patch.object(main, "get_xperp_instruments", return_value={"AAPL": 1}),              patch.object(main, "get_klines_xperp", return_value={"close": [1.0]}),              patch.object(main, "get_open_signals", return_value=[]),              patch.object(main, "_edit_admin_text",
                          side_effect=lambda c, m, t, k=None: shown.append(t)):
            main._handle_admin_callback("cb", 1, 1, "adm_health", user_id=671071896)
        self.assertTrue(shown, "health panel produced no text")
        self.assertIn("Проверка сервисов", shown[0])
        self.assertIn("ЗАПРЕЩЕНЫ КОДОМ", shown[0])

    def test_menu_has_no_autotrading_button(self):
        labels = str(main._USER_KB) + str(main._ADMIN_KB) + str(main._KB_PEOPLE)
        self.assertNotIn("Автотрейдинг", labels)
        self.assertIn("Paper-сделки", labels)

    def test_one_scan_cannot_publish_an_unbounded_signal_batch(self):
        self.assertLessEqual(config.VENUE_MAX_SIGNALS_PER_SCAN, 2)
        self.assertLessEqual(config.MAX_SAME_DIRECTION_POSITIONS, 4)
        candidates = [
            ({"symbol": f"S{i}USDT", "direction": "SHORT"}, i + 1)
            for i in range(config.VENUE_MAX_SIGNALS_PER_SCAN + 2)
        ]
        with patch.object(main, "send_signal", return_value=True) as send, \
             patch.object(main, "get_recent_signals", return_value=[]), \
             patch.object(main, "mark_setup_sent") as sent, \
             patch.object(main, "mark_setup_blocked") as blocked, \
             patch.object(main, "_cache_signal"):
            published = main._publish_stock_venue_candidates(candidates)
        self.assertEqual(published, config.VENUE_MAX_SIGNALS_PER_SCAN)
        self.assertEqual(send.call_count, config.VENUE_MAX_SIGNALS_PER_SCAN)
        self.assertEqual(sent.call_count, config.VENUE_MAX_SIGNALS_PER_SCAN)
        self.assertEqual(blocked.call_count, 2)
        self.assertTrue(all(call.args[1] == "scan_cap"
                            for call in blocked.call_args_list))

    def test_native_rolling_hour_cap_blocks_correlated_batch(self):
        candidate = ({
            "symbol": "AAPLUSDT", "direction": "SHORT",
            "source": "stock_xperp_native_shadow",
        }, 9)
        recent = [{
            "source": "stock_xperp_native_shadow", "opened_at": stamp,
        } for stamp in (7000, 8000, 9000, 9500)]
        with patch.object(main.time, "time", return_value=10000), \
             patch.object(main, "get_recent_signals", return_value=recent), \
             patch.object(main, "send_signal") as send, \
             patch.object(main, "mark_setup_blocked") as blocked:
            published = main._publish_stock_venue_candidates([candidate])
        self.assertEqual(published, 0)
        send.assert_not_called()
        blocked.assert_called_once_with(9, "rolling_hour_cap")

    def test_five_minute_analysis_is_permanently_shadow_only(self):
        setup = {
            "direction": "SHORT", "entry": 100.0, "atr": 2.0,
            "target_r": .5, "signal_bar_ts": 1_788_876_000.0,
            "module": "shadow_stock_5m_range_reversion_short",
            "qqq_regime": "5m_regime", "volume_ratio": .7,
            "_shadow_experiment": True, "_five_minute_shadow": True,
        }
        with patch.object(main, "DEPLOYMENT_MODE", "live"):
            analysis = main._stock_venue_analysis("AAPLUSDT", setup)
        self.assertTrue(analysis["_shadow_only"])
        self.assertTrue(analysis["_five_minute_shadow"])
        self.assertEqual(analysis["source"], "stock_venue_shadow_5m")
        self.assertEqual(analysis["fixed_runner_target_r"], 1.0)
        self.assertEqual(analysis["strategy_tp1_close_frac"], .70)
        self.assertEqual(analysis["strategy_runner_mode"], "fixed_be")
        self.assertEqual(analysis["strategy_bar_seconds"], 300)
        self.assertEqual(analysis["strategy_entry_bar_ts"], 1_788_876_300.0)
        self.assertFalse(analysis["strategy_stop_on_close"])
        self.assertGreater(analysis["strategy_expire_at"], 1_788_876_300.0)

    def test_pullback_five_minute_analysis_uses_four_r_runner(self):
        setup = {
            "direction": "SHORT", "entry": 100.0, "atr": 2.0,
            "target_r": .25, "signal_bar_ts": 1_800_000_000.0,
            "module": "shadow_stock_5m_trend_pullback_short",
            "qqq_regime": "5m_regime", "volume_ratio": .7,
            "_shadow_experiment": True, "_five_minute_shadow": True,
        }
        analysis = main._stock_venue_analysis("AAPLUSDT", setup)
        self.assertEqual(analysis["fixed_runner_target_r"], 4.0)

    def test_five_minute_fixed_runner_uses_5m_bars_and_next_bar_breakeven(self):
        start = 2_000_000_100.0
        frame = {
            "time": [start, start + 300],
            "open": [99.8, 99.6], "high": [99.9, 100.0],
            "low": [99.4, 99.0], "close": [99.6, 100.0],
            "confirmed": [True, False],
        }
        signal = {
            "id": 71, "symbol": "AAPLUSDT", "direction": "SHORT",
            "entry_price": 100.0, "sl": 102.0, "tp1": 99.5, "tp2": 92.0,
            "atr": 2.0, "opened_at": start - 60, "tp1_hit_at": start + 30,
            "status": "TP1_PARTIAL", "strategy_tp1_close_frac": .70,
            "strategy_runner_mode": "fixed_be", "strategy_bar_seconds": 300,
            "strategy_entry_bar_ts": start,
            "runner_activation_bar_ts": start,
            "strategy_stop_on_close": 0,
            "strategy_expire_at": start + 3600,
        }
        with patch.object(main, "get_open_signals", return_value=[signal]), \
             patch.object(main.time, "time", return_value=start + 360), \
             patch.object(main, "get_klines_xperp", return_value=frame) as candles, \
             patch.object(main, "update_signal_status") as update, \
             patch.object(main, "send_signal_update"), \
             patch.object(main.autotrader, "mirror_transition"):
            main._check_open_signals()
        self.assertEqual(candles.call_args.kwargs["interval"], "5m")
        self.assertEqual(update.call_args.args[1], "BREAKEVEN")
        self.assertAlmostEqual(update.call_args.kwargs["realized_r"], .175)

    def test_five_minute_stop_is_wick_touch_even_when_forming_close_recovers(self):
        start = 2_000_001_000.0
        frame = {
            "time": [start], "open": [100.0], "high": [100.5],
            "low": [97.8], "close": [100.0], "confirmed": [False],
        }
        signal = {
            "id": 72, "symbol": "AAPLUSDT", "direction": "LONG",
            "entry_price": 100.0, "sl": 98.0, "tp1": 101.0, "tp2": 104.0,
            "atr": 2.0, "opened_at": start, "status": "OPEN",
            "strategy_tp1_close_frac": .70, "strategy_runner_mode": "fixed_be",
            "strategy_bar_seconds": 300, "strategy_entry_bar_ts": start,
            "strategy_stop_on_close": 0,
            "strategy_expire_at": start + 3600,
        }
        with patch.object(main, "get_open_signals", return_value=[signal]), \
             patch.object(main.time, "time", return_value=start + 60), \
             patch.object(main, "get_klines_xperp", return_value=frame), \
             patch.object(main, "_deep_feed_breached", return_value=True), \
             patch.object(main, "set_sl_xperp_only"), \
             patch.object(main, "update_signal_status") as update, \
             patch.object(main, "send_signal_update"), \
             patch.object(main.autotrader, "mirror_transition"):
            main._check_open_signals()
        self.assertEqual(update.call_args.args[1], "SL_HIT")
        self.assertEqual(update.call_args.kwargs["realized_r"], -1.0)

    def test_stock_target_after_cash_session_close_is_ignored(self):
        start = 2_000_010_000.0
        frame = {
            "time": [start, start + 300], "open": [100.0, 100.0],
            "high": [100.2, 100.2], "low": [99.8, 90.0],
            "close": [100.0, 90.0], "confirmed": [True, False],
        }
        signal = {
            "id": 73, "symbol": "AAPLUSDT", "direction": "SHORT",
            "entry_price": 100.0, "sl": 102.0, "tp1": 99.0, "tp2": 98.0,
            "atr": 2.0, "opened_at": start, "status": "OPEN",
            "strategy_tp1_close_frac": .70, "strategy_runner_mode": "fixed_be",
            "strategy_bar_seconds": 300, "strategy_entry_bar_ts": start,
            "strategy_stop_on_close": 0,
            "strategy_expire_at": start + 300,
        }
        with patch.object(main, "get_open_signals", return_value=[signal]), \
             patch.object(main.time, "time", return_value=start + 360), \
             patch.object(main, "get_klines_xperp", return_value=frame), \
             patch.object(main, "update_signal_status") as update, \
             patch.object(main, "send_signal_update"), \
             patch.object(main.autotrader, "mirror_transition"):
            main._check_open_signals()
        self.assertEqual(update.call_args.args[1], "EXPIRED")

    def test_persistent_scan_cap_counts_previous_five_minute_tick(self):
        candidates = [({
            "symbol": "AAPLUSDT", "direction": "SHORT",
            "source": "stock_venue_shadow_5m",
        }, 7)]
        now = main.time.time()
        prior = int(now // 900) * 900 + 1
        with patch.object(main, "get_recent_signals", return_value=[
                {"opened_at": prior}, {"opened_at": prior + 60},
             ]), patch.object(main, "send_signal") as send, \
             patch.object(main, "mark_setup_blocked") as blocked:
            self.assertEqual(main._publish_stock_venue_candidates(candidates), 0)
        send.assert_not_called()
        blocked.assert_called_once_with(7, "scan_cap")

    def test_production_candidate_precedes_five_minute_shadow(self):
        candidates = [
            ({"symbol": "AAPLUSDT", "direction": "SHORT",
              "source": "stock_venue_shadow_5m"}, 8),
            ({"symbol": "NVDAUSDT", "direction": "SHORT",
              "source": "stock_venue_regime"}, 9),
        ]
        with patch.object(main, "get_recent_signals", return_value=[
                {"opened_at": int(main.time.time() // 900) * 900 + 1},
             ]), patch.object(main, "send_signal", return_value=True) as send, \
             patch.object(main, "mark_setup_sent"), \
             patch.object(main, "mark_setup_blocked") as blocked, \
             patch.object(main, "_cache_signal"):
            self.assertEqual(main._publish_stock_venue_candidates(candidates), 1)
        self.assertEqual(send.call_args.args[0]["source"], "stock_venue_regime")
        blocked.assert_called_once_with(8, "scan_cap")

    def test_stale_environment_cannot_raise_direction_cap(self):
        env = {**os.environ, "MAX_SAME_DIRECTION_POSITIONS": "99"}
        value = subprocess.check_output(
            [sys.executable, "-c",
             "import config; print(config.MAX_SAME_DIRECTION_POSITIONS)"],
            env=env, text=True,
        ).strip()
        self.assertEqual(value, "4")

    def test_venue_cooldown_is_per_symbol_and_direction(self):
        now = 2_000_000.0
        with patch.dict(main._signal_cache, {
                "AAPLUSDT|SHORT": ("SHORT", now - 60)}, clear=True):
            self.assertTrue(main._venue_signal_on_cooldown("AAPLUSDT", "SHORT", now))
            self.assertFalse(main._venue_signal_on_cooldown("AAPLUSDT", "LONG", now))

    def test_existing_direction_exposure_blocks_whole_new_batch(self):
        candidates = [({"symbol": "AAPLUSDT", "direction": "SHORT"}, 1)]
        with patch.object(main, "send_signal", return_value=True) as send, \
             patch.object(main, "get_recent_signals", return_value=[]), \
             patch.object(main, "mark_setup_blocked") as blocked:
            published = main._publish_stock_venue_candidates(
                candidates, {"SHORT": config.MAX_SAME_DIRECTION_POSITIONS})
        self.assertEqual(published, 0)
        send.assert_not_called()
        blocked.assert_called_once_with(1, "dir_cap")

    def test_daily_loss_pause_uses_only_closed_signal_streak(self):
        with patch.object(main, "get_today_sl_streak",
                          return_value=config.VENUE_LOSS_PAUSE_STREAK):
            self.assertTrue(main._venue_daily_loss_pause())


if __name__ == "__main__":
    unittest.main()
