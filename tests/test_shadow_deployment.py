import os
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
        # The legacy strategy is not a flag any more - it was deleted on
        # 2026-09-16, so there is nothing left to switch on.
        self.assertFalse(hasattr(config, "LEGACY_STRATEGY_ENABLED"))
        self.assertEqual(config.STOCK_VENUE_FILTER_MODE, "paper")

    def test_scan_dispatches_only_frozen_router(self):
        with patch.object(main, "_run_stock_venue_strategy_scan") as scan:
            main.run_scan()
        scan.assert_called_once_with()

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


if __name__ == "__main__":
    unittest.main()
