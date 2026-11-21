import json
from datetime import datetime, timezone
from unittest.mock import patch

import main
from src import autotrader
from src.drawdown_recovery import checked_equity_guard, drawdown_reset_version, manual_reset_equity_guard


NOW = datetime(2026, 11, 21, 12, tzinfo=timezone.utc).timestamp()
PAUSED = {"day": "2026-11-21", "peak": 130., "daily_start": 120., "paused": True}
LIMITS = {"max_daily_loss": .03, "max_drawdown": .03}


def test_reset_rebases_peak_and_preserves_daily_latch():
    state = manual_reset_equity_guard(dict(PAUSED, daily_paused=True), equity=119., now=NOW, **LIMITS)
    assert state["peak"] == 119. and state["pre_reset_peak"] == 130.
    assert state["paused"] is False and state["daily_paused"] is True
    assert state["last_block"] == "daily loss limit"
    next_day, blocked = checked_equity_guard(state, equity=119., now=NOW + 86400,
                                            peak_floor=130., **LIMITS)
    assert blocked is None and next_day["peak"] == 119.
    next_pause, blocked = checked_equity_guard(next_day, equity=115., now=NOW + 86401,
                                              peak_floor=130., **LIMITS)
    assert blocked == "drawdown limit"
    assert drawdown_reset_version(next_pause) != drawdown_reset_version(PAUSED)


def test_reset_needs_current_pause_and_fresh_positive_balance():
    raw = json.dumps(PAUSED)
    version = drawdown_reset_version(PAUSED)
    with (patch.object(autotrader, "at_get", return_value={"allowed": True}),
          patch.object(autotrader, "_creds_of", return_value={"key": "fixture"}),
          patch.object(autotrader, "get_bot_state", return_value=raw),
          patch.object(autotrader.time, "time", return_value=NOW),
          patch.object(autotrader.okx, "get_balance") as get_balance,
          patch.object(autotrader, "compare_and_set_bot_state", return_value=True) as save):
        assert autotrader.reset_user_drawdown(15, "old") == ("stale", None)
        get_balance.assert_not_called()
        get_balance.return_value = False, None
        assert autotrader.reset_user_drawdown(15, version) == ("balance_unavailable", None)
        get_balance.return_value = True, float("nan")
        assert autotrader.reset_user_drawdown(15, version) == ("unavailable", None)
        save.assert_not_called()
        get_balance.return_value = True, 120.
        status, state = autotrader.reset_user_drawdown(15, version)
        assert status == "reset" and state["peak"] == 120.
        assert save.call_args.args[:2] == ("risk_guard:15", raw)
        assert json.loads(save.call_args.args[2]) == state
        save.return_value = False
        assert autotrader.reset_user_drawdown(15, version) == ("stale", None)


def test_button_shown_only_for_a_drawdown_and_works_in_private_chat_only():
    with (patch.object(main, "get_bot_state", return_value=json.dumps(PAUSED)),
          patch.object(main, "_at_reply_kb") as reply):
        main._at_show_menu(15, {})
    reset_button = reply.call_args.args[2][0][0]["callback_data"]
    assert reset_button == f"at_reset_dd:{drawdown_reset_version(PAUSED)}"
    with (patch.object(main, "get_bot_state", return_value=json.dumps(dict(PAUSED, paused=False, daily_paused=True))),
          patch.object(main, "_at_reply_kb") as reply):
        main._at_show_menu(15, {})
    assert not any(button["callback_data"].startswith("at_reset_dd:")
                   for row in reply.call_args.args[2] for button in row)
    with (patch.object(main, "at_get", return_value={"allowed": True}),
          patch.object(main, "_answer_callback"),
          patch.object(main, "_at_show_menu") as menu,
          patch.object(autotrader, "reset_user_drawdown", return_value=("reset", {
              "equity": 120., "daily_paused": True,
          })) as reset):
        assert main._at_handle_callback("callback", -15, 15, reset_button)
        reset.assert_not_called()
        assert main._at_handle_callback("callback", 15, 15, reset_button)
        reset.assert_called_once_with(15, drawdown_reset_version(PAUSED))
        assert "Дневная пауза" in menu.call_args.kwargs["notice"]
