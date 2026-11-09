import json
import inspect
import time
from unittest.mock import patch

import main
from src import autotrader


def test_entry_skip_is_persisted_and_sent_to_user():
    signal = {"id": 17, "symbol": "AAVEUSDT", "direction": "LONG"}
    with (patch.object(autotrader, "set_bot_state") as save,
          patch.object(autotrader, "_dm") as dm):
        autotrader._entry_skip(
            7, signal, "AAVE-XPERP", "цена уже прошла TP1", "пропущено")

    key, raw = save.call_args.args
    payload = json.loads(raw)
    assert key == "autotrade_last_skip:7"
    assert payload["signal_id"] == 17
    assert payload["symbol"] == "AAVEUSDT"
    assert payload["reason"] == "цена уже прошла TP1"
    dm.assert_called_once_with(7, "пропущено")


def test_autotrade_menu_shows_last_skip_reason():
    payload = json.dumps({
        "ts": time.time() - 120,
        "symbol": "AAVEUSDT",
        "direction": "LONG",
        "reason": "цена уже вышла за диапазон между стопом и TP1",
    })
    with (patch.object(main, "get_bot_state", return_value=payload),
          patch.object(main, "_at_reply_kb") as reply):
        main._at_show_menu(7, {
            "active": 1, "size_value": 3, "last_balance": 130,
            "tp1_close_pct": 5,
        })

    text = reply.call_args.args[1]
    assert "Последний пропуск" in text
    assert "AAVE LONG" in text
    assert "цена уже вышла за диапазон" in text


def test_book_audit_runs_only_after_protection_and_position_activation():
    source = inspect.getsource(autotrader._open_for_account_locked)
    assert source.index("place_protection_oco") < source.index("at_activate_entry_intent")
    assert source.index("at_activate_entry_intent") < source.rindex(
        "record_signal_execution_book")
