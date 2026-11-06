from contextlib import ExitStack
from unittest.mock import patch

import main


def test_onboarding_rejects_key_without_trade_before_saving():
    chat_id = user_id = 918273
    main._at_onboarding[chat_id] = {
        "step": "passphrase",
        "data": {"api_key": "key", "api_secret": "secret"},
    }
    replies = []
    try:
        with ExitStack() as stack:
            stack.enter_context(patch.object(main, "_at_delete_msg"))
            stack.enter_context(patch.object(
                main._okx_trade, "get_balance", return_value=(True, 130.0)))
            stack.enter_context(patch.object(
                main._okx_trade, "validate_autotrade_account", return_value=(
                    False, {"reason": "trade_permission_missing"})))
            save = stack.enter_context(patch.object(main, "at_set_keys"))
            stack.enter_context(patch.object(
                main, "_reply", side_effect=lambda _chat, text, *a, **k: replies.append(text)))
            assert main._at_handle_text(chat_id, user_id, "pass", 77)
        save.assert_not_called()
        assert chat_id not in main._at_onboarding
        assert any("Trade" in text and "не сохранены" in text for text in replies)
    finally:
        main._at_onboarding.pop(chat_id, None)
