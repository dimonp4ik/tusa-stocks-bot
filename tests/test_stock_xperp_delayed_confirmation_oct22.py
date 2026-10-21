from stock_xperp_delayed_confirmation_oct22_audit import confirmation_index


def candles():
    return {
        "time": [0, 900, 1800, 2700, 3600],
        "open": [100, 99, 98, 99, 100],
        "high": [101, 100, 99, 101, 102],
        "low": [99, 97, 97, 98, 99],
        "close": [100, 98, 98.5, 100.5, 101],
    }


def test_directional_confirmation_uses_closed_bar_then_next_entry():
    assert confirmation_index(candles(), 1, 1, "directional_close", 4) == 2


def test_previous_bar_break_requires_stronger_close():
    assert confirmation_index(candles(), 1, 1, "previous_bar_break", 4) == 3


def test_wait_window_fails_closed_before_future_confirmation():
    assert confirmation_index(candles(), 1, 1, "previous_bar_break", 2) is None
