import stock_profit_first_high_move_oct17_audit as audit


def test_selection_is_profit_first_without_win_rate_gate():
    value = {"return_pct": 12.0, "dd_pct": 15.0,
             "trades_per_calendar_day": 3.1, "wr": 22.0}
    assert audit.selection_ok(value)


def test_profile_grid_contains_material_moves():
    assert min(audit.TARGETS_PCT) >= 2.5
    assert max(audit.TARGETS_PCT) == 4.0
