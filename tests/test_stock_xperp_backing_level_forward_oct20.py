import stock_xperp_backing_level_forward_oct20_audit as audit


def test_forward_audit_uses_current_backing_guard_module():
    assert audit.backing.guarded is not None
    assert audit.backing.add_backing_distance is not None


def test_forward_audit_has_no_trading_client():
    assert not hasattr(audit, "OkxTrader")


def test_diagnostics_counts_exit_order_loss_streak():
    rows = [
        {"entry_time": 1, "exit_time": 4, "symbol": "A",
         "signed_return": -.01, "outcome_code": -1, "module_priority": 0},
        {"entry_time": 2, "exit_time": 3, "symbol": "B",
         "signed_return": -.01, "outcome_code": -1, "module_priority": 0},
        {"entry_time": 3, "exit_time": 5, "symbol": "C",
         "signed_return": .02, "outcome_code": 1, "module_priority": 0},
    ]
    found = audit.diagnostics(rows, .03)
    assert found["max_loss_streak"] == 2
    assert found["win_rate_pct"] == 33.3333
