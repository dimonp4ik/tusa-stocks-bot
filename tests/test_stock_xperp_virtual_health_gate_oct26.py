import pandas as pd

import stock_xperp_virtual_health_gate_oct26_audit as audit


def test_pending_outcome_cannot_block_an_earlier_entry():
    rows = pd.DataFrame([
        {"symbol": "A", "entry_time": 100.0, "exit_time": 500.0,
         "signed_return": -1.0},
        {"symbol": "B", "entry_time": 200.0, "exit_time": 300.0,
         "signed_return": 1.0},
    ])
    accepted = audit.apply_gate(rows, window=1, threshold=0.0)
    assert accepted.symbol.tolist() == ["A", "B"]


def test_completed_virtual_rows_update_health_even_when_gate_blocks():
    rows = pd.DataFrame([
        {"symbol": "A", "entry_time": 100.0, "exit_time": 110.0,
         "signed_return": -1.0},
        {"symbol": "B", "entry_time": 200.0, "exit_time": 210.0,
         "signed_return": 1.0},
        {"symbol": "C", "entry_time": 300.0, "exit_time": 310.0,
         "signed_return": 1.0},
    ])
    # Force immediate gating by monkeying the window's normal warm-up length.
    accepted = audit.apply_gate(rows, window=1, threshold=0.0)
    # The production warm-up is min(20, window)=1: B is blocked by A, but B's
    # virtual win is still learned and allows C.
    assert accepted.symbol.tolist() == ["A", "C"]
