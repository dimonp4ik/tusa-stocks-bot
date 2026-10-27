import pandas as pd

import stock_xperp_mature_supplement_oct27_audit as mature
import stock_xperp_modular_supplement_oct27_audit as modular


def row(symbol, module, entry, exit_time, signed_return=-0.01):
    return {
        "symbol": symbol,
        "module": module,
        "entry_time": entry,
        "exit_time": exit_time,
        "direction": "LONG",
        "cross_z": 3.0,
        "signed_return": signed_return,
    }


def test_module_pause_does_not_pause_independent_module():
    rows = [
        row("NEW1", "supplement", 100, 110),
        row("NEW2", "supplement", 1_000, 1_010),
        row("NEW3", "supplement", 2_000, 2_010),
        row("CORE1", "core", 3_000, 3_010, 0.02),
    ]
    accepted = modular.module_pause_gate(rows, loss_streak_cap=2)
    assert [item["symbol"] for item in accepted] == ["NEW1", "NEW2", "CORE1"]


def test_minimum_age_uses_only_candles_before_entry():
    candles = {"NEW": {"time": [1_000, 2_000]}}
    frame = pd.DataFrame([
        {"symbol": "NEW", "entry_time": 1_000 + 6 * 86400},
        {"symbol": "NEW", "entry_time": 1_000 + 7 * 86400},
    ])
    result = mature.apply_age(frame, candles, 7)
    assert result.entry_time.tolist() == [1_000 + 7 * 86400]
