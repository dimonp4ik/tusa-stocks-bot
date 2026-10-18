import numpy as np
import pandas as pd

import stock_xperp_native_model_quality_oct19_audit as audit


def test_matrix_uses_frozen_numeric_and_symbol_schema():
    frame = pd.DataFrame({
        **{name: [1.0, np.nan] for name in audit.NUMERIC},
        "symbol": ["AAPLUSDT", "NEWUSDT"],
    })
    design = {
        "symbols": ["AAPLUSDT", "MSFTUSDT"],
        "medians": {name: 0.25 for name in audit.NUMERIC},
    }

    values = audit.matrix(frame, design)

    assert values.shape == (2, len(audit.NUMERIC) + 2)
    assert np.isfinite(values).all()
    assert values[0, -2:].tolist() == [1.0, 0.0]
    assert values[1, -2:].tolist() == [0.0, 0.0]


def test_selection_gate_requires_projected_monthly_objective():
    value = {
        "n": 66,
        "dd_pct": 4.0,
        "trades_per_market_day": 3.0,
        "projected_22_session_pct": 30.0,
    }

    assert audit.selection_ok(value)
    value["projected_22_session_pct"] = 29.9999
    assert not audit.selection_ok(value)


def test_full_month_gate_requires_actual_return_and_no_negative_month():
    value = {
        "n": 66,
        "dd_pct": 8.0,
        "trades_per_market_day": 3.0,
        "return_pct": 30.0,
        "negative_months": 0,
    }

    assert audit.full_month_ok(value)
    value["return_pct"] = 29.9999
    assert not audit.full_month_ok(value)
    value["return_pct"] = 30.0
    value["negative_months"] = 1
    assert not audit.full_month_ok(value)
