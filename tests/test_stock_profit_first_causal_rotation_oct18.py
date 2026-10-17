import pandas as pd

import stock_profit_first_causal_rotation_oct18_audit as audit


def test_history_uses_only_trades_closed_before_month_start():
    frame = pd.DataFrame({
        "entry_dt": pd.to_datetime(["2025-01-20", "2025-01-25"], utc=True),
        "exit_dt": pd.to_datetime(["2025-01-31", "2025-02-02"], utc=True),
        "signed_return": [0.02, 0.50],
    })
    found = audit.history_before(frame, pd.Timestamp("2025-02-01", tz="UTC"), 3)
    assert found.signed_return.tolist() == [0.02]


def test_rotation_grid_is_small_and_predeclared():
    assert len(audit.policy_grid()) == (len(audit.LOOKBACK_MONTHS)
                                        * len(audit.TOP_K)
                                        * len(audit.SCORE_MODES))


def test_development_score_ignores_win_rate():
    value = {"n": 600, "return_pct": 40.0, "dd_pct": 8.0,
             "negative_months": 2, "literal_target_hit_rate": 10.0}
    assert audit.development_score(value) == 24.0
