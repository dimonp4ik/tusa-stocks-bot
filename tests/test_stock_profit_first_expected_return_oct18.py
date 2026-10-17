import stock_profit_first_expected_return_oct18_audit as audit


def test_development_gate_does_not_require_win_rate():
    years = {
        "2022": {"n": 300, "return_pct": 12.0, "dd_pct": 10.0,
                 "trades_per_calendar_day": 1.0, "wr": 10.0},
        "2023": {"n": 250, "return_pct": 8.0, "dd_pct": 12.0,
                 "trades_per_calendar_day": .8, "wr": 15.0},
    }
    assert audit.development_ok(years)


def test_development_score_is_limited_by_weaker_year():
    years = {
        "2022": {"equivalent_monthly_return_pct": 10.0, "dd_pct": 4.0,
                 "negative_months": 0, "trades_per_calendar_day": 2.0},
        "2023": {"equivalent_monthly_return_pct": 3.0, "dd_pct": 5.0,
                 "negative_months": 1, "trades_per_calendar_day": 1.0},
    }
    assert audit.development_score(years) == 2.55


def test_grid_only_uses_material_targets():
    assert min(audit.TARGETS_PCT) >= 2.0
    assert max(audit.TARGETS_PCT) == 4.0
