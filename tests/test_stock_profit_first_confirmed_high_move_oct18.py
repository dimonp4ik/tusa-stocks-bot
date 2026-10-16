import stock_profit_first_confirmed_high_move_oct18_audit as audit


def test_development_gate_ignores_win_rate():
    years = {
        "2022": {"n": 100, "return_pct": 1.0, "dd_pct": 14.0,
                 "trades_per_calendar_day": 1.0, "wr": 20.0},
        "2023": {"n": 101, "return_pct": 2.0, "dd_pct": 10.0,
                 "trades_per_calendar_day": 1.1, "wr": 25.0},
    }
    assert audit.development_ok(years)


def test_development_requires_both_years_profitable():
    years = {
        "2022": {"n": 100, "return_pct": 1.0, "dd_pct": 4.0,
                 "trades_per_calendar_day": 1.0},
        "2023": {"n": 100, "return_pct": -0.1, "dd_pct": 4.0,
                 "trades_per_calendar_day": 1.0},
    }
    assert not audit.development_ok(years)


def test_score_uses_weaker_development_year():
    years = {
        "2022": {"equivalent_monthly_return_pct": 8.0, "dd_pct": 4.0,
                 "negative_months": 1},
        "2023": {"equivalent_monthly_return_pct": 3.0, "dd_pct": 2.0,
                 "negative_months": 0},
    }
    assert audit.development_score(years) == 2.58
