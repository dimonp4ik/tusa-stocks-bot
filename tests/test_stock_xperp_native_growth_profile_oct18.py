import stock_xperp_native_growth_profile_oct18_audit as audit


def test_growth_gate_requires_every_complete_month():
    months = {
        "jul": {"return_pct": 31},
        "aug": {"return_pct": 45},
        "sep": {"return_pct": 30},
    }
    pre = {"dd_pct": 9.9, "negative_months": 0}
    assert audit.qualifies(months, pre)
    months["sep"]["return_pct"] = 29.99
    assert not audit.qualifies(months, pre)


def test_growth_gate_rejects_excess_drawdown_or_negative_month():
    months = {name: {"return_pct": 35} for name in ("jul", "aug", "sep")}
    assert not audit.qualifies(months, {"dd_pct": 10.01, "negative_months": 0})
    assert not audit.qualifies(months, {"dd_pct": 5, "negative_months": 1})
