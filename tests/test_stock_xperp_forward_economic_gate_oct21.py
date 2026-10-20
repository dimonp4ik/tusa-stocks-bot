from stock_xperp_forward_economic_gate_oct21 import projected_month_pct


def test_projection_matches_constant_trade_compounding():
    found = projected_month_pct(.01, 3.0, .03)
    expected = 100 * ((1 + .03 * 10 * .01) ** (22 * 3) - 1)
    assert abs(found - expected) < 1e-12
