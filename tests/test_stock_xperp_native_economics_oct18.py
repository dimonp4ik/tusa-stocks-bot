import pytest

import stock_xperp_native_economics_oct18_audit as audit


def row(entry, exit_, signed_return, symbol="A"):
    return {
        "entry_time": entry, "exit_time": exit_,
        "signed_return": signed_return, "module_priority": 0,
        "symbol": symbol,
    }


def test_overlapping_positions_use_equity_known_at_entry():
    result = audit.simulate([
        row(0, 100, .10, "A"), row(0, 200, .10, "B")], .03)
    assert result["return_pct"] == pytest.approx(6.0)


def test_margin_scenario_scales_nonoverlapping_compounded_returns():
    low = audit.simulate([row(0, 100, .02), row(200, 300, .02)], .02)
    high = audit.simulate([row(0, 100, .02), row(200, 300, .02)], .05)
    assert low["return_pct"] == pytest.approx(.8016)
    assert high["return_pct"] == pytest.approx(2.01)
