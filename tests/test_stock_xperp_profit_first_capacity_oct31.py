import json
from pathlib import Path

from stock_module_cluster_audit import portfolio_gate
import stock_xperp_profit_first_capacity_oct31_audit as audit


def _row(symbol: str, direction: str, entry: float, exit_time: float) -> dict:
    return {
        "symbol": symbol, "direction": direction,
        "entry_time": entry, "exit_time": exit_time,
        "module": "test", "signed_return": .01,
    }


def test_direction_cap_is_explicit_and_backward_compatible():
    rows = [_row(f"S{i}", "LONG", 1000 + 900 * i, 10000)
            for i in range(6)]
    assert len(portfolio_gate(rows, None)) == 4
    assert len(portfolio_gate(rows, None, direction_cap=6)) == 6


def test_capacity_report_keeps_hidden_periods_out_of_selection():
    report = json.loads(Path(audit.OUT).read_text(encoding="utf-8"))
    assert report["protocol"]["development"] == "2026-07"
    assert report["protocol"]["validation"] == "2026-08"
    assert report["protocol"]["hidden_reveal"] == [
        "2026-09", "2026-10-01..16"]
    assert report["baseline"]["direction_cap"] == 4
    assert report["baseline"]["rolling_hour_cap"] == 4


def test_accepted_policy_cannot_exceed_authorized_margin():
    report = json.loads(Path(audit.OUT).read_text(encoding="utf-8"))
    if report["accepted"]:
        for values in report["selected_before_hidden"]["periods"].values():
            assert values["peak_margin_pct_at_3pct"] <= 90
