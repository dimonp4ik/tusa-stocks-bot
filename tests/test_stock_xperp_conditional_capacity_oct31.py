import json
from pathlib import Path

import stock_xperp_conditional_capacity_oct31_audit as audit


def _row(symbol: str, entry: float, z: float) -> dict:
    return {
        "symbol": symbol, "direction": "LONG", "entry_time": entry,
        "exit_time": 10000, "module": "test", "signed_return": .01,
        "cross_z": z,
    }


def test_overflow_slot_requires_candidate_confirmation():
    rows = [_row(f"S{i}", 1000 + 900 * i, 2.1) for i in range(4)]
    rows.append(_row("STRONG", 4600, 4.0))
    rejected = audit.conditional_gate(rows, lambda row: False, "z_high", 3)
    accepted = audit.conditional_gate(
        rows, lambda row: abs(float(row["cross_z"])) >= 3.5, "z_high", 3)
    assert len(rejected) == 4
    assert len(accepted) == 5


def test_report_preserves_hidden_reveal_and_margin_boundary():
    report = json.loads(Path(audit.OUT).read_text(encoding="utf-8"))
    assert report["protocol"]["hidden_reveal"] == [
        "2026-09", "2026-10-01..16"]
    for values in report["selected_before_hidden"]["periods"].values():
        assert values["peak_margin_pct_at_3pct"] <= 90
