import json

import stock_xperp_regime_target_router_oct30_forward as audit


def test_forward_report_preserves_prefix_and_stays_shadow_only():
    value = json.loads(audit.OUT.read_text(encoding="utf-8"))
    assert value["status"] == "FORWARD_GATE_FAILED_SMALL_SAMPLE"
    assert value["ready_for_live_orders"] is False
    assert value["orders_possible"] is False
    assert value["integrity"]["raw_candle_overlap"]["passes"] is True
    assert value["integrity"]["stable_accepted_row_parity"]["passes"] is True
    assert value["policy"]["orders"] == "market only"
    assert value["forward_gate"]["passes"] is False
    assert value["forward_gate"]["positive_return"] is False


def test_forward_report_has_only_fixed_router_targets():
    value = json.loads(audit.OUT.read_text(encoding="utf-8"))
    assert set(value["target_distribution_full"]) <= {"2.0", "2.5", "3.0"}
    assert value["newly_resolved_rows"] >= 0
    assert value["new_entry_rows"] >= 0
