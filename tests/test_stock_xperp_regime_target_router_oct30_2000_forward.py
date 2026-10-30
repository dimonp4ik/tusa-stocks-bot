import json

import stock_xperp_regime_target_router_oct30_2000_forward as audit


def report():
    return json.loads(audit.OUT.read_text(encoding="utf-8"))


def test_evening_refresh_preserves_every_stable_row_and_candle():
    value = report()
    overlap = value["integrity"]["raw_candle_overlap"]
    parity = value["integrity"]["stable_accepted_row_parity"]
    assert overlap["passes"] is True
    assert overlap["symbols"] == 46
    assert overlap["overlap_bars"] == 444601
    assert overlap["value_mismatches"] == 0
    assert parity["passes"] is True
    assert parity["old_rows"] == parity["new_rows"] == 907
    assert parity["value_mismatches"] == 0


def test_zero_new_setups_are_recorded_without_inventing_performance():
    value = report()
    assert value["new_entry_rows"] == 0
    assert value["newly_resolved_rows"] == 0
    assert value["newly_resolved_after_old_snapshot"]["margin_3pct"]["n"] == 0
    assert value["forward_gate"]["passes"] is False
    assert value["ready_for_live_orders"] is False
    assert value["orders_possible"] is False


def test_october_router_result_is_unchanged():
    value = report()["october_to_date"]
    assert value["margin_3pct"]["n"] == 223
    assert value["margin_3pct"]["return_pct"] == 25.939
    assert value["margin_3pct"]["dd_pct"] == 2.3607
    assert value["margin_3pct"]["economic_gate"]["passes"] is False
    assert value["margin_5pct"]["return_pct"] == 46.2028
    assert value["margin_5pct"]["economic_gate"]["passes"] is True
