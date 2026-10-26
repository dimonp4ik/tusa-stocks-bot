import json

import stock_xperp_basket_trend_forward_oct27_audit as audit


def test_oct27_forward_has_exact_raw_and_stable_signal_parity():
    report = json.loads(audit.OUT.read_text(encoding="utf-8"))
    assert report["integrity"]["raw_candle_overlap"]["passes"] is True
    assert report["integrity"]["raw_candle_overlap"]["value_mismatches"] == 0
    assert report["integrity"]["stable_accepted_prefix"]["passes"] is True
    assert report["integrity"]["stable_accepted_prefix"]["value_mismatches"] == 0
    assert report["policy"]["retuned_on_oct27"] is False


def test_forward_counts_only_entries_after_the_old_snapshot_end():
    report = json.loads(audit.OUT.read_text(encoding="utf-8"))
    assert report["new_rows"] == 0
    assert report["new_resolved_rows"] == 0
    assert report["activation_gate"]["met"] is False
    assert report["ready_for_live_orders"] is False


def test_october_economic_gate_reports_material_objective_failure():
    report = json.loads(audit.OUT.read_text(encoding="utf-8"))
    value = report["october_to_date"]["margin_3pct"]
    assert value["n"] == value["resolved"]
    assert value["trades_per_active_day"] >= 3
    assert value["economic_gate"]["frequency_pass"] is True
    assert value["economic_gate"]["return_pass"] is False
    assert value["economic_gate"]["passes"] is False
    assert value["return_pct"] < 30
