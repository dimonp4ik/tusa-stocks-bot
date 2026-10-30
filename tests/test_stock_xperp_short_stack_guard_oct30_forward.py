import json

import stock_xperp_short_stack_guard_oct30_forward as audit


def test_frozen_short_stack_guard_improves_tiny_forward_return_only():
    report = json.loads(audit.OUT.read_text(encoding="utf-8"))
    comparison = report["new_outcomes_after_freeze"]
    assert comparison["baseline"]["resolved"] == 14
    assert comparison["guarded"]["resolved"] == 13
    assert comparison["return_delta_pp"] == .2998
    assert comparison["dd_delta_pp"] == 0.0
    assert report["forward_gate"]["enough_outcomes"] is False
    assert report["status"] == "AWAIT_MORE_FORWARD_OUTCOMES"
    assert report["ready_for_live_orders"] is False


def test_oct30_guard_monitor_preserves_both_parity_checks():
    report = json.loads(audit.OUT.read_text(encoding="utf-8"))
    assert report["integrity"]["published_baseline"]["passes"] is True
    assert report["integrity"]["append_only_stable_prefix"]["passes"] is True
