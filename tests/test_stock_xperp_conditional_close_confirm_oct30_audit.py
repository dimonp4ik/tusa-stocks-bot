import json

import stock_xperp_conditional_close_confirm_oct30_audit as audit


def report():
    return json.loads(audit.OUT.read_text(encoding="utf-8"))


def test_stop_path_attribution_is_recorded_without_runtime_change():
    value = report()
    path = value["baseline_stop_path"]
    assert path == {
        "stops": 467,
        "closed_back_inside_pct": 23.7687,
        "target_remaining_hold_pct": 28.4797,
        "double_hold_complete_stops": 460,
        "target_double_hold_pct_of_complete": 38.4783,
    }
    assert value["runtime_change_allowed"] is False
    assert value["ready_for_live_orders"] is False
    assert value["orders_possible"] is False


def test_uniform_close_confirmation_damages_each_month():
    value = report()
    assert value["status"] == "REJECTED_TEST_AND_UNIFORM_DAMAGE"
    for month in audit.MONTHS:
        delayed = value["uniform_close_confirmation"][month]
        baseline = value["baseline"][month]
        assert delayed["return_pct"] < baseline["return_pct"]
        assert delayed["dd_pct"] > baseline["dd_pct"]


def test_july_selected_cohort_fails_august_validation():
    value = report()
    selected = value["selected_on_july"]
    assert selected["config"] == {
        "direction": "LONG",
        "feature": "abs_z",
        "threshold": 3.5,
        "operator": "at_least",
    }
    assert selected["july_return_gain_pp"] == 2.6573
    assert selected["metrics"]["2026-08"]["return_pct"] == 22.8192
    assert value["baseline"]["2026-08"]["return_pct"] == 23.6995
    assert value["validation_pass"] is False
    assert value["test_pass"] is False
    assert value["all_visible_qualified_candidates"] == 0
