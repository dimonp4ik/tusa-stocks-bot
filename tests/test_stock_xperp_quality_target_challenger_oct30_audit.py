import json

import stock_xperp_quality_target_challenger_oct30_audit as audit


def report():
    return json.loads(audit.OUT.read_text(encoding="utf-8"))


def test_only_one_visible_candidate_improves_every_month_without_worse_dd():
    value = report()
    assert value["status"] == "SHADOW_CHALLENGER_AWAIT_GENUINELY_NEW_FORWARD"
    assert value["qualified_visible_candidates"] == 1
    challenger = value["frozen_challenger"]
    assert challenger["config"] == {
        "feature": "atr14_fraction",
        "threshold": 0.002,
        "operator": "at_most",
        "direction": "SHORT",
        "target_pct": 3.0,
    }
    assert challenger["minimum_return_delta_pp"] == 0.1478
    assert challenger["total_return_delta_pp"] == 5.5246
    assert all(item["return_delta_pp"] >= 0
               and item["dd_delta_pp"] <= 0
               for item in challenger["deltas"].values())


def test_challenger_is_frozen_without_changing_runtime():
    value = report()
    assert value["runtime_change_allowed"] is False
    assert value["ready_for_live_orders"] is False
    assert value["orders_possible"] is False
    gate = value["forward_activation_gate"]
    assert gate["freeze_candle_end_utc"] == "2026-10-30T07:45:00+00:00"
    assert gate["minimum_new_resolved_trades"] == 30
    assert gate["observed_new_resolved_trades"] == 0
    assert gate["met"] is False


def test_visible_metrics_match_frozen_challenger_report():
    value = report()
    baseline = value["baseline"]
    candidate = value["frozen_challenger"]["metrics"]
    assert baseline["october"]["return_pct"] == 25.939
    assert candidate["october"]["return_pct"] == 27.7853
    assert candidate["september"]["return_pct"] == 31.8304
    assert candidate["october"]["dd_pct"] == baseline["october"]["dd_pct"]
