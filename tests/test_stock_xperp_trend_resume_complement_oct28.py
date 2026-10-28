import json

import stock_xperp_trend_resume_complement_oct28_audit as audit


def report():
    return json.loads(audit.OUT.read_text(encoding="utf-8"))


def test_trend_resume_search_covers_declared_independent_grid():
    value = report()
    expected = (len(audit.HORIZONS)*len(audit.PULLBACKS)*len(audit.TARGETS)
                * len(audit.STOPS)*len(audit.HOLDS)*len(audit.TREND_MINIMUMS)
                * len(audit.EFFICIENCIES)*len(audit.PULLBACK_MINIMUMS)
                * len(audit.PULLBACK_RATIOS)*len(audit.CONFIRMATIONS)
                * len(audit.BASKET_ALIGNMENTS)*len(audit.SIDES))
    assert value["grid"]["candidate_rules"] == expected == 13_824
    assert value["grid"]["resolved_geometries"] == 72
    assert value["protocol"]["limit_orders"] is False


def test_no_selection_survivor_cannot_reach_holdouts_or_runtime():
    value = report()
    assert value["grid"]["raw_survivors"] > 0
    assert value["grid"]["exact_survivors"] == 0
    assert value["selected_before_holdouts"] is None
    assert value["test_pass"] is False
    assert value["confirmation_pass"] is False
    assert value["objective_30pct_every_month_at_3pct"] is False
    assert value["status"] == "REJECTED_NO_ROBUST_IMPROVEMENT"
    assert value["ready_for_live_orders"] is False
    assert value["orders_possible"] is False
