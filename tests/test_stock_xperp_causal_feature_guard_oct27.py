import json

import pandas as pd

import stock_xperp_causal_feature_guard_oct27_audit as audit


def frame():
    return pd.DataFrame([
        {"direction": "LONG", "quality": 1.0},
        {"direction": "LONG", "quality": 3.0},
        {"direction": "SHORT", "quality": 1.0},
        {"direction": "SHORT", "quality": 3.0},
    ])


def test_side_specific_guard_keeps_opposite_side_unchanged():
    result = audit.apply(frame(), "quality", "high", "LONG", 2.0)
    assert list(result.index) == [1, 2, 3]


def test_all_side_low_guard_applies_to_every_row():
    result = audit.apply(frame(), "quality", "low", "ALL", 2.0)
    assert list(result.index) == [0, 2]


def test_report_rejects_rule_that_failed_later_months():
    report = json.loads(audit.OUT.read_text(encoding="utf-8"))
    assert report["protocol"]["selection"] == ["2026-07", "2026-08"]
    assert report["grid"]["unique_candidates"] > 500
    assert report["grid"]["selection_survivors"] > 0
    assert report["checks"]["validation_september"] is False
    assert report["checks"]["accepted"] is False
    assert report["status"] == "REJECTED_NO_ROBUST_IMPROVEMENT"


def test_report_baseline_matches_frozen_basket_route():
    report = json.loads(audit.OUT.read_text(encoding="utf-8"))
    baseline = report["baseline"]
    assert baseline["selection_july"]["margin_3pct"]["return_pct"] == 31.3702
    assert baseline["selection_august"]["margin_3pct"]["return_pct"] == 23.8007
    assert baseline["validation_september"]["margin_3pct"]["return_pct"] == 22.8547
