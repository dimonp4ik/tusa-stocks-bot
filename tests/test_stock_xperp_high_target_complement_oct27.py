import json

import pandas as pd

import stock_xperp_high_target_complement_oct27_audit as audit


def metric(return_pct, dd_pct=2.0, trades=4.0, negative_months=0):
    return {
        "return_pct": return_pct,
        "dd_pct": dd_pct,
        "trades_per_active_day": trades,
        "negative_months": negative_months,
    }


def test_filter_rule_applies_strength_move_and_side_together():
    rows = pd.DataFrame([
        {"cross_z": 2.2, "signal_return": .009, "direction": "LONG"},
        {"cross_z": -2.4, "signal_return": -.010, "direction": "SHORT"},
        {"cross_z": 1.9, "signal_return": .009, "direction": "LONG"},
        {"cross_z": 2.2, "signal_return": .007, "direction": "LONG"},
    ])
    found = audit.filter_rule(rows, 2.0, .008, "LONG")
    assert found.index.tolist() == [0]


def test_combine_keeps_frozen_base_on_duplicate_entry():
    base = pd.DataFrame([{
        "symbol": "TEST", "entry_time": 100, "direction": "LONG",
        "module": "base", "signed_return": .02,
    }])
    supplement = pd.DataFrame([
        {"symbol": "TEST", "entry_time": 100, "direction": "LONG",
         "module": "supplement", "signed_return": .04},
        {"symbol": "OTHER", "entry_time": 100, "direction": "SHORT",
         "module": "supplement", "signed_return": .04},
    ])
    combined = audit.combine(base, supplement)
    assert combined[["symbol", "module"]].to_dict("records") == [
        {"symbol": "TEST", "module": "base"},
        {"symbol": "OTHER", "module": "supplement"},
    ]


def test_selection_requires_improvement_in_both_training_months():
    baseline = {
        "selection_july": metric(10),
        "selection_august": metric(10),
    }
    assert audit.selection_ok({
        "selection_july": metric(11),
        "selection_august": metric(11),
    }, baseline)
    assert not audit.selection_ok({
        "selection_july": metric(12),
        "selection_august": metric(9),
    }, baseline)


def test_report_rejects_training_winner_that_failed_september():
    report = json.loads(audit.OUT.read_text(encoding="utf-8"))
    assert report["grid"]["candidate_rules"] == 2592
    assert report["grid"]["selection_survivors"] == 85
    assert report["selected_before_holdouts"]["lookback_bars"] == 1
    assert report["checks"] == {
        "accepted": False,
        "observed_october": True,
        "validation_september": False,
    }
    assert report["status"] == "REJECTED_NO_ROBUST_IMPROVEMENT"
