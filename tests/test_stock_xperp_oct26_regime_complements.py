import json

import stock_xperp_basket_direction_router_oct26_audit as direction
import stock_xperp_multi_horizon_complement_oct26_audit as complement


def test_direction_router_was_selected_before_validation_and_rejected():
    report = json.loads(direction.OUT.read_text(encoding="utf-8"))
    assert report["grid"]["tested"] == 216
    assert report["protocol"]["retune_on_validation_or_october"] is False
    assert report["selected_before_september"]["selection_score"] > report[
        "baseline"]["selection_score"]
    assert report["validation_improves_baseline"] is False
    assert report["robust_improvement"] is False
    assert report["status"] == "REJECTED_NO_ROBUST_IMPROVEMENT"


def test_multi_horizon_complement_does_not_replace_the_frozen_route():
    report = json.loads(complement.OUT.read_text(encoding="utf-8"))
    assert report["grid"]["tested"] == 81
    assert report["protocol"]["retune_on_validation_or_october"] is False
    assert report["selected_before_september"]["selection_score"] > report[
        "baseline"]["selection_score"]
    assert report["validation_improves_baseline"] is False
    assert report["robust_improvement"] is False
    assert report["status"] == "REJECTED_NO_ROBUST_IMPROVEMENT"
