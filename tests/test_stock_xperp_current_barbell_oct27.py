import json

import pytest

import stock_xperp_current_barbell_oct27_audit as audit


def row(direction: int = 1) -> dict:
    return {
        "symbol": "TESTUSDT", "entry_index": 0, "entry_time": 0,
        "direction_n": direction, "direction": "LONG" if direction > 0 else "SHORT",
    }


def candles(open_, high, low, close):
    return {
        "time": [900 * index for index in range(len(open_))],
        "open": open_, "high": high, "low": low, "close": close,
    }


def test_stop_wins_same_bar_near_target_ambiguity():
    value = audit.resolve_row(
        row(), candles([100], [101.5], [99], [100.5]), 1.0, .2)
    assert value["outcome"] == "SL"
    assert value["signed_return"] == pytest.approx(-.01)
    assert value["near_activated"] is False


def test_runner_can_fill_on_near_activation_bar_after_stop_check():
    value = audit.resolve_row(
        row(), candles([100], [102], [99.5], [101.5]), 1.0, .2)
    assert value["outcome"] == "TP"
    assert value["signed_return"] == pytest.approx(.018)


def test_break_even_starts_on_bar_after_near_activation():
    value = audit.resolve_row(
        row(), candles([100, 101], [101.2, 102], [99.5, 100], [101, 100.5]),
        1.0, 0.0)
    assert value["outcome"] == "BE"
    assert value["signed_return"] == pytest.approx(0.0)
    assert value["exit_time"] == 1800


def test_unfinished_horizon_stays_pending():
    value = audit.resolve_row(
        row(), candles([100, 100], [100.2, 100.2], [99.8, 99.8], [100, 100]),
        1.0, 0.0)
    assert value["outcome"] == "PENDING"
    assert value["signed_return"] is None


def test_report_candidate_was_selected_without_september_or_october():
    report = json.loads(audit.OUT.read_text(encoding="utf-8"))
    assert report["protocol"]["selection"] == ["2026-07", "2026-08"]
    assert report["selected"]["near_target_pct"] in audit.NEAR_TARGETS_PCT
    assert report["selected"]["near_fraction"] in audit.NEAR_FRACTIONS
    assert report["status"] == "REJECTED_NO_ROBUST_IMPROVEMENT"
    assert report["acceptance"]["accepted"] is False


def test_report_baseline_matches_frozen_basket_route():
    report = json.loads(audit.OUT.read_text(encoding="utf-8"))
    assert report["baseline"]["selection_july"]["margin_3pct"]["return_pct"] == 31.3702
    assert report["baseline"]["selection_august"]["margin_3pct"]["return_pct"] == 23.8007
    assert report["baseline"]["validation_september"]["margin_3pct"]["return_pct"] == 22.8547
