import json

import stock_xperp_short_stack_regime_oct30_audit as audit


def test_veto_requires_a_second_short_and_matching_regime():
    admitted = [{"direction": "SHORT"}]
    rising = {"direction": "SHORT", "basket_trend_96": .01,
              "basket_trend_16": 0.0}
    exhausted = {"direction": "SHORT", "basket_trend_96": -.01,
                 "basket_trend_16": -.025}
    long = {"direction": "LONG", "basket_trend_96": .01,
            "basket_trend_16": -.025}
    assert audit.veto_second_short(
        rising, admitted, "second_short_unstable_2pct") is True
    assert audit.veto_second_short(
        exhausted, admitted, "second_short_unstable_2pct") is True
    assert audit.veto_second_short(
        long, admitted, "second_short_unstable_2pct") is False
    assert audit.veto_second_short(
        rising, [], "second_short_unstable_2pct") is False


def test_sequential_report_rejects_october_drawdown_regression():
    report = json.loads(audit.OUT.read_text(encoding="utf-8"))
    selected = report["selected_on_july"]
    assert selected["policy"] == "second_short_unstable_2pct"
    assert report["validation_pass"] is True
    assert report["test_pass"] is True
    assert report["confirmation_pass"] is False
    assert report["forward_better_return_no_worse_dd"] is True
    assert report["status"] == "REJECTED_SEQUENTIAL_VALIDATION"
    assert report["ready_for_live_orders"] is False
    assert report["orders_possible"] is False


def test_forward_diagnostic_is_improved_but_too_small_for_activation():
    report = json.loads(audit.OUT.read_text(encoding="utf-8"))
    baseline = report["baseline"]["new_outcomes_after_freeze"]
    candidate = report["selected_on_july"]["periods"][
        "new_outcomes_after_freeze"]
    assert baseline["resolved"] == 14
    assert baseline["return_pct"] == -.6159
    assert baseline["dd_pct"] == 1.4816
    assert baseline["max_loss_streak"] == 4
    assert candidate["resolved"] == 13
    assert candidate["return_pct"] == .6141
    assert candidate["dd_pct"] == 1.1623
    assert candidate["max_loss_streak"] == 3
