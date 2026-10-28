import json

import stock_xperp_short_stack_guard_oct28_audit as audit


def row(symbol, direction="SHORT", trend=.01, signed=.02, entry_time=9000):
    return {
        "symbol": symbol, "direction": direction,
        "entry_time": entry_time, "exit_time": entry_time + 900,
        "signed_return": signed, "outcome": "TP" if signed > 0 else "SL",
        "module": "native_xperp_basket_trend_guard",
        "cross_z": 3.0 if direction == "SHORT" else -3.0,
        "basket_trend_96": trend,
    }


def test_guard_rejects_only_the_second_rising_basket_short():
    rows = [row("AUSDT"), row("BUSDT")]
    baseline = audit.portfolio_gate(rows, False)
    guarded = audit.portfolio_gate(rows, True)
    assert [item["symbol"] for item in baseline] == ["AUSDT", "BUSDT"]
    assert [item["symbol"] for item in guarded] == ["AUSDT"]


def test_guard_keeps_opposite_direction_or_nonpositive_basket_short():
    opposite = audit.portfolio_gate(
        [row("AUSDT"), row("BUSDT", direction="LONG")], True)
    falling = audit.portfolio_gate(
        [row("AUSDT", trend=-.01), row("BUSDT", trend=-.01)], True)
    assert len(opposite) == 2
    assert len(falling) == 2


def test_report_preserves_baseline_and_does_not_promote_tiny_forward_sample():
    report = json.loads(audit.OUT.read_text(encoding="utf-8"))
    assert report["integrity"]["published_baseline"]["passes"] is True
    assert report["integrity"]["append_only_stable_prefix"]["passes"] is True
    assert report["forward_gate"]["baseline_resolved"] == 4
    assert report["forward_gate"]["guarded_resolved"] == 3
    assert report["forward_gate"]["enough_outcomes"] is False
    assert report["ready_for_live_orders"] is False
    assert report["status"] == "AWAIT_MORE_FORWARD_OUTCOMES"


def test_visible_month_replay_exposes_the_september_backfill_regression():
    report = json.loads(audit.OUT.read_text(encoding="utf-8"))
    september = report["retrospective_visible_months"]["2026-09"]
    assert september["return_delta_pp"] < 0
    assert september["dd_delta_pp"] > 0
