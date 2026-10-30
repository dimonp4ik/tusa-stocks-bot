import json

import stock_5m_material_forward_oct30_audit as audit


def report():
    return json.loads(audit.OUT.read_text(encoding="utf-8"))


def test_oct30_snapshot_preserves_confirmed_candles_and_signal_events():
    value = report()["integrity"]
    assert all(
        not any(item["revisions"].values())
        for item in value["candle_overlap"].values()
    )
    parity = value["legacy_research_event_parity"]
    assert parity["legacy_events"] == parity["current_runtime_events"] == 91
    assert parity["matched"] == 91
    assert parity["legacy_missing_from_runtime"] == 0
    assert parity["runtime_unexpected"] == 0


def test_frozen_material_5m_route_fails_the_new_forward_gate():
    value = report()
    fresh = value["genuinely_post_material_freeze"]["result"]
    assert fresh["n"] == 1
    assert fresh["outcomes"] == {"SL": 1}
    assert fresh["return_pct"] == -0.395
    assert fresh["dd_pct"] == 0.395
    assert fresh["negative_months"] == 1
    assert value["readiness"] == {
        "required_genuinely_new_accepted": 30,
        "observed_genuinely_new_accepted": 1,
        "statistically_ready": False,
        "positive_return": False,
        "no_negative_months": False,
        "met": False,
        "live_activation_allowed": False,
    }
    assert value["status"] == "FORWARD_GATE_FAILED"
    assert value["ready_for_live_orders"] is False
    assert value["orders_possible"] is False


def test_material_move_requirement_rejects_most_small_5m_signals():
    material = report()["material_filter"]
    assert material["all_current_runtime_events"] == 148
    assert material["all_events_passing_0_5pct"] == 1
    assert material["rejected_below_0_5pct"] == 147
    assert material["pass_rate_pct"] == 0.6757

