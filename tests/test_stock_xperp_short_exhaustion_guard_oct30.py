import json

import pandas as pd

import stock_xperp_short_exhaustion_guard_oct30_audit as audit


def test_guard_only_vetoes_exhausted_shorts():
    frame = pd.DataFrame([
        {"direction": "SHORT", "basket_trend_16": -.011},
        {"direction": "SHORT", "basket_trend_16": -.009},
        {"direction": "LONG", "basket_trend_16": -.02},
    ])
    kept = audit.apply_guard(frame, -.01)
    assert kept.index.tolist() == [1, 2]


def test_report_keeps_experiment_shadow_only():
    value = json.loads(audit.OUT.read_text(encoding="utf-8"))
    assert value["status"] == "SHADOW_ONLY_FORWARD_EXPERIMENT"
    assert value["ready_for_live_orders"] is False
    assert value["orders_possible"] is False
    assert value["activation_gate"]["passes"] is True
    assert value["policy"]["entry"] == "unchanged next 15-minute market open"
    assert value["guarded"]["new_resolved_after_freeze"]["margin_3pct"][
        "return_pct"] > value["baseline"]["new_resolved_after_freeze"][
            "margin_3pct"]["return_pct"]
