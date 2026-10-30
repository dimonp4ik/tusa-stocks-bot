import json

import stock_xperp_multihorizon_short_veto_oct30_audit as audit


def report():
    return json.loads(audit.OUT.read_text(encoding="utf-8"))


def test_selected_veto_is_rejected_on_older_october_confirmation():
    value = report()
    assert value["status"] == "REJECTED_CONFIRMATION_AND_POSTHOC"
    assert value["test_pass"] is True
    assert value["confirmation_pass"] is False
    assert value["runtime_change_allowed"] is False
    assert value["ready_for_live_orders"] is False
    assert value["orders_possible"] is False

    selected = value["selected_on_july_august"]
    assert selected["config"] == {
        "basket_32_at_most": -0.02,
        "basket_64_at_most": -0.06,
        "basket_96_at_most": -0.06,
    }
    assert selected["selection_return_gain_pp"] == 7.1165
    confirmation = value["selected_deltas"]["confirmation_old_october"]
    assert confirmation["return_delta_pp"] == -0.4372
    assert confirmation["dd_delta_pp"] == 0.0515


def test_visible_latest_slice_cannot_be_used_to_accept_the_rule():
    value = report()
    latest = value["selected_deltas"]["visible_post_hypothesis_slice"]
    assert latest["baseline"]["resolved"] == 5
    assert latest["baseline"]["return_pct"] == -0.5973
    assert latest["candidate"]["resolved"] == 0
    assert value["protocol"]["latest_slice"].startswith("diagnostic only")
