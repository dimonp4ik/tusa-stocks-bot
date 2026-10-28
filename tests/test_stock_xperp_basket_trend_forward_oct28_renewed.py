import json

import pandas as pd

import stock_xperp_basket_trend_forward_oct28_renewed_audit as audit


def report():
    return json.loads(audit.OUT.read_text(encoding="utf-8"))


def test_renewed_stock_snapshot_has_exact_raw_and_signal_parity():
    integrity = report()["integrity"]
    assert integrity["raw_candle_overlap"]["passes"] is True
    assert integrity["raw_candle_overlap"]["value_mismatches"] == 0
    assert integrity["stable_accepted_prefix"]["passes"] is True
    assert integrity["stable_accepted_prefix"]["value_mismatches"] == 0


def test_saved_rows_reproduce_current_october_metrics():
    rows = pd.read_csv(audit.ROWS_OUT)
    stamp = pd.to_datetime(rows.entry_time, unit="s", utc=True)
    october = rows[stamp.ge(pd.Timestamp("2026-10-01", tz="UTC"))]
    saved = report()["october_to_date"]
    for margin in (.03, .05):
        key = f"margin_{int(100 * margin)}pct"
        observed = audit.metrics(october, margin)
        for field in (
                "n", "resolved", "pending", "return_pct", "dd_pct",
                "profitable_exit_rate_pct", "literal_tp_rate_pct",
                "trades_per_active_day", "max_loss_streak"):
            assert observed[field] == saved[key][field]
        assert observed["economic_gate"] == saved[key]["economic_gate"]


def test_stock_october_remains_below_the_requested_return_gate():
    value = report()["october_to_date"]["margin_3pct"]
    assert value["n"] == value["resolved"] == 210
    assert value["pending"] == 1
    assert value["return_pct"] == 15.1832
    assert value["dd_pct"] == 2.6878
    assert value["economic_gate"]["frequency_pass"] is True
    assert value["economic_gate"]["return_pass"] is False
    assert value["economic_gate"]["passes"] is False


def test_new_slice_and_activation_gate_are_not_overstated():
    value = report()
    fresh = value["newly_revealed_after_oct27_2156_snapshot"]["margin_3pct"]
    assert fresh["resolved"] == 4
    assert fresh["pending"] == 1
    assert fresh["return_pct"] == -0.3726
    assert value["activation_gate"] == {
        "required_resolved_trades": 60,
        "observed_resolved_trades": 10,
        "required_calendar_days": 14,
        "observed_calendar_days": 2,
        "met": False,
    }
    assert value["ready_for_live_orders"] is False
