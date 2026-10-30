import json

import pandas as pd

import stock_xperp_basket_trend_forward_oct30_audit as audit


def report():
    return json.loads(audit.OUT.read_text(encoding="utf-8"))


def test_oct30_stock_snapshot_preserves_raw_and_signal_history():
    integrity = report()["integrity"]
    assert integrity["raw_candle_overlap"]["passes"] is True
    assert integrity["raw_candle_overlap"]["value_mismatches"] == 0
    assert integrity["stable_accepted_prefix"]["passes"] is True
    assert integrity["stable_accepted_prefix"]["value_mismatches"] == 0
    assert integrity["stable_accepted_prefix"]["old_rows"] == 957
    assert integrity["stable_accepted_prefix"]["new_rows"] == 957


def test_saved_rows_reproduce_october_metrics():
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


def test_stock_route_still_fails_the_profit_gate():
    value = report()["october_to_date"]["margin_3pct"]
    assert value["resolved"] == 220
    assert value["pending"] == 1
    assert value["return_pct"] == 14.9019
    assert value["dd_pct"] == 2.6878
    assert value["economic_gate"]["frequency_pass"] is True
    assert value["economic_gate"]["return_pass"] is False


def test_new_stock_slice_exposes_loss_cluster_without_overstating_readiness():
    value = report()
    fresh = value["newly_revealed_after_oct27_2156_snapshot"]["margin_3pct"]
    assert fresh["resolved"] == 9
    assert fresh["pending"] == 1
    assert fresh["return_pct"] == -0.2941
    assert fresh["max_loss_streak"] == 4
    assert value["activation_gate"] == {
        "required_resolved_trades": 60,
        "observed_resolved_trades": 20,
        "required_calendar_days": 14,
        "observed_calendar_days": 4,
        "met": False,
    }
    assert value["ready_for_live_orders"] is False
