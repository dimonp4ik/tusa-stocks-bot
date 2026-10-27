import json

import pandas as pd

import stock_xperp_basket_trend_forward_oct28_audit as audit


def test_oct28_forward_has_exact_raw_and_stable_signal_parity():
    report = json.loads(audit.OUT.read_text(encoding="utf-8"))
    assert report["integrity"]["raw_candle_overlap"]["passes"] is True
    assert report["integrity"]["raw_candle_overlap"]["value_mismatches"] == 0
    assert report["integrity"]["stable_accepted_prefix"]["passes"] is True
    assert report["integrity"]["stable_accepted_prefix"]["value_mismatches"] == 0
    assert report["policy"]["retuned_on_oct27"] is False
    assert report["policy"]["retuned_on_oct28"] is False


def test_forward_keeps_the_original_selection_cutoff():
    report = json.loads(audit.OUT.read_text(encoding="utf-8"))
    rows = pd.read_csv(audit.ROWS_OUT)
    cutoff = int(pd.Timestamp(
        report["integrity"]["original_selection_cutoff_utc"]).timestamp())
    cumulative = rows[rows.entry_time.gt(cutoff)]
    assert report["new_rows"] == len(cumulative)
    assert report["new_resolved_rows"] == int(
        cumulative.outcome.ne("PENDING").sum())
    assert report["activation_gate"]["observed_resolved_trades"] == int(
        cumulative.outcome.ne("PENDING").sum())
    assert report["activation_gate"]["met"] is False
    assert report["ready_for_live_orders"] is False


def test_october_economic_gate_reports_material_objective_failure():
    report = json.loads(audit.OUT.read_text(encoding="utf-8"))
    value = report["october_to_date"]["margin_3pct"]
    assert value["n"] == value["resolved"]
    assert value["trades_per_active_day"] >= 3
    assert value["economic_gate"]["frequency_pass"] is True
    assert value["economic_gate"]["return_pass"] is False
    assert value["economic_gate"]["passes"] is False
    assert value["return_pct"] < 30
