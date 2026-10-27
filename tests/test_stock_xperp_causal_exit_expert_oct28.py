import json

import pandas as pd

import stock_xperp_causal_exit_expert_oct28_audit as audit
import stock_xperp_basket_trend_forward_oct28_audit as forward


def report():
    return json.loads(audit.OUT.read_text(encoding="utf-8"))


def test_virtual_expert_uses_only_strictly_completed_prior_outcomes():
    value = report()
    assert value["integrity"]["strictly_prior_exit_violations"] == 0
    assert value["integrity"]["policies"] == 24
    assert value["integrity"]["entries"] == 12_266
    assert value["protocol"]["candidate_router_configs"] == 192
    assert value["protocol"]["limit_orders"] is False


def test_selected_rows_reproduce_saved_monthly_metrics():
    value = report()["selected"]["all_months"]
    rows = pd.read_csv(audit.ROWS_OUT)
    months = pd.to_datetime(rows.entry_time, unit="s", utc=True).dt.strftime("%Y-%m")
    for month in audit.MONTHS:
        part = rows[months.eq(month)]
        for margin in (.03, .05):
            key = f"margin_{int(100*margin)}pct"
            observed = forward.metrics(part, margin)
            for field in (
                    "n", "return_pct", "dd_pct", "negative_months",
                    "trades_per_active_day", "profitable_exit_rate_pct"):
                assert observed[field] == value[month][key][field]


def test_out_of_sample_failure_cannot_activate_the_route():
    value = report()
    assert value["status"] == "REJECTED_OUT_OF_SAMPLE"
    assert value["test_pass"] is False
    assert value["confirmation_pass"] is False
    assert value["objective_30pct_every_month_at_3pct"] is False
    assert value["ready_for_live_orders"] is False
    assert value["orders_possible"] is False
