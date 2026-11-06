import json

import stock_xperp_direction_hour_guard_nov6_audit as audit


def test_frozen_report_supports_paper_direction_guard():
    report = json.loads(audit.OUT.read_text(encoding="utf-8"))
    assert report["status"] == "PAPER_GUARD_ACCEPTED"
    assert report["fresh_november_used_for_acceptance"] is False
    assert report["rule"]["same_direction_entries_per_rolling_hour"] == 2
    baseline = report["baseline"]["summary"]
    selected = report["selected"]["summary"]
    assert selected["sum_monthly_return_pct"] >= .95 * baseline[
        "sum_monthly_return_pct"]
    assert selected["worst_month_return_pct"] > baseline[
        "worst_month_return_pct"]
    assert selected["max_monthly_drawdown_pct"] <= baseline[
        "max_monthly_drawdown_pct"]
