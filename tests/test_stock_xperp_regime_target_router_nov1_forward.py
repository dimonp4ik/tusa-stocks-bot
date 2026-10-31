import json
from pathlib import Path

import stock_xperp_regime_target_router_nov1_forward as audit


def test_nov1_refresh_preserves_append_only_integrity():
    report = json.loads(Path(audit.OUT).read_text(encoding="utf-8"))
    assert report["integrity"]["raw_candle_overlap"]["passes"] is True
    assert report["integrity"]["stable_accepted_row_parity"]["passes"] is True
    assert set(report["complete_months"]) == {
        "2026-07", "2026-08", "2026-09", "2026-10"}


def test_nov1_refresh_reports_actual_objective_without_claiming_live_ready():
    report = json.loads(Path(audit.OUT).read_text(encoding="utf-8"))
    objective = report["monthly_30pct_objective"]
    for margin_key in ("margin_3pct", "margin_5pct"):
        assert objective[margin_key]["all_complete_months_pass"] == all(
            objective[margin_key][month] for month in audit.MONTHS)
    assert objective["margin_3pct"]["all_complete_months_pass"] is False
    assert objective["margin_5pct"]["all_complete_months_pass"] is True
    assert report["ready_for_live_orders"] is False
    assert report["orders_possible"] is False
