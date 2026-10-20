import json
from pathlib import Path

import config
import stock_xperp_backing_hour_cap_oct20_audit as audit


def test_frozen_runtime_cap_matches_confirmed_audit():
    report = json.loads(Path(audit.OUT).read_text(encoding="utf-8"))
    assert report["objective_complete_months_pass"] is True
    assert report["selected_rolling_hour_cap"] == 4
    assert config.NATIVE_XPERP_ROLLING_HOUR_CAP == 4
    assert all(
        report["margin_scenarios"]["margin_5pct"][month]["return_pct"] >= 30
        for month in ("2026-07", "2026-08", "2026-09")
    )


def test_untouched_forward_did_not_regress_after_hour_cap():
    path = Path("reports/audit_2026_09_08/stock_xperp_backing_hour_cap_forward_oct20.json")
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["rolling_hour_cap"] == 4
    assert report["pending_signals"] == 0
    assert report["resolved_signals"] == 27
    assert report["resolved_metrics_3pct"]["return_pct"] > 0
    assert report["resolved_metrics_3pct"]["dd_pct"] < 1
