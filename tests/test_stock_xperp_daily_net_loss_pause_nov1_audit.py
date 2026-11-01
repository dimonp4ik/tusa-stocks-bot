import json
from pathlib import Path

import stock_xperp_daily_net_loss_pause_nov1_audit as audit


def test_daily_net_brake_has_low_historical_incremental_impact():
    report = json.loads(Path(audit.OUT).read_text(encoding="utf-8"))
    selected = report["selected"]
    assert selected["daily_net_loss_cap_r"] == 2.0
    assert selected["rejected"] == 1
    assert report["ready_for_live_orders"] is False
