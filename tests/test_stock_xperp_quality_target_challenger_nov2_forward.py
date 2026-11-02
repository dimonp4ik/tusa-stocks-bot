import json
from pathlib import Path

import stock_xperp_quality_target_challenger_nov2_forward as audit


def test_frozen_challenger_changes_only_low_atr_shorts():
    base = {"entry_time": 13 * 3600, "basket_trend_32": 0.0}
    assert audit.challenger_target({
        **base, "direction": "SHORT", "atr14_fraction": .0019}) == 3.0
    assert audit.challenger_target({
        **base, "direction": "SHORT", "atr14_fraction": .0021}) == 2.5
    assert audit.challenger_target({
        **base, "direction": "LONG", "atr14_fraction": .0019}) == 2.5


def test_nov2_forward_report_preserves_integrity_and_gate_truth():
    report = json.loads(Path(audit.OUT).read_text(encoding="utf-8"))
    assert report["frozen_config"] == audit.FROZEN_CONFIG
    assert report["integrity"]["raw_candle_overlap"]["passes"] is True
    assert report["integrity"]["baseline_route_parity"]["passes"] is True
    observed = report["new_forward"]["observed_resolved_trades"]
    assert report["new_forward"]["enough_observations"] == (
        observed >= audit.MINIMUM_NEW_RESOLVED)
    assert report["runtime_change_allowed"] == report["new_forward"]["gate_passes"]
    assert report["ready_for_live_orders"] is False
