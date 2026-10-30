import json

import stock_xperp_short_exhaustion_target_oct30_audit as audit
from src.stock_xperp_native_shadow import TARGET_FRACTION


def test_selection_score_prioritizes_total_return_with_drawdown_penalty():
    values = {
        "selection_july": {"margin_3pct": {
            "return_pct": 35, "dd_pct": 4, "trades_per_active_day": 5}},
        "selection_august": {"margin_3pct": {
            "return_pct": 25, "dd_pct": 3, "trades_per_active_day": 5}},
    }
    assert audit.selection_score(values) == 58.0


def test_runtime_uses_audited_shadow_target():
    value = json.loads(audit.OUT.read_text(encoding="utf-8"))
    assert value["status"] == "SHADOW_ONLY_FORWARD_EXPERIMENT"
    assert value["ready_for_live_orders"] is False
    assert value["orders_possible"] is False
    assert value["selected"] == {"hold_bars": 64, "target_pct": 2.5}
    assert value["runtime_target_parity"] is True
    assert TARGET_FRACTION == .025
    assert value["test_pass"] is True
    assert value["confirmation_pass"] is True
    assert value["forward_pass"] is True
