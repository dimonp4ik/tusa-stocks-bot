import json

import stock_xperp_regime_target_router_oct30_audit as audit
from src.stock_xperp_native_shadow import routed_target_fraction


def test_runtime_target_router_precedence():
    assert routed_target_fraction(-.05, 8 * 3600) == .02
    assert routed_target_fraction(-.02, 8 * 3600) == .03
    assert routed_target_fraction(-.02, 14 * 3600) == .025


def test_report_matches_runtime_and_passes_sequential_gates():
    value = json.loads(audit.OUT.read_text(encoding="utf-8"))
    assert value["status"] == "SHADOW_ONLY_FORWARD_EXPERIMENT"
    assert value["ready_for_live_orders"] is False
    assert value["orders_possible"] is False
    assert value["runtime_parity"] is True
    assert value["selected"]["threshold"] == -.04
    assert value["selected"]["expansion"] == [6, 12]
    assert value["test_pass"] is True
    assert value["confirmation_pass"] is True
    assert value["forward_pass"] is True
