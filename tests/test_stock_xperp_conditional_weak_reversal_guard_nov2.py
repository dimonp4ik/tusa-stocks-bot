import json
from pathlib import Path

import pandas as pd

import stock_xperp_conditional_weak_reversal_guard_nov2_audit as audit


def test_guard_rejects_only_weak_reversal_against_long_basket_state():
    base = {
        "direction_n": [1, 1, 1],
        "signal_return": [-.0045, -.006, -.0045],
        "basket_trend_64": [-.02, -.02, .02],
    }
    frame = pd.DataFrame(base)
    assert audit.guard_mask(frame).tolist() == [False, True, True]


def test_report_keeps_challenger_out_of_runtime_until_new_forward():
    report = json.loads(Path(audit.OUT).read_text(encoding="utf-8"))
    assert report["runtime_change_allowed"] is False
    assert report["ready_for_live_orders"] is False
    assert report["integrity"]["raw_candle_overlap"]["passes"] is True
    assert report["integrity"]["baseline_route_parity"]["passes"] is True
