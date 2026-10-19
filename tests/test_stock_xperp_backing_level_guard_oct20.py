import json
from pathlib import Path

import pandas as pd

import stock_xperp_backing_level_guard_oct20_audit as audit


def test_backing_distance_uses_only_bars_before_entry():
    frame = pd.DataFrame({
        "symbol": ["LONG", "SHORT"],
        "entry_index": [2, 2],
        "direction_n": [1, -1],
    })
    candles = {
        "LONG": {"open": [100, 100, 98], "low": [99, 97, 1],
                 "high": [101, 102, 999]},
        "SHORT": {"open": [100, 100, 103], "low": [99, 97, 1],
                  "high": [101, 102, 999]},
    }
    found = audit.add_backing_distance(frame, candles, 2)
    assert found.backing_level_distance.iloc[0] > 0
    assert found.backing_level_distance.iloc[1] < 0


def test_guard_keeps_zero_distance():
    frame = pd.DataFrame({"backing_level_distance": [-.1, 0, .1]})
    assert audit.guarded(frame).backing_level_distance.tolist() == [0, .1]


def test_report_sources_match_and_live_is_false():
    report = json.loads(Path(audit.OUT).read_text(encoding="utf-8"))
    assert report["ready_for_live_orders"] is False
    assert report["selected_lookback_bars"] == 32
    for source, expected in report["sources"].items():
        assert audit.digest(Path(source)) == expected
