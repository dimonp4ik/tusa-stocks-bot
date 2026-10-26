import json
from pathlib import Path

from stock_xperp_forward_loss_attribution_oct22_audit import attribute_stop


def test_post_stop_path_excludes_ambiguous_stop_bar_and_tracks_recovery():
    row = {
        "entry_index": 0, "hold_bars": 4, "direction": "LONG",
        "entry": 100.0, "stop_pct": 1.0, "target_pct": 2.0,
        "backing_level_distance": .005,
    }
    candles = {
        "time": [1, 2, 3, 4],
        "open": [100, 99, 98, 100],
        "high": [100.5, 103, 99, 103],
        "low": [99.5, 98.5, 97.5, 99],
        "close": [100, 99, 98, 102],
    }

    found = attribute_stop(row, candles)

    assert found["stop_bar_time"] == 2
    assert found["original_target_hit_after_stop"]
    assert found["entry_reclaimed_after_stop"]
    assert found["another_1r_adverse_after_stop"]
    assert found["backing_inside_stop_distance"]


def test_short_stop_path_uses_directional_levels():
    row = {
        "entry_index": 0, "hold_bars": 3, "direction": "SHORT",
        "entry": 100.0, "stop_pct": 1.0, "target_pct": 2.0,
        "backing_level_distance": .02,
    }
    candles = {
        "time": [1, 2, 3],
        "open": [100, 101, 102],
        "high": [100.5, 101.5, 103],
        "low": [99.5, 100, 97.5],
        "close": [100, 101, 98],
    }

    found = attribute_stop(row, candles)

    assert found["stop_bar_time"] == 2
    assert found["original_target_hit_after_stop"]
    assert found["entry_reclaimed_after_stop"]
    assert found["another_1r_adverse_after_stop"]
    assert not found["backing_inside_stop_distance"]


def test_oct26_refresh_has_material_sample_and_rejects_blind_stop_widening():
    report = json.loads(Path(
        "reports/audit_2026_09_08/stock_xperp_forward_loss_attribution_oct26.json"
    ).read_text(encoding="utf-8"))
    summary = report["summary"]
    assert summary["stops"] == 42
    assert summary["entry_reclaimed_after_stop_pct"] > 60
    assert summary["another_1r_adverse_after_stop_pct"] > 60
    assert summary["original_target_hit_after_stop_pct"] < 40
    assert summary["mean_horizon_signed_return_pct"] < 0
    assert not report["ready_for_live_orders"]
