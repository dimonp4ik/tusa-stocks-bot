import stock_xperp_native_growth_forward_monitor as monitor
import pytest


def candles(open_, high, low, close):
    return {
        "time": [0, 900, 1800, 2700],
        "open": open_, "high": high, "low": low, "close": close,
    }


def row(direction=1):
    return {"symbol": "A", "entry_index": 1, "entry_time": 900,
            "direction_n": direction}


def test_resolver_is_stop_first_on_same_bar_ambiguity():
    candle = candles([100, 100, 100, 100], [100, 102, 100, 100],
                     [100, 99, 100, 100], [100, 100, 100, 100])
    result = monitor.resolve_row(row(), candle, 2, 1, 3)
    assert result["outcome"] == "SL"
    assert result["signed_return"] == -0.01


def test_resolver_keeps_incomplete_horizon_pending():
    candle = candles([100] * 4, [100.5] * 4, [99.5] * 4, [100] * 4)
    result = monitor.resolve_row(row(), candle, 2, 1, 64)
    assert result["outcome"] == "PENDING"
    assert result["signed_return"] is None


def test_resolver_times_out_only_after_full_hold():
    candle = {
        "time": [index * 900 for index in range(5)],
        "open": [100] * 5, "high": [100.5] * 5,
        "low": [99.5] * 5, "close": [100, 100, 100, 100, 101],
    }
    result = monitor.resolve_row(row(), candle, 2, 1, 4)
    assert result["outcome"] == "TIME"
    assert result["signed_return"] == pytest.approx(0.01)


def test_overlap_verifier_counts_new_bars_and_rejects_revision():
    frozen = candles([100] * 4, [101] * 4, [99] * 4, [100] * 4)
    current = {
        key: list(value) + ([3600] if key == "time" else [value[-1]])
        for key, value in frozen.items()
    }
    current["volume"] = [1] * 5
    frozen["volume"] = [1] * 4
    stats = monitor.overlap_stats(frozen, current)
    assert stats["overlap_bars"] == 4
    assert stats["new_bars_after_frozen_end"] == 1
    current["close"][1] = 101
    with pytest.raises(ValueError, match="overlap mismatch"):
        monitor.overlap_stats(frozen, current)
