import stock_liquidity_sweep_walkforward_oct22_audit as audit


def test_resample_15m_preserves_market_bar_ohlcv_and_epoch_seconds():
    source = {
        "time": [1_800_000_000 + 300*i for i in range(6)],
        "open": [10, 11, 12, 13, 14, 15],
        "high": [11, 12, 13, 14, 15, 16],
        "low": [9, 10, 11, 12, 13, 14],
        "close": [10.5, 11.5, 12.5, 13.5, 14.5, 15.5],
        "volume": [1, 2, 3, 4, 5, 6],
    }
    result = audit.resample_15m(source)

    assert result["time"] == [1_800_000_000, 1_800_000_900]
    assert result["open"] == [10.0, 13.0]
    assert result["high"] == [13.0, 16.0]
    assert result["low"] == [9.0, 12.0]
    assert result["close"] == [12.5, 15.5]
    assert result["volume"] == [6.0, 15.0]
