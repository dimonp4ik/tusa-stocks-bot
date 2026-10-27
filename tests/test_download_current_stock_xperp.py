from download_current_stock_xperp import equity_mapping, snapshot_anchor_ms


def instrument(base="AAPL", *, category="3", state="live",
               listed="1000", rule="xperp", kind="FUTURES"):
    return {
        "ctValCcy": base, "instId": f"{base}-USD_UM_XPERP-311003",
        "instCategory": category, "state": state, "listTime": listed,
        "settleCcy": "USD", "instType": kind, "ruleType": rule,
    }


def test_equity_mapping_uses_current_equity_xperp_futures():
    values = [
        instrument(), instrument("BTC", category="1"),
        instrument("OLD", state="suspend"), instrument("REG", rule="normal"),
    ]
    assert equity_mapping(values) == {"AAPLUSDT": "AAPL-USD_UM_XPERP-311003"}


def test_equity_mapping_can_freeze_listing_cutoff():
    values = [instrument("OLD", listed="1000"), instrument("NEW", listed="2000")]
    assert equity_mapping(values, 1500) == {"OLDUSDT": "OLD-USD_UM_XPERP-311003"}


def test_snapshot_anchor_is_current_candle_open():
    assert snapshot_anchor_ms(901) == 900_000
    assert snapshot_anchor_ms(1799.999) == 900_000
    assert snapshot_anchor_ms(1801) == 1800_000
