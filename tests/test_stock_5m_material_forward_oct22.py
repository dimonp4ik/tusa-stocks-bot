import stock_5m_material_forward_oct13_audit as audit


def test_append_parity_counts_new_bars_without_revisions(monkeypatch):
    monkeypatch.setattr(audit.audit.runtime, "AUDITED_SYMBOLS", {"AAA"})
    old = {"AAA": {"time": [1, 2], "open": [1, 2], "high": [1, 2],
                   "low": [1, 2], "close": [1, 2], "volume": [1, 2]}}
    new = {"AAA": {"time": [1, 2, 3], "open": [1, 2, 3],
                   "high": [1, 2, 3], "low": [1, 2, 3],
                   "close": [1, 2, 3], "volume": [1, 2, 3]}}

    result = audit.append_parity(old, new)["AAA"]

    assert result["overlap_bars_verified"] == 2
    assert result["new_bars_after_previous"] == 1
    assert not any(result["field_mismatches"].values())
