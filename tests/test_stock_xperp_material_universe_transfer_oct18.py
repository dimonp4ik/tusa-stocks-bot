import stock_cash_xperp_leadlag_oct6_audit as lead
import stock_xperp_material_universe_transfer_oct18_audit as audit


def test_expansion_is_disjoint_from_fitted_universe():
    assert set(audit.EXPANSION_SYMBOLS).isdisjoint(lead.SYMBOLS)


def test_ambiguous_crypto_stock_tickers_are_excluded():
    assert set(audit.EXPANSION_SYMBOLS).isdisjoint(
        audit.AMBIGUOUS_TICKER_COLLISIONS)


def test_expansion_contains_only_explicit_verified_symbols():
    assert len(audit.EXPANSION_SYMBOLS) == 8
    assert len(set(audit.EXPANSION_SYMBOLS)) == len(audit.EXPANSION_SYMBOLS)
