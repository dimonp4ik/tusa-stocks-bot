import pytest
from stock_prospective_quote_quality_nov18 import first_book_check,quote_geometry


def observation(spread):
    return dict(quote_ok=True,payload=dict(received_at=100.,spec=dict(ctVal='1',lotSz='1',minSz='1'),
        book=dict(ok=True,exchange_at=99.9,bid=9.99,ask=9.99+spread,
            bids=[[9.99,100]],asks=[[9.99+spread,100]],spread_bps=spread/10*10000)))


def test_first_wide_book_does_not_invent_the_two_following_retry_books_or_a_fill():
    result=first_book_check(observation(.1),direction='SHORT')
    assert result['status']=='first_snapshot_wide_spread'
    assert result['further_two_retry_books_unobserved'] and result['final_execution_unverified']


def test_fresh_narrow_book_passes_only_the_public_first_snapshot_checks():
    result=first_book_check(observation(.001),direction='LONG')
    assert result['status']=='first_snapshot_pass' and result['final_execution_unverified']


def test_observed_qcom_displacement_cannot_preserve_the_signal_close_risk_reward():
    g=dict(direction_n=-1,stop=169.87811,target=165.13575,risk_fraction=.003,target_fraction=.025)
    actual=quote_geometry(g,167.19)
    assert actual['signal_rr']==pytest.approx(8.3333333333)
    assert actual['quoted_risk_fraction']==pytest.approx(.01607817453)
    assert actual['quoted_rr']==pytest.approx(.7641986377)
