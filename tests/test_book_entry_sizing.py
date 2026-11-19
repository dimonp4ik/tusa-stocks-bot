from unittest.mock import Mock

import pytest

from src.execution_guard import execution_quote_with_retry
from src.okx_trader import calc_contracts


def test_quantity_uses_new_book_each_retry_and_keeps_client_margin_cap():
    spec = dict(lotSz=1, minSz=1, ctVal=1)
    fetch = Mock(side_effect=[
        dict(ts=1000, asks=[[21, 5]], bids=[[20, 5]]),
        dict(ts=1001, asks=[[10, 2], [10.003, 5]], bids=[[9.998, 5]]),
    ])
    _, quote = execution_quote_with_retry(fetch, 'LONG', None, clock_ms=lambda: 1001,
        sleep=Mock(), size_for_price=lambda px: calc_contracts(3, 10, px, spec))
    assert quote.contracts == 2  # Third lot would require more than the $30 budget.
    assert quote.average == 10


def test_short_bracket_checks_first_level_as_well_as_worst_level():
    book = dict(ts=1000, bids=[[100.01, 1], [99.98, 1]], asks=[[100.02, 2]])
    _, quote = execution_quote_with_retry(lambda: book, 'SHORT', 2, clock_ms=lambda: 1001)
    assert quote.worst < 100.0 < quote.best
    with pytest.raises(ValueError, match='outside signal bracket'):
        quote.validate_bracket(100.0, 98.0)
