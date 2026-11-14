from unittest.mock import Mock
import pytest
from src.execution_guard import execution_quote_with_retry, SpreadLimitError


def book(ask):
    return dict(ts=10000, bids=[[99.99, 100]], asks=[[ask, 100]])


def test_transient_spread_reloads_and_uses_only_passing_book():
    fetch = Mock(side_effect=[book(101), book(100.01)])
    pause = Mock()
    b, quote = execution_quote_with_retry(fetch, 'LONG', 1, clock_ms=lambda: 10001, sleep=pause)
    assert quote.best == 100.01
    assert fetch.call_count == 2
    pause.assert_called_once_with(.2)


def test_persistent_spread_stops_after_three_checks_and_exposes_numbers():
    fetch = Mock(return_value=book(101))
    with pytest.raises(SpreadLimitError) as error:
        execution_quote_with_retry(fetch, 'SHORT', 1, clock_ms=lambda: 10001, sleep=Mock())
    assert fetch.call_count == 3
    assert error.value.measured > error.value.limit


def test_retry_does_not_bypass_freshness_or_depth():
    stale = book(100.01)
    stale['ts'] = 0
    for second, match in ((stale, 'stale'), (book(100.01), 'insufficient')):
        fetch = Mock(side_effect=[book(101), second])
        with pytest.raises(ValueError, match=match):
            execution_quote_with_retry(fetch, 'LONG', 101, clock_ms=lambda: 10001, sleep=Mock())
        assert fetch.call_count == 2
