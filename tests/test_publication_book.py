from unittest.mock import Mock, patch

import pytest

import main
from src.execution_guard import execution_quote_with_retry
from src.publication_book import publication_book


def book(spread_bps, stamp=100_000):
    offset = spread_bps / 200
    return dict(ts=stamp, asks=[[100 + offset, 10]], bids=[[100 - offset, 10]])


@pytest.mark.parametrize('second,blocked', [
    (book(80, 100_100), True), (book(3, 100_100), False),
    (book(80), False), (book(80, 10), False), (None, False),
])
def test_only_two_distinct_fresh_wide_books_block_publication(second, blocked):
    fetch = Mock(side_effect=[book(80), second])
    wait = Mock()
    assert publication_book(fetch, clock=lambda: 100.2, sleep=wait)[0] is blocked
    wait.assert_called_once_with(.35)


def test_execution_rechecks_are_staggered_and_cannot_run_past_the_deadline():
    fetch = Mock(side_effect=[book(80), book(80, 100_200), book(3, 100_800)])
    now = 100_000
    elapsed = []

    def sleep(seconds):
        nonlocal now
        elapsed.append(seconds)
        now += seconds * 1000

    with pytest.raises(ValueError, match='entry deadline elapsed'):
        execution_quote_with_retry(fetch, 'LONG', 1, clock_ms=lambda: now, sleep=sleep,
                                   deadline_ms=100_500)
    assert fetch.call_count == 2
    assert elapsed == pytest.approx([.2, .6])


def test_wide_candidate_does_not_take_the_remaining_signal_slot():
    stock = hasattr(main, '_publish_stock_venue_candidates')
    publish = main._publish_stock_venue_candidates if stock else main._publish_venue_candidates
    source = 'stock_xperp_native_shadow' if stock else 'venue_shadow_profit_first_2r'
    candidates = [(dict(symbol=f'{s}USDT', direction='LONG', source=source,
                        _shadow_only=False, strategy_entry_bar_ts=980.), i)
                  for i, s in enumerate(('BILL', 'LINK'), 1)]
    books = [book(80, 999_100), book(79, 999_200), book(2, 999_300)]
    with patch.object(main.time, 'time', return_value=1000.), \
            patch('src.publication_book.time.sleep'), \
            patch.object(main, 'get_recent_signals', return_value=[]), \
            patch.object(main, 'send_signal', return_value=True) as send, \
            patch.object(main, 'mark_setup_blocked') as blocked, \
            patch.object(main, 'mark_setup_sent') as sent, \
            patch.object(main, '_cache_signal'), \
            patch.object(main._okx_trade, 'get_order_book', side_effect=books):
        if stock:
            with patch.object(main, 'is_earnings_blackout', return_value=(False, '')):
                result = publish(candidates, instruments={'BILL': 'BILL-X', 'LINK': 'LINK-X'})
        else:
            result = publish(candidates, instruments={'BILL': 'BILL-X', 'LINK': 'LINK-X'})
    assert result == 1
    blocked.assert_called_once_with(1, 'spread_at_publish')
    sent.assert_called_once_with(2)
    assert send.call_args.args[0]['symbol'] == 'LINKUSDT'
