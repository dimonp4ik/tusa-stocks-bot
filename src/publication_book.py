"""Screen persistent spread before an entry can take a live signal slot."""
import time

from src.execution_guard import MAX_ENTRY_SPREAD, order_book_snapshot, validate_book_freshness


def publication_book(fetch_book, *, clock=time.time, sleep=time.sleep):
    """Return (confirmed wide, snapshot, receipt time) for public evidence.

    Missing or stale data remain inconclusive. The account-specific check
    still requires fresh, executable depth at the actual order size.
    """
    def sample():
        book = fetch_book()
        observed = clock()
        validate_book_freshness(book, observed * 1000)
        # Contract value cancels out of spread. Depth is scaled by the caller
        # once the contract spec is known.
        return book, order_book_snapshot(book, 1), observed

    try:
        first, snapshot, observed = sample()
        if snapshot.spread_bps <= MAX_ENTRY_SPREAD * 10_000:
            return False, snapshot, observed
        sleep(.35)
        second, snapshot, observed = sample()
    except (TypeError, ValueError, KeyError, IndexError, OverflowError):
        return False, None, None
    independent = float(second['ts']) > float(first['ts'])
    wide = snapshot.spread_bps > MAX_ENTRY_SPREAD * 10_000
    return independent and wide, snapshot, observed
