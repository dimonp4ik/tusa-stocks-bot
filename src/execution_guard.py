"""Executable order-book checks, independent of signal profitability estimates."""
from dataclasses import dataclass
import math
import time

MAX_ENTRY_SPREAD = .0005
MAX_BOOK_AGE_MS = 3000


class SpreadLimitError(ValueError):
    def __init__(self, measured, limit):
        super().__init__('spread exceeds limit')
        self.measured = measured
        self.limit = limit


def execution_quote_with_retry(fetch_book, direction, contracts, *, clock_ms,
                               sleep=time.sleep, attempts=3, retry_delay=.2,
                               deadline_ms=None):
    """Retry only a wide spread; each fresh book must pass every original gate."""
    if attempts not in (1, 2, 3):
        raise ValueError('invalid execution attempts')
    for attempt in range(attempts):
        if deadline_ms is not None and clock_ms() >= deadline_ms:
            raise ValueError('entry deadline elapsed')
        book = fetch_book()
        if not book:
            raise ValueError('order book unavailable')
        try:
            now = clock_ms()
            if deadline_ms is not None and now >= deadline_ms:
                raise ValueError('entry deadline elapsed')
            return book, execution_quote(book, direction, contracts, now)
        except SpreadLimitError:
            if attempt + 1 == attempts:
                raise
            # Separate checks enough to catch a temporarily wide market.
            sleep(retry_delay * (1 if attempt == 0 else 3))

@dataclass(frozen=True)
class ExecutionQuote:
    best: float
    average: float
    worst: float
    spread_fraction: float
    slippage_fraction: float


@dataclass(frozen=True)
class OrderBookSnapshot:
    spread_bps: float
    depth_usd: float
    imbalance: float


def order_book_snapshot(book, contract_value):
    """Summarise the already-fetched execution book for forward diagnostics.

    OKX X-Perp quantities are contracts, so visible USD depth is
    ``price * contracts * ctVal``. This performs no network I/O and stays
    separate from the execution gate: an audit write must never decide whether
    an otherwise valid market order is sent.
    """
    ct_val = float(contract_value)
    if not math.isfinite(ct_val) or ct_val <= 0:
        raise ValueError('invalid contract value')
    sides = {}
    for key in ('bids', 'asks'):
        levels = [(float(row[0]), float(row[1])) for row in book[key]]
        if not levels or any(
                not math.isfinite(price) or not math.isfinite(quantity)
                or price <= 0 or quantity <= 0
                for price, quantity in levels):
            raise ValueError('invalid depth')
        sides[key] = levels
    bid = sides['bids'][0][0]
    ask = sides['asks'][0][0]
    if ask <= bid:
        raise ValueError('crossed order book')
    midpoint = (ask + bid) / 2.0
    bid_usd = sum(price * quantity * ct_val
                  for price, quantity in sides['bids'])
    ask_usd = sum(price * quantity * ct_val
                  for price, quantity in sides['asks'])
    total = bid_usd + ask_usd
    if not math.isfinite(total) or total <= 0:
        raise ValueError('empty notional depth')
    return OrderBookSnapshot(
        spread_bps=(ask - bid) / midpoint * 10_000.0,
        depth_usd=total,
        imbalance=(bid_usd - ask_usd) / total,
    )


def validate_book_freshness(book, now_ms, max_age_ms=MAX_BOOK_AGE_MS):
    age = now_ms - float(book['ts'])
    if not math.isfinite(age) or age < -1000 or age > max_age_ms:
        raise ValueError('stale order book')


def execution_quote(book, direction, contracts, now_ms, *, max_age_ms=MAX_BOOK_AGE_MS,
                    max_spread=MAX_ENTRY_SPREAD, max_slippage=.0005):
    if direction not in ('LONG','SHORT'):raise ValueError('invalid direction')
    if not math.isfinite(contracts) or contracts<=0:raise ValueError('invalid size')
    validate_book_freshness(book, now_ms, max_age_ms)
    sides={}
    for key,descending in [('bids',True),('asks',False)]:
        levels=[(float(r[0]),float(r[1])) for r in book[key]]
        if not levels or any(not math.isfinite(p) or not math.isfinite(q) or p<=0 or q<=0 for p,q in levels):
            raise ValueError('invalid depth')
        prices=[p for p,q in levels]
        if prices!=sorted(prices,reverse=descending):raise ValueError('unordered depth')
        sides[key]=levels
    bid=sides['bids'][0][0];ask=sides['asks'][0][0]
    if ask<=bid:raise ValueError('crossed order book')
    spread=(ask-bid)/((ask+bid)/2)
    if spread>max_spread:raise SpreadLimitError(spread, max_spread)
    levels=sides['asks' if direction=='LONG' else 'bids'];best=levels[0][0]
    remaining=contracts;value=0.;worst=best
    for price,quantity in levels:
        take=min(remaining,quantity);value+=take*price;remaining-=take;worst=price
        if remaining<=contracts*1e-12:break
    if remaining>contracts*1e-12:raise ValueError('insufficient displayed depth')
    sign=1 if direction=='LONG' else -1
    slip=sign*(worst/best-1)
    if slip>max_slippage:raise ValueError('depth slippage exceeds limit')
    return ExecutionQuote(best,value/contracts,worst,spread,slip)
