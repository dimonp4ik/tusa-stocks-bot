"""Executable order-book checks, independent of signal profitability estimates."""
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class ExecutionQuote:
    best: float
    average: float
    worst: float
    spread_fraction: float
    slippage_fraction: float


def execution_quote(book, direction, contracts, now_ms, *, max_age_ms=3000,
                    max_spread=.0005, max_slippage=.0005):
    if direction not in ('LONG','SHORT'):raise ValueError('invalid direction')
    if not math.isfinite(contracts) or contracts<=0:raise ValueError('invalid size')
    age=now_ms-float(book['ts'])
    if not math.isfinite(age) or age < -1000 or age>max_age_ms:raise ValueError('stale order book')
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
    if spread>max_spread:raise ValueError('spread exceeds limit')
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
