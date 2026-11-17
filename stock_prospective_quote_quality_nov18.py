"""Inspect archived first books with production checks; no retry/fill claims."""
import hashlib
import json
import math
from pathlib import Path

from src.execution_guard import execution_quote,SpreadLimitError
from src.okx_trader import calc_contracts

FOLDER=Path('reports/structure_observer_prospective_nov18')


def quote_geometry(geometry,price):
    sign=geometry['direction_n']
    if sign not in (-1,1) or not math.isfinite(price) or price<=0:raise ValueError('invalid quote geometry')
    risk=sign*(price-geometry['stop'])/price
    reward=sign*(geometry['target']-price)/price
    return dict(signal_rr=geometry['target_fraction']/geometry['risk_fraction'],
        quoted_risk_fraction=risk,quoted_reward_fraction=reward,quoted_rr=reward/risk if risk>0 else None)


def first_book_check(row,*,direction,capital=130.,margin_fraction=.03):
    p=row['payload'];book=p['book'];spec=p.get('spec')
    if not row['quote_ok'] or not book.get('ok'):
        return dict(status='unknown_book',final_execution_unverified=True)
    if not spec:return dict(status='unknown_specification',final_execution_unverified=True)
    px=float(book['ask'] if direction=='LONG' else book['bid'])
    contracts=calc_contracts(capital*margin_fraction,10,px,spec)
    result=dict(reference_price=px,hypothetical_contracts=contracts,spread_bps=book['spread_bps'],
                final_execution_unverified=True)
    if contracts<=0:return dict(result,status='below_minimum_size')
    raw=dict(bids=book['bids'],asks=book['asks'],ts=book['exchange_at']*1000)
    try:
        quote=execution_quote(raw,direction,contracts,p['received_at']*1000)
        result.update(status='first_snapshot_pass',average=quote.average,worst=quote.worst)
    except SpreadLimitError:
        result.update(status='first_snapshot_wide_spread',further_two_retry_books_unobserved=True)
    except ValueError as exc:
        result.update(status='first_snapshot_refused',reason=str(exc))
    return result


def main():
    watch_path=FOLDER/'frozen_hypothesis_observations.json'
    watch=json.loads(watch_path.read_text(encoding='utf-8'))
    scans={};sources={};snapshots=[]
    for path in sorted(FOLDER.glob('bucket_*.json')):
        raw=path.read_bytes();scan=json.loads(raw);scans[scan['bucket']]=scan
        sources[str(path)]=hashlib.sha256(raw).hexdigest()
        if scan['status']!='complete':continue
        checks=[]
        for row in scan['observations']:
            for side in ('LONG','SHORT'):
                checks.append(dict(symbol=row['symbol'],direction=side,**first_book_check(row,direction=side)))
        snapshots.append(dict(bucket=scan['bucket'],checks=checks,
            first_snapshot_pass=sum(r['status']=='first_snapshot_pass' for r in checks),
            wide_spread=sum(r['status']=='first_snapshot_wide_spread' for r in checks)))
    candidates=[]
    for window in watch['rows']:
        if window['status']!='observed':continue
        by_symbol={r['symbol']:r for r in scans[window['bucket']]['observations']}
        for name,rows in window['strategies'].items():
            for r in rows:
                quote=r['reference_quote'];g=r['geometry'];sign=g['direction_n']
                result=dict(bucket=window['bucket'],hypothesis=name,symbol=r['symbol'],direction=r['direction'],
                    signal_entry=g['entry'],stop=g['stop'],target=g['target'],quote_reference=quote,
                    first_book=first_book_check(by_symbol[r['symbol']],direction=r['direction']))
                if quote is not None:
                    result.update(quote_geometry(g,quote))
                candidates.append(result)
    report=dict(status='FIRST_PROSPECTIVE_BOOK_DIAGNOSTIC_NOT_EXECUTION_OR_PNL',objective_verified=False,
        live_activation_supported=False,hypothetical_capital=130.,margin_fraction=.03,leverage=10,
        snapshots=snapshots,candidates=candidates,sources=sources,
        limitations=['First archived public book only; real execution retries up to three times for spread.',
            'Only five archived levels; missing deeper liquidity is unknown, not proof of a private rejection.',
            'Hypothetical size uses quoted best and $130 reference account, not an actual client balance or ticker.',
            'No account capacity, readiness, actual order/fill, exit or PnL is established.'])
    for p in (watch_path,Path(__file__),Path('src/execution_guard.py'),Path('src/okx_trader.py')):
        report['sources'][str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    (FOLDER/'quote_quality_diagnostic.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(snapshots=[{k:v for k,v in r.items() if k!='checks'} for r in snapshots],
        candidates=candidates)),flush=True)


if __name__=='__main__':main()
