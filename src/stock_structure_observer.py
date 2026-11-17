"""Prospective public-quote archive. Never opens or publishes trading signals."""
import collections
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import logging
import math
from pathlib import Path
import threading
import time

from src import stock_xperp_native_shadow as native
from src.structure_observation_store import ObservationStore,closed_frame,encoded

PROFILE='stock_swing2_original_target_quote_observer_v1'
FROZEN_AT=1794948873.  # 2026-11-17 20:54:33 UTC, before prospective collection.
NATIVE_SOURCE_SHA='bef374601c9599b9df506e3ac9abf4e05a9d4d80e66daf07625edc04eb524b0d'
log=logging.getLogger(__name__)


def observer_path(main_database):
    path=Path(main_database)
    if not path.is_absolute() or not path.is_file():raise ValueError('durable_database_required')
    return path.with_name('stock_structure_observer_v1.sqlite3')


def book_evidence(book,received_at):
    if not isinstance(book,dict):return dict(ok=False,rejection='missing_book')
    try:
        sides={key:[[float(r[0]),float(r[1])] for r in book[key][:5]] for key in ('bids','asks')}
        if any(not rows or any(not math.isfinite(p) or not math.isfinite(q) or min(p,q)<=0 for p,q in rows)
               for rows in sides.values()):raise ValueError('invalid_levels')
        for key,reverse in (('bids',True),('asks',False)):
            prices=[r[0] for r in sides[key]]
            if prices!=sorted(prices,reverse=reverse):raise ValueError('unordered_levels')
        ts=float(book['ts'])/1000
        if not math.isfinite(ts):raise ValueError('invalid_book_time')
        bid,ask=sides['bids'][0][0],sides['asks'][0][0]
        if ask<=bid:raise ValueError('crossed_book')
    except (KeyError,TypeError,ValueError,IndexError,OverflowError):
        return dict(ok=False,rejection='malformed_book')
    age=received_at-ts
    return dict(ok=-1<=age<=3,rejection=None if -1<=age<=3 else 'book_age',
        exchange_at=ts,age_seconds=age,bid=bid,ask=ask,spread_bps=(ask-bid)/((ask+bid)/2)*10000,**sides)


def signal_geometry(setup,frame):
    """Keep the frozen known-close levels; the observed entry quote is separate."""
    entry=float(setup['entry']);sign=1 if setup['direction']=='LONG' else -1
    if setup['direction'] not in ('LONG','SHORT'):raise ValueError('invalid_direction')
    atr=float(setup['atr14_fraction'])*entry
    if not math.isfinite(entry) or entry<=0 or not math.isfinite(atr) or atr<0:
        raise ValueError('invalid_geometry')
    extreme=min(frame['low'][-2:])-.25*atr if sign>0 else max(frame['high'][-2:])+.25*atr
    risk=max(.003,sign*(entry-extreme)/entry)
    target_fraction=float(setup['target_r'])*.01
    if not all(math.isfinite(x) for x in (entry,atr,risk,target_fraction)) or entry<=0 or atr<0:
        raise ValueError('invalid_geometry')
    if risk>.02:return None
    if not .02<=target_fraction<=.03:raise ValueError('target_protocol_changed')
    return dict(signal_entry=entry,stop=entry*(1-sign*risk),target=entry*(1+sign*target_fraction),
        signal_risk_fraction=risk,target_fraction=target_fraction,hold_seconds=64*900,
        side=setup['direction'])


class PublicSource:
    def __init__(self):
        self._pace_lock=threading.Lock();self._next_request=0.

    def _pace(self):
        # Research traffic has its own low rate; it never throttles trading calls.
        with self._pace_lock:
            delay=self._next_request-time.monotonic()
            if delay>0:time.sleep(delay)
            self._next_request=time.monotonic()+.15

    def inventory(self):
        from src.binance_client import get_xperp_instruments,_okx_get
        mapping=get_xperp_instruments()
        self._pace()
        rows=_okx_get('/api/v5/public/instruments',dict(instType='FUTURES',ruleType='xperp'),timeout=5).get('data',[])
        wanted=set(mapping.values());specs={}
        for row in rows:
            if row.get('instId') in wanted:
                specs[row['instId']]={k:row.get(k) for k in
                    ('instId','state','ctVal','ctValCcy','lotSz','minSz','tickSz','settleCcy')}
        return mapping,specs

    def candles(self,symbol):
        from src.binance_client import get_klines_xperp
        self._pace()
        return get_klines_xperp(symbol,limit=native.REQUIRED_HISTORY_BARS)

    def book(self,instrument):
        from src.binance_client import _okx_get
        self._pace()
        answer=_okx_get('/api/v5/market/books',dict(instId=instrument,sz=5),timeout=5)
        row=(answer.get('data') or [None])[0]
        # This single-instrument endpoint normally omits instId. Its request
        # identifies the contract; reject a conflicting echoed id if supplied.
        if row is not None and row.get('instId') not in (None,instrument):
            raise ValueError('book_instrument_mismatch')
        return row

    def candidates(self,frames,now,diagnostics):
        return native.native_shadow_setups(frames,now=now,diagnostics=diagnostics)


class StockStructureObserver:
    def __init__(self,path,*,source=None,clock=time.time,symbols=None,revision='unknown'):
        self.store=ObservationStore(path)
        self.source=source or PublicSource();self.clock=clock
        self.symbols=tuple(sorted(symbols if symbols is not None else native.AUDITED_SYMBOLS))
        self.revision=str(revision)[:64]

    def run(self):
        started=self.clock();bucket=int(started//900)*900
        if started<FROZEN_AT:return dict(status='before_freeze')
        try:
            self.store.initialize()
            self.store.prune(started)
            if not self.store.claim(PROFILE,bucket,started,self.revision):return dict(status='already_recorded')
        except Exception as exc:
            log.warning('Structure observer archive unavailable (%s)',type(exc).__name__)
            return dict(status='archive_unavailable',error_type=type(exc).__name__)
        summary=dict(expected=len(self.symbols),loaded=0,quotes=0,candidates=0,quoted_candidates=0)
        try:
            mapping,specs=self.source.inventory()
            instruments={s:mapping[s[:-4]] for s in self.symbols if s[:-4] in mapping}
            frames={};history_errors={}
            def fetch(symbol):
                if self.clock()-started>80:raise ValueError('history_deadline')
                return closed_frame(self.source.candles(symbol),bucket,native.REQUIRED_HISTORY_BARS)
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures={pool.submit(fetch,s):s for s in instruments}
                for future in as_completed(futures):
                    symbol=futures[future]
                    try:frames[symbol]=future.result()
                    except Exception as exc:history_errors[symbol]=type(exc).__name__
            references={s:self.store.put_frame(instruments[s],f) for s,f in frames.items()}
            decision_at=self.clock();diagnostics={}
            native_sha=hashlib.sha256(Path(native.__file__).read_text(encoding='utf-8').encode()).hexdigest()
            complete=len(instruments)==len(frames)==len(self.symbols)
            protocol_ok=native_sha==NATIVE_SOURCE_SHA
            forward_boundary=bucket>=math.ceil(FROZEN_AT/900)*900
            setups=(self.source.candidates(frames,decision_at,diagnostics)
                    if complete and protocol_ok and forward_boundary else [])
            candidates={s['symbol']:s for s in setups}
            summary.update(available=len(instruments),loaded=len(frames),decision_at=decision_at,
                native_source_sha256=native_sha,protocol_ok=protocol_ok,history_errors=history_errors,
                history_complete=complete,native_result=diagnostics.get('result'),raw_setups=len(setups),
                universe_sha256=hashlib.sha256(encoded(self.symbols).encode()).hexdigest(),
                basis='public quote evidence, no orders, no portfolio admission, no account PnL')
            def spec_usable(instrument):
                try:
                    s=specs[instrument]
                    return s.get('state')=='live' and all(math.isfinite(float(s[k])) and float(s[k])>0
                        for k in ('ctVal','lotSz','minSz','tickSz'))
                except (KeyError,TypeError,ValueError):return False
            summary['specs_available']=sum(spec_usable(i) for i in instruments.values())
            errors=collections.Counter()
            def quote(symbol):
                requested=self.clock()
                if requested-started>150:return requested,requested,None,'quote_deadline'
                try:
                    raw=self.source.book(instruments[symbol])
                    return requested,self.clock(),raw,None
                except Exception as exc:return requested,self.clock(),None,type(exc).__name__
            # Keep every receipt time immediately after its own response, not
            # after waiting for the other instruments in this scan.
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures={pool.submit(quote,s):s for s in instruments}
                for future in as_completed(futures):
                    symbol=futures[future];requested,received,raw,error=future.result()
                    book=dict(ok=False,rejection=error) if error else book_evidence(raw,received)
                    geometry_error=None
                    try:geometry=signal_geometry(candidates[symbol],frames[symbol]) if symbol in candidates else None
                    except (KeyError,TypeError,ValueError,ZeroDivisionError) as exc:
                        geometry=None;geometry_error=type(exc).__name__
                    payload=dict(requested_at=requested,received_at=received,book=book,
                        spec=specs.get(instruments[symbol]),frame=references.get(symbol),
                        native_candidate=candidates.get(symbol),geometry=geometry,decision_at=decision_at,
                        geometry_error=geometry_error,
                        market_reference_only=True,private_fill_unverified=True)
                    if geometry and book['ok']:
                        px=book['ask'] if geometry['side']=='LONG' else book['bid']
                        sign=1 if geometry['side']=='LONG' else -1
                        risk=sign*(px-geometry['stop'])/px
                        payload.update(quote_reference_price=px,quote_risk_fraction=risk,
                            quote_between_levels=sign*(px-geometry['stop'])>0 and sign*(geometry['target']-px)>0,
                            quote_within_signal_age=0<=received-bucket<=native.MAX_SIGNAL_AGE_SECONDS)
                    self.store.put_observation(PROFILE,bucket,symbol,instruments[symbol],received,
                        book['ok'],geometry is not None,payload)
                    summary['quotes']+=int(book['ok'])
                    summary['candidates']+=int(geometry is not None)
                    summary['quoted_candidates']+=int(geometry is not None and book['ok'])
                    if not book['ok']:errors[book['rejection']]+=1
            summary['quote_errors']=dict(errors)
            status='complete' if (complete and summary['quotes']==len(self.symbols)
                and summary['specs_available']==len(self.symbols) and protocol_ok) else 'partial'
            self.store.finish(PROFILE,bucket,self.clock(),status,summary)
            log.info('Structure observer: status=%s history=%s/%s quotes=%s candidates=%s',
                     status,len(frames),len(self.symbols),summary['quotes'],summary['candidates'])
            return dict(status=status,**summary)
        except Exception as exc:
            summary['error_type']=type(exc).__name__
            self.store.finish(PROFILE,bucket,self.clock(),'failed',summary)
            log.warning('Structure observer deferred (%s)',type(exc).__name__)
            return dict(status='failed',**summary)
