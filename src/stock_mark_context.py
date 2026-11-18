"""Public mark/trade/book context for research, with no strategy or orders."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import logging
import math
import time

from src.stock_structure_observer import PublicSource,book_evidence
from src.structure_observation_store import ObservationStore,closed_frame,encoded

PROFILE='stock_liquid_mark_context_v1'
SYMBOLS=('AMZNUSDT','GOOGLUSDT','INTCUSDT','MSFTUSDT','MSTRUSDT','NVDAUSDT',
         'QQQUSDT','SOXLUSDT','SPCXUSDT','SPYUSDT','TSLAUSDT')
HISTORY_BARS=163
log=logging.getLogger(__name__)


def mark_frame(rows,bucket,minimum=HISTORY_BARS):
    if not isinstance(rows,list):raise ValueError('missing_mark_history')
    unique={}
    for row in rows:
        if not isinstance(row,(list,tuple)) or len(row)!=6 or str(row[5]) not in ('0','1'):
            raise ValueError('invalid_mark_candle_schema')
        stamp=int(row[0])
        if stamp%900000:raise ValueError('unaligned_mark_candle')
        if str(row[5])=='0':continue
        if stamp//1000>=bucket:raise ValueError('future_confirmed_mark_bar')
        unique.setdefault(stamp,row)
    selected=[unique[t] for t in sorted(unique)][-minimum:]
    frame=dict(time=[int(r[0])//1000 for r in selected],confirmed=[1]*len(selected),
        **{key:[float(r[i]) for r in selected] for i,key in enumerate(('open','high','low','close'),1)})
    return closed_frame(frame,bucket,minimum,kind='mark')


class MarkContextSource(PublicSource):
    def mark_candles(self,instrument,bucket):
        from src.binance_client import _okx_get
        rows=[];after=None;wanted=HISTORY_BARS+2
        while len(rows)<wanted:
            params=dict(instId=instrument,bar='15m',limit=min(100,wanted-len(rows)))
            if after is not None:params['after']=after
            self._pace()
            page=_okx_get('/api/v5/market/mark-price-candles',params,timeout=5).get('data',[])
            if not page:break
            rows.extend(page)
            oldest=min(int(r[0]) for r in page)
            if after is not None and oldest>=int(after):raise ValueError('mark_pagination_not_advancing')
            after=str(oldest)
            if len(page)<params['limit']:break
        return mark_frame(rows,bucket)


class StockMarkContextObserver:
    def __init__(self,path,*,source=None,clock=time.time,symbols=SYMBOLS,revision='unknown'):
        self.store=ObservationStore(path);self.source=source or MarkContextSource()
        self.clock=clock;self.symbols=tuple(sorted(symbols));self.revision=str(revision)[:64]

    def run(self):
        started=self.clock();bucket=int(started//900)*900
        try:
            self.store.initialize();self.store.prune(started)
            if not self.store.claim(PROFILE,bucket,started,self.revision):return dict(status='already_recorded')
        except Exception as exc:
            log.warning('Mark context archive unavailable (%s)',type(exc).__name__)
            return dict(status='archive_unavailable',error_type=type(exc).__name__)
        summary=dict(expected=len(self.symbols),mark_loaded=0,trade_loaded=0,quotes=0,
            basis='mark features, trade trigger data, public quotes; no portfolio or PnL')
        try:
            mapping,specs=self.source.inventory()
            instruments={s:mapping[s[:-4]] for s in self.symbols if s[:-4] in mapping}
            summary['available']=len(instruments)
            def spec_available(instrument):
                try:
                    s=specs[instrument]
                    return s.get('state')=='live' and all(math.isfinite(float(s[k])) and float(s[k])>0
                        for k in ('ctVal','lotSz','minSz','tickSz'))
                except (KeyError,TypeError,ValueError):return False
            summary['specs_available']=sum(spec_available(i) for i in instruments.values())
            summary['universe_sha256']=hashlib.sha256(encoded(self.symbols).encode()).hexdigest()
            frames={};errors={}
            def fetch(symbol):
                result={}
                for kind in ('mark','trade'):
                    if self.clock()-started>100:
                        result[kind+'_error']='history_deadline';continue
                    try:
                        result[kind]=(self.source.mark_candles(instruments[symbol],bucket) if kind=='mark'
                            else closed_frame(self.source.candles(symbol),bucket,HISTORY_BARS))
                    except Exception as exc:result[kind+'_error']=type(exc).__name__
                return result
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures={pool.submit(fetch,s):s for s in instruments}
                for future in as_completed(futures):
                    symbol=futures[future];frames[symbol]=future.result()
            references={}
            for symbol,pair in frames.items():
                references[symbol]={}
                for kind in ('mark','trade'):
                    if kind in pair:
                        references[symbol][kind]=self.store.put_frame(instruments[symbol],pair[kind],kind=kind)
                        summary[kind+'_loaded']+=1
                    else:errors[symbol+':'+kind]=pair[kind+'_error']
            summary['history_errors']=errors;summary['decision_at']=self.clock()
            def fetch_book(symbol):
                requested=self.clock()
                if requested-started>150:return requested,requested,None,'quote_deadline'
                try:
                    raw=self.source.book(instruments[symbol]);return requested,self.clock(),raw,None
                except Exception as exc:return requested,self.clock(),None,type(exc).__name__
            quote_errors={}
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures={pool.submit(fetch_book,s):s for s in instruments}
                for future in as_completed(futures):
                    symbol=futures[future];requested,received,raw,error=future.result()
                    book=dict(ok=False,rejection=error) if error else book_evidence(raw,received)
                    payload=dict(requested_at=requested,received_at=received,book=book,
                        frame=references[symbol].get('mark'),trade_frame=references[symbol].get('trade'),
                        series_kind='mark',mark_volume_available=False,spec=specs.get(instruments[symbol]),
                        market_reference_only=True,private_fill_unverified=True)
                    if 'mark' in frames[symbol] and 'trade' in frames[symbol]:
                        mark=frames[symbol]['mark']['close'][-1];trade=frames[symbol]['trade']['close'][-1]
                        payload.update(mark_close=mark,trade_close=trade,trade_to_mark_fraction=trade/mark-1)
                    self.store.put_observation(PROFILE,bucket,symbol,instruments[symbol],received,
                        book['ok'],False,payload)
                    summary['quotes']+=int(book['ok'])
                    if not book['ok']:quote_errors[symbol]=book['rejection']
            summary['quote_errors']=quote_errors
            status=('complete' if len(instruments)==summary['mark_loaded']==summary['trade_loaded']
                ==summary['quotes']==summary['specs_available']==len(self.symbols) else 'partial')
            self.store.finish(PROFILE,bucket,self.clock(),status,summary)
            log.info('Mark context: status=%s mark=%s/%s trade=%s quotes=%s',status,
                summary['mark_loaded'],len(self.symbols),summary['trade_loaded'],summary['quotes'])
            return dict(status=status,**summary)
        except Exception as exc:
            summary['error_type']=type(exc).__name__
            self.store.finish(PROFILE,bucket,self.clock(),'failed',summary)
            log.warning('Mark context collection deferred (%s)',type(exc).__name__)
            return dict(status='failed',**summary)
