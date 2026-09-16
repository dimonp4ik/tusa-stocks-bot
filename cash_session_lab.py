"""Fixed US cash-session entry hypotheses on 2026 stock-linked swaps."""
import json,hashlib
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from filter_lab import load,gate,metrics,timestamp
US_BASES=set('AAPL AMZN GOOGL META MSFT NVDA TSLA INTC MRVL MSTR MU SNDK SOXL QQQ SPY CRCL'.split())
HOLIDAYS=set('2026-01-01 2026-01-19 2026-02-16 2026-04-03 2026-05-25 2026-06-19 2026-07-03 2026-09-07 2026-11-26 2026-12-25'.split())
EARLY=set('2026-11-27 2026-12-24'.split())
NY=ZoneInfo('America/New_York')
def session(ts):
 d=datetime.fromtimestamp(ts,NY)
 if d.year!=2026:raise ValueError('Calendar only verified for2026')
 if d.weekday()>=5 or d.date().isoformat() in HOLIDAYS:return 'closed'
 minute=d.hour*60+d.minute;end=780 if d.date().isoformat() in EARLY else 960
 if not 570<=minute<end:return 'outside'
 if minute<630:return 'first_hour'
 if minute>=end-60:return 'last_hour'
 return 'middle'
def main():
 folder=Path('reports/audit_2026_09_08');p=folder/'all_structure_timing_raw.csv';rows=load(p)
 rows=[r for r in rows if r['symbol'].removesuffix('USDT') in US_BASES]
 rules={'all':lambda s:True,'core':lambda s:s in ('first_hour','middle','last_hour'),
        'first_hour':lambda s:s=='first_hour','middle':lambda s:s=='middle','last_hour':lambda s:s=='last_hour'}
 out={'status':'RESEARCH_ONLY','raw_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
 'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 'calendar_source':'https://www.nyse.com/trade/hours-calendars',
 'universe':sorted(US_BASES),'results':[],
 'limitations':['Entry-time filter only: existing exits may occur outside cash session.',
 'Stock-linked SWAP candles, not underlying equity or X-Perp executions.',
 'No new threshold optimized; already inspected2026 history; funding excluded.']}
 for name,accept in rules.items():
  accepted=gate([r for r in rows if accept(session(r['entry_time']))],4)
  early=timestamp('2026-06-01');late=timestamp('2026-07-01')
  result={'rule':name,'all':metrics(accepted),
          'train':metrics([r for r in accepted if r['exit_time']<early]),
          'validation':metrics([r for r in accepted if early<=r['entry_time'] and r['exit_time']<late]),
          'test':metrics([r for r in accepted if r['entry_time']>=late])}
  out['results'].append(result);print(json.dumps(result))
 (folder/'cash_session_review.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
if __name__=='__main__':main()
