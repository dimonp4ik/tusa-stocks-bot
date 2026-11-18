"""Attribute the November 18 OKTA PAPER gap without claiming an exchange fill."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT=Path('reports/audit_2026_09_08')
ARCHIVE=Path('reports/structure_observer_prospective_nov18/bucket_1794986100.json')
TRADES=ROOT/'okta_gap_nov18_1m.json'
MARK=ROOT/'okta_gap_nov18_mark.json'


def main():
    scan=json.loads(ARCHIVE.read_text(encoding='utf-8'))
    observation=next(r for r in scan['observations'] if r['symbol']=='OKTAUSDT')
    candidate=observation['payload']['native_candidate']
    if candidate['direction']!='SHORT' or candidate['entry_time']!=1794986100:
        raise ValueError('unexpected_archived_candidate')
    entry=candidate['entry'];stop=entry*1.01
    trades=json.loads(TRADES.read_text(encoding='utf-8'))
    active=[i for i,t in enumerate(trades['time']) if t>=scan['bucket'] and trades['volume'][i]>0]
    if not active:raise ValueError('no_positive_volume_bar_observed')
    first=active[0];gap_open=trades['open'][first]
    if gap_open<=stop:raise ValueError('unexpected_first_trade_reference')
    mark=json.loads(MARK.read_text(encoding='utf-8'))['frame']
    prior=mark['time'].index(scan['bucket']-900)
    hits=[i for i,t in enumerate(mark['time']) if t>=scan['bucket'] and mark['high'][i]>=stop]
    if not hits:raise ValueError('no_mark_stop_crossing')
    mark_hit=hits[0];book=observation['payload']['book']
    report=dict(status='OBSERVED_PUBLIC_REFERENCE_GAP_NOT_PRIVATE_FILL',symbol='OKTAUSDT',
        modeled_entry=entry,modeled_one_percent_stop=stop,
        archived_candidate=dict(timestamp=candidate['entry_time'],signal_volume=candidate['signal_volume'],
            original_book_spread_bps=book['spread_bps'],bid=book['bid'],ask=book['ask']),
        first_later_positive_volume_bar=dict(timestamp=trades['time'][first],
            utc=datetime.fromtimestamp(trades['time'][first],timezone.utc).isoformat(),
            open=gap_open,close=trades['close'][first],volume=trades['volume'][first]),
        elapsed_hours_to_first_positive_volume_bar=(trades['time'][first]-scan['bucket'])/3600,
        hypothetical_open_gap_r=(entry-gap_open)/(entry*.01),
        latest_closed_mark_at_signal=mark['close'][prior],
        signal_trade_to_mark_fraction=entry/mark['close'][prior]-1,
        first_later_mark_stop_crossing_bar=dict(timestamp=mark['time'][mark_hit],
            high=mark['high'][mark_hit],low=mark['low'][mark_hit]),
        actual_order_or_fill_confirmed=False,live_activation_supported=False,
        limitations=[
            'Archived candidate and known 1% native geometry match the public rounded -15.03R report; private signal row not exported.',
            'Zero reported candle volume is a data observation, not proof that the order book was empty.',
            'The first later positive-volume minute open is a public PAPER fill proxy, not the client actual stop fill.',
            'Fifteen-minute mark candles cannot identify an exact trigger instant or an executable bid/ask.',
            'Historical public data fetched after the event may include vendor revisions.',
            'The archive quote precedes the gap; it proves no later liquidation or gap-time spread.',
            'Changing from last to mark trigger requires a separately validated strategy and matching exchange orders.'],
        sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ARCHIVE,TRADES,MARK,Path(__file__))})
    output=ROOT/'stock_okta_gap_nov18.json'
    output.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('status','hypothetical_open_gap_r',
        'elapsed_hours_to_first_positive_volume_bar','first_later_mark_stop_crossing_bar')},indent=2))


if __name__=='__main__':main()
