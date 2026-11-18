"""Aggregate frozen candidate books by observed session, never infer fills."""
import collections
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import statistics

from src.market_hours import session_phase

ROOT=Path('reports/structure_observer_prospective_nov18')


def main():
    source=ROOT/'quote_quality_diagnostic.json'
    q=json.loads(source.read_text(encoding='utf-8'));sessions={}
    for window in q['snapshots']:
        phase=session_phase(datetime.fromtimestamp(window['bucket'],timezone.utc))
        row=sessions.setdefault(phase,dict(windows=0,directional_checks=0,first_book_pass=0,wide_spread=0))
        row['windows']+=1;row['directional_checks']+=len(window['checks'])
        row['first_book_pass']+=window['first_snapshot_pass'];row['wide_spread']+=window['wide_spread']
    strategies={}
    for name in sorted({r['hypothesis'] for r in q['candidates']}):
        selected=[r for r in q['candidates'] if r['hypothesis']==name]
        strategies[name]=dict(candidates=len(selected),first_book_status=dict(collections.Counter(
            r['first_book']['status'] for r in selected)),median_spread_bps=statistics.median(
                r['first_book']['spread_bps'] for r in selected),by_session={})
        for phase in sessions:
            rows=[r for r in selected if session_phase(datetime.fromtimestamp(r['bucket'],timezone.utc))==phase]
            strategies[name]['by_session'][phase]=dict(candidates=len(rows),first_book_pass=sum(
                r['first_book']['status']=='first_snapshot_pass' for r in rows))
    unique={(r['bucket'],r['symbol'],r['direction']) for r in q['candidates']}
    report=dict(status='OBSERVED_BOOK_QUALITY_BY_SESSION_NOT_RETURNS',objective_verified=False,
        live_activation_supported=False,sessions=sessions,strategies=strategies,
        unique_candidate_events=len(unique),hypothesis_candidate_rows=len(q['candidates']),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        limitations=['Hypothesis counts overlap; they are not distinct fills.',
            'First book only, not final retry outcome or a verified private entry.',
            'One partial calendar day and its preceding overnight period; no monthly or all-regime claim.',
            'Session labels use the existing DST-aware market_hours module; X-Perps themselves trade 24/7.'])
    (ROOT/'session_summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
