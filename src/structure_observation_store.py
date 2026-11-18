"""Immutable public-market evidence in a separate, bounded SQLite file."""
import hashlib
import json
import math
import sqlite3
from contextlib import contextmanager
from pathlib import Path

FIELDS=('open','high','low','close','volume')
RETENTION_SECONDS=30*86400


def series_fields(kind):
    if kind=='trade':return FIELDS
    if kind=='mark':return FIELDS[:-1]
    raise ValueError('unknown_price_series_kind')


def series_key(instrument,kind):
    series_fields(kind)
    return instrument if kind=='trade' else 'mark:'+instrument


def encoded(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def frame_hash(frame):
    return hashlib.sha256(encoded(frame).encode()).hexdigest()


def closed_frame(frame,bucket,minimum,*,kind='trade'):
    if not isinstance(frame,dict):raise ValueError('missing_history')
    times=frame.get('time') or []
    if len(times)<minimum or len(times)>300:raise ValueError('history_length')
    if (any(int(t)!=t for t in times) or times[-1]!=bucket-900
            or any(b-a!=900 for a,b in zip(times,times[1:]))):
        raise ValueError('history_alignment')
    if frame.get('confirmed')!=[1]*len(times):raise ValueError('unconfirmed_history')
    out=dict(time=[int(t) for t in times])
    for key in series_fields(kind):
        values=[float(x) for x in frame[key]]
        if len(values)!=len(times) or not all(math.isfinite(x) and (x>=0 if key=='volume' else x>0) for x in values):
            raise ValueError('invalid_history_values')
        out[key]=values
    for o,h,l,c in zip(*(out[k] for k in ('open','high','low','close'))):
        if not l<=min(o,c)<=max(o,c)<=h:raise ValueError('invalid_history_ohlc')
    return out


class ObservationStore:
    def __init__(self,path):self.path=Path(path).resolve()

    @contextmanager
    def connect(self):
        c=sqlite3.connect(str(self.path),timeout=2)
        c.row_factory=sqlite3.Row
        c.execute('PRAGMA journal_mode=WAL')
        c.execute('PRAGMA busy_timeout=2000')
        c.execute('PRAGMA synchronous=NORMAL')
        page_size=c.execute('PRAGMA page_size').fetchone()[0]
        c.execute(f'PRAGMA max_page_count={200*1024*1024//page_size}')
        try:
            with c:yield c
        finally:c.close()

    def initialize(self):
        with self.connect() as c:
            c.executescript('''
                CREATE TABLE IF NOT EXISTS scans(
                    profile TEXT NOT NULL,bucket INTEGER NOT NULL,started_at REAL NOT NULL,
                    completed_at REAL,status TEXT NOT NULL,revision TEXT NOT NULL,summary TEXT,
                    PRIMARY KEY(profile,bucket));
                CREATE TABLE IF NOT EXISTS candles(
                    instrument TEXT NOT NULL,stamp INTEGER NOT NULL,payload TEXT NOT NULL,
                    PRIMARY KEY(instrument,stamp));
                CREATE TABLE IF NOT EXISTS observations(
                    profile TEXT NOT NULL,bucket INTEGER NOT NULL,symbol TEXT NOT NULL,
                    instrument TEXT NOT NULL,received_at REAL NOT NULL,quote_ok INTEGER NOT NULL,
                    candidate INTEGER NOT NULL,payload TEXT NOT NULL,
                    PRIMARY KEY(profile,bucket,symbol));
                CREATE INDEX IF NOT EXISTS observation_time ON observations(received_at);
            ''')

    def claim(self,profile,bucket,started_at,revision):
        with self.connect() as c:
            return bool(c.execute('INSERT OR IGNORE INTO scans(profile,bucket,started_at,status,revision) '
                'VALUES(?,?,?,\'collecting\',?)',(profile,bucket,started_at,revision)).rowcount)

    def put_frame(self,instrument,frame,*,kind='trade'):
        overrides=[]
        fields=series_fields(kind);storage_key=series_key(instrument,kind)
        with self.connect() as c:
            for i,stamp in enumerate(frame['time']):
                values=[frame[k][i] for k in fields];payload=encoded(values)
                inserted=c.execute('INSERT OR IGNORE INTO candles VALUES(?,?,?)',(storage_key,stamp,payload)).rowcount
                if not inserted:
                    old=c.execute('SELECT payload FROM candles WHERE instrument=? AND stamp=?',(storage_key,stamp)).fetchone()[0]
                    if old!=payload:overrides.append([stamp,*values])
        reference=dict(start=frame['time'][0],end=frame['time'][-1],bars=len(frame['time']),
                       sha256=frame_hash(frame),overrides=overrides)
        if kind!='trade':reference['kind']=kind
        return reference

    def reconstruct(self,instrument,reference):
        with self.connect() as c:
            return self._reconstruct(c,instrument,reference)

    @staticmethod
    def _reconstruct(c,instrument,reference):
        kind=reference.get('kind','trade');fields=series_fields(kind)
        rows=c.execute('SELECT stamp,payload FROM candles WHERE instrument=? AND stamp BETWEEN ? AND ? '
            'ORDER BY stamp',(series_key(instrument,kind),reference['start'],reference['end'])).fetchall()
        values={r['stamp']:json.loads(r['payload']) for r in rows}
        values.update({r[0]:r[1:] for r in reference['overrides']})
        times=list(range(reference['start'],reference['end']+900,900))
        if len(times)!=reference['bars'] or any(t not in values for t in times):raise ValueError('missing_archived_history')
        frame=dict(time=times,**{k:[values[t][i] for t in times] for i,k in enumerate(fields)})
        if frame_hash(frame)!=reference['sha256']:raise ValueError('archived_history_hash_mismatch')
        return frame

    def export(self,profile,after_bucket=0,limit=1):
        """Bounded read-only snapshot of this archive, never the client DB."""
        if not self.path.exists():return dict(scans=[],next_bucket=after_bucket)
        c=sqlite3.connect(self.path.as_uri()+'?mode=ro',uri=True,timeout=2)
        c.row_factory=sqlite3.Row
        try:
            c.execute('BEGIN')
            scans=c.execute('SELECT * FROM scans WHERE profile=? AND bucket>? '
                'ORDER BY bucket LIMIT ?',(profile,after_bucket,limit)).fetchall()
            output=[]
            for scan in scans:
                rows=c.execute('SELECT * FROM observations WHERE profile=? AND bucket=? '
                    'ORDER BY symbol LIMIT 101',(profile,scan['bucket'])).fetchall()
                if len(rows)>100:raise ValueError('archive_scan_too_large')
                observations=[]
                for row in rows:
                    payload=json.loads(row['payload'])
                    item={k:row[k] for k in ('symbol','instrument','received_at','quote_ok','candidate')}
                    item['payload']=payload
                    if payload.get('frame'):
                        try:item['closed_frame']=self._reconstruct(c,row['instrument'],payload['frame'])
                        except (KeyError,TypeError,ValueError):item['frame_error']='archived_history_unavailable'
                    if payload.get('trade_frame'):
                        try:item['trade_closed_frame']=self._reconstruct(c,row['instrument'],payload['trade_frame'])
                        except (KeyError,TypeError,ValueError):item['trade_frame_error']='archived_trade_history_unavailable'
                    observations.append(item)
                output.append(dict(profile=scan['profile'],bucket=scan['bucket'],started_at=scan['started_at'],
                    completed_at=scan['completed_at'],status=scan['status'],revision=scan['revision'],
                    summary=json.loads(scan['summary']) if scan['summary'] else None,observations=observations))
            return dict(scans=output,next_bucket=output[-1]['bucket'] if output else after_bucket)
        finally:c.close()

    def put_observation(self,profile,bucket,symbol,instrument,received_at,quote_ok,candidate,payload):
        with self.connect() as c:
            return bool(c.execute('INSERT OR IGNORE INTO observations VALUES(?,?,?,?,?,?,?,?)',
                (profile,bucket,symbol,instrument,received_at,int(quote_ok),int(candidate),encoded(payload))).rowcount)

    def prune(self,now):
        # Reclaim old pages before a new claim, even if the previous run failed.
        with self.connect() as c:
            self._prune(c,now)

    @staticmethod
    def _prune(c,now):
        c.execute('DELETE FROM observations WHERE received_at<?',(now-RETENTION_SECONDS,))
        c.execute('DELETE FROM scans WHERE started_at<?',(now-RETENTION_SECONDS,))
        c.execute('DELETE FROM candles WHERE stamp<?',(now-RETENTION_SECONDS-2*86400-300*900,))

    def finish(self,profile,bucket,completed_at,status,summary):
        with self.connect() as c:
            c.execute('UPDATE scans SET completed_at=?,status=?,summary=? WHERE profile=? AND bucket=? '
                'AND completed_at IS NULL',(completed_at,status,encoded(summary),profile,bucket))
            self._prune(c,completed_at)

    def summary(self,profile):
        if not self.path.exists():return dict(status='not_started',scans=0,observations=0,quotes=0,candidates=0)
        # Readers do not initialize or mutate an archive while serving /status.
        c=sqlite3.connect(self.path.as_uri()+'?mode=ro',uri=True,timeout=1)
        c.row_factory=sqlite3.Row
        try:
            last=c.execute('SELECT bucket,started_at,completed_at,status,summary FROM scans WHERE profile=? '
                           'ORDER BY bucket DESC LIMIT 1',(profile,)).fetchone()
            counts=c.execute('SELECT COUNT(*) n,COALESCE(SUM(quote_ok),0) q,COALESCE(SUM(candidate),0) p '
                             'FROM observations WHERE profile=?',(profile,)).fetchone()
            scans=c.execute('SELECT COUNT(*) FROM scans WHERE profile=?',(profile,)).fetchone()[0]
            return dict(status=last['status'] if last else 'not_started',scans=scans,
                observations=counts['n'],quotes=counts['q'],candidates=counts['p'],
                last=dict(last) if last else None)
        finally:c.close()
