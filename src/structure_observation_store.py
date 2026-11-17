"""Immutable public-market evidence in a separate, bounded SQLite file."""
import hashlib
import json
import math
import sqlite3
from contextlib import contextmanager
from pathlib import Path

FIELDS=('open','high','low','close','volume')
RETENTION_SECONDS=30*86400


def encoded(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def frame_hash(frame):
    return hashlib.sha256(encoded(frame).encode()).hexdigest()


def closed_frame(frame,bucket,minimum):
    if not isinstance(frame,dict):raise ValueError('missing_history')
    times=frame.get('time') or []
    if len(times)<minimum or len(times)>300:raise ValueError('history_length')
    if (any(int(t)!=t for t in times) or times[-1]!=bucket-900
            or any(b-a!=900 for a,b in zip(times,times[1:]))):
        raise ValueError('history_alignment')
    if frame.get('confirmed')!=[1]*len(times):raise ValueError('unconfirmed_history')
    out=dict(time=[int(t) for t in times])
    for key in FIELDS:
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

    def put_frame(self,instrument,frame):
        overrides=[]
        with self.connect() as c:
            for i,stamp in enumerate(frame['time']):
                values=[frame[k][i] for k in FIELDS];payload=encoded(values)
                inserted=c.execute('INSERT OR IGNORE INTO candles VALUES(?,?,?)',(instrument,stamp,payload)).rowcount
                if not inserted:
                    old=c.execute('SELECT payload FROM candles WHERE instrument=? AND stamp=?',(instrument,stamp)).fetchone()[0]
                    if old!=payload:overrides.append([stamp,*values])
        return dict(start=frame['time'][0],end=frame['time'][-1],bars=len(frame['time']),
                    sha256=frame_hash(frame),overrides=overrides)

    def reconstruct(self,instrument,reference):
        with self.connect() as c:
            rows=c.execute('SELECT stamp,payload FROM candles WHERE instrument=? AND stamp BETWEEN ? AND ? '
                'ORDER BY stamp',(instrument,reference['start'],reference['end'])).fetchall()
        values={r['stamp']:json.loads(r['payload']) for r in rows}
        values.update({r[0]:r[1:] for r in reference['overrides']})
        times=list(range(reference['start'],reference['end']+900,900))
        if len(times)!=reference['bars'] or any(t not in values for t in times):raise ValueError('missing_archived_history')
        frame=dict(time=times,**{k:[values[t][i] for t in times] for i,k in enumerate(FIELDS)})
        if frame_hash(frame)!=reference['sha256']:raise ValueError('archived_history_hash_mismatch')
        return frame

    def put_observation(self,profile,bucket,symbol,instrument,received_at,quote_ok,candidate,payload):
        with self.connect() as c:
            return bool(c.execute('INSERT OR IGNORE INTO observations VALUES(?,?,?,?,?,?,?,?)',
                (profile,bucket,symbol,instrument,received_at,int(quote_ok),int(candidate),encoded(payload))).rowcount)

    def finish(self,profile,bucket,completed_at,status,summary):
        with self.connect() as c:
            c.execute('UPDATE scans SET completed_at=?,status=?,summary=? WHERE profile=? AND bucket=? '
                'AND completed_at IS NULL',(completed_at,status,encoded(summary),profile,bucket))
            c.execute('DELETE FROM observations WHERE received_at<?',(completed_at-RETENTION_SECONDS,))
            c.execute('DELETE FROM scans WHERE started_at<?',(completed_at-RETENTION_SECONDS,))
            # Keep the extra two-day indicator prefix for the oldest retained scan.
            c.execute('DELETE FROM candles WHERE stamp<?',(completed_at-RETENTION_SECONDS-2*86400-300*900,))

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
