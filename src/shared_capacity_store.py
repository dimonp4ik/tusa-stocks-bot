"""Durable, transactional capacity claims shared by the two trusted services.

No exchange orders or API credentials enter this module. Exchange snapshots
and close evidence must be collected by the authenticated client adapter.
Submitted/open claims never expire automatically.
"""
from contextlib import contextmanager
import math
from pathlib import Path
import re
import sqlite3
import time
import uuid

OWNERS={'crypto':4,'stocks':8}
ACTIVE=('RESERVED','SUBMITTING','OPEN')
PEER_MAX_AGE=180
RESERVE_TTL=180
STARTUP_GRACE=180
SNAPSHOT_MAX_AGE=15


class CapacityError(ValueError):
    """A rejected request, without credentials or account identifiers in text."""


def text_id(value):
    if not isinstance(value,str) or not 1<=len(value)<=128 or not value.isascii() or not value.isprintable():
        raise CapacityError('invalid identifier')
    return value


def number(value,*,positive=False):
    if isinstance(value,bool):
        raise CapacityError('invalid number')
    try:
        result=float(value)
    except (TypeError,ValueError,OverflowError) as exc:
        raise CapacityError('invalid number') from exc
    if not math.isfinite(result) or result<0 or (positive and result==0):
        raise CapacityError('invalid number')
    return result


def integer(value):
    result=number(value)
    if result!=int(result):
        raise CapacityError('invalid integer')
    return int(result)


def owner_id(value):
    if not isinstance(value,str) or value not in OWNERS:
        raise CapacityError('unknown service')
    return value


def account_id(value):
    if not isinstance(value,str) or re.fullmatch('[0-9a-f]{64}',value) is None:
        raise CapacityError('invalid account identity')
    return value


def binding(row):
    return {k:row[k] for k in ('owner','account','instrument','intent')}


def snapshot(value,now):
    """Validate a complete account snapshot, including per-instrument margin."""
    if not isinstance(value,dict):
        raise CapacityError('invalid snapshot')
    captured=number(value.get('captured_at'))
    if not -2<=now-captured<=SNAPSHOT_MAX_AGE:
        raise CapacityError('stale snapshot')
    equity=number(value.get('equity'),positive=True)
    count=integer(value.get('count'))
    instruments=value.get('inst_ids')
    margins=value.get('margin_by_inst')
    if not isinstance(instruments,(list,tuple)) or not isinstance(margins,dict):
        raise CapacityError('incomplete snapshot')
    ids={text_id(v) for v in instruments}
    if len(ids)!=len(instruments) or count<len(ids) or count>10000 or set(margins)!=ids:
        raise CapacityError('inconsistent snapshot')
    by_inst={text_id(k):number(v,positive=True) for k,v in margins.items()}
    total=number(value.get('margin_usd'))
    if not math.isclose(sum(by_inst.values()),total,rel_tol=1e-9,abs_tol=1e-7):
        raise CapacityError('inconsistent snapshot margin')
    if bool(count)!=bool(ids):
        raise CapacityError('inconsistent position count')
    return dict(equity=equity,count=count,inst_ids=ids,margin_by_inst=by_inst,margin_usd=total)


class CapacityStore:
    def __init__(self,path,*,clock=time.time,startup_grace=STARTUP_GRACE,
                 max_positions=12,max_margin=.90):
        self.path=str(Path(path).resolve())
        self.clock=clock
        self.startup_grace=number(startup_grace)
        self.max_positions=integer(max_positions)
        self.max_margin=number(max_margin,positive=True)
        if not 1<=self.max_positions<=12 or self.max_margin>.90:
            raise CapacityError('unsafe capacity configuration')
        with self._transaction() as c:
            c.execute('CREATE TABLE IF NOT EXISTS capacity_meta (key TEXT PRIMARY KEY,value TEXT NOT NULL)')
            c.execute('CREATE TABLE IF NOT EXISTS capacity_peers '
                '(owner TEXT PRIMARY KEY,instance TEXT NOT NULL,last_sync REAL NOT NULL)')
            c.execute('CREATE TABLE IF NOT EXISTS capacity_claims '
                '(id TEXT PRIMARY KEY,owner TEXT NOT NULL,account TEXT NOT NULL,instrument TEXT NOT NULL,'
                'intent TEXT NOT NULL,margin REAL NOT NULL,state TEXT NOT NULL,created REAL NOT NULL,'
                'updated REAL NOT NULL,release_reason TEXT,UNIQUE(owner,intent))')
            c.execute("CREATE UNIQUE INDEX IF NOT EXISTS capacity_active_instrument "
                "ON capacity_claims(account,instrument) WHERE state IN ('RESERVED','SUBMITTING','OPEN')")
            c.execute('CREATE INDEX IF NOT EXISTS capacity_account_state ON capacity_claims(account,state)')
            c.execute('CREATE INDEX IF NOT EXISTS capacity_state_created ON capacity_claims(state,created)')
            c.execute('INSERT OR IGNORE INTO capacity_meta VALUES (?,?)',('epoch',uuid.uuid4().hex))
            c.execute('INSERT OR IGNORE INTO capacity_meta VALUES (?,?)',('boot_at',str(self.clock())))

    @contextmanager
    def _transaction(self):
        c=sqlite3.connect(self.path,timeout=5,isolation_level=None)
        c.row_factory=sqlite3.Row
        try:
            c.execute('BEGIN IMMEDIATE')
            yield c
            c.commit()
        except BaseException:
            c.rollback()
            raise
        finally:
            c.close()

    def _epoch(self,c):
        return c.execute("SELECT value FROM capacity_meta WHERE key='epoch'").fetchone()[0]

    def _ready(self,c,now):
        boot=float(c.execute("SELECT value FROM capacity_meta WHERE key='boot_at'").fetchone()[0])
        peers={r['owner']:r['last_sync'] for r in c.execute('SELECT * FROM capacity_peers')}
        return now-boot>=self.startup_grace and all(
            owner in peers and 0<=now-peers[owner]<=PEER_MAX_AGE for owner in OWNERS)

    def _expire(self,c,now):
        # A submitted request may become visible after a timeout or restart.
        # Only an unsubmitted reservation can be retired by age.
        c.execute("UPDATE capacity_claims SET state='RELEASED',updated=?,release_reason='unsent_ttl' "
            "WHERE state='RESERVED' AND created<=?",(now,now-RESERVE_TTL))

    def _claim(self,value):
        if not isinstance(value,dict):
            raise CapacityError('invalid claim')
        return dict(account=account_id(value.get('account')),instrument=text_id(value.get('instrument')),
            intent=text_id(value.get('intent')),margin=number(value.get('margin'),positive=True))

    def sync(self,owner,instance,positions):
        """Seed real legacy positions; absence never releases an existing claim."""
        owner=owner_id(owner);instance=text_id(instance)
        if not isinstance(positions,list) or len(positions)>5000:
            raise CapacityError('invalid synchronization batch')
        checked=[]
        for value in positions:
            item=self._claim(value)
            state=value.get('state')
            if state not in {'SUBMITTING','OPEN','CLOSED'}:
                raise CapacityError('cannot seed unsent entry')
            if state=='CLOSED' and not isinstance(value.get('was_submitted'),bool):
                raise CapacityError('closed intent submission history missing')
            checked.append((item,state,value.get('was_submitted')))
        now=self.clock()
        with self._transaction() as c:
            prior=c.execute('SELECT * FROM capacity_peers WHERE owner=?',(owner,)).fetchone()
            if prior and prior['instance']!=instance and now-prior['last_sync']<=PEER_MAX_AGE:
                raise CapacityError('service instance already active')
            if prior and prior['instance']!=instance:
                c.execute("UPDATE capacity_meta SET value=? WHERE key='boot_at'",(str(now),))
            identities={};released=[]
            for item,state,submitted in checked:
                found=c.execute('SELECT * FROM capacity_claims WHERE owner=? AND intent=?',
                    (owner,item['intent'])).fetchone()
                if found:
                    if found['account']!=item['account'] or found['instrument']!=item['instrument']:
                        raise CapacityError('synchronization conflicts with durable intent')
                    if found['state']=='RELEASED':
                        if state!='CLOSED':
                            raise CapacityError('synchronization conflicts with durable intent')
                        identities[item['intent']]=found['id'];released.append(item['intent'])
                        continue
                    next_state=(found['state'] if state=='CLOSED' else
                        ('OPEN' if state=='OPEN' or found['state']=='OPEN' else 'SUBMITTING'))
                    c.execute('UPDATE capacity_claims SET state=?,margin=?,updated=? WHERE id=?',
                        (next_state,max(found['margin'],item['margin']),now,found['id']))
                    identities[item['intent']]=found['id']
                else:
                    occupied=c.execute("SELECT 1 FROM capacity_claims WHERE account=? AND instrument=? "
                        "AND state IN ('RESERVED','SUBMITTING','OPEN')",(item['account'],item['instrument'])).fetchone()
                    if occupied and not (state=='CLOSED' and submitted is False):
                        raise CapacityError('instrument already claimed by another intent')
                    claim_id=uuid.uuid4().hex
                    if state=='CLOSED':
                        state='SUBMITTING' if submitted else 'RELEASED'
                    if state=='RELEASED':
                        released.append(item['intent'])
                    c.execute('INSERT INTO capacity_claims VALUES (?,?,?,?,?,?,?,?,?,NULL)',
                        (claim_id,owner,item['account'],item['instrument'],item['intent'],item['margin'],state,now,now))
                    identities[item['intent']]=claim_id
            c.execute('INSERT INTO capacity_peers VALUES (?,?,?) ON CONFLICT(owner) '
                'DO UPDATE SET instance=excluded.instance,last_sync=excluded.last_sync',(owner,instance,now))
            return dict(epoch=self._epoch(c),claims=identities,released=released,ready=self._ready(c,now))

    def _capacity(self,c,account,view,claims):
        ids=view['inst_ids']
        occupied=view['count']+len({r['instrument'] for r in claims}-ids)
        margin=view['margin_usd']+sum(max(0.,r['margin']-view['margin_by_inst'].get(r['instrument'],0.))
            for r in claims)
        if occupied>self.max_positions:
            return 'position_limit'
        if margin>view['equity']*self.max_margin+1e-9:
            return 'margin_limit'
        for owner,limit in OWNERS.items():
            if sum(r['owner']==owner for r in claims)>limit:
                return 'service_position_limit'
        return None

    def reserve(self,owner,epoch,value,account_snapshot):
        owner=owner_id(owner);item=self._claim(value)
        with self._transaction() as c:
            now=self.clock();view=snapshot(account_snapshot,now)
            self._expire(c,now)
            if epoch!=self._epoch(c) or not self._ready(c,now):
                return dict(allowed=False,reason='coordinator_not_ready')
            existing=c.execute('SELECT * FROM capacity_claims WHERE owner=? AND intent=?',
                (owner,item['intent'])).fetchone()
            if existing:
                if any(existing[k]!=item[k] for k in ('account','instrument','margin')):
                    raise CapacityError('intent payload changed')
                return dict(allowed=existing['state']=='RESERVED',claim_id=existing['id'],
                    state=existing['state'],reason='existing_intent',epoch=self._epoch(c),binding=binding(existing))
            claims=[dict(r) for r in c.execute("SELECT * FROM capacity_claims WHERE account=? AND state IN ('RESERVED','SUBMITTING','OPEN')",
                (item['account'],))]
            if item['instrument'] in view['inst_ids'] or any(r['instrument']==item['instrument'] for r in claims):
                return dict(allowed=False,reason='instrument_occupied')
            candidate=dict(item,owner=owner)
            reason=self._capacity(c,item['account'],view,claims+[candidate])
            if reason:
                return dict(allowed=False,reason=reason)
            claim_id=uuid.uuid4().hex
            c.execute('INSERT INTO capacity_claims VALUES (?,?,?,?,?,?,?,?,?,NULL)',
                (claim_id,owner,item['account'],item['instrument'],item['intent'],item['margin'],
                 'RESERVED',now,now))
            return dict(allowed=True,claim_id=claim_id,state='RESERVED',epoch=self._epoch(c),binding=binding(candidate))

    def begin(self,owner,epoch,claim_id,account_snapshot,*,identity):
        owner=owner_id(owner);claim_id=text_id(claim_id)
        with self._transaction() as c:
            now=self.clock();view=snapshot(account_snapshot,now)
            self._expire(c,now)
            row=c.execute('SELECT * FROM capacity_claims WHERE id=? AND owner=?',(claim_id,owner)).fetchone()
            if epoch!=self._epoch(c) or not self._ready(c,now):
                return dict(allowed=False,reason='coordinator_not_ready')
            if not row or row['state'] not in {'RESERVED','SUBMITTING'}:
                return dict(allowed=False,reason='reservation_unavailable')
            if identity!=binding(row):
                raise CapacityError('claim identity does not match')
            if row['instrument'] in view['inst_ids']:
                return dict(allowed=False,reason='instrument_already_visible')
            claims=[dict(r) for r in c.execute("SELECT * FROM capacity_claims WHERE account=? AND state IN ('RESERVED','SUBMITTING','OPEN')",
                (row['account'],))]
            reason=self._capacity(c,row['account'],view,claims)
            if reason:
                return dict(allowed=False,reason=reason)
            c.execute("UPDATE capacity_claims SET state='SUBMITTING',updated=? WHERE id=?",(now,claim_id))
            return dict(allowed=True,claim_id=claim_id,state='SUBMITTING',epoch=self._epoch(c))

    def release(self,owner,epoch,claim_id,evidence,*,identity):
        owner=owner_id(owner);claim_id=text_id(claim_id);now=self.clock()
        if not isinstance(evidence,dict) or evidence.get('local_status')!='CLOSED':
            raise CapacityError('local closure not confirmed')
        with self._transaction() as c:
            if epoch!=self._epoch(c):
                raise CapacityError('coordinator epoch changed')
            row=c.execute('SELECT * FROM capacity_claims WHERE id=? AND owner=?',(claim_id,owner)).fetchone()
            if not row:
                raise CapacityError('claim unavailable')
            if identity!=binding(row):
                raise CapacityError('claim identity does not match')
            if row['state']=='RELEASED':
                return dict(released=True)
            if evidence.get('kind')=='never_submitted':
                if evidence.get('submit_started_at')!=0 or row['state']=='OPEN':
                    raise CapacityError('unsent release not proven')
            elif evidence.get('kind')=='terminal_flat':
                if (evidence.get('terminal_order_state') not in {'filled','canceled'}
                        or integer(evidence.get('flat_reads',0))<2
                        or evidence.get('pending_entries_clear') is not True
                        or now-row['created']<90):
                    raise CapacityError('terminal flatness not proven')
            elif evidence.get('kind')=='rejected_flat':
                if (evidence.get('rejection_code') not in {'51007','51008'}
                        or integer(evidence.get('flat_reads',0))<2
                        or evidence.get('pending_entries_clear') is not True
                        or now-row['created']<90):
                    raise CapacityError('definitive rejection not proven')
            else:
                raise CapacityError('unknown release proof')
            c.execute("UPDATE capacity_claims SET state='RELEASED',updated=?,release_reason=? WHERE id=?",
                (now,evidence['kind'],claim_id))
            return dict(released=True)

    def status(self):
        now=self.clock()
        with self._transaction() as c:
            self._expire(c,now)
            peers={r['owner']:dict(instance=r['instance'],age_seconds=max(0.,now-r['last_sync']))
                for r in c.execute('SELECT * FROM capacity_peers')}
            active=c.execute("SELECT COUNT(*) FROM capacity_claims WHERE state IN ('RESERVED','SUBMITTING','OPEN')").fetchone()[0]
            return dict(epoch=self._epoch(c),ready=self._ready(c,now),peers=peers,active_claims=active)
