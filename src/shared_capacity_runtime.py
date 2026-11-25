"""Glue durable local entry intents to the shared coordinator, without orders."""
import hashlib
import logging
import os
from pathlib import Path
import re
import threading
import time

from src import db,okx_trader as okx
from src.shared_capacity_store import CapacityError
from src.shared_capacity_transport import from_environment,register_capacity_api

log=logging.getLogger(__name__)


def opaque(value):
    if not isinstance(value,str) or re.fullmatch('[0-9a-f]{32}',value) is None:
        raise CapacityError('invalid shared coordinator response identity')
    return value


class SharedCapacityRuntime:
    def __init__(self,owner,*,transport=None,clock=time.time):
        self.owner=owner;self.transport=transport;self.clock=clock
        self.enabled=transport is not None or os.getenv('SHARED_CAPACITY_MODE','off').strip().lower()!='off'
        self.initialized=transport is not None
        self.epoch=None;self.ready=False;self.last_sync=0.;self.error=None
        self.lock=threading.RLock();self.cursor=0
        self.last_region_probe=0.
        self.legacy_unbound_count=0

    def initialize(self,app=None):
        if not self.initialized:
            self.initialized=True
            try:
                self.transport=from_environment(self.owner)
            except Exception:
                self.error='configuration_unavailable';self.enabled=True
                log.error('shared capacity configuration refused for %s',self.owner)
        if app is not None and self.transport is not None:
            register_capacity_api(app,self.transport)

    def status_text(self):
        if not self.enabled:
            return 'Общая бронь: не настроена; работают локальные и биржевые проверки.'
        if self.error or not self.ready or self.clock()-self.last_sync>180:
            return 'Общая бронь: не готова; новые входы запрещены, сопровождение продолжается.'
        result='Общая бронь: включена; последняя синхронизация успешна.'
        if self.legacy_unbound_count:
            result+=(f' Старых записей отключённых подключений без привязки: {self.legacy_unbound_count}.'
                     ' Перед входом обязателен свежий снимок счёта OKX.')
        return result

    def persistent_storage_ready(self):
        """Non-secret deployment diagnostic; never exposes a file path."""
        try:
            volume=Path(os.environ['RAILWAY_VOLUME_MOUNT_PATH'])
            database=Path(os.environ['DB_PATH'])
            return (volume.is_absolute() and volume.is_dir() and database.is_absolute()
                and database.is_file() and database.resolve().is_relative_to(volume.resolve())
                and database.resolve()!=volume.resolve())
        except (KeyError,OSError,ValueError):
            return False

    def _intent(self,row):
        if row.get('client_order_id'):
            return row['client_order_id']
        material=f"{self.owner}:{row['id']}:{row['user_id']}:{row.get('opened_at')}"
        return 'legacy-'+hashlib.sha256(material.encode()).hexdigest()[:32]

    def _account(self,row,credentials):
        if row.get('account_key'):
            return row['account_key']
        user=db.at_get(row['user_id'])
        if not user:
            raise CapacityError('position user missing')
        creds=credentials(user)
        if not creds:
            raise CapacityError('position credentials unavailable')
        good,config=okx.get_account_configuration(creds)
        if not good:
            code=re.search(r'\((?:sCode|code) ([0-9]{5})(?:[,\)])',str(config))
            log.warning('shared legacy account lookup refused for %s (exchange code %s)',
                self.owner,code.group(1) if code else 'unknown')
            if (code and code.group(1)=='50119' and okx._base_url()=='https://my.okx.com'
                    and self.clock()-self.last_region_probe>=300):
                self.last_region_probe=self.clock()
                # Read only. A successful probe does not authorize entry or
                # redirect an order; deployment config must be corrected.
                good_eu,config_eu=okx.get_account_configuration(creds,base_url='https://eea.okx.com')
                log.warning('shared EU read-only region probe for %s: %s',
                    self.owner,bool(good_eu and okx.account_key(config_eu)))
            raise CapacityError('position exchange configuration unavailable')
        account=okx.account_key(config)
        if not account:
            raise CapacityError('position exchange uid unavailable')
        return account

    def _inactive_unbound_legacy(self,row):
        """Retain historical rows without pretending they identify an account.

        Only pre-intent OPEN rows of explicitly inactive connections qualify.
        No existing account/claim, order ID, or submission evidence is ignored.
        This does not declare a position flat: reserve/begin still require the
        complete current account snapshot, including pending-entry checks.
        Any real position on that account counts even without a local claim.
        """
        if row.get('status')!='OPEN' or row.get('entry_submit_started_at') is not None:
            return False
        if any(row.get(key) for key in ('account_key','client_order_id','shared_claim_id','shared_epoch')):
            return False
        user=db.at_get(row['user_id'])
        return user is not None and user.get('active')==0

    def maintenance(self,credentials,*,reconcile=True):
        if not self.enabled:
            return
        checks=[]
        with self.lock:
            self.initialize()
            if self.transport is None:
                self.ready=False
                return
            try:
                stage='local_intents'
                items=[];local={};legacy_unbound=0
                for row in db.at_shared_positions():
                    # No market request can precede the local submission CAS.
                    # An in-flight, unsubmitted reservation already exists at
                    # the coordinator and must not be seeded as a real fill.
                    if row['status']=='ENTRY_PENDING' and row.get('entry_submit_started_at')==0:
                        continue
                    if self._inactive_unbound_legacy(row):
                        legacy_unbound+=1
                        continue
                    identity=self._intent(row)
                    payload=dict(account=self._account(row,credentials),instrument=row['inst_id'],
                        intent=identity,margin=float(row.get('margin_usd') or 0))
                    if row['status']=='CLOSED':
                        payload.update(state='CLOSED',was_submitted=row.get('entry_submit_started_at')!=0)
                    else:
                        payload['state']='OPEN' if row['status']=='OPEN' else 'SUBMITTING'
                    items.append(payload);local[identity]=(row,payload['account'])
                stage='peer_sync'
                answer=self.transport.call('sync',dict(owner=self.owner,instance=db.at_shared_instance_id(),positions=items))
                stage='local_binding'
                epoch=opaque(answer['epoch']);mapping=answer['claims']
                released=answer.get('released',[])
                if (not isinstance(mapping,dict) or not isinstance(released,list)
                        or any(not isinstance(v,str) or v not in local for v in released)
                        or not isinstance(answer.get('ready'),bool)):
                    raise CapacityError('invalid synchronization response')
                for identity,(row,account) in local.items():
                    claim_id=opaque(mapping[identity])
                    if not db.at_set_shared_claim(row['id'],claim_id,epoch,account):
                        raise CapacityError('local claim binding refused')
                    if identity in released:
                        db.at_mark_shared_released(row['id'],claim_id,epoch)
                self.epoch=epoch;self.ready=answer.get('ready') is True
                if legacy_unbound!=self.legacy_unbound_count:
                    log.info('shared capacity retained %d inactive unbound legacy rows for %s; '
                             'current broker snapshot remains mandatory',legacy_unbound,self.owner)
                self.legacy_unbound_count=legacy_unbound
                self.last_sync=self.clock();self.error=None
                closed=[r for r in db.at_shared_positions() if r['status']=='CLOSED'] if reconcile else []
                if closed:
                    start=self.cursor%len(closed)
                    ordered=closed[start:]+closed[:start]
                    self.cursor=(start+2)%len(closed)
                    checks=ordered[:2]
            except Exception as exc:
                self.ready=False;self.error='synchronization_unavailable'
                # Never log payloads, account identities, keys or HTTP errors.
                # Only fixed diagnostics from our validation code, never the
                # exception payload (which could contain private identifiers).
                reasons={'position account identity unavailable':'account_identity_unavailable',
                    'position user missing':'legacy_user_missing',
                    'position credentials unavailable':'legacy_credentials_unavailable',
                    'position exchange configuration unavailable':'exchange_configuration_unavailable',
                    'position exchange uid unavailable':'exchange_uid_missing',
                    'invalid account identity':'account_identity_invalid',
                    'invalid number':'numeric_field_invalid',
                    'invalid identifier':'identifier_invalid',
                    'instrument already claimed by another intent':'duplicate_instrument',
                    'service instance already active':'duplicate_instance',
                    'local claim binding refused':'binding_refused'}
                reason=reasons.get(str(exc),'unclassified') if type(exc) is CapacityError else type(exc).__name__
                log.warning('shared capacity synchronization refused for %s (%s; %s)',self.owner,stage,reason)
        # Read-only exchange evidence can be slow. Do not hold the entry/sync
        # mutex while collecting it, and never run it in the TP/SL job.
        for row in checks:
            try:
                self._release_closed(row,credentials)
            except Exception as exc:
                log.warning('shared close proof deferred for %s (%s)',self.owner,type(exc).__name__)

    def _release_closed(self,row,credentials):
        epoch,claim_id=row.get('shared_epoch'),row.get('shared_claim_id')
        if epoch!=self.epoch or not claim_id:
            return
        if row.get('entry_submit_started_at')==0:
            evidence=dict(kind='never_submitted',local_status='CLOSED',submit_started_at=0)
        else:
            if self.clock()-float(row.get('opened_at') or self.clock())<90:
                return
            if not row.get('client_order_id'):
                # A legacy record without an entry-order id needs an explicit
                # audit, not an invented terminal order or automatic expiry.
                return
            user=db.at_get(row['user_id']);creds=credentials(user) if user else None
            if not creds:
                return
            good,config=okx.get_account_configuration(creds)
            if not good or okx.account_key(config)!=row['account_key']:
                return
            good,size=okx.get_position_size(creds,row['inst_id'])
            if not good or size!=0:
                return
            rejected=(re.search(r'\((?:sCode|code) (51007|51008)\)$',str(row.get('error') or ''))
                if row.get('close_reason')=='ENTRY_REJECTED' else None)
            if not rejected:
                good,order=okx.get_entry_order_for_release(creds,row['inst_id'],row['client_order_id'],
                    row.get('entry_submit_started_at'),now=self.clock())
                if not good or order.get('state') not in {'filled','canceled'}:
                    return
            clear,_=okx.pending_entries_clear(creds,inst_id=row['inst_id'])
            if not clear:
                return
            good,size=okx.get_position_size(creds,row['inst_id'])
            if not good or size!=0:
                return
            evidence=dict(local_status='CLOSED',flat_reads=2,pending_entries_clear=True)
            if rejected:
                evidence.update(kind='rejected_flat',rejection_code=rejected.group(1))
            else:
                evidence.update(kind='terminal_flat',terminal_order_state=order['state'])
        if epoch!=self.epoch:
            return
        answer=self.transport.call('release',dict(owner=self.owner,epoch=epoch,
            claim_id=claim_id,evidence=evidence,identity=dict(owner=self.owner,
                account=row['account_key'],instrument=row['inst_id'],intent=self._intent(row))))
        if answer.get('released') is True:
            # Apply the local release after any sync that already read this row.
            # If that sync rebound it to a new coordinator epoch, its new claim
            # still needs reconciliation. Do not hold the lock during network IO.
            with self.lock:
                if epoch == self.epoch:
                    db.at_mark_shared_released(row['id'],claim_id,epoch)

    def prepare(self,credentials):
        if not self.enabled:
            return True
        if self.clock()-self.last_sync>20 or not self.ready:
            self.maintenance(credentials,reconcile=False)
        return self.ready and self.transport is not None and self.clock()-self.last_sync<=180

    def _snapshot(self,positions,equity,captured_at):
        capacity_fields = {key:value for key,value in positions.items() if key!='side_by_inst'}
        return dict(capacity_fields,equity=equity,captured_at=captured_at)

    def reserve(self,position_id,account,instrument,intent,margin,positions,equity,captured_at):
        if not self.enabled:
            return None
        if self.transport is None or not self.epoch:
            raise CapacityError('shared coordinator unavailable')
        answer=self.transport.call('reserve',dict(owner=self.owner,epoch=self.epoch,
            claim=dict(account=account,instrument=instrument,intent=intent,margin=margin),
            snapshot=self._snapshot(positions,equity,captured_at)))
        if answer.get('allowed') is not True:
            raise CapacityError('shared reservation refused')
        claim_id=opaque(answer.get('claim_id'))
        identity=dict(owner=self.owner,account=account,instrument=instrument,intent=intent)
        if (answer.get('epoch')!=self.epoch or answer.get('state')!='RESERVED'
                or answer.get('binding')!=identity):
            raise CapacityError('invalid shared reservation response')
        if not db.at_set_shared_claim(position_id,claim_id,self.epoch,account):
            raise CapacityError('local shared reservation persistence failed')
        return dict(claim_id=claim_id,epoch=self.epoch,identity=identity)

    def begin(self,claim,positions,equity,captured_at):
        if not self.enabled:
            return
        if not isinstance(claim,dict) or claim.get('epoch')!=self.epoch:
            raise CapacityError('shared claim epoch changed')
        claim_id=opaque(claim.get('claim_id'))
        answer=self.transport.call('begin',dict(owner=self.owner,epoch=self.epoch,claim_id=claim_id,
            identity=claim.get('identity'),
            snapshot=self._snapshot(positions,equity,captured_at)))
        if (answer.get('allowed') is not True or answer.get('claim_id')!=claim_id
                or answer.get('epoch')!=self.epoch or answer.get('state')!='SUBMITTING'):
            raise CapacityError('shared submission refused')
