"""Authenticated bounded model-trade export. No users, keys or broker orders."""
import hmac
import time
from src.shared_capacity_transport import valid_token

SIGNAL_FIELDS=frozenset('id symbol direction entry_price tp1 tp2 sl status opened_at '
    'tp1_hit_at closed_at exit_price realized_r source autotrade_eligible atr '
    'strategy_tp1_close_frac strategy_runner_mode strategy_runner_target_r '
    'strategy_bar_seconds strategy_entry_bar_ts strategy_stop_on_close strategy_max_bars '
    'strategy_expire_at runner_activation_bar_ts trend_1h trend_4h rsi volume_ratio '
    'session eff_ratio vol_atr_pct sl_xperp_only'.split())
SETUP_FIELDS=frozenset('id ts symbol direction entry_price tp1 tp2 sl sent source signal_id '
    'signal_bar_ts block_reason outcome net_r resolved_ts decision_spread_bps '
    'decision_depth_usd decision_imbalance book_spread_bps book_depth_usd '
    'book_imbalance book_sample_kind book_observed_ts decision_context_json'.split())


def register_trade_audit_api(app,token):
    if not token:
        return
    try:
        valid_token(token)
    except ValueError:
        return
    from flask import Blueprint,request,jsonify
    from src import db
    api=Blueprint('trade_audit',__name__)

    @api.route('/capacity-summary',methods=['POST'])
    def capacity_summary():
        auth=request.headers.get('Authorization','')
        if not auth.isascii() or not hmac.compare_digest(auth,'Bearer '+token):
            return jsonify(ok=False,error='unauthorized'),401
        if request.content_length is None or request.content_length>1000:
            return jsonify(ok=False,error='request_size_invalid'),413
        if request.get_json(silent=True)!={}:
            return jsonify(ok=False,error='invalid_request'),400
        try:
            with db._conn() as c:
                # Counts only. No user IDs, keys, account hashes, private order
                # IDs or free-form exchange errors leave the service.
                rows=c.execute("""SELECT p.status,p.opened_at,
                    COALESCE(LENGTH(p.account_key),0)=64 AS account_bound,
                    p.client_order_id IS NOT NULL AND p.client_order_id!='' AS has_client_order,
                    p.entry_submit_started_at,
                    u.user_id IS NOT NULL AS owner_exists,
                    u.active AS owner_active,
                    u.allowed AS owner_allowed,
                    p.status='OPEN' AND p.entry_submit_started_at IS NULL
                      AND COALESCE(p.account_key,'')=''
                      AND COALESCE(p.client_order_id,'')=''
                      AND COALESCE(p.shared_claim_id,'')=''
                      AND COALESCE(p.shared_epoch,'')=''
                      AND u.active=0 AS inactive_unbound_legacy,
                    COALESCE(u.allowed,0)=1 AND COALESCE(u.active,0)=1
                      AND COALESCE(u.api_key_enc,'')!=''
                      AND COALESCE(u.api_secret_enc,'')!=''
                      AND COALESCE(u.passphrase_enc,'')!='' AS owner_connected
                    FROM autotrade_positions p LEFT JOIN autotrade_users u ON u.user_id=p.user_id
                    WHERE p.status IN ('OPEN','ENTRY_PENDING') OR
                      (p.shared_claim_id IS NOT NULL AND COALESCE(p.shared_released_at,0)=0)
                    ORDER BY p.id""").fetchall()
                active=c.execute("""SELECT COUNT(*) FROM autotrade_users WHERE allowed=1
                    AND active=1 AND api_key_enc IS NOT NULL""").fetchone()[0]
            eligible=[r for r in rows if not (r['status']=='ENTRY_PENDING' and r['entry_submit_started_at']==0)]
            unknown=[r for r in eligible if not r['account_bound']]
            first=unknown[0] if unknown else None
            return jsonify(ok=True,summary=dict(active_connected_traders=active,
                local_intents=len(rows),synchronization_intents=len(eligible),
                unknown_account_intents=len(unknown),
                unknown_account_with_connected_owner=sum(bool(r['owner_connected']) for r in unknown),
                unknown_account_missing_owner=sum(not r['owner_exists'] for r in unknown),
                unknown_account_without_order_id=sum(not r['has_client_order'] for r in unknown),
                unknown_account_owner_inactive=sum(r['owner_active']==0 for r in unknown),
                unknown_account_owner_disallowed=sum(r['owner_allowed']==0 for r in unknown),
                inactive_unbound_legacy_rows=sum(bool(r['inactive_unbound_legacy']) for r in rows),
                first_unknown_owner_connected=bool(first['owner_connected']) if first else None,
                first_unknown_age_over_seven_days=(time.time()-float(first['opened_at'])>7*86400) if first else None,
                statuses={state:sum(r['status']==state for r in rows) for state in ('OPEN','ENTRY_PENDING','CLOSED')}))
        except Exception:
            return jsonify(ok=False,error='audit_unavailable'),503

    @api.route('/signals',methods=['POST'])
    def signals():
        auth=request.headers.get('Authorization','')
        if not auth.isascii() or not hmac.compare_digest(auth,'Bearer '+token):
            return jsonify(ok=False,error='unauthorized'),401
        if request.content_length is None or request.content_length>10000:
            return jsonify(ok=False,error='request_size_invalid'),413
        body=request.get_json(silent=True)
        if not isinstance(body,dict) or set(body)-{'ids','limit'}:
            return jsonify(ok=False,error='invalid_request'),400
        ids=body.get('ids');limit=body.get('limit',20)
        if (not isinstance(limit,int) or isinstance(limit,bool) or not 1<=limit<=30
                or (ids is not None and (not isinstance(ids,list) or not 1<=len(ids)<=30
                    or any(not isinstance(i,int) or isinstance(i,bool) or i<1 for i in ids)))):
            return jsonify(ok=False,error='invalid_request'),400
        try:
            with db._conn() as c:
                if ids is None:
                    rows=c.execute('SELECT * FROM signals ORDER BY id DESC LIMIT ?',(limit,)).fetchall()
                else:
                    rows=c.execute('SELECT * FROM signals WHERE id IN ('+','.join('?' for _ in ids)+') '
                        'ORDER BY id DESC',ids).fetchall()
                selected=[int(r['id']) for r in rows]
                setups=(c.execute('SELECT * FROM setup_log WHERE signal_id IN ('
                    +','.join('?' for _ in selected)+') ORDER BY id DESC LIMIT 150',selected).fetchall()
                    if selected else [])
            return jsonify(ok=True,
                signals=[{k:r[k] for k in r.keys() if k in SIGNAL_FIELDS} for r in rows],
                setups=[{k:r[k] for k in r.keys() if k in SETUP_FIELDS} for r in setups])
        except Exception:
            return jsonify(ok=False,error='audit_unavailable'),503

    app.register_blueprint(api,url_prefix='/internal/trade-audit/v1')
