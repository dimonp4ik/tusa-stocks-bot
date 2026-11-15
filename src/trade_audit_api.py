"""Authenticated bounded model-trade export. No users, keys or broker orders."""
import hmac
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
    'book_imbalance book_sample_kind book_observed_ts'.split())


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
