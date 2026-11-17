"""Authenticated, bounded export of isolated public-market research evidence."""
import hmac
from src.shared_capacity_transport import valid_token


def register_structure_observation_api(app,token,store_provider):
    if not token:return
    try:valid_token(token)
    except ValueError:return
    from flask import Blueprint,request,jsonify
    from src.stock_structure_observer import PROFILE
    api=Blueprint('structure_observer_audit',__name__)

    @api.route('/export',methods=['POST'])
    def export():
        auth=request.headers.get('Authorization','')
        if not auth.isascii() or not hmac.compare_digest(auth,'Bearer '+token):
            return jsonify(ok=False,error='unauthorized'),401
        if request.content_length is None or request.content_length>1000:
            return jsonify(ok=False,error='request_size_invalid'),413
        body=request.get_json(silent=True)
        if not isinstance(body,dict) or set(body)-{'after_bucket','limit'}:
            return jsonify(ok=False,error='invalid_request'),400
        after=body.get('after_bucket',0);limit=body.get('limit',1)
        if (not isinstance(after,int) or isinstance(after,bool) or after<0 or after%900
                or not isinstance(limit,int) or isinstance(limit,bool) or not 1<=limit<=4):
            return jsonify(ok=False,error='invalid_request'),400
        try:
            store=store_provider()
            if store is None:return jsonify(ok=False,error='archive_not_started'),503
            return jsonify(ok=True,export=store.export(PROFILE,after,limit))
        except Exception:
            return jsonify(ok=False,error='archive_unavailable'),503

    app.register_blueprint(api,url_prefix='/internal/structure-observer/v1')
