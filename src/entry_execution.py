"""Attach broker-reported entry fill times to protected positions."""

import logging
import math
import time

from src import db
from src import okx_trader as okx


log = logging.getLogger(__name__)


def confirmed_last_fill_ms(order, position, now):
    direction = position['direction']
    if direction not in ('LONG', 'SHORT') or not isinstance(order, dict):
        return None
    if (order.get('instId') != position['inst_id'] or order.get('clOrdId') != position['client_order_id']
            or not order.get('ordId') or order.get('ordType') != 'market' or order.get('tdMode') != 'isolated'
            or order.get('side') != ('buy' if direction == 'LONG' else 'sell')
            or order.get('state') not in ('filled', 'canceled')
            or str(order.get('reduceOnly', 'false')).lower() != 'false'):
        return None
    try:
        filled = float(order['accFillSz'])
        stamp = str(order['fillTime'])
        submitted = float(position['entry_submit_started_at'])
        if (not math.isfinite(filled) or filled <= 0 or not math.isfinite(submitted) or submitted <= 0
                or len(stamp) != 13 or not stamp.isascii() or not stamp.isdecimal()):
            return None
        filled_ms = int(stamp)
        return filled_ms if submitted*1000-2000 <= filled_ms <= now*1000+2000 else None
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


def audit_entry_fills():
    from src.autotrader import _creds_of

    for position in db.at_entry_fill_checks(time.time()):
        try:
            user = db.at_get(position['user_id'])
            creds = _creds_of(user) if user and user.get('allowed') else None
            fill_ms = None
            if creds:
                ok, order = okx.get_order_by_client_id(
                    creds, position['inst_id'], position['client_order_id'], timeout=2)
                if ok:
                    fill_ms = confirmed_last_fill_ms(order, position, time.time())
            recorded = db.at_record_entry_fill_check(position['id'], position['client_order_id'],
                                                     time.time(), fill_ms)
            if fill_ms is not None and recorded:
                tp1_time = position['tp1_hit_at']
                close_time = position['signal_closed_at']
                log.info('entry fill confirmed signal=%s inst=%s last_fill_ms=%s published_to_fill_ms=%s '
                         'submit_to_fill_ms=%s tp1_observed_before_last_fill=%s close_observed_before_last_fill=%s',
                         position['signal_id'], position['inst_id'], fill_ms,
                         round(fill_ms-position['published_at']*1000),
                         round(fill_ms-position['entry_submit_started_at']*1000),
                         bool(tp1_time and tp1_time*1000 < fill_ms), bool(close_time and close_time*1000 < fill_ms))
        except Exception as exc:
            log.warning('Entry fill verification deferred (%s)', type(exc).__name__)
