"""Verify closed positions against their actual entry and protective orders."""

import json
import logging
import math
import time

from src import db
from src import okx_trader as okx
from src.entry_execution import confirmed_last_fill_ms


log = logging.getLogger(__name__)
LOOKBACK_SECONDS = 3 * 86400
ORDER_TIMEOUT_SECONDS = 2


def finite_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def protective_order_fill(order, position, order_id, *, entry_ms, latest_ms):
    close_side = 'sell' if position['direction'] == 'LONG' else 'buy'
    if (not isinstance(order, dict) or order.get('ordId') != order_id
            or order.get('instId') != position['inst_id'] or order.get('side') != close_side
            or order.get('tdMode') != 'isolated' or order.get('ordType') != 'market'
            or str(order.get('reduceOnly')).lower() != 'true'
            or order.get('state') not in ('filled', 'canceled')):
        return None
    size = finite_number(order.get('accFillSz'))
    if size == 0 and order['state'] == 'canceled':
        # Split protection can spawn a child that never executed.
        return 0., 0., 0., 0
    price = finite_number(order.get('avgPx'))
    pnl = finite_number(order.get('pnl'))
    stamp = str(order.get('fillTime'))
    if (size is None or size <= 0 or price is None or price <= 0 or pnl is None
            or order.get('feeCcy') != 'USDC' or order.get('ccy') not in (None, '', 'USDC')
            or len(stamp) != 13 or not stamp.isascii() or not stamp.isdecimal()
            or not entry_ms <= int(stamp) <= latest_ms):
        return None
    return size, price, pnl, int(stamp)


def get_exit_receipt(position, creds, broker, now):
    inst = position['inst_id']
    deadline = dict(timeout=ORDER_TIMEOUT_SECONDS)
    ok, entry = broker.get_order_by_client_id(creds, inst, position['client_order_id'], **deadline)
    if not ok:
        return None
    entry_ms = confirmed_last_fill_ms(entry, position, now)
    if entry_ms is None or entry_ms != position['entry_last_fill_ms']:
        return None
    entry_size = finite_number(entry['accFillSz'])
    entry_price = finite_number(entry.get('avgPx'))
    saved_size = finite_number(position.get('entry_filled_sz'))
    if (entry_price is None or entry_price <= 0 or entry_size is None
            or saved_size is not None and not math.isclose(entry_size, saved_size, rel_tol=1e-9)):
        return None

    # The history links the exact OCO and TP1 algos to their child market orders.
    runner_ids = broker.get_protection_order_ids(creds, inst, position['sl_algo_id'], **deadline)
    tp1_ids = ()
    if position.get('tp1_algo_id'):
        tp1_ids = broker.get_protection_order_ids(
            creds, inst, position['tp1_algo_id'], ord_type='conditional', **deadline)
    if runner_ids is None or tp1_ids is None or set(runner_ids) & set(tp1_ids):
        return None
    order_ids = runner_ids + tp1_ids
    if not 0 < len(order_ids) <= 4 or entry.get('ordId') in order_ids:
        return None

    latest_ms = round(min(now, float(position['closed_at'])) * 1000) + 2000
    fills = {}
    for order_id in order_ids:
        ok, order = broker.get_order_by_exchange_id(creds, inst, order_id, **deadline)
        receipt = protective_order_fill(order, position, order_id, entry_ms=entry_ms, latest_ms=latest_ms) if ok else None
        if receipt is None:
            return None
        fills[order_id] = receipt
    closed_size = math.fsum(receipt[0] for receipt in fills.values())
    tp1_size = math.fsum(fills[order_id][0] for order_id in tp1_ids)
    planned_tp1 = finite_number(position.get('tp1_sz'))
    if (closed_size <= 0 or not math.isclose(closed_size, entry_size, rel_tol=1e-9)
            or tp1_size > 0 and (planned_tp1 is None or (
                tp1_size > planned_tp1 and not math.isclose(tp1_size, planned_tp1, rel_tol=1e-9)))):
        return None
    exit_price = math.fsum(size * price for size, price, _, _ in fills.values()) / closed_size
    pnl = math.fsum(receipt[2] for receipt in fills.values())
    runner_size = closed_size - tp1_size
    runner_exit_price = (math.fsum(fills[order_id][0] * fills[order_id][1] for order_id in runner_ids) / runner_size
                         if runner_size > 0 else None)
    tp1_exit_price = (math.fsum(fills[order_id][0] * fills[order_id][1] for order_id in tp1_ids) / tp1_size
                      if tp1_size > 0 else None)
    return dict(entry_px=entry_price, exit_px=exit_price, contracts=closed_size,
                tp1_contracts=tp1_size, tp1_exit_px=tp1_exit_price, runner_exit_px=runner_exit_price,
                pnl_usdc=pnl, last_exit_ms=max(receipt[3] for receipt in fills.values()))


def audit_exit_fills():
    from src.autotrader import _creds_of

    for position in db.at_exit_fill_checks(time.time()):
        try:
            user = db.at_get(position['user_id'])
            creds = _creds_of(user) if user and user.get('allowed') else None
            receipt = get_exit_receipt(position, creds, okx, time.time()) if creds else None
            if db.at_record_exit_fill_check(position['id'], position['client_order_id'], time.time(), receipt) and receipt:
                pnl = receipt['pnl_usdc']
                outcome = 'positive' if pnl > 0 else 'negative' if pnl < 0 else 'flat'
                log.info('exit receipt verified signal=%s inst=%s broker_pnl=%s signal_status=%s tp1_filled=%s position=%s',
                         position['signal_id'], position['inst_id'], outcome, position['signal_status'],
                         receipt['tp1_contracts'] > 0, position['id'])
        except (ValueError, TypeError, KeyError, ArithmeticError, OSError) as exc:
            log.warning('Exit fill verification deferred (%s)', type(exc).__name__)


def cash_status(user_id, now):
    rows = db.at_exit_receipts_for_user(user_id, now - LOOKBACK_SECONDS)
    if not rows:
        return ''
    verified = [(row['inst_id'], json.loads(row['exit_receipt_json']))
                for row in rows if row['exit_receipt_json'] is not None]
    if not verified:
        return f'OKX за 72 ч · сверено 0 из {len(rows)} закрытий.\n'
    pnl = math.fsum(row['pnl_usdc'] for _, row in verified)
    partial = sum(row['tp1_contracts'] > 0 for _, row in verified)
    instrument, last = max(verified, key=lambda item: item[1]['last_exit_ms'])
    last_symbol = instrument.split('-')[0]
    runner_price = last["runner_exit_px"]
    runner_line = f" · выход остатка {runner_price:.8g}" if last["tp1_contracts"] > 0 and runner_price else ""
    return (f'OKX за 72 ч · сверено {len(verified)} из {len(rows)} закрытий · TP1 исполнен в {partial}.\n'
            f'Результат сверенных сделок: *{pnl:+.2f} USDC* до комиссий и фандинга.\n'
            f'Последняя сверка · {last_symbol} · TP1 {last["tp1_contracts"]:g} '
            f'из {last["contracts"]:g} контр. · средний выход {last["exit_px"]:.8g}{runner_line}.\n')
