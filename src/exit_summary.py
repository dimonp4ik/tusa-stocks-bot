"""Read-only, order-bound full protective exit summary; no account PnL guesses."""
import math


def positive(value):
    try:
        number=float(value)
        return number if math.isfinite(number) and number>0 else None
    except (TypeError,ValueError,OverflowError):
        return None


def confirmed_partial_tp1(position, remaining, creds, broker):
    """Attribute a reduction only to verified fills of this position's TP1."""
    tracked = positive(position.get('sz'))
    original = positive(position.get('entry_filled_sz')) or tracked
    planned = positive(position.get('tp1_sz'))
    opened = positive(position.get('opened_at'))
    remaining = positive(remaining)
    direction = position.get('direction')
    algo_id = position.get('tp1_algo_id')
    if (None in (tracked, original, planned, opened, remaining) or not algo_id
            or direction not in ('LONG', 'SHORT') or not remaining < tracked <= original):
        return None
    closed = original - remaining
    if closed > planned and not math.isclose(closed, planned, rel_tol=1e-9):
        return None
    instrument = position['inst_id']
    order_ids = broker.get_protection_order_ids(creds, instrument, algo_id, ord_type='conditional')
    if not order_ids:
        return None
    price = positive(broker.get_last_fill_px(
        creds, instrument, since_ts=opened, side='sell' if direction == 'LONG' else 'buy',
        order_ids=order_ids, expected_size=closed))
    return dict(closed=closed, original=original, remaining=remaining, price=price) if price else None


def protective_exit_summary(position,creds,broker):
    direction=str(position.get('direction') or '').upper()
    if direction not in ('LONG','SHORT') or not position.get('sl_algo_id'):return None
    original=positive(position.get('entry_filled_sz'));entry=None
    client_id=position.get('client_order_id')
    if client_id:
        ok,order=broker.get_order_by_client_id(creds,position['inst_id'],client_id)
        if ok:
            if (not isinstance(order,dict) or order.get('instId')!=position['inst_id']
                    or order.get('clOrdId')!=client_id
                    or order.get('side')!=('buy' if direction=='LONG' else 'sell')):
                return None
            # A terminal actual entry, not the requested size or a quote saved
            # when the position's average fill was unavailable.
            filled=positive(order.get('accFillSz'))
            if order.get('state') not in ('filled','canceled') or filled is None:return None
            if original is not None and not math.isclose(original,filled,rel_tol=1e-9):return None
            original=filled;entry=positive(order.get('avgPx'))
    if original is None:return None
    groups=[];tp1_filled=False
    for field,kind in (('sl_algo_id','oco'),('tp1_algo_id','conditional')):
        algo_id=position.get(field)
        if not algo_id:continue
        ids=broker.get_protection_order_ids(creds,position['inst_id'],algo_id,ord_type=kind)
        if ids is None:return None
        groups.extend(ids)
        if field=='tp1_algo_id':tp1_filled=bool(ids)
    order_ids=tuple(dict.fromkeys(groups))
    if not order_ids:return None
    exit_px=positive(broker.get_last_fill_px(creds,position['inst_id'],since_ts=position.get('opened_at'),
        side='sell' if direction=='LONG' else 'buy',order_ids=order_ids,expected_size=original))
    if exit_px is None:return None
    sign=1 if direction=='LONG' else -1
    return dict(exit_px=exit_px,entry_px=entry,contracts=original,tp1_filled=tp1_filled,
        price_move_pct=sign*(exit_px/entry-1)*100 if entry is not None else None)


def format_exit_summary(summary,fmt_price):
    if summary is None:
        return 'Полный результат по исполнениям пока не подтверждён. Средняя цена и доходность не рассчитаны.'
    lines=[f"Средняя цена всех закрытий: ≈`{fmt_price(summary['exit_px'])}`.",
        f"Подтверждённый закрытый объём: {summary['contracts']:g} контр."]
    if summary['price_move_pct'] is not None:
        lines.append(f"Вход по ордеру: `{fmt_price(summary['entry_px'])}` · движение всей позиции *{summary['price_move_pct']:+.3f}%*.")
    else:
        lines.append('Цена входа по ордеру не подтверждена; доходность не рассчитана.')
    if summary['tp1_filled']:lines.append('TP1 включён в общий результат.')
    lines.append('Это ценовой результат всей позиции, а не процент изменения депозита.')
    return '\n'.join(lines)
