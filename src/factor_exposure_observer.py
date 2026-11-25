"""Observe shared BTC exposure after a position has exchange protection."""

from functools import lru_cache
import logging
import math
from queue import Full, Queue
import re
import threading

from src.binance_client import _okx_get, get_xperp_instruments


_LOG = logging.getLogger(__name__)
_BAR = 900
_WINDOW = 96
_INSTRUMENT = re.compile(r'[A-Z0-9_-]{1,64}')
_VERSION = 'btc_two_phase_v1'
_JOBS = Queue(maxsize=24)
_START_LOCK = threading.Lock()
_worker = None


def closed_returns(candles, epoch):
    """Align traded 15-minute returns to the instant before the signal bar."""
    first = epoch - (_WINDOW + 1) * _BAR
    bars = {}
    for row in candles:
        if len(row) < 9 or row[8] != '1':
            continue
        stamp_ms = int(row[0])
        if stamp_ms % (_BAR * 1000):
            raise ValueError('unaligned public candle')
        stamp = stamp_ms // 1000
        if not first <= stamp < epoch:
            continue
        value = float(row[4]), float(row[5])
        if (not all(math.isfinite(x) for x in value) or value[0] <= 0 or value[1] < 0
                or bars.setdefault(stamp, value) != value):
            raise ValueError('invalid or conflicting public candle')
    result = {}
    for stamp in range(first + _BAR, epoch, _BAR):
        earlier, current = bars.get(stamp - _BAR), bars.get(stamp)
        if earlier and current and min(earlier[1], current[1]) > 0:
            result[stamp] = math.log(current[0]) - math.log(earlier[0])
    return result


@lru_cache(maxsize=256)
def _public_returns(instrument, epoch):
    candles = _okx_get('/api/v5/market/history-candles', dict(
        instId=instrument, bar='15m', after=str(epoch * 1000), limit=_WINDOW + 1), timeout=3)['data']
    return closed_returns(candles, epoch)


def correlation(left, right):
    common = sorted(left.keys() & right.keys())
    if len(common) < 32:
        return None
    a, b = [left[k] for k in common], [right[k] for k in common]
    mean_a, mean_b = math.fsum(a) / len(a), math.fsum(b) / len(b)
    ac, bc = [v - mean_a for v in a], [v - mean_b for v in b]
    denominator = math.sqrt(math.fsum(v*v for v in ac) * math.fsum(v*v for v in bc))
    if denominator == 0:
        return None
    value = math.fsum(x*y for x, y in zip(ac, bc, strict=True)) / denominator
    return max(-1., min(1., value))


def market_phase(returns, epoch, loading):
    split = epoch - 16 * _BAR
    recent = [v for k, v in returns.items() if k >= split]
    past = [v for k, v in returns.items() if k < split]
    if len(recent) < 12 or len(past) < 48:
        return 'unknown'
    if loading * math.fsum(recent) < 0:
        return 'opposed'
    if math.fsum(v*v for v in recent)/len(recent) > 1.25**2 * math.fsum(v*v for v in past)/len(past):
        return 'volatile'
    return 'calm'


def evaluate(instrument, side, held, epoch, reference, load):
    """Return only aggregate exposure evidence. Missing data stays visible."""
    reference_returns = load(reference, epoch)
    candidate = correlation(load(instrument, epoch), reference_returns)
    if candidate is None:
        return 'unknown', 'candidate_history', 0, len(held)
    if abs(candidate) < .65:
        return 'would_allow', 'low_btc_loading', 0, 0
    loading = (1 if side == 'LONG' else -1) * candidate
    matching = missing = 0
    for other, direction in held:
        if direction not in ('long', 'short'):
            missing += 1
            continue
        try:
            coefficient = correlation(load(other, epoch), reference_returns)
        except Exception:
            coefficient = None
        if coefficient is None:
            missing += 1
        elif abs(coefficient) >= .65 and loading * (1 if direction == 'long' else -1) * coefficient > 0:
            matching += 1
    if matching >= 2:
        phase = market_phase(reference_returns, epoch, loading)
        return ('would_allow' if phase == 'calm' else 'would_block'), phase, matching, missing
    if matching + missing >= 2:
        return 'unknown', 'incomplete_positions', matching, missing
    return 'would_allow', 'below_factor_limit', matching, missing


def _emit(position, signal, instrument, epoch, count, result):
    decision, basis, matching, missing = result
    _LOG.info('factor exposure shadow version=%s position=%d signal=%d inst=%s epoch=%d '
              'occupied=%d matching=%d unknown=%d decision=%s basis=%s',
              _VERSION, position, signal, instrument, epoch, count, matching, missing, decision, basis)


def _process():
    while True:
        position, signal, instrument, side, epoch, held = _JOBS.get()
        try:
            reference = get_xperp_instruments().get('BTC')
            if not reference:
                raise ValueError('BTC reference unavailable')
            result = evaluate(instrument, side, held, epoch, reference, _public_returns)
        except Exception:
            result = 'unknown', 'public_history', 0, len(held)
        try:
            _emit(position, signal, instrument, epoch, len(held), result)
        finally:
            _JOBS.task_done()


def record(position, signal, instrument, snapshot, snapshot_at):
    """Record a pre-order OKX snapshot after successful entry protection."""
    global _worker
    if not _INSTRUMENT.fullmatch(instrument):
        return
    signal_id = int(signal['id'])
    side = signal.get('direction')
    stamp = float(signal.get('strategy_entry_bar_ts') or signal.get('opened_at') or snapshot_at)
    if not math.isfinite(stamp) or not math.isfinite(snapshot_at):
        return
    epoch = int(stamp // _BAR) * _BAR
    count = int(snapshot['count'])
    members = tuple(snapshot['inst_ids'])
    same_bar = epoch <= snapshot_at < epoch + _BAR
    if not same_bar or side not in ('LONG','SHORT') or count != len(members):
        _emit(position, signal_id, instrument, epoch, count, ('unknown', 'snapshot_context', 0, count))
        return
    if count < 2:
        _emit(position, signal_id, instrument, epoch, count, ('would_allow', 'less_than_two_positions', 0, 0))
        return
    directions = snapshot.get('side_by_inst', {})
    held = tuple((member, directions.get(member) if _INSTRUMENT.fullmatch(member) else None)
                 for member in members)
    if not any(side in ('long', 'short') for _, side in held):
        _emit(position, signal_id, instrument, epoch, count, ('unknown', 'incomplete_positions', 0, count))
        return
    with _START_LOCK:
        if _worker is None or not _worker.is_alive():
            _worker = threading.Thread(target=_process, name='factor-exposure-shadow', daemon=True)
            _worker.start()
    try:
        _JOBS.put_nowait((position, signal_id, instrument, side, epoch, held))
    except Full:
        _emit(position, signal_id, instrument, epoch, count, ('unknown', 'queue_full', 0, count))
