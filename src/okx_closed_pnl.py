"""Reconcile closed futures PnL with the bot's own exchange entry order IDs."""

from collections import defaultdict
from decimal import Decimal, InvalidOperation
import logging
import re
import threading
import time


_HISTORY = "/api/v5/account/positions-history"
_FILLS = "/api/v5/trade/fills-history"
_PAGE = 100
_MAX_PAGES = 8
_DAY_MS = 86_400_000
_TOLERANCE_MS = 1000
_CLOSE_TYPES = {"2", "3", "5"}
_PARTIAL_TYPES = {"1", "4"}
_CLOSING = {"long": {"5", "100", "104", "112", "125"},
            "short": {"6", "101", "105", "113", "126"}}
_CURRENCY = re.compile(r"[A-Z][A-Z0-9]{1,9}\Z")
_ACTIVE = set()
_LOCK = threading.Lock()
_LOG = logging.getLogger(__name__)


def _number(value):
    number = Decimal(str(value))
    if not number.is_finite():
        raise ValueError("nonfinite exchange value")
    return number


def _page(request, path, extra):
    ok, rows = request(path, {"instType": "FUTURES", "limit": str(_PAGE), **extra})
    if not ok or not isinstance(rows, list) or any(not isinstance(r, dict) for r in rows):
        raise ValueError("exchange history unavailable")
    return rows


def _positions(request, since, now):
    result = []
    cursor = None
    for _ in range(_MAX_PAGES):
        rows = _page(request, _HISTORY, {"after": str(cursor)} if cursor is not None else {})
        stamps = [int(r["uTime"]) for r in rows]
        if (any(t <= 0 for t in stamps) or any(a < b for a, b in zip(stamps, stamps[1:]))
                or (cursor is not None and any(t >= cursor for t in stamps))):
            raise ValueError("invalid position pagination")
        result.extend(r for r, at in zip(rows, stamps) if since <= at <= now)
        if len(rows) < _PAGE or stamps[-1] < since:
            return result
        cursor = stamps[-1]
    raise ValueError("position history exceeds report page limit")


def _fills(request, begin, end):
    query = {"begin": str(begin), "end": str(end)}
    result = []
    cursor = None
    for _ in range(_MAX_PAGES):
        rows = _page(request, _FILLS, {**query, **({"after": str(cursor)} if cursor is not None else {})})
        ids = [int(r["billId"]) for r in rows]
        if (len(rows) > _PAGE or any(i <= 0 for i in ids) or any(a <= b for a, b in zip(ids, ids[1:]))
                or (cursor is not None and any(i >= cursor for i in ids))):
            raise ValueError("invalid fill pagination")
        result.extend(rows)
        if len(rows) < _PAGE:
            return result
        cursor = ids[-1]
    raise ValueError("fill history exceeds report page limit")


def _lifetime(row):
    opened, closed = int(row["cTime"]), int(row["uTime"])
    instrument, pos_id, side, direction = (row[key] for key in ("instId", "posId", "posSide", "direction"))
    currency = row["ccy"]
    if (not instrument or not pos_id or not 0 < opened <= closed or side not in {"net", "long", "short"}
            or direction not in {"long", "short"} or not _CURRENCY.fullmatch(currency)):
        raise ValueError("unverifiable position identity")
    return instrument, pos_id, opened, closed, side, direction, currency


def _owns_lifetime(row, lifetime, fills, registered):
    instrument, _, opened, closed, _, direction, _ = lifetime
    if row.get("mgnMode") != "isolated":
        return False
    opening_type = "3" if direction == "long" else "4"
    opening_side = "buy" if direction == "long" else "sell"
    entry_ids, order_ids = set(), set()
    opened_size = closed_size = remaining = Decimal(0)
    for executed in fills:
        kind = executed["subType"]
        side = executed["side"]
        size = _number(executed["fillSz"])
        if size <= 0:
            return False
        if kind == opening_type and side == opening_side and not closed_size:
            entry_ids.add(executed.get("clOrdId"))
            order_ids.add(executed.get("ordId"))
            opened_size += size
            remaining += size
        elif kind in _CLOSING[direction] and side != opening_side and side in {"buy", "sell"}:
            closed_size += size
            remaining -= size
            if remaining < 0:
                return False
        else:
            return False
    if len(entry_ids) != 1 or len(order_ids) != 1 or not all(order_ids):
        return False
    client_id = next(iter(entry_ids))
    if registered.get(client_id) != (instrument, direction.upper()):
        return False
    return (remaining == 0 and opened_size > 0 and opened_size == _number(row["openMaxPos"])
            and closed_size == _number(row["closeTotalPos"]))


def reconcile(request, registered, *, now_ms):
    """Read the last 30 days. Ambiguous lifetimes never enter the per-bot sum."""
    since = now_ms - 30 * _DAY_MS
    try:
        rows = _positions(request, since, now_ms)
        lifetimes = []
        seen = set()
        ignored = 0
        account = defaultdict(Decimal)
        for row in rows:
            if row.get("type") in _PARTIAL_TYPES:
                ignored += 1
                continue
            if row.get("type") not in _CLOSE_TYPES or row.get("instType") != "FUTURES":
                raise ValueError("unknown closed-position type")
            life = _lifetime(row)
            identity = life[:3]
            if identity in seen:
                raise ValueError("duplicate position lifetime")
            seen.add(identity)
            opened_size, closed_size = _number(row["openMaxPos"]), _number(row["closeTotalPos"])
            if opened_size <= 0 or closed_size <= 0:
                raise ValueError("invalid closed position quantities")
            if closed_size < opened_size:
                ignored += 1
                continue
            amount = _number(row["realizedPnl"])
            lifetimes.append((row, life, amount))
            account[life[-1]] += amount
    except (ValueError, InvalidOperation, OverflowError, KeyError, TypeError):
        return None

    account_result = dict(account=dict(account), positions=len(lifetimes), partial=ignored,
                          bot={}, matched=0, attribution_available=False)
    if not lifetimes:
        return dict(account_result, attribution_available=True)
    if not registered:
        return account_result
    try:
        minimum = now_ms - 89 * _DAY_MS
        registered_markets = set(registered.values())
        possible = [(row, life, value) for row, life, value in lifetimes
                    if life[2]-_TOLERANCE_MS >= minimum
                    and (life[0], life[5].upper()) in registered_markets]
        if not possible:
            return account_result
        history = _fills(request, min(life[2] for _, life, _ in possible)-_TOLERANCE_MS, now_ms)
        grouped = defaultdict(list)
        for fill in history:
            if fill["instType"] != "FUTURES":
                raise ValueError("wrong market in execution history")
            grouped[fill["instId"], fill["posSide"]].append(fill)
        bot = defaultdict(Decimal)
        matched = 0
        used = set()
        for row, life, value in possible:
            instrument, _, opened, closed, side, _, currency = life
            fills = [f for f in grouped[instrument, side]
                     if opened-_TOLERANCE_MS <= int(f["fillTime"]) <= min(closed+_TOLERANCE_MS, now_ms)]
            fills.sort(key=lambda f: (int(f["fillTime"]), int(f["billId"])))
            ids = {f["billId"] for f in fills}
            if not _owns_lifetime(row, life, fills, registered):
                continue
            if ids & used:
                raise ValueError("one exchange fill matched two lifetimes")
            used.update(ids)
            matched += 1
            bot[currency] += value
        return dict(account_result, bot=dict(bot), matched=matched, attribution_available=True)
    except (ValueError, InvalidOperation, OverflowError, KeyError, TypeError):
        return account_result


def format_closed_pnl(report):
    lines = ["💵 *Реальный результат OKX · 30 дней*"]
    if report is None:
        return "\n\n".join([*lines, "OKX пока не подтвердил полную историю закрытых позиций."])

    def amounts(values):
        return " · ".join(f"{amount:+.4f} {currency}" for currency, amount in sorted(values.items()))

    count = report["positions"]
    account = amounts(report["account"])
    lines.append(f"*Общий счёт · закрытые фьючерсы* — {account} · позиций {count}" if count else
                 "*Общий счёт* — за этот период закрытых фьючерсных позиций нет.")
    if report["matched"]:
        lines.append(f"*Входы этого бота* — {amounts(report['bot'])} · позиций {report['matched']}")
        lines.append("Результат этого бота уже входит в итог общего счёта.")
    elif count:
        lines.append("Результат входов этого бота пока не подтверждён исполнениями OKX.")
    if report["partial"]:
        lines.append(f"Частичные закрытия открытых позиций: {report['partial']}.")
    return "\n\n".join(lines)


def start_user_report(user_id, connection, deliver):
    """Queue a read-only report without holding the Telegram webhook open."""
    with _LOCK:
        if user_id in _ACTIVE:
            return False
        _ACTIVE.add(user_id)

    def run():
        try:
            from src import okx_trader
            from src.autotrader import _creds_of
            from src.db import at_entry_order_identities, at_get

            credentials = _creds_of(connection)
            if credentials is None:
                deliver("История OKX недоступна. Проверь подключение в «Автотрейдинге».")
                return

            def request(path, params):
                return okx_trader._request(credentials, "GET", path, params=params, timeout=5)

            report = reconcile(request, at_entry_order_identities(user_id), now_ms=int(time.time()*1000))
            current = at_get(user_id) or {}
            if not current.get("allowed") or any(current.get(key) != connection.get(key)
                    for key in ("api_key_enc", "api_secret_enc", "passphrase_enc")):
                return
            deliver(format_closed_pnl(report))
        except Exception as exc:
            _LOG.warning("Closed OKX report unavailable (%s)", type(exc).__name__)
            deliver(format_closed_pnl(None))
        finally:
            with _LOCK:
                _ACTIVE.discard(user_id)

    try:
        threading.Thread(target=run, name="okx-closed-pnl", daemon=True).start()
    except RuntimeError:
        with _LOCK:
            _ACTIVE.discard(user_id)
        return False
    return True
