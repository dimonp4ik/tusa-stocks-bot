#!/usr/bin/env python3
"""Does the LIVE MONITOR agree with the model and the shadow?

The exit rule exists three times (see tools_exit_parity.py, which compares the
other two). This one covers the implementation that actually decides real
money: main._check_open_signals.

It was left out of that comparison because it needs a database and a live
feed. Both can be supplied: a temp DB holds one OPEN signal, the feed
functions are replaced with a synthetic series, and the monitor is stepped
until the signal reaches a final status. That makes the third engine testable
on exactly the candles the other two see.

Run: python tools_monitor_parity.py [runs]
"""
import os, sys, tempfile, sqlite3, time, random, importlib

os.environ.setdefault("DB_PATH", os.path.join(tempfile.mkdtemp(), "parity.db"))
import src.db as db          # noqa: E402
import backtest as bt        # noqa: E402
import config as C           # noqa: E402


def _series(n, start, vol):
    o = h = l = c = None
    px = start
    o, h, l, c = [], [], [], []
    for _ in range(n):
        op = px
        px = px * (1.0 + random.gauss(0.0, vol))
        hi = max(op, px) * (1.0 + abs(random.gauss(0.0, vol / 2)))
        lo = min(op, px) * (1.0 - abs(random.gauss(0.0, vol / 2)))
        o.append(op); h.append(hi); l.append(lo); c.append(px)
    return o, h, l, c


def _run_monitor(main, dbmod, direction, entry, tp1, tp2, sl, atr, candles, max_steps=40):
    """Step the live monitor until the signal closes. Returns its final status."""
    now = candles["time"][-1]
    cx = sqlite3.connect(dbmod.DB_PATH)
    cx.execute("DELETE FROM signals")
    cx.execute(
        "INSERT INTO signals (symbol,direction,entry_price,tp1,tp2,sl,opened_at,"
        "status,confidence,mtf_score,atr) VALUES (?,?,?,?,?,?,?,'OPEN','MEDIUM',15,?)",
        ("PARITYUSDT", direction, entry, tp1, tp2, sl, candles["time"][0], atr))
    cx.commit(); cx.close()
    for _ in range(max_steps):
        main._check_open_signals()
        cx = sqlite3.connect(dbmod.DB_PATH); cx.row_factory = sqlite3.Row
        row = cx.execute("SELECT status FROM signals WHERE symbol='PARITYUSDT'").fetchone()
        cx.close()
        st = row["status"] if row else None
        if st in (None, "OPEN", "TP1_PARTIAL"):
            if st == "OPEN":
                return "OPEN"          # nothing happened at all; caller skips
            continue                    # TP1 booked, keep stepping for the runner
        return st
    return "TP1_PARTIAL"


def main_() -> int:
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    random.seed(11)
    import main as m
    m.get_klines_xperp = lambda *a, **k: _CUR[0]
    m.get_klines = lambda *a, **k: _CUR[0]
    m.send_signal_update = lambda *a, **k: True
    try:
        m.autotrader.on_signal_status = lambda *a, **k: None
    except Exception:
        pass

    agree = dis = skipped = 0
    kinds = {}
    for k in range(runs):
        n = 60
        o, h, l, c = _series(n, 100.0, 0.006)
        now = time.time()
        candles = {"open": o, "high": h, "low": l, "close": c,
                   "time": [now - (n - i) * 900 for i in range(n)],
                   "volume": [1] * n, "confirmed": [1] * (n - 1) + [0]}
        _CUR[0] = candles
        entry, atr = o[0], o[0] * 0.004
        direction = "LONG" if k % 2 == 0 else "SHORT"
        tp1, tp2, sl = bt.calculate_tp_sl_local(
            entry, direction, atr=atr,
            recent_high=max(h[:5]), recent_low=min(l[:5]))

        live = _run_monitor(m, db, direction, entry, tp1, tp2, sl, atr, candles)
        if live in ("OPEN", "TP1_PARTIAL"):
            skipped += 1
            continue
        setup = {"direction": direction, "current_price": entry, "atr": atr,
                 "recent_high": max(h[:5]), "recent_low": min(l[:5]),
                 "entry_low": entry * 0.999, "entry_high": entry * 1.001,
                 "session": "OPEN", "volume_ratio": 2.0,
                 "eff_ratio": 0.4, "vol_atr_pct": 0.004}
        trade = bt.simulate_trade_direct(
            # Full series, not a 48-bar window: the monitor sees every candle,
            # so capping the model here made it report EXPIRED where the live
            # engine had already stopped out. That disagreement was the
            # harness, not the bots.
            "PARITY", setup, candles, 0, len(candles["close"]) - 1, 0.0005, 0.0005,
            exit_policy="trail", trail_atr_mult=max(0.0, float(C.TRAIL_ATR_MULT)))
        if trade is None:
            skipped += 1
            continue
        # Same categorical collapse the sister tool uses: reaching TP1 and not
        # running to TP2 is one outcome, however each engine labels it.
        def _norm(x):
            if x in ("TP1", "TRAIL", "TP1_TRAIL", "TP1_HIT", "BREAKEVEN"):
                return "TP1"
            if x in ("SL", "SL_HIT"):
                return "SL"
            if x in ("TP2", "TP2_HIT"):
                return "TP2"
            return str(x)
        a, b = _norm(live), _norm(trade.outcome)
        if a == b:
            agree += 1
        else:
            dis += 1
            kinds[f"live={a} model={b}"] = kinds.get(f"live={a} model={b}", 0) + 1

    total = agree + dis
    if total == 0:
        print("АВАРИЯ: ни одной сравнимой пары — проверь стенд")
        return 1
    print(f"сравнено пар: {total}  (пропущено {skipped})")
    print(f"  совпало:   {agree} ({100*agree/total:.1f}%)")
    print(f"  разошлось: {dis} ({100*dis/total:.1f}%)")
    for kind, cnt in sorted(kinds.items(), key=lambda x: -x[1]):
        print(f"    {kind}: {cnt}")
    return 0


_CUR = [None]

if __name__ == "__main__":
    raise SystemExit(main_())
