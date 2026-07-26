"""Replay backtest trades on ONE shared account.

Why this exists
---------------
backtest.py measures a single trade line: `max_drawdown_r()` walks trades one at
a time and adds each trade's full result as if it had closed before the next one
opened. The real account does not work that way. It holds several positions at
once, on alts that are ~0.7-0.9 correlated to BTC, so one adverse BTC move can
resolve five of them within the same minute. Losses that the single-line curve
spreads out — cushioned by winners that happened "between" them in entry order —
land simultaneously in reality, and the trough is deeper.

The single line also ignores every live guardrail:
  * the live bot never holds two positions on the SAME symbol, but backtest.py
    scans every bar and happily opens overlapping trades on one symbol
  * MAX_SAME_DIRECTION_POSITIONS caps correlated same-side exposure
  * MAX_SIGNALS_PER_SCAN caps new entries per scan
  * the kill switch halts new entries for the rest of the Riga day after
    KILL_SWITCH_SL_STREAK consecutive stop-outs

Those pull in opposite directions: the caps REMOVE trades (less drawdown, less
profit), while concurrency-aware accounting DEEPENS the trough. Only simulating
both together gives a number that means anything for the actual account.

Usage
-----
    python portfolio_sim.py <trades.csv> [--max-same-dir 4] [--no-caps]

Reads a CSV exported by `backtest.py --export-trades`.
"""

import argparse
import csv
import sys
from datetime import datetime, timezone

try:
    from zoneinfo import ZoneInfo
    _RIGA = ZoneInfo("Europe/Riga")
except Exception:  # pragma: no cover - zoneinfo always present on 3.9+
    _RIGA = timezone.utc


def _riga_day(ts: float) -> str:
    """Calendar day in Riga — the kill switch resets at local midnight."""
    return datetime.fromtimestamp(float(ts), tz=_RIGA).strftime("%Y-%m-%d")


def load_trades(path: str) -> list:
    out = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                et, xt = row.get("entry_time"), row.get("exit_time")
                if not et or not xt:
                    continue
                out.append({
                    "symbol":    row["symbol"],
                    "direction": row["direction"],
                    "outcome":   row["outcome"],
                    "entry_time": float(et),
                    "exit_time":  float(xt),
                    "net_r":      float(row["net_r"]),
                })
            except (KeyError, TypeError, ValueError):
                continue
    out.sort(key=lambda t: t["entry_time"])
    return out


def naive_drawdown(trades: list, key: str = "entry_time") -> float:
    """What backtest.py reports: one trade at a time, no overlap, no caps."""
    equity = peak = 0.0
    dd = 0.0
    for t in sorted(trades, key=lambda x: x[key]):
        equity += t["net_r"]
        peak = max(peak, equity)
        dd = min(dd, equity - peak)
    return dd


def simulate(trades: list, *, max_same_dir: int, max_per_scan: int,
             kill_streak: int, scan_seconds: int, apply_caps: bool) -> dict:
    """Walk the timeline holding positions concurrently on one account.

    Events are processed in time order; a position's P&L lands on the equity
    curve at its EXIT, which is what makes simultaneous stop-outs show up as a
    single deep step instead of being smoothed across the trade sequence.
    """
    open_pos = []          # dicts with exit_time / symbol / direction / net_r
    equity = peak = 0.0
    max_dd = 0.0
    taken = skipped = 0
    skip_reasons = {"symbol_busy": 0, "dir_cap": 0, "scan_cap": 0, "kill_switch": 0}

    taken_outcomes = []
    sl_streak = 0
    streak_day = None
    halted_day = None
    per_scan = {}          # scan bucket -> entries opened

    worst_moment = {"ts": None, "n": 0, "r": 0.0}

    def close_due(until_ts):
        """Realise every position that exited at or before until_ts."""
        nonlocal equity, peak, max_dd, sl_streak, streak_day, halted_day
        due = [p for p in open_pos if p["exit_time"] <= until_ts]
        if not due:
            return
        due.sort(key=lambda p: p["exit_time"])
        for p in due:
            open_pos.remove(p)
            equity += p["net_r"]
            peak = max(peak, equity)
            drop = equity - peak
            if drop < max_dd:
                max_dd = drop
            # Kill switch counts CONSECUTIVE stop-outs inside one Riga day.
            day = _riga_day(p["exit_time"])
            if day != streak_day:
                streak_day, sl_streak = day, 0
            if p["outcome"] == "SL":
                sl_streak += 1
                if kill_streak > 0 and sl_streak >= kill_streak:
                    halted_day = day
            else:
                sl_streak = 0
        # Track the single worst simultaneous cluster for the report.
        by_ts = {}
        for p in due:
            by_ts.setdefault(p["exit_time"], []).append(p["net_r"])
        for ts, rs in by_ts.items():
            tot = sum(rs)
            if tot < worst_moment["r"]:
                worst_moment.update({"ts": ts, "n": len(rs), "r": tot})

    for t in trades:
        close_due(t["entry_time"])

        if apply_caps:
            day = _riga_day(t["entry_time"])
            if halted_day == day:
                skipped += 1
                skip_reasons["kill_switch"] += 1
                continue
            # Live never runs two positions on one symbol; backtest.py does.
            if any(p["symbol"] == t["symbol"] for p in open_pos):
                skipped += 1
                skip_reasons["symbol_busy"] += 1
                continue
            same_dir = sum(1 for p in open_pos if p["direction"] == t["direction"])
            if max_same_dir > 0 and same_dir >= max_same_dir:
                skipped += 1
                skip_reasons["dir_cap"] += 1
                continue
            bucket = int(t["entry_time"] // scan_seconds)
            if max_per_scan > 0 and per_scan.get(bucket, 0) >= max_per_scan:
                skipped += 1
                skip_reasons["scan_cap"] += 1
                continue
            per_scan[bucket] = per_scan.get(bucket, 0) + 1

        open_pos.append(dict(t))
        taken_outcomes.append(t["outcome"])
        taken += 1

    close_due(float("inf"))

    wins = sum(1 for o in taken_outcomes if o not in ("SL", "EXPIRED"))
    return {
        "taken": taken, "skipped": skipped, "skip_reasons": skip_reasons,
        "net_r": equity, "max_dd": max_dd, "worst_moment": worst_moment,
        "wins": wins,
        "win_rate": (wins / taken * 100) if taken else 0.0,
        "r_per_trade": (equity / taken) if taken else 0.0,
        "sl": sum(1 for o in taken_outcomes if o == "SL"),
    }


def peak_concurrency(trades: list) -> tuple:
    """Highest number of positions open at once, and same-direction."""
    events = []
    for t in trades:
        events.append((t["entry_time"], 1, t["direction"]))
        events.append((t["exit_time"], -1, t["direction"]))
    events.sort(key=lambda e: (e[0], e[1]))
    cur = {"LONG": 0, "SHORT": 0}
    best_total = best_dir = 0
    for _ts, delta, d in events:
        if d in cur:
            cur[d] += delta
        total = cur["LONG"] + cur["SHORT"]
        best_total = max(best_total, total)
        best_dir = max(best_dir, cur["LONG"], cur["SHORT"])
    return best_total, best_dir


def main():
    ap = argparse.ArgumentParser(description="Replay backtest trades on one shared account")
    ap.add_argument("csv", help="trades CSV from backtest.py --export-trades")
    ap.add_argument("--max-same-dir", type=int, default=4)
    ap.add_argument("--max-per-scan", type=int, default=3)
    ap.add_argument("--kill-streak", type=int, default=3)
    ap.add_argument("--scan-seconds", type=int, default=300)
    ap.add_argument("--no-caps", action="store_true",
                    help="concurrency-aware accounting only, no live guardrails")
    args = ap.parse_args()

    trades = load_trades(args.csv)
    if not trades:
        print("No usable trades in CSV (need entry_time/exit_time/net_r).")
        return 1

    dd_entry = naive_drawdown(trades, "entry_time")
    dd_exit = naive_drawdown(trades, "exit_time")
    total_r = sum(t["net_r"] for t in trades)
    pk_total, pk_dir = peak_concurrency(trades)

    print(f"Trades in file : {len(trades)}")
    print(f"Total net R    : {total_r:+.2f}R")
    print()
    print("--- single trade line (what backtest.py reports) ---")
    print(f"  maxDD by entry order : {dd_entry:+.2f}R")
    print(f"  maxDD by exit order  : {dd_exit:+.2f}R")
    print(f"  peak positions open at once : {pk_total}  (same direction: {pk_dir})")
    print()

    for label, caps in (("no caps, concurrency-aware", False),
                        ("full live guardrails", True)):
        if args.no_caps and caps:
            continue
        r = simulate(trades, max_same_dir=args.max_same_dir,
                     max_per_scan=args.max_per_scan, kill_streak=args.kill_streak,
                     scan_seconds=args.scan_seconds, apply_caps=caps)
        print(f"--- portfolio: {label} ---")
        print(f"  trades taken : {r['taken']}  skipped: {r['skipped']}")
        if caps:
            for k, v in r["skip_reasons"].items():
                if v:
                    print(f"      {k}: {v}")
        print(f"  win rate     : {r['win_rate']:.1f}%  "
              f"({r['wins']} won / {r['sl']} stopped)")
        print(f"  net R        : {r['net_r']:+.2f}R  ({r['r_per_trade']:+.3f}R/trade)")
        print(f"  maxDD        : {r['max_dd']:+.2f}R")
        w = r["worst_moment"]
        if w["ts"]:
            when = datetime.fromtimestamp(w["ts"], tz=_RIGA).strftime("%Y-%m-%d %H:%M")
            print(f"  worst single moment: {w['n']} positions closed together "
                  f"for {w['r']:+.2f}R  ({when} Riga)")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
