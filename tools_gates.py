#!/usr/bin/env python3
"""Replay the live-capacity gates over an exported RAW trade list.

Every gate sweep used to cost a full backtest — eight minutes on the crypto
book — because the gates run inside the run. They do not need to: the gates are
a pure function of the trade list, so exporting once with BT_EXPORT_RAW=1 and
replaying here makes a sweep instant and lets a whole grid be measured in the
time one run used to take.

Written 2026-08-29 while fixing the kill-switch lookahead. Keep the two modes:
--lookahead reproduces the old peeking replay, which is the anchor with a known
answer for anything that changes this file.

  python backtest.py --candles 18000 --end-date 2026-08-26 --quiet \
      --export-trades raw.csv        # with BT_EXPORT_RAW=1 in the environment
  python tools_gates.py raw.csv
  python tools_gates.py raw.csv --grid
"""
from __future__ import annotations

import argparse
import csv
import sys


def _num(row: dict, key: str, default: float = 0.0) -> float:
    v = row.get(key)
    if v is None or v == "":
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def risk_profile(rs: list[float], k: int = 5, n: int = 25) -> dict:
    """Same measures the backtest prints: max DD, worst windows, ulcer."""
    if not rs:
        return {"max_dd": 0.0, "worst_windows": 0.0, "ulcer": 0.0}
    cum = 0.0
    peak = 0.0
    worst = 0.0
    sq: list[float] = []
    curve: list[float] = []
    for r in rs:
        cum += r
        curve.append(cum)
        peak = max(peak, cum)
        worst = min(worst, cum - peak)
        sq.append((cum - peak) ** 2)
    # Match backtest.risk_profile exactly: mean of the k smallest rolling sums,
    # negated — NOT filtered to losing ones. A negative ww means no 25-trade
    # stretch was underwater at all, which the caller must read as "not
    # applicable" rather than as zero risk.
    ww = 0.0
    if len(rs) >= n:
        sums = sorted(sum(rs[i:i + n]) for i in range(len(rs) - n + 1))
        ww = -(sum(sums[:k]) / k)
    return {"max_dd": -worst,
            "worst_windows": ww,
            "ulcer": (sum(sq) / len(sq)) ** 0.5}


def apply_gates(rows: list[dict], *, cooldown_h: float, per_scan: int,
                per_dir: int, kill: int, bar_sec: int = 900,
                lookahead: bool = False) -> list[dict]:
    """Replay cooldown / per-scan / per-direction / kill-switch in entry order.

    The kill-switch is the one gate that cannot be replayed naively. Live it
    fires on today's CLOSED signals; walking entries and reading the eventual
    outcome pauses the day at the entry of a trade that only stops out later,
    which is knowledge the live bot does not have. Losses cluster, so that peek
    deletes the rest of a bad patch and understates drawdown several-fold.
    """
    ordered = sorted(rows, key=lambda r: (_num(r, "entry_time"),
                                          r.get("symbol", ""),
                                          _num(r, "entry_bar")))
    last_sig: dict = {}
    per_bar: dict = {}
    open_by_dir: dict = {}
    kept: list[dict] = []
    closed: list[tuple[float, str]] = []
    streak = 0
    cur_day = None
    blocked_day = None

    def _ts(v: float) -> float:
        return v / 1000 if v > 1e11 else v

    def _sl_streak_at(now: float, dy: int) -> int:
        done = sorted((e, o) for e, o in closed if e <= now and int(e // 86400) == dy)
        n = 0
        for _, outcome in reversed(done):
            if outcome != "SL":
                break
            n += 1
        return n

    for t in ordered:
        raw = _num(t, "entry_time")
        ts = _ts(raw)
        day = int(ts // 86400)
        if day != cur_day:
            cur_day, streak, blocked_day, closed = day, 0, None, []
        if kill > 0:
            if blocked_day == day:
                continue
            if not lookahead and _sl_streak_at(ts, day) >= kill:
                blocked_day = day
                continue
        key = (t.get("symbol", ""), t.get("direction", ""))
        if cooldown_h > 0 and key in last_sig and (ts - last_sig[key]) / 3600 < cooldown_h:
            continue
        bar = int(ts // (bar_sec or 900))
        if per_scan > 0 and per_bar.get(bar, 0) >= per_scan:
            continue
        if per_dir > 0:
            live = [o for o in open_by_dir.get(t.get("direction", ""), [])
                    if _num(o, "exit_time") > raw]
            if len(live) >= per_dir:
                continue
            live.append(t)
            open_by_dir[t.get("direction", "")] = live
        last_sig[key] = ts
        per_bar[bar] = per_bar.get(bar, 0) + 1
        kept.append(t)
        if kill > 0:
            ex = _ts(_num(t, "exit_time"))
            if ex > 0:
                closed.append((ex, t.get("outcome", "")))
            if lookahead:
                streak = streak + 1 if t.get("outcome") == "SL" else 0
                if streak >= kill:
                    blocked_day = day
    return kept


def report(label: str, kept: list[dict]) -> dict:
    rs = [_num(r, "net_r") for r in kept]
    wins = sum(1 for r in rs if r > 0)
    net = sum(rs)
    rp = risk_profile(rs)
    ww = rp["worst_windows"]
    ul = rp["ulcer"]
    if ww <= 0:   # no losing 25-trade stretch — ratio is meaningless
        ww = float("nan")
    print(f"{label:<28} {len(kept):>5}сд  WR {100*wins/max(1,len(kept)):>5.1f}%  "
          f"net {net:>+9.2f}R  худш {ww:>6.2f} ({net/ww if ww else 0:>6.1f})  "
          f"ulcer {ul:>5.2f} ({net/ul if ul else 0:>6.1f})")
    return {"n": len(kept), "net": net, "ww": ww, "ulcer": ul}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--cooldown", type=float, default=3.0)
    ap.add_argument("--per-scan", type=int, default=3)
    ap.add_argument("--per-dir", type=int, default=5)
    ap.add_argument("--kill", type=int, default=3)
    ap.add_argument("--lookahead", action="store_true")
    ap.add_argument("--grid", action="store_true", help="Sweep every capacity gate.")
    ap.add_argument("--out", default=None,
                    help="Write the honestly-gated book to CSV, for the analysis tools.")
    a = ap.parse_args()

    with open(a.csv, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        print("АВАРИЯ: выгрузка пуста")
        return 1
    print(f"сырых сделок: {len(rows)}")

    if a.out:
        kept = apply_gates(rows, cooldown_h=a.cooldown, per_scan=a.per_scan,
                           per_dir=a.per_dir, kill=a.kill, lookahead=a.lookahead)
        with open(a.out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(kept)
        report("записано", kept)
        print(f"-> {a.out}")
        return 0

    if not a.grid:
        report("подглядка",
               apply_gates(rows, cooldown_h=a.cooldown, per_scan=a.per_scan,
                           per_dir=a.per_dir, kill=a.kill, lookahead=True))
        report("честно",
               apply_gates(rows, cooldown_h=a.cooldown, per_scan=a.per_scan,
                           per_dir=a.per_dir, kill=a.kill, lookahead=False))
        return 0

    def run(**kw):
        base = dict(cooldown_h=a.cooldown, per_scan=a.per_scan,
                    per_dir=a.per_dir, kill=a.kill, lookahead=False)
        base.update(kw)
        return apply_gates(rows, **base)

    print("\n--- килл-свитч (серия стопов) ---")
    for k in (0, 2, 3, 4, 5):
        report(f"kill={k}", run(kill=k))
    print("\n--- пауза по символу+направлению, часов ---")
    for c in (0, 1.0, 2.0, 3.0, 6.0, 12.0):
        report(f"cooldown={c}", run(cooldown_h=c))
    print("\n--- сигналов за скан ---")
    for p in (1, 2, 3, 4, 6, 0):
        report(f"per_scan={p or 'нет'}", run(per_scan=p))
    print("\n--- одновременных в одну сторону ---")
    for d in (2, 3, 4, 5, 8, 0):
        report(f"per_dir={d or 'нет'}", run(per_dir=d))
    return 0


if __name__ == "__main__":
    sys.exit(main())
