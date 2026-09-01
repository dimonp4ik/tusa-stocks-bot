#!/usr/bin/env python3
"""Do the THREE exit implementations agree? Now yes, to within ~1%.

The exit rule exists three times in this project:
  1. main.py _check_open_signals      — what the live bot does
  2. backtest.py simulate_trade_direct — what every reported figure measures
  3. main.py _simulate_setup_outcome   — what the shadow tracker resolves
     unsent setups with, and therefore what Claude is shown as his own history

Nothing asserted they agree. parity_check.py covers TP/SL placement only, and
the divergence that mattered most this week — shadow outcomes computed under a
different exit policy — sat in (3) for months without anyone noticing.

(1) needs a database and a live feed, so this compares (2) against (3) on
synthetic candle series with known answers. Both are pure functions of the
candles, so a disagreement here is a real disagreement.

Run it after touching any exit rule. A rising number means the shadow history
Claude learns from is drifting away from the book being measured.

State when this tool was written: 13% disagreement — of which most turned out to
be the tool's own fault. simulate_trade_direct defaults to exit_policy="classic"
with a 0.75 ATR trail, and calling it bare compared the shadow against an exit
rule the bot does not run. Passing the production policy took crypto to 0.0% and
stocks to 6.0%. The rest was a real defect: the shadow started its trailing peak
at the TP1 bar instead of at the fill, leaving the trail near entry so runners
survived pullbacks the live trail would have closed. Fixed; stocks now 0.9% over
800 cases, all in the same direction (model TP1 / shadow TP2), consistent with
the model's in-session expiry clock which this simplification does not model.

    python tools_exit_parity.py [runs]
"""
from __future__ import annotations

import os
import random
import sys
import tempfile

os.environ.setdefault("DB_PATH", os.path.join(tempfile.mkdtemp(), "parity.db"))

import backtest as bt  # noqa: E402
import config as C  # noqa: E402
import main  # noqa: E402


def _series(n: int, start: float, vol: float):
    o, h, l, c = [], [], [], []
    px = start
    for _ in range(n):
        op = px
        px = px * (1 + random.gauss(0, vol))
        o.append(op)
        h.append(max(op, px) * (1 + abs(random.gauss(0, vol / 2))))
        l.append(min(op, px) * (1 - abs(random.gauss(0, vol / 2))))
        c.append(px)
    return o, h, l, c


def compare(runs: int = 400):
    """Compare the model's exit against the shadow's on synthetic series.

    Split out of main_ so parity_check.py can assert on it. The drift this
    catches went unnoticed for months precisely because running it was
    something a person had to remember.
    """
    random.seed(7)
    agree = dis = skipped = 0
    kinds: dict[str, int] = {}
    for k in range(runs):
        o, h, l, c = _series(60, 100.0, 0.006)
        entry, atr = o[0], o[0] * 0.004
        direction = "LONG" if k % 2 == 0 else "SHORT"
        tp1, tp2, sl = bt.calculate_tp_sl_local(
            entry, direction, atr=atr,
            recent_high=max(h[:5]), recent_low=min(l[:5]),
        )
        shadow, _, _ = main._simulate_setup_outcome(
            direction, entry, tp1, tp2, sl, h, l, c, atr=atr,
        )
        if shadow is None or shadow == "NO_FILL":
            skipped += 1
            continue
        setup = {
            "direction": direction, "current_price": entry, "atr": atr,
            "recent_high": max(h[:5]), "recent_low": min(l[:5]),
            "entry_low": entry * 0.999, "entry_high": entry * 1.001,
            "session": "OPEN", "volume_ratio": 2.0,
            "eff_ratio": 0.4, "vol_atr_pct": 0.004,
        }
        candles = {"open": o, "high": h, "low": l, "close": c,
                   "time": list(range(len(o))), "volume": [1] * len(o)}
        # These two MUST be passed. simulate_trade_direct defaults to
        # exit_policy="classic" with a 0.75 ATR trail — neither is what the bot
        # runs, and calling it bare compares the shadow against an exit rule
        # that exists nowhere. The real scan loop passes "trail" and the
        # configured multiplier; mirror it or the whole comparison is void.
        trade = bt.simulate_trade_direct(
            "PARITY", setup, candles, 0, 48, 0.0005, 0.0005,
            exit_policy="trail",
            trail_atr_mult=max(0.0, float(C.TRAIL_ATR_MULT)),
        )
        if trade is None:
            skipped += 1
            continue
        # TRAIL and TP1 are the same categorical result: "reached TP1, did not
        # run to TP2". The engines label that exit differently by design.
        model = "TP1" if trade.outcome in ("TP1", "TRAIL") else trade.outcome
        shad = "TP1" if shadow in ("TP1", "TRAIL") else shadow
        if model == shad:
            agree += 1
        else:
            dis += 1
            kinds[f"model={model} shadow={shad}"] = kinds.get(
                f"model={model} shadow={shad}", 0) + 1

    return agree, dis, skipped, kinds


def main_() -> int:
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    agree, dis, skipped, kinds = compare(runs)
    total = agree + dis
    if total == 0:
        print("АВАРИЯ: ни одной сравнимой пары — проверь генератор серий")
        return 1
    print(f"сравнено пар: {total}  (пропущено {skipped})")
    print(f"  совпало:   {agree} ({100*agree/total:.1f}%)")
    print(f"  разошлось: {dis} ({100*dis/total:.1f}%)")
    for kind, n in sorted(kinds.items(), key=lambda x: -x[1]):
        print(f"    {kind}: {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main_())
