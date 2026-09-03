#!/usr/bin/env python3
"""
Fast parity checks between live bot helpers and research/backtest helpers.

Run before trusting a new backtest change:
  python parity_check.py

Ported from the crypto desk 2026-08-28. This desk had no parity guard at all —
every live/backtest agreement was assumed rather than asserted, on a bot that
trades real money. The crypto version had also been silently RED for weeks
because it hardcoded expectations that a later exit change invalidated, so the
R-model expectations here are DERIVED from config and survive such a change.
"""

from __future__ import annotations

import importlib.util
from importlib.machinery import ModuleSpec
import os
import sys
import types


ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

if importlib.util.find_spec("dotenv") is None:
    dotenv_stub = types.ModuleType("dotenv")
    dotenv_stub.__spec__ = ModuleSpec("dotenv", loader=None)
    dotenv_stub.load_dotenv = lambda *args, **kwargs: None
    sys.modules["dotenv"] = dotenv_stub

if importlib.util.find_spec("requests") is None:
    requests_stub = types.ModuleType("requests")
    requests_stub.__spec__ = ModuleSpec("requests", loader=None)
    requests_stub.post = lambda *args, **kwargs: None
    requests_stub.get = lambda *args, **kwargs: None
    sys.modules["requests"] = requests_stub


from backtest import calculate_tp_sl_local, gross_r_for_outcome  # noqa: E402
from src.telegram_notifier import calculate_tp_sl  # noqa: E402


def _almost_equal(a: float, b: float, eps: float = 1e-8) -> bool:
    return abs(float(a) - float(b)) <= eps


def check_tp_sl_parity() -> list[str]:
    """Live and backtest must place TP1/TP2/SL identically for the same input.

    Cases are chosen at stock-perp price scales and around the risk clamps
    (RISK_MIN_PCT 0.4%, RISK_MAX_PCT 1.5%), which is where the two paths would
    most plausibly drift: over half this book has its stop set by a clamp
    rather than by structure, so the clamp arithmetic is load-bearing.
    """
    cases = [
        dict(price=100.0, direction="LONG", atr=0.6, recent_high=103.0, recent_low=99.2),
        dict(price=100.0, direction="SHORT", atr=0.6, recent_high=100.8, recent_low=97.0),
        # structure closer than RISK_MIN_PCT -> floor must bind, both sides
        dict(price=100.0, direction="LONG", atr=0.05, recent_high=100.4, recent_low=99.95),
        dict(price=100.0, direction="SHORT", atr=0.05, recent_high=100.05, recent_low=99.6),
        # structure wider than RISK_MAX_PCT -> ceiling must bind, both sides
        dict(price=100.0, direction="LONG", atr=3.0, recent_high=112.0, recent_low=94.0),
        dict(price=100.0, direction="SHORT", atr=3.0, recent_high=106.0, recent_low=88.0),
        # no ATR and no structure -> percentage fallback
        dict(price=250.0, direction="LONG", atr=0.0, recent_high=0.0, recent_low=0.0),
        dict(price=250.0, direction="SHORT", atr=0.0, recent_high=0.0, recent_low=0.0),
        # explicit structural targets passed through
        dict(price=512.30, direction="LONG", atr=1.8, recent_high=520.0, recent_low=505.0,
             tp1_level=516.0, tp2_level=522.0),
        dict(price=512.30, direction="SHORT", atr=1.8, recent_high=519.0, recent_low=504.0,
             tp1_level=508.0, tp2_level=502.0),
    ]

    failures = []
    for case in cases:
        live = calculate_tp_sl(**case)
        test = calculate_tp_sl_local(**case)
        if not all(_almost_equal(a, b) for a, b in zip(live, test)):
            failures.append(f"TP/SL mismatch {case}: live={live} backtest={test}")
    return failures


def check_r_model() -> list[str]:
    """R-model check, with expectations DERIVED from config rather than written in.

    The crypto version of this test hardcoded the answers for TP1_CLOSE_FRAC=0.5
    and went red the moment the exit profile moved to keeping the full position
    past TP1 — then stayed red, unnoticed, for weeks. A guard that reports a
    permanent false alarm is worse than no guard, because the red light gets
    ignored. Deriving keeps it honest across exit changes.
    """
    from config import TP1_CLOSE_FRAC

    failures = []
    entry, tp1, tp2, sl = 100.0, 101.0, 102.0, 99.0
    risk = entry - sl
    tp1_r = (tp1 - entry) / risk
    tp2_r = (tp2 - entry) / risk
    frac = max(0.0, min(1.0, float(TP1_CLOSE_FRAC)))
    runner = 1.0 - frac
    expected = {
        "TP2": frac * tp1_r + runner * tp2_r,
        "TP1": frac * tp1_r,          # runner exits at breakeven
        "SL": -1.0,
        "EXPIRED": 0.0,
    }
    for outcome, value in expected.items():
        actual = gross_r_for_outcome(outcome, entry, tp1, tp2, sl)
        if not _almost_equal(actual, value):
            failures.append(
                f"R model mismatch {outcome}: got {actual}, expected {value} "
                f"(TP1_CLOSE_FRAC={frac})"
            )
    return failures


EXIT_PARITY_MAX = 0.03   # known residual is ~1%; a real break is much larger


def check_exit_parity() -> list[str]:
    """The exit rule exists three times; assert two of them still agree.

    This check covers backtest.simulate_trade_direct against the shadow
    tracker's _simulate_setup_outcome, which is what unsent setups are
    resolved with and therefore what Claude is shown as his own record. The
    comparison already existed as tools_exit_parity.py, but running it was
    something a person had to remember -- and the divergence that mattered
    most sat there for months. A residual around 1% is expected and
    documented; anything past EXIT_PARITY_MAX means the two have drifted.
    """
    try:
        from tools_exit_parity import compare
    except Exception as e:
        return [f"exit parity unavailable: {e}"]
    agree, dis, _skipped, kinds = compare(200)
    total = agree + dis
    if total == 0:
        return ["exit parity: no comparable pairs — the series generator is broken"]
    rate = dis / total
    if rate > EXIT_PARITY_MAX:
        return [f"exit rule drift: {dis}/{total} ({rate*100:.1f}%) disagree, max {EXIT_PARITY_MAX*100:.0f}% — {kinds}"]
    return []


def check_live_r_matches_model() -> list[str]:
    """The monitor's R against the backtest's, on the same trade.

    check_r_model above proves the BACKTEST's function agrees with config. It
    said nothing about the live monitor, which computed its own blended R
    inline — eleven copies of the same arithmetic in the loop that decides what
    goes into realized_r. Nothing compared the two, so the model could have
    been re-derived and the live books would have gone on using the old shape.

    The copies are now one function (src/r_model.blended_r) and this is the
    guard that keeps it honest.
    """
    from config import TP1_CLOSE_FRAC
    from src.r_model import blended_r

    failures = []
    frac = max(0.0, min(1.0, float(TP1_CLOSE_FRAC)))
    runner = 1.0 - frac
    for entry, tp1, tp2, sl in ((100.0, 101.0, 102.0, 99.0),
                                (100.0, 100.6, 105.0, 99.0),
                                (50.0, 50.9, 53.5, 48.2),
                                (100.0, 99.4, 95.0, 101.0),   # short
                                (7.5, 7.44, 7.10, 7.62)):     # short, small px
        risk = abs(entry - sl)
        tp1_r = abs(tp1 - entry) / risk
        tp2_r = abs(tp2 - entry) / risk

        live_tp2 = blended_r(frac, tp1_r, runner, tp2_r)
        model_tp2 = gross_r_for_outcome("TP2", entry, tp1, tp2, sl)
        if not _almost_equal(live_tp2, model_tp2, eps=5e-5):
            failures.append(
                f"live/model R mismatch TP2 at entry={entry}: "
                f"monitor {live_tp2}, backtest {model_tp2}"
            )

        # TP1 reached, runner gave nothing back — the monitor's BREAKEVEN and
        # TP1_EXPIRED closes, and the backtest's "TP1" outcome.
        live_tp1 = blended_r(frac, tp1_r, 0.0, 0.0)
        model_tp1 = gross_r_for_outcome("TP1", entry, tp1, tp2, sl)
        if not _almost_equal(live_tp1, model_tp1, eps=5e-5):
            failures.append(
                f"live/model R mismatch TP1 at entry={entry}: "
                f"monitor {live_tp1}, backtest {model_tp1}"
            )
    return failures


def main() -> int:
    failures = (check_tp_sl_parity() + check_r_model()
                + check_live_r_matches_model() + check_exit_parity())
    if failures:
        print("FAIL")
        for item in failures:
            print(f"- {item}")
        return 1
    print("PASS: TP/SL parity, R model, live-vs-model R and "
          "exit-rule parity are OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
