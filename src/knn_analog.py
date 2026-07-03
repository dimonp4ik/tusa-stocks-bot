"""
k-NN price-shape analog scorer (Kronos-inspired, CPU-only, no ML).

Idea borrowed from foundation forecasters (Kronos): a recent candle window is a
"pattern". Instead of a neural net, we do nearest-neighbour matching against the
symbol's own past: find the K most-similar historical windows and measure how
often the move that followed matched our intended trade direction.

Output is a probability-like score in [0, 1]:
    score = fraction of the K nearest past analogs whose forward path favoured
            `direction` (MFE > MAE over `horizon` bars).

Strictly causal — only uses bars whose forward outcome had already resolved
before the query bar `i`, so it is safe to call inside a backtest with no
look-ahead.
"""

from __future__ import annotations

import math


def _log_returns(closes: list[float]) -> list[float]:
    out = []
    prev = None
    for c in closes:
        if prev is not None and prev > 0 and c > 0:
            out.append(math.log(c / prev))
        elif prev is not None:
            out.append(0.0)
        prev = c
    return out


def _zscore(vec: list[float]) -> list[float]:
    n = len(vec)
    if n == 0:
        return vec
    m = sum(vec) / n
    var = sum((x - m) ** 2 for x in vec) / n
    sd = math.sqrt(var) if var > 1e-18 else 1e-9
    return [(x - m) / sd for x in vec]


def knn_direction_score(
    candles: dict[str, list],
    i: int,
    direction: str,
    *,
    shape_len: int = 12,
    horizon: int = 16,
    k: int = 40,
    min_history: int = 120,
    max_history: int | None = None,
) -> float | None:
    """
    Fraction of the k most-similar past price-shape analogs (each resolved
    strictly before bar i) whose forward path favoured `direction`.

    Args:
        candles: dict with 'close','high','low' lists (full series).
        i:       current scan bar (entry context = closes[:i], last bar i-1).
        direction: "LONG" or "SHORT".
        shape_len: bars of recent return-shape used as the query vector.
        horizon: forward bars over which the analog outcome is measured.
        k:       neighbours to average.
        min_history: minimum bars before scoring is attempted.

    Returns float in [0,1], or None when history is insufficient.
    """
    closes = candles.get("close") or []
    highs  = candles.get("high")  or []
    lows   = candles.get("low")   or []
    n = len(closes)
    if i <= min_history or i > n:
        return None

    rets = _log_returns(closes[:i])           # causal: up to bar i-1
    if len(rets) < shape_len + horizon + 60:
        return None

    query = _zscore(rets[-shape_len:])

    # Candidate window ends e: window returns rets[e-shape_len:e], entry close
    # index = e, forward outcome over closes/highs/lows[e .. e+horizon].
    # No look-ahead: require e + horizon <= i - 1.
    max_e = min(len(rets) - horizon, i - 1 - horizon)
    # Cap the analog pool to the most-recent max_history bars to mirror the
    # candle depth available live (the bot only fetches KLINES_LIMIT 15m bars).
    min_e = shape_len
    if max_history is not None:
        min_e = max(min_e, max_e - int(max_history))
    neighbors: list[tuple[float, int]] = []

    for e in range(min_e, max_e + 1):
        win = _zscore(rets[e - shape_len:e])
        d = 0.0
        for a, b in zip(query, win):
            diff = a - b
            d += diff * diff

        c0 = closes[e]
        if c0 <= 0:
            continue
        fwd_hi = max(highs[e + 1:e + 1 + horizon])
        fwd_lo = min(lows[e + 1:e + 1 + horizon])
        if direction == "LONG":
            mfe = fwd_hi - c0
            mae = c0 - fwd_lo
        else:
            mfe = c0 - fwd_lo
            mae = fwd_hi - c0
        label = 1 if mfe > mae else 0
        neighbors.append((d, label))

    if len(neighbors) < k:
        return None

    neighbors.sort(key=lambda x: x[0])
    top = neighbors[:k]
    return sum(lbl for _, lbl in top) / float(k)


def knn_risk_mult(
    score: float | None,
    *,
    high_score: float,
    high_mult: float,
    low_score: float,
    low_mult: float,
) -> tuple[float, str]:
    """
    Map a k-NN analog score to a position-size multiplier.

    Backtest (2026-06-13, 90d, live-like 800-bar pool):
      score >= 0.55 → WR ~68% (size up);  score < 0.50 → WR ~59% (size down).
    Keeps trade frequency unchanged (no gating), lifts total R ~+6%.

    Returns (multiplier, tag). multiplier is 1.0 when score is None/neutral.
    """
    if score is None:
        return 1.0, ""
    if score >= high_score:
        return float(high_mult), f"kNN+{score:.2f}:x{high_mult:.2f}"
    if score < low_score:
        return float(low_mult), f"kNN-{score:.2f}:x{low_mult:.2f}"
    return 1.0, ""
