"""Research-only direction router from closed entry-time features."""
import math


def direction_for(*, signal_return, basket_trend_64, policy):
    move, trend = float(signal_return), float(basket_trend_64)
    if not math.isfinite(move) or not math.isfinite(trend) or move == 0:
        return None
    momentum = 1 if move > 0 else -1
    if policy == "reversal":
        return -momentum
    if policy == "momentum":
        return momentum
    if policy == "momentum_aligned":
        return momentum if trend * momentum > 0 else None
    if policy == "mixed_strong_trend":
        return momentum if abs(trend) >= .04 else -momentum
    raise ValueError("unknown direction policy")
