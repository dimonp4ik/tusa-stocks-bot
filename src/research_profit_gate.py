"""Economic acceptance math for literal-move trading research.

This module deliberately has no exchange or strategy dependency.  Research
scripts use it to reject attractive hit rates whose price edge and frequency
cannot support the client's requested account growth at the stated margin and
leverage.
"""
from __future__ import annotations

import math
from collections.abc import Iterable


DEFAULT_MARGIN_FRACTION = 0.03
DEFAULT_LEVERAGE = 10.0
DEFAULT_MARKET_DAYS_PER_MONTH = 22
DEFAULT_TRADES_PER_DAY = 3.0
DEFAULT_MONTHLY_TARGET = 0.30


def exposure(margin_fraction: float, leverage: float) -> float:
    if not (0 < margin_fraction <= 1):
        raise ValueError("margin_fraction must be in (0, 1]")
    if not math.isfinite(leverage) or leverage <= 0:
        raise ValueError("leverage must be positive")
    value = margin_fraction*leverage
    if value <= 0:
        raise ValueError("invalid exposure")
    return value


def trades_per_month(trades_per_day: float = DEFAULT_TRADES_PER_DAY,
                     market_days: int = DEFAULT_MARKET_DAYS_PER_MONTH) -> float:
    if not math.isfinite(trades_per_day) or trades_per_day <= 0:
        raise ValueError("trades_per_day must be positive")
    if market_days <= 0:
        raise ValueError("market_days must be positive")
    return trades_per_day*market_days


def required_signed_price_return_pct(
        monthly_target: float = DEFAULT_MONTHLY_TARGET,
        margin_fraction: float = DEFAULT_MARGIN_FRACTION,
        leverage: float = DEFAULT_LEVERAGE,
        trades_per_day: float = DEFAULT_TRADES_PER_DAY,
        market_days: int = DEFAULT_MARKET_DAYS_PER_MONTH) -> float:
    """Constant signed price return required on every trade, with compounding."""
    if monthly_target <= -1 or not math.isfinite(monthly_target):
        raise ValueError("monthly_target must be finite and greater than -1")
    count = trades_per_month(trades_per_day, market_days)
    account_return = math.expm1(math.log1p(monthly_target)/count)
    return 100*account_return/exposure(margin_fraction, leverage)


def minimum_binary_hit_rate_pct(
        target_pct: float, stop_pct: float,
        monthly_target: float = DEFAULT_MONTHLY_TARGET,
        margin_fraction: float = DEFAULT_MARGIN_FRACTION,
        leverage: float = DEFAULT_LEVERAGE,
        trades_per_day: float = DEFAULT_TRADES_PER_DAY,
        market_days: int = DEFAULT_MARKET_DAYS_PER_MONTH) -> float | None:
    """Hit rate needed for fixed TP/SL outcomes to compound to the objective.

    Returns ``None`` when even a 100% target-hit rate cannot reach the target.
    Gap losses, partial exits, fees and slippage are intentionally outside this
    idealized ceiling calculation.
    """
    if target_pct <= 0 or stop_pct <= 0:
        raise ValueError("target_pct and stop_pct must be positive")
    count = trades_per_month(trades_per_day, market_days)
    gross = exposure(margin_fraction, leverage)
    win_factor = 1+gross*target_pct/100
    loss_factor = 1-gross*stop_pct/100
    if loss_factor <= 0:
        raise ValueError("stop would lose the entire account")
    required_log = math.log1p(monthly_target)/count
    win_log, loss_log = math.log(win_factor), math.log(loss_factor)
    if win_log+1e-15 < required_log:
        return None
    rate = (required_log-loss_log)/(win_log-loss_log)
    return 100*min(1.0, max(0.0, rate))


def projected_monthly_return_pct(
        signed_price_returns: Iterable[float], trades_per_day: float,
        margin_fraction: float = DEFAULT_MARGIN_FRACTION,
        leverage: float = DEFAULT_LEVERAGE,
        market_days: int = DEFAULT_MARKET_DAYS_PER_MONTH) -> float:
    """Project a sample's geometric per-trade growth at a stated frequency.

    ``signed_price_returns`` are decimal price returns: ``0.005`` is +0.5%.
    The projection is a diagnostic, not a forecast.
    """
    values = [float(value) for value in signed_price_returns]
    if not values:
        return 0.0
    gross = exposure(margin_fraction, leverage)
    logs = []
    for value in values:
        factor = 1+gross*value
        if not math.isfinite(factor) or factor <= 0:
            raise ValueError("trade return produces an invalid account factor")
        logs.append(math.log(factor))
    count = trades_per_month(trades_per_day, market_days)
    return 100*math.expm1(sum(logs)/len(logs)*count)


def economic_gate(
        signed_price_returns: Iterable[float], observed_trades_per_day: float,
        monthly_target: float = DEFAULT_MONTHLY_TARGET,
        required_trades_per_day: float = DEFAULT_TRADES_PER_DAY,
        margin_fraction: float = DEFAULT_MARGIN_FRACTION,
        leverage: float = DEFAULT_LEVERAGE,
        market_days: int = DEFAULT_MARKET_DAYS_PER_MONTH) -> dict:
    values = [float(value) for value in signed_price_returns]
    projected = projected_monthly_return_pct(
        values, observed_trades_per_day, margin_fraction, leverage, market_days)
    required = required_signed_price_return_pct(
        monthly_target, margin_fraction, leverage,
        required_trades_per_day, market_days)
    mean_signed = 100*sum(values)/len(values) if values else 0.0
    return {
        "n": len(values),
        "margin_fraction": margin_fraction,
        "leverage": leverage,
        "observed_trades_per_day": observed_trades_per_day,
        "required_trades_per_day": required_trades_per_day,
        "market_days_per_month": market_days,
        "monthly_target_pct": 100*monthly_target,
        "mean_signed_price_return_pct": round(mean_signed, 6),
        "required_constant_signed_price_return_pct": round(required, 6),
        "projected_monthly_return_pct": round(projected, 4),
        "frequency_pass": observed_trades_per_day >= required_trades_per_day,
        "return_pass": projected >= 100*monthly_target,
        "passes": (observed_trades_per_day >= required_trades_per_day
                   and projected >= 100*monthly_target),
        "interpretation": "diagnostic projection from observed geometric trade return; not a forecast",
    }
