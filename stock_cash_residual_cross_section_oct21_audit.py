"""Cross-era audit of residual cross-sectional mean reversion.

The current stock shadow route detects an extreme with cross-sectional z-score,
but reverses the instrument's absolute move.  This experiment makes the signal
internally consistent: it reverses the instrument's return minus the same-bar
cross-sectional mean.  Signals use only closed 15-minute cash candles and enter
the next market open.  Development chooses the rule; later eras are revealed in
order and remain sealed as soon as one gate fails.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from stock_module_cluster_audit import portfolio_gate
import stock_cash_xperp_leadlag_oct6_audit as resolver
import stock_xperp_current_route_cash_transfer_oct21_audit as parent
import stock_xperp_native_economics_oct18_audit as economics


ROOT = Path("reports/audit_2026_09_08")
OUT = ROOT / "stock_cash_residual_cross_section_oct21.json"
ROWS_OUT = ROOT / "stock_cash_residual_cross_section_oct21.csv"
DEVELOPMENT = (2017, 2018, 2019, 2020)
VALIDATION = (2021, 2022)
TEST = (2023, 2024)
CONFIRMATION = (2025, 2026)
LOOKBACKS = (1, 2, 4, 8)
Z_THRESHOLDS = (1.5, 2.0, 2.5)
RESIDUAL_THRESHOLDS = (.002, .004, .006)
TARGETS = (.5, 1.0, 1.5, 2.0)
STOPS = (.5, .75, 1.0)
HOLDS = (8, 16, 32)
MIN_UNIVERSE = 8
MARGIN = .03
LEVERAGE = 10.0
MONTH_MARKET_DAYS = 22
MONTH_TARGET = .30
MINIMUM_DAILY_TRADES = 3.0
TOP_EXACT = 80
_MARKET_DAY_CACHE: dict[tuple[int, ...], int] = {}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def feature_rows(candles: dict[str, dict], lookback: int) -> pd.DataFrame:
    """Build causal same-timestamp residuals and next-open entry indexes."""
    rows = []
    for symbol, candle in candles.items():
        times = np.asarray(candle["time"], dtype=np.int64)
        close = np.asarray(candle["close"], dtype=float)
        for index in range(lookback, len(times) - 1):
            if times[index + 1] - times[index] != 900:
                continue
            move = close[index] / close[index - lookback] - 1.0
            rows.append((symbol, index + 1, int(times[index + 1]), move))
    frame = pd.DataFrame(rows, columns=(
        "symbol", "entry_index", "entry_time", "signal_return"))
    grouped = frame.groupby("entry_time", sort=False).signal_return
    frame["universe_n"] = grouped.transform("size")
    frame["cross_mean"] = grouped.transform("mean")
    frame["cross_std"] = grouped.transform("std")
    frame["residual_return"] = frame.signal_return - frame.cross_mean
    frame["cross_z"] = (
        frame.residual_return / frame.cross_std.replace(0, np.nan))
    frame["direction_n"] = -np.sign(
        frame.residual_return.to_numpy(float)).astype(np.int8)
    frame["year"] = pd.to_datetime(
        frame.entry_time, unit="s", utc=True).dt.year.astype(np.int16)
    return frame.replace([np.inf, -np.inf], np.nan).dropna()


def select_entries(frame: pd.DataFrame, z_threshold: float,
                   residual_threshold: float) -> pd.DataFrame:
    return frame[
        frame.cross_z.abs().ge(z_threshold)
        & frame.residual_return.abs().ge(residual_threshold)
        & frame.universe_n.ge(MIN_UNIVERSE)
    ].copy()


def year_of_time(values) -> np.ndarray:
    converted = pd.to_datetime(values, unit="s", utc=True)
    return (converted.dt.year.to_numpy() if isinstance(converted, pd.Series)
            else converted.year.to_numpy())


def years_mask(frame: pd.DataFrame, years: tuple[int, ...]) -> np.ndarray:
    if "year" in frame.columns:
        return frame.year.isin(years).to_numpy()
    return np.isin(year_of_time(frame.entry_time), np.asarray(years))


def market_days(candles: dict[str, dict], years: tuple[int, ...]) -> int:
    key = tuple(years)
    if key not in _MARKET_DAY_CACHE:
        _MARKET_DAY_CACHE[key] = parent.market_days(candles, years)
    return _MARKET_DAY_CACHE[key]


def raw_metrics(frame: pd.DataFrame, years: tuple[int, ...],
                market_days: int) -> dict:
    part = frame.loc[years_mask(frame, years)].copy()
    if part.empty:
        return {"n": 0, "return_pct": 0.0, "dd_pct": 0.0,
                "profitable_exit_rate_pct": 0.0,
                "trades_per_market_day": 0.0,
                "mean_signed_price_return_pct": 0.0}
    part.sort_values(["exit_time", "entry_time", "symbol"], inplace=True)
    factors = 1 + MARGIN * LEVERAGE * part.signed_return.to_numpy(float)
    equity = np.cumprod(factors)
    peaks = np.maximum.accumulate(np.maximum(equity, 1.0))
    drawdown = float(np.max((peaks - equity) / peaks))
    return {
        "n": int(len(part)),
        "return_pct": round(100 * (float(equity[-1]) - 1), 4),
        "dd_pct": round(100 * drawdown, 4),
        "profitable_exit_rate_pct": round(
            100 * float((part.signed_return > 0).mean()), 4),
        "trades_per_market_day": round(len(part) / max(market_days, 1), 4),
        "mean_signed_price_return_pct": round(
            100 * float(part.signed_return.mean()), 6),
    }


def raw_bundle(frame: pd.DataFrame, candles: dict[str, dict],
               years: tuple[int, ...]) -> dict:
    return {
        "combined": raw_metrics(frame, years, market_days(candles, years)),
        "annual": {
            str(year): raw_metrics(
                frame, (year,), market_days(candles, (year,)))
            for year in years
        },
    }


def rank_key(row: dict) -> tuple:
    return (-abs(float(row["cross_z"])),
            -abs(float(row["residual_return"])), str(row["symbol"]))


def gated(frame: pd.DataFrame, years: tuple[int, ...]) -> list[dict]:
    wanted = set(years)
    rows = [row for row in frame.to_dict("records")
            if pd.Timestamp(float(row["entry_time"]), unit="s", tz="UTC").year
            in wanted]
    return portfolio_gate(rows, None, rolling_hour_cap=4,
                          rank_key=rank_key, loss_streak_cap=2)


def metrics(rows: list[dict], margin: float, market_days: int) -> dict:
    value = economics.simulate(rows, margin)
    if not rows:
        value.update({
            "profitable_exit_rate_pct": 0.0, "literal_tp_rate_pct": 0.0,
            "max_loss_streak": 0, "trades_per_market_day": 0.0,
            "mean_signed_price_return_pct": 0.0,
            "projected_22_market_day_pct": 0.0,
            "required_mean_signed_move_at_actual_frequency_pct": None,
            "mandatory_three_daily_mean_signed_move_pct": round(
                required_mean_move(MINIMUM_DAILY_TRADES, margin), 6),
            "economic_gap_multiple": None,
            "economic_30pct_gate_pass": False,
        })
        return value
    signed = np.asarray([float(row["signed_return"]) for row in rows])
    frequency = len(rows) / max(market_days, 1)
    periods = frequency * MONTH_MARKET_DAYS
    mean_signed = float(signed.mean())
    trade_factor = 1 + margin * LEVERAGE * mean_signed
    projected = (100 * (trade_factor ** periods - 1)
                 if periods > 0 and trade_factor > 0 else -100.0)
    required = required_mean_move(frequency, margin) if frequency > 0 else None
    ordered = sorted(rows, key=lambda row: (
        float(row["exit_time"]), float(row["entry_time"]), str(row["symbol"])))
    streak = maximum = 0
    for row in ordered:
        streak = streak + 1 if float(row["signed_return"]) <= 0 else 0
        maximum = max(maximum, streak)
    value.update({
        "profitable_exit_rate_pct": round(100 * float(np.mean(signed > 0)), 4),
        "literal_tp_rate_pct": round(100 * float(np.mean(
            np.asarray([int(row["outcome_code"]) for row in rows]) == 1)), 4),
        "max_loss_streak": maximum,
        "trades_per_market_day": round(frequency, 4),
        "mean_signed_price_return_pct": round(100 * mean_signed, 6),
        "projected_22_market_day_pct": round(projected, 6),
        "required_mean_signed_move_at_actual_frequency_pct": (
            round(required, 6) if required is not None else None),
        "mandatory_three_daily_mean_signed_move_pct": round(
            required_mean_move(MINIMUM_DAILY_TRADES, margin), 6),
        "economic_gap_multiple": (
            round(required / (100 * mean_signed), 4)
            if required is not None and mean_signed > 0 else None),
        "economic_30pct_gate_pass": bool(projected >= 30),
    })
    return value


def required_mean_move(trades_per_day: float, margin: float) -> float:
    periods = MONTH_MARKET_DAYS * trades_per_day
    return (((1 + MONTH_TARGET) ** (1 / periods) - 1)
            / (margin * LEVERAGE) * 100)


def exact_bundle(frame: pd.DataFrame, candles: dict[str, dict],
                 years: tuple[int, ...]) -> tuple[dict, list[dict]]:
    accepted = gated(frame, years)
    result = {
        "accepted_rows": len(accepted),
        "margin_3pct": metrics(
            accepted, .03, market_days(candles, years)),
        "margin_5pct": metrics(
            accepted, .05, market_days(candles, years)),
        "annual_3pct": {},
    }
    for year in years:
        annual = [row for row in accepted if
                  pd.Timestamp(float(row["entry_time"]), unit="s", tz="UTC").year
                  == year]
        result["annual_3pct"][str(year)] = metrics(
            annual, .03, market_days(candles, (year,)))
    return result, accepted


def development_ok(stage: dict) -> bool:
    value = stage["margin_3pct"]
    years = list(stage["annual_3pct"].values())
    return bool(
        value["n"] >= 300 and value["return_pct"] > 0
        and value["dd_pct"] <= 12
        and value["trades_per_market_day"] >= 2.5
        and sum(item["return_pct"] > 0 for item in years) >= 3
        and min(item["return_pct"] for item in years) >= -3
    )


def later_ok(stage: dict) -> bool:
    value = stage["margin_3pct"]
    years = list(stage["annual_3pct"].values())
    return bool(
        value["n"] >= 100 and value["return_pct"] > 0
        and value["dd_pct"] <= 10
        and value["trades_per_market_day"] >= 2.5
        and all(item["return_pct"] > 0 and item["dd_pct"] <= 10
                for item in years)
    )


def objective_ok(stage: dict) -> bool:
    value = stage["margin_3pct"]
    months = len(value.get("monthly_return_pct", {}))
    return bool(
        value["dd_pct"] <= 5 and value["negative_months"] == 0
        and value["trades_per_market_day"] >= 3
        and months > 0 and value["months_ge_30pct"] == months
        and value["economic_30pct_gate_pass"]
    )


def candidate_parameters(lookback, z_threshold, residual_threshold,
                         target, stop, hold) -> dict:
    return {
        "lookback_bars": lookback, "z_threshold": z_threshold,
        "minimum_residual_pct": 100 * residual_threshold,
        "target_pct": target, "stop_pct": stop, "hold_bars": hold,
    }


def resolve_candidate(parameters: dict, features: pd.DataFrame,
                      candles: dict[str, dict],
                      years: tuple[int, ...]) -> pd.DataFrame:
    entries = select_entries(
        features, float(parameters["z_threshold"]),
        float(parameters["minimum_residual_pct"]) / 100)
    entries = entries.loc[years_mask(entries, years)]
    resolved = resolver.add_outcomes(
        entries, candles, float(parameters["target_pct"]),
        float(parameters["stop_pct"]), int(parameters["hold_bars"]))
    resolved["module"] = "cash_residual_cross_section_reversion"
    resolved["module_priority"] = 0
    return resolved


def public(item: dict) -> dict:
    return {key: value for key, value in item.items()
            if key not in ("_rows", "_accepted")}


def main() -> None:
    candles, sources = parent.load_history()
    raw_candidates = []
    searched = 0
    feature_count = 0
    features_by_lookback = {}
    for lookback in LOOKBACKS:
        features = feature_rows(candles, lookback)
        features_by_lookback[lookback] = features
        feature_count += len(features)
        broad = select_entries(features, min(Z_THRESHOLDS),
                               min(RESIDUAL_THRESHOLDS))
        broad = broad.loc[years_mask(broad, DEVELOPMENT)]
        for target, stop, hold in itertools.product(TARGETS, STOPS, HOLDS):
            resolved_broad = resolver.add_outcomes(
                broad, candles, target, stop, hold)
            for z_threshold, residual_threshold in itertools.product(
                    Z_THRESHOLDS, RESIDUAL_THRESHOLDS):
                searched += 1
                resolved = resolved_broad[
                    resolved_broad.cross_z.abs().ge(z_threshold)
                    & resolved_broad.residual_return.abs().ge(
                        residual_threshold)
                ]
                if len(resolved) < 300:
                    continue
                dev = raw_bundle(resolved, candles, DEVELOPMENT)
                combined = dev["combined"]
                annual = list(dev["annual"].values())
                if (combined["n"] < 300 or combined["return_pct"] <= 0
                        or combined["dd_pct"] > 18
                        or combined["trades_per_market_day"] < 2.5
                        or sum(item["return_pct"] > 0 for item in annual) < 2):
                    continue
                score = (min(item["return_pct"] for item in annual)
                         + .25 * combined["return_pct"]
                         - combined["dd_pct"])
                raw_candidates.append({
                    "parameters": candidate_parameters(
                        lookback, z_threshold, residual_threshold,
                        target, stop, hold),
                    "development_raw": dev, "development_raw_score": score,
                })
        print(json.dumps({"lookback": lookback, "searched": searched,
                          "raw_passed": len(raw_candidates)}), flush=True)

    raw_candidates.sort(key=lambda item: item["development_raw_score"],
                        reverse=True)
    exact = []
    for item in raw_candidates[:TOP_EXACT]:
        parameters = item["parameters"]
        resolved = resolve_candidate(
            parameters, features_by_lookback[parameters["lookback_bars"]],
            candles, DEVELOPMENT)
        stage, accepted = exact_bundle(resolved, candles, DEVELOPMENT)
        exact_item = {**item, "development": stage, "_accepted": accepted}
        if development_ok(stage):
            value = stage["margin_3pct"]
            exact_item["development_score"] = (
                min(v["return_pct"] for v in stage["annual_3pct"].values())
                + .25 * value["return_pct"] - value["dd_pct"])
            exact.append(exact_item)
    exact.sort(key=lambda item: item["development_score"], reverse=True)

    validated = []
    for item in exact:
        parameters = item["parameters"]
        validation_rows = resolve_candidate(
            parameters, features_by_lookback[parameters["lookback_bars"]],
            candles, VALIDATION)
        validation, _ = exact_bundle(validation_rows, candles, VALIDATION)
        item["validation"] = validation
        if later_ok(validation):
            validated.append(item)

    frozen = validated[0] if validated else None
    test = confirmation = None
    status = "REJECTED_DEVELOPMENT"
    if exact:
        status = "REJECTED_VALIDATION"
    if frozen:
        parameters = frozen["parameters"]
        test_rows = resolve_candidate(
            parameters, features_by_lookback[parameters["lookback_bars"]],
            candles, TEST)
        test, _ = exact_bundle(test_rows, candles, TEST)
        status = "REJECTED_TEST"
        if later_ok(test):
            confirmation, confirmation_rows = exact_bundle(
                resolve_candidate(
                    parameters,
                    features_by_lookback[parameters["lookback_bars"]],
                    candles, CONFIRMATION),
                candles, CONFIRMATION)
            status = ("CROSS_ERA_CANDIDATE"
                      if later_ok(confirmation) else "REJECTED_CONFIRMATION")
            frozen["_accepted"] = confirmation_rows

    selected_rows = (frozen["_accepted"] if frozen else
                     exact[0]["_accepted"] if exact else [])
    pd.DataFrame(selected_rows).to_csv(ROWS_OUT, index=False)
    selected = public(frozen) if frozen else (public(exact[0]) if exact else None)
    stages = [stage for stage in (
        selected and selected.get("development"),
        selected and selected.get("validation"), test, confirmation) if stage]
    objective = bool(stages and len(stages) == 4
                     and all(objective_ok(stage) for stage in stages))
    report = {
        "status": status,
        "ready_for_live_orders": False,
        "orders_possible": False,
        "protocol": {
            "hypothesis": (
                "reverse the stock return minus the simultaneous universe mean; "
                "direction and cross-sectional z-score use the same residual"),
            "development": list(DEVELOPMENT),
            "validation": list(VALIDATION), "test": list(TEST),
            "confirmation": list(CONFIRMATION),
            "selection": "development only; later eras revealed sequentially",
            "entry": "next 15m market open after a fully closed signal bar",
            "exit": "literal 0.5-2.0% targets; stop-first ambiguity",
            "orders": "market only",
            "money": "client 3% or 5% current-equity margin; fixed 10x",
            "portfolio": (
                "two per scan, four same-direction open, four rolling-hour, "
                "three-hour symbol/direction cooldown, pause after two losses"),
            "economic_gate": (
                "+30% over 22 market days; 1.327709% constant signed price "
                "move per trade at 3% margin, 10x and 3 trades/day"),
        },
        "grid": {
            "feature_rows_across_lookbacks": feature_count,
            "searched_candidates": searched,
            "raw_development_passes": len(raw_candidates),
            "exact_development_passes": len(exact),
            "validation_passes": len(validated),
            "exact_replayed": min(TOP_EXACT, len(raw_candidates)),
        },
        "selected": selected,
        "test": test, "confirmation": confirmation,
        "all_four_stages_meet_monthly_objective": objective,
        "top_development": [public(item) for item in exact[:10]],
        "sources": {
            **sources,
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "Cash-proxy history tests structure, not historical X-Perp fills.",
            "The historical universe has ten equities and only eight before MU.",
            "Session gaps terminate the position instead of carrying overnight.",
            "Gross returns exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": status, "grid": report["grid"], "selected": selected,
        "test": test, "confirmation": confirmation,
        "all_four_stages_meet_monthly_objective": objective,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
