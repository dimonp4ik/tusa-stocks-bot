"""Cross-era cash-proxy transfer of the exact current stock shadow route.

The native equity X-Perp route has direct venue history only from July 2026.
This audit freezes its current signal and execution parameters, then applies
the same closed-bar cross-sectional reversal to hashed 15-minute underlying
stock history.  No transfer parameter is fitted.  The only mechanical
adaptation is a minimum of eight aligned equities because the historical cash
universe contains ten non-index symbols and two have shorter histories.

Later periods remain unevaluated after the first failed objective gate.
"""
from __future__ import annotations

import hashlib
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from stock_module_cluster_audit import portfolio_gate
import stock_cash_xperp_leadlag_oct6_audit as resolver
import stock_xperp_backing_hour_cap_oct20_audit as hour_cap
import stock_xperp_backing_level_guard_oct20_audit as backing
import stock_xperp_native_cross_section_oct18_audit as native
import stock_xperp_native_economics_oct18_audit as economics
import stock_xperp_native_growth_profile_oct18_audit as growth
import stock_xperp_native_quality_guard_oct18_audit as quality
import stock_xperp_native_volatility_guard_oct18_audit as volatility


ROOT = Path("reports/audit_2026_09_08")
MANIFEST = ROOT / "stock_extended_history_manifest.json"
OUT = ROOT / "stock_xperp_current_route_cash_transfer_oct21.json"
ROWS_OUT = ROOT / "stock_xperp_current_route_cash_transfer_oct21.csv"
EXCLUDED = {"QQQUSDT", "SPYUSDT"}
MIN_UNIVERSE = 8
DEVELOPMENT = (2017, 2018, 2019, 2020)
VALIDATION = (2021, 2022)
TEST = (2023, 2024)
CONFIRMATION = (2025, 2026)
BASELINE_MARGIN = .03
LEVERAGE = 10.0
MONTH_MARKET_DAYS = 22
MINIMUM_DAILY_TRADES = 3.0
MONTH_TARGET = .30
MANDATORY_SIGNED_MOVE_PCT = 1.327709


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_history() -> tuple[dict[str, dict], dict[str, str]]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    candles = {}
    sources = {str(MANIFEST): digest(MANIFEST)}
    for symbol, item in sorted(manifest["symbols"].items()):
        if symbol in EXCLUDED:
            continue
        path = Path(item["path"])
        raw = path.read_bytes()
        found = hashlib.sha256(raw).hexdigest()
        if found != item["sha256"]:
            raise ValueError(f"source hash mismatch: {symbol}")
        candle = pickle.loads(raw)
        lengths = {key: len(candle[key]) for key in (
            "time", "open", "high", "low", "close", "volume")}
        if len(set(lengths.values())) != 1 or not lengths["time"]:
            raise ValueError(f"invalid history: {symbol} {lengths}")
        times = np.asarray(candle["time"], dtype=np.int64)
        if np.any(np.diff(times) <= 0):
            raise ValueError(f"non-increasing history: {symbol}")
        candles[symbol] = candle
        sources[str(path)] = found
    if len(candles) != 10:
        raise ValueError(f"expected ten non-index equities, got {len(candles)}")
    return candles, sources


def feature_rows(candles: dict[str, dict], lookback: int) -> pd.DataFrame:
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
    frame["cross_z"] = (
        (frame.signal_return - frame.cross_mean)
        / frame.cross_std.replace(0, np.nan)
    )
    frame["direction_n"] = -np.sign(
        frame.signal_return.to_numpy(float)).astype(np.int8)
    return frame.replace([np.inf, -np.inf], np.nan).dropna()


def select_entries(frame: pd.DataFrame, z_threshold: float,
                   move_threshold: float) -> pd.DataFrame:
    return frame[
        frame.cross_z.abs().ge(z_threshold)
        & frame.signal_return.abs().ge(move_threshold)
        & frame.universe_n.ge(MIN_UNIVERSE)
    ].copy()


def year_of(row: dict) -> int:
    return pd.Timestamp(float(row["entry_time"]), unit="s", tz="UTC").year


def rows_for_years(rows: list[dict], years: tuple[int, ...]) -> list[dict]:
    wanted = set(years)
    return [row for row in rows if year_of(row) in wanted]


def market_days(candles: dict[str, dict], years: tuple[int, ...]) -> int:
    wanted = set(years)
    reference = candles["AAPLUSDT"]
    stamps = pd.to_datetime(
        np.asarray(reference["time"], dtype=np.int64), unit="s", utc=True)
    return len({stamp.date() for stamp in stamps if stamp.year in wanted})


def gated(rows: list[dict], years: tuple[int, ...]) -> list[dict]:
    return portfolio_gate(
        rows_for_years(rows, years), None,
        rolling_hour_cap=4, rank_key=quality.ranker("z_high"),
        loss_streak_cap=2)


def metrics(rows: list[dict], margin: float, market_day_count: int) -> dict:
    value = economics.simulate(rows, margin)
    if not rows:
        value.update({
            "profitable_exit_rate_pct": 0.0,
            "literal_tp_rate_pct": 0.0,
            "max_loss_streak": 0,
            "trades_per_market_day": 0.0,
            "active_market_days": 0,
            "mean_signed_price_return_pct": 0.0,
            "projected_22_market_day_pct": 0.0,
            "required_mean_signed_move_at_actual_frequency_pct": None,
            "economic_gap_multiple": None,
            "economic_30pct_gate_pass": False,
            "concentration": {},
        })
        return value
    ordered = sorted(rows, key=lambda row: (
        float(row["exit_time"]), float(row["entry_time"]), str(row["symbol"])))
    streak = maximum = 0
    for row in ordered:
        streak = streak + 1 if float(row["signed_return"]) <= 0 else 0
        maximum = max(maximum, streak)
    signed = np.asarray([float(row["signed_return"]) for row in rows])
    frequency = len(rows) / max(market_day_count, 1)
    mean_signed = float(signed.mean())
    periods = frequency * MONTH_MARKET_DAYS
    trade_factor = 1.0 + margin * LEVERAGE * mean_signed
    projected = (100 * (trade_factor ** periods - 1)
                 if periods > 0 and trade_factor > 0 else 0.0)
    required = (
        ((1 + MONTH_TARGET) ** (1 / periods) - 1)
        / (margin * LEVERAGE) * 100 if periods > 0 else None)
    mandatory = (
        ((1 + MONTH_TARGET) **
         (1 / (MONTH_MARKET_DAYS * MINIMUM_DAILY_TRADES)) - 1)
        / (margin * LEVERAGE) * 100)
    frame = pd.DataFrame(rows)
    scans = frame.groupby("entry_time").size()
    hours = frame.assign(hour=(frame.entry_time // 3600).astype(int)).groupby(
        "hour").size()
    active_days = len({
        pd.Timestamp(float(row["entry_time"]), unit="s", tz="UTC").date()
        for row in rows
    })
    value.update({
        "profitable_exit_rate_pct": round(100 * float(np.mean(signed > 0)), 4),
        "literal_tp_rate_pct": round(100 * float(np.mean(
            frame.outcome_code.to_numpy(int) == 1)), 4),
        "max_loss_streak": maximum,
        "trades_per_market_day": round(frequency, 4),
        "active_market_days": active_days,
        "market_days": market_day_count,
        "mean_signed_price_return_pct": round(100 * mean_signed, 6),
        "projected_22_market_day_pct": round(projected, 6),
        "required_mean_signed_move_at_actual_frequency_pct":
            round(required, 6) if required is not None else None,
        "mandatory_three_daily_mean_signed_move_pct":
            round(mandatory, 6),
        "economic_gap_multiple": (
            round(required / (100 * mean_signed), 4)
            if required is not None and mean_signed > 0 else None),
        "economic_30pct_gate_pass": bool(projected >= 30),
        "concentration": {
            "max_entries_same_15m": int(scans.max()),
            "max_entries_same_clock_hour": int(hours.max()),
            "entries_in_multi_signal_scans_pct": round(
                100 * int(scans[scans > 1].sum()) / len(rows), 4),
            "entries_in_hours_with_3plus_pct": round(
                100 * int(hours[hours >= 3].sum()) / len(rows), 4),
        },
    })
    return value


def bundle(rows: list[dict], candles: dict[str, dict],
           years: tuple[int, ...]) -> dict:
    accepted = gated(rows, years)
    return {
        "accepted_rows": len(accepted),
        "margin_3pct": metrics(
            accepted, .03, market_days(candles, years)),
        "margin_5pct": metrics(
            accepted, .05, market_days(candles, years)),
        "annual_3pct": {
            str(year): metrics(
                [row for row in accepted if year_of(row) == year], .03,
                market_days(candles, (year,)))
            for year in years
        },
    }


def objective_ok(value: dict) -> bool:
    months = len(value.get("monthly_return_pct", {}))
    return bool(
        value["n"] >= 100
        and value["return_pct"] > 0
        and value["dd_pct"] <= 5
        and value["negative_months"] == 0
        and value["trades_per_market_day"] >= MINIMUM_DAILY_TRADES
        and months > 0
        and value["months_ge_30pct"] == months
        and value["economic_30pct_gate_pass"]
    )


def stage_ok(stage: dict, years: tuple[int, ...]) -> bool:
    combined = stage["margin_3pct"]
    return bool(
        objective_ok(combined)
        and all(stage["annual_3pct"][str(year)]["return_pct"] > 0
                and stage["annual_3pct"][str(year)]["dd_pct"] <= 5
                for year in years)
    )


def frozen_parameters() -> dict:
    profile = json.loads(growth.OUT.read_text(encoding="utf-8"))
    parameters = profile["selection"]["selected"]["parameters"]
    vol = json.loads(volatility.OUT.read_text(encoding="utf-8"))
    quality_report = json.loads(quality.OUT.read_text(encoding="utf-8"))
    backing_report = json.loads(backing.OUT.read_text(encoding="utf-8"))
    cap_report = json.loads(hour_cap.OUT.read_text(encoding="utf-8"))
    frozen = {
        **{key: parameters[key] for key in (
            "lookback_bars", "z_threshold", "minimum_move_pct",
            "target_pct", "stop_pct", "hold_bars")},
        "maximum_atr14_fraction": vol["selection"][
            "selected_maximum_atr14_fraction"],
        "ranker": quality_report["selection"]["selected_ranker"],
        "loss_streak_cap": quality_report["selection"][
            "selected_loss_streak_cap"],
        "backing_level_lookback_bars": backing_report[
            "selected_lookback_bars"],
        "rolling_hour_cap": cap_report["selected_rolling_hour_cap"],
    }
    expected = {
        "lookback_bars": 2, "z_threshold": 2.0,
        "minimum_move_pct": .4, "target_pct": 2.0,
        "stop_pct": 1.0, "hold_bars": 64,
        "maximum_atr14_fraction": .01, "ranker": "z_high",
        "loss_streak_cap": 2, "backing_level_lookback_bars": 32,
        "rolling_hour_cap": 4,
    }
    if frozen != expected:
        raise ValueError(f"current route drifted: {frozen}")
    return frozen


def main() -> None:
    parameters = frozen_parameters()
    candles, sources = load_history()
    features = feature_rows(candles, int(parameters["lookback_bars"]))
    entries = select_entries(
        features, float(parameters["z_threshold"]),
        float(parameters["minimum_move_pct"]) / 100)
    entries = volatility.add_atr14(entries, candles)
    entries = entries[entries.atr14_fraction.le(
        float(parameters["maximum_atr14_fraction"]))]
    entries = backing.guarded(backing.add_backing_distance(
        entries, candles, int(parameters["backing_level_lookback_bars"])))
    resolved = resolver.add_outcomes(
        entries, candles, float(parameters["target_pct"]),
        float(parameters["stop_pct"]), int(parameters["hold_bars"]))
    resolved["module"] = "current_stock_route_cash_transfer"
    resolved["module_priority"] = 0
    records = resolved.to_dict("records")

    development = bundle(records, candles, DEVELOPMENT)
    validation = test = confirmation = None
    status = "REJECTED_DEVELOPMENT"
    if stage_ok(development, DEVELOPMENT):
        validation = bundle(records, candles, VALIDATION)
        status = "REJECTED_VALIDATION"
        if stage_ok(validation, VALIDATION):
            test = bundle(records, candles, TEST)
            status = "REJECTED_TEST"
            if stage_ok(test, TEST):
                confirmation = bundle(records, candles, CONFIRMATION)
                status = ("CROSS_ERA_TRANSFER_CANDIDATE"
                          if stage_ok(confirmation, CONFIRMATION)
                          else "REJECTED_CONFIRMATION")
    accepted_development = gated(records, DEVELOPMENT)
    pd.DataFrame(accepted_development).to_csv(ROWS_OUT, index=False)
    report = {
        "status": status,
        "ready_for_live_orders": False,
        "orders_possible": False,
        "protocol": {
            "route": "exact current stock shadow parameters",
            "direct_venue_history": "July-October 2026 only",
            "transfer_source": "hashed 15m underlying cash-proxy OHLCV",
            "development": list(DEVELOPMENT),
            "validation": list(VALIDATION),
            "test": list(TEST),
            "confirmation": list(CONFIRMATION),
            "later_periods": "not evaluated after first failed objective gate",
            "entry": "next 15m market open after closed signal bar",
            "orders": "market only",
            "money": "client 3% or 5% current-equity margin; fixed 10x",
            "economic_gate": (
                "+30% over 22 market days; 1.327709% constant signed "
                "price move per trade at 3% margin, 10x and 3 trades/day"),
        },
        "frozen_parameters": parameters,
        "transfer_adaptation": {
            "cash_non_index_symbols": len(candles),
            "minimum_aligned_universe": MIN_UNIVERSE,
            "reason": (
                "80% coverage of the ten-symbol cash universe; MU and MRVL "
                "have shorter histories"),
        },
        "raw_feature_rows": len(features),
        "selected_entry_rows": len(entries),
        "resolved_rows": len(resolved),
        "development": development,
        "development_objective_pass": stage_ok(development, DEVELOPMENT),
        "validation": validation,
        "test": test,
        "confirmation": confirmation,
        "sources": {
            **sources,
            str(growth.OUT): digest(growth.OUT),
            str(volatility.OUT): digest(volatility.OUT),
            str(quality.OUT): digest(quality.OUT),
            str(backing.OUT): digest(backing.OUT),
            str(hour_cap.OUT): digest(hour_cap.OUT),
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "Cash-proxy transfer tests structure, not historical X-Perp fills.",
            "The cash universe is ten equities versus 44 current X-Perps.",
            "Cash sessions end daily, so a 64-bar hold expires at the session gap.",
            "Gross results exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True),
                   encoding="utf-8")
    print(json.dumps({
        "status": status,
        "features": len(features), "entries": len(entries),
        "development": development,
        "later_periods_sealed": validation is None,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
