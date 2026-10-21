"""Cross-era audit of volatility-scaled residual mean reversion.

Raw cross-sectional returns systematically nominate high-volatility names such
as TSLA and NVDA.  This route first scales each closed-bar return by the stock's
causal ATR, then forms the simultaneous cross-sectional residual.  The raw
price residual remains a minimum-move guard.  Development now includes the
already-inspected 2021-2022 failure; 2023-2024 is the untouched validation.
"""
from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import stock_cash_residual_cross_section_oct21_audit as base
import stock_cash_xperp_leadlag_oct6_audit as resolver


OUT = base.ROOT / "stock_vol_scaled_residual_cross_section_oct21.json"
ROWS_OUT = base.ROOT / "stock_vol_scaled_residual_cross_section_oct21.csv"
DEVELOPMENT = (2017, 2018, 2019, 2020, 2021, 2022)
VALIDATION = (2023, 2024)
CONFIRMATION = (2025, 2026)
ATR_BARS = 14


def feature_rows(candles: dict[str, dict], lookback: int) -> pd.DataFrame:
    rows = []
    for symbol, candle in candles.items():
        times = np.asarray(candle["time"], dtype=np.int64)
        open_ = np.asarray(candle["open"], dtype=float)
        high = np.asarray(candle["high"], dtype=float)
        low = np.asarray(candle["low"], dtype=float)
        close = np.asarray(candle["close"], dtype=float)
        previous = np.r_[open_[0], close[:-1]]
        true_range = np.maximum(
            high - low, np.maximum(abs(high - previous), abs(low - previous)))
        atr = pd.Series(true_range).rolling(ATR_BARS).mean().to_numpy()
        for index in range(max(lookback, ATR_BARS - 1), len(times) - 1):
            if (times[index + 1] - times[index] != 900
                    or times[index] - times[index - lookback] != 900 * lookback):
                continue
            move = close[index] / close[index - lookback] - 1.0
            atr_fraction = atr[index] / close[index]
            if not np.isfinite(atr_fraction) or atr_fraction <= 0:
                continue
            scaled = move / (atr_fraction * math.sqrt(lookback))
            rows.append((symbol, index + 1, int(times[index + 1]), move,
                         atr_fraction, scaled))
    frame = pd.DataFrame(rows, columns=(
        "symbol", "entry_index", "entry_time", "signal_return",
        "atr14_fraction", "scaled_return"))
    scaled_group = frame.groupby("entry_time", sort=False).scaled_return
    raw_group = frame.groupby("entry_time", sort=False).signal_return
    frame["universe_n"] = scaled_group.transform("size")
    frame["cross_mean"] = raw_group.transform("mean")
    frame["residual_return"] = frame.signal_return - frame.cross_mean
    frame["scaled_cross_mean"] = scaled_group.transform("mean")
    frame["scaled_cross_std"] = scaled_group.transform("std")
    frame["scaled_residual"] = frame.scaled_return - frame.scaled_cross_mean
    frame["cross_z"] = (
        frame.scaled_residual / frame.scaled_cross_std.replace(0, np.nan))
    frame["direction_n"] = -np.sign(
        frame.scaled_residual.to_numpy(float)).astype(np.int8)
    frame["year"] = pd.to_datetime(
        frame.entry_time, unit="s", utc=True).dt.year.astype(np.int16)
    return frame.replace([np.inf, -np.inf], np.nan).dropna()


def select_entries(frame: pd.DataFrame, z_threshold: float,
                   residual_threshold: float) -> pd.DataFrame:
    return frame[
        frame.cross_z.abs().ge(z_threshold)
        & frame.residual_return.abs().ge(residual_threshold)
        & frame.universe_n.ge(base.MIN_UNIVERSE)
    ].copy()


def parameters(lookback, z_threshold, residual_threshold,
               target, stop, hold) -> dict:
    return {
        "lookback_bars": lookback, "atr_bars": ATR_BARS,
        "z_threshold": z_threshold,
        "minimum_raw_residual_pct": 100 * residual_threshold,
        "target_pct": target, "stop_pct": stop, "hold_bars": hold,
    }


def resolve_candidate(rule: dict, features: pd.DataFrame,
                      candles: dict[str, dict], years: tuple[int, ...]) -> pd.DataFrame:
    entries = select_entries(
        features, float(rule["z_threshold"]),
        float(rule["minimum_raw_residual_pct"]) / 100)
    entries = entries[entries.year.isin(years)]
    found = resolver.add_outcomes(
        entries, candles, float(rule["target_pct"]),
        float(rule["stop_pct"]), int(rule["hold_bars"]))
    found["module"] = "vol_scaled_residual_cross_section_reversion"
    found["module_priority"] = 0
    return found


def public(item: dict) -> dict:
    return {key: value for key, value in item.items()
            if key not in ("_accepted",)}


def main() -> None:
    candles, sources = base.parent.load_history()
    feature_cache = {}
    raw_candidates = []
    searched = feature_count = 0
    for lookback in base.LOOKBACKS:
        features = feature_rows(candles, lookback)
        feature_cache[lookback] = features
        feature_count += len(features)
        broad = select_entries(
            features, min(base.Z_THRESHOLDS), min(base.RESIDUAL_THRESHOLDS))
        broad = broad[broad.year.isin(DEVELOPMENT)]
        for target, stop, hold in itertools.product(
                base.TARGETS, base.STOPS, base.HOLDS):
            resolved_broad = resolver.add_outcomes(
                broad, candles, target, stop, hold)
            resolved_broad["module"] = (
                "vol_scaled_residual_cross_section_reversion")
            resolved_broad["module_priority"] = 0
            for z_threshold, residual_threshold in itertools.product(
                    base.Z_THRESHOLDS, base.RESIDUAL_THRESHOLDS):
                searched += 1
                resolved = resolved_broad[
                    resolved_broad.cross_z.abs().ge(z_threshold)
                    & resolved_broad.residual_return.abs().ge(
                        residual_threshold)]
                if len(resolved) < 450:
                    continue
                dev = base.raw_bundle(resolved, candles, DEVELOPMENT)
                value = dev["combined"]
                annual = list(dev["annual"].values())
                if (value["return_pct"] <= 0 or value["dd_pct"] > 18
                        or value["trades_per_market_day"] < 2.5
                        or sum(v["return_pct"] > 0 for v in annual) < 4
                        or min(v["return_pct"] for v in annual) < -3):
                    continue
                score = (min(v["return_pct"] for v in annual)
                         + .2 * value["return_pct"] - value["dd_pct"])
                raw_candidates.append({
                    "parameters": parameters(
                        lookback, z_threshold, residual_threshold,
                        target, stop, hold),
                    "development_raw": dev,
                    "development_raw_score": score,
                })
        print(json.dumps({"lookback": lookback, "searched": searched,
                          "raw_passed": len(raw_candidates)}), flush=True)

    raw_candidates.sort(key=lambda item: item["development_raw_score"],
                        reverse=True)
    exact = []
    for item in raw_candidates[:base.TOP_EXACT]:
        rule = item["parameters"]
        resolved = resolve_candidate(
            rule, feature_cache[rule["lookback_bars"]], candles, DEVELOPMENT)
        stage, accepted = base.exact_bundle(resolved, candles, DEVELOPMENT)
        value = stage["margin_3pct"]
        annual = list(stage["annual_3pct"].values())
        if not (value["n"] >= 450 and value["return_pct"] > 0
                and value["dd_pct"] <= 12
                and value["trades_per_market_day"] >= 2.5
                and sum(v["return_pct"] > 0 for v in annual) >= 5
                and min(v["return_pct"] for v in annual) >= -3):
            continue
        exact.append({
            **item, "development": stage, "_accepted": accepted,
            "development_score": (
                min(v["return_pct"] for v in annual)
                + .2 * value["return_pct"] - value["dd_pct"]),
        })
    exact.sort(key=lambda item: item["development_score"], reverse=True)

    validated = []
    for item in exact:
        rule = item["parameters"]
        resolved = resolve_candidate(
            rule, feature_cache[rule["lookback_bars"]], candles, VALIDATION)
        stage, _ = base.exact_bundle(resolved, candles, VALIDATION)
        item["validation"] = stage
        if base.later_ok(stage):
            validated.append(item)

    frozen = validated[0] if validated else None
    confirmation = None
    status = "REJECTED_DEVELOPMENT" if not exact else "REJECTED_VALIDATION"
    if frozen:
        rule = frozen["parameters"]
        resolved = resolve_candidate(
            rule, feature_cache[rule["lookback_bars"]], candles, CONFIRMATION)
        confirmation, accepted = base.exact_bundle(
            resolved, candles, CONFIRMATION)
        frozen["_accepted"] = accepted
        status = ("CROSS_ERA_CANDIDATE" if base.later_ok(confirmation)
                  else "REJECTED_CONFIRMATION")

    chosen = frozen if frozen else (exact[0] if exact else None)
    pd.DataFrame(chosen["_accepted"] if chosen else []).to_csv(
        ROWS_OUT, index=False)
    objective = bool(
        frozen and confirmation and base.objective_ok(frozen["development"])
        and base.objective_ok(frozen["validation"])
        and base.objective_ok(confirmation))
    report = {
        "status": status, "ready_for_live_orders": False,
        "orders_possible": False,
        "protocol": {
            "hypothesis": (
                "normalize each stock move by causal ATR before measuring and "
                "reversing its simultaneous cross-sectional residual"),
            "development": list(DEVELOPMENT),
            "untouched_validation": list(VALIDATION),
            "confirmation": list(CONFIRMATION),
            "confirmation_sealed_until_validation_pass": True,
            "signal_window_requires_contiguous_15m_bars": True,
            "entry": "next 15m market open after the closed signal bar",
            "orders": "market only",
            "money": "client 3% or 5% current-equity margin; fixed 10x",
            "targets": "literal 0.5-2.0% price moves; stop-first ambiguity",
        },
        "grid": {
            "feature_rows_across_lookbacks": feature_count,
            "searched_candidates": searched,
            "raw_development_passes": len(raw_candidates),
            "exact_development_passes": len(exact),
            "validation_passes": len(validated),
            "exact_replayed": min(base.TOP_EXACT, len(raw_candidates)),
        },
        "selected": public(chosen) if chosen else None,
        "confirmation": confirmation,
        "all_stages_meet_monthly_objective": objective,
        "top_development": [public(item) for item in exact[:10]],
        "sources": {
            **sources,
            str(Path(base.__file__).resolve()):
                base.digest(Path(base.__file__).resolve()),
            str(Path(resolver.__file__).resolve()):
                base.digest(Path(resolver.__file__).resolve()),
            str(Path(__file__).resolve()): base.digest(Path(__file__).resolve()),
        },
        "limitations": [
            "Cash-proxy history tests structure, not historical X-Perp fills.",
            "The historical universe has ten equities and only eight before MU.",
            "Session gaps terminate positions and cannot form a signal window.",
            "Gross returns exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": status, "grid": report["grid"],
        "selected": report["selected"], "confirmation": confirmation,
        "all_stages_meet_monthly_objective": objective,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
