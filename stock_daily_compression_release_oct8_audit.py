"""Chronological stock compression/release audit with literal price moves.

Signals use completed US cash sessions from the deep Dukascopy history.  Entry
is the next session's first 15-minute market open.  This is a daily volatility
expansion hypothesis, distinct from the rejected intraday opening routes.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

import stock_15m_multisession_regime_oct6_audit as source
from src.research_profit_gate import economic_gate
from stock_5m_family_oct3_audit import regular_groups


ROOT = Path("reports/audit_2026_09_08")
OUT = ROOT / "stock_daily_compression_release_oct8.json"
TARGETS = (1.5, 2.0)
STOPS = (.5, .75, 1.0)
HOLDS = (1, 2, 3)
DEVELOPMENT = (2017, 2018, 2019, 2020)
VALIDATION = (2021,)
CONFIRMATION = (2022,)
HIDDEN = (2023, 2024, 2025, 2026)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def daily_features(symbol: str, candle: dict) -> pd.DataFrame:
    groups = regular_groups(candle)
    lengths = [len(indices) for indices in groups.values()]
    expected = int(round(float(np.median(lengths))))
    complete = [(day, indices) for day, indices in sorted(groups.items())
                if len(indices) >= int(.9 * expected)]
    rows = []
    daily = []
    for day, indices in complete:
        first, last = indices[0], indices[-1]
        opening = float(candle["open"][first])
        high = max(float(candle["high"][index]) for index in indices)
        low = min(float(candle["low"][index]) for index in indices)
        close = float(candle["close"][last])
        volume = sum(float(candle["volume"][index]) for index in indices)
        daily.append({"day": day, "first": first, "last": last,
                      "open": opening, "high": high, "low": low,
                      "close": close, "volume": volume})
    for position in range(21, len(daily) - 1):
        current = daily[position]
        following = daily[position + 1]
        history = daily[:position]
        ranges = [(row["high"] - row["low"]) / row["close"]
                  for row in history[-20:]]
        current_range = (current["high"] - current["low"]) / current["close"]
        median_range = float(np.median(ranges)) or 1e-12
        recent_ranges = [(row["high"] - row["low"]) / row["close"]
                         for row in history[-3:]]
        span = max(current["high"] - current["low"], 1e-12)
        closes = [row["close"] for row in daily]
        path5 = sum(abs(closes[index] - closes[index - 1])
                    for index in range(position - 4, position + 1))
        prior3_high = max(row["high"] for row in history[-3:])
        prior3_low = min(row["low"] for row in history[-3:])
        prior5_high = max(row["high"] for row in history[-5:])
        prior5_low = min(row["low"] for row in history[-5:])
        rows.append({
            "symbol": symbol, "signal_date": current["day"],
            "session_date": following["day"],
            "entry_index": following["first"],
            "entry_time": int(candle["time"][following["first"]]),
            "year": int(following["day"][:4]),
            "range_ratio": current_range / median_range,
            "prior3_range_ratio": float(np.mean(recent_ranges)) / median_range,
            "body_ratio": (current["close"] - current["open"]) / span,
            "close_pos": (current["close"] - current["low"]) / span,
            "ret1": current["close"] / history[-1]["close"] - 1,
            "ret5": current["close"] / history[-5]["close"] - 1,
            "ret20": current["close"] / history[-20]["close"] - 1,
            "eff5": abs(current["close"] - history[-5]["close"]) / path5
            if path5 else 0.0,
            "inside_day": int(current["high"] <= history[-1]["high"]
                              and current["low"] >= history[-1]["low"]),
            "breakout3_up": current["close"] / prior3_high - 1,
            "breakout3_down": prior3_low / current["close"] - 1,
            "breakout5_up": current["close"] / prior5_high - 1,
            "breakout5_down": prior5_low / current["close"] - 1,
            "volume_ratio": current["volume"] /
            (float(np.median([row["volume"] for row in history[-20:]])) or 1.0),
        })
    return pd.DataFrame(rows)


def build_features(candles: dict) -> tuple[pd.DataFrame, dict]:
    frames = [daily_features(symbol, candles[symbol])
              for symbol in source.SYMBOLS]
    qqq = daily_features("QQQUSDT", candles["QQQUSDT"])[
        ["signal_date", "ret1", "ret5", "ret20", "range_ratio"]
    ].rename(columns={name: f"qqq_{name}" for name in
                      ("ret1", "ret5", "ret20", "range_ratio")})
    frame = pd.concat(frames, ignore_index=True).merge(
        qqq, on="signal_date", how="inner", validate="many_to_one")
    group = frame.groupby("signal_date", sort=False)
    frame["breadth1"] = group.ret1.transform(lambda value: (value > 0).mean())
    mean = group.ret5.transform("mean")
    std = group.ret5.transform("std").replace(0, np.nan)
    frame["cross_z5"] = (frame.ret5 - mean) / std
    metadata = {}
    for symbol in source.SYMBOLS:
        groups = regular_groups(candles[symbol])
        lengths = [len(value) for value in groups.values()]
        expected = int(round(float(np.median(lengths))))
        regular = [index for _, indices in sorted(groups.items())
                   if len(indices) >= int(.9 * expected) for index in indices]
        metadata[symbol] = {"bars_per_session": expected,
                            "regular": np.asarray(regular, dtype=np.int64)}
    return frame.replace([np.inf, -np.inf], np.nan).dropna(), metadata


def profiles() -> list[dict]:
    output = []
    for comp, close_pos, align in itertools.product(
            (.5, .7, .85), (.7, .85), (False, True)):
        output.append({"family": "compression_pressure", "compression": comp,
                       "close_pos": close_pos, "align_ret5": align})
    for close_pos, align in itertools.product((.7, .85), (False, True)):
        output.append({"family": "inside_pressure", "close_pos": close_pos,
                       "align_ret5": align})
    for prior_comp, release, close_pos, body in itertools.product(
            (.6, .8), (1.25, 1.5), (.7, .85), (.3, .5)):
        output.append({"family": "volatility_release",
                       "prior_compression": prior_comp, "release": release,
                       "close_pos": close_pos, "body": body})
    for lookback, comp, close_pos in itertools.product(
            (3, 5), (.8, 1.0), (.7, .85)):
        output.append({"family": "compression_breakout", "lookback": lookback,
                       "compression": comp, "close_pos": close_pos})
    return output


def directed(frame: pd.DataFrame, direction: np.ndarray) -> pd.DataFrame:
    found = frame.copy()
    found["direction_n"] = direction.astype(np.int8)
    found["close_pos_dir"] = np.where(
        found.direction_n > 0, found.close_pos, 1 - found.close_pos)
    found["ret5_dir"] = found.ret5 * found.direction_n
    found["ret20_dir"] = found.ret20 * found.direction_n
    found["qqq_ret5_dir"] = found.qqq_ret5 * found.direction_n
    found["cross_z5_dir"] = found.cross_z5 * found.direction_n
    found["breadth_dir"] = np.where(
        found.direction_n > 0, found.breadth1, 1 - found.breadth1)
    return found


def select(frame: pd.DataFrame, profile: dict) -> pd.DataFrame:
    family = profile["family"]
    if family in ("compression_pressure", "inside_pressure", "volatility_release"):
        direction = np.sign(frame.body_ratio.to_numpy(float))
    else:
        up = frame[f"breakout{profile['lookback']}_up"].gt(0)
        down = frame[f"breakout{profile['lookback']}_down"].gt(0)
        direction = np.where(up & ~down, 1, np.where(down & ~up, -1, 0))
    found = directed(frame, direction)
    if family == "compression_pressure":
        mask = (found.direction_n.ne(0)
                & found.range_ratio.le(profile["compression"])
                & found.close_pos_dir.ge(profile["close_pos"])
                & found.body_ratio.abs().ge(.25))
        if profile["align_ret5"]:
            mask &= found.ret5_dir.ge(0)
    elif family == "inside_pressure":
        mask = (found.direction_n.ne(0) & found.inside_day.eq(1)
                & found.close_pos_dir.ge(profile["close_pos"])
                & found.body_ratio.abs().ge(.25))
        if profile["align_ret5"]:
            mask &= found.ret5_dir.ge(0)
    elif family == "volatility_release":
        mask = (found.direction_n.ne(0)
                & found.prior3_range_ratio.le(profile["prior_compression"])
                & found.range_ratio.ge(profile["release"])
                & found.close_pos_dir.ge(profile["close_pos"])
                & found.body_ratio.abs().ge(profile["body"]))
    else:
        mask = (found.direction_n.ne(0)
                & found.range_ratio.le(profile["compression"])
                & found.close_pos_dir.ge(profile["close_pos"]))
    selected = found.loc[mask].copy()
    selected["entry_spec"] = family
    selected["score"] = (selected.close_pos_dir + selected.body_ratio.abs()
                         + .1 * selected.cross_z5_dir.clip(lower=0))
    return selected


def evaluate(frame: pd.DataFrame, features: pd.DataFrame,
             years: tuple[int, ...]) -> dict:
    chosen = frame[frame.year.isin(years)].copy()
    market_days = features[features.year.isin(years)].session_date.nunique()
    accepted = source.portfolio_gate(
        chosen.to_dict("records"), None,
        rank_key=lambda row: (-float(row["score"]), str(row["symbol"])))
    metrics = source.account_metrics(chosen, market_days)
    hits = sum(int(row["outcome_code"]) == 1 for row in accepted)
    signed = [float(row["signed_return"]) for row in accepted]
    frequency = len(accepted) / max(1, market_days)
    metrics.update({
        "literal_target_hits": hits,
        "literal_target_hit_rate": round(100 * hits / len(accepted), 4)
        if accepted else 0.0,
        "economic_gate": economic_gate(signed, frequency),
    })
    return metrics


def stage_ok(value: dict, minimum: int) -> bool:
    return bool(value["n"] >= minimum and value["return_pct"] > 0
                and value["literal_target_hit_rate"] >= 55
                and value["dd_pct"] <= 10
                and value["max_loss_streak"] <= 8)


def objective_ok(value: dict) -> bool:
    return bool(value["economic_gate"]["passes"]
                and 75 <= value["literal_target_hit_rate"] <= 95
                and value["dd_pct"] <= 5 and value["negative_months"] == 0
                and value["months_ge_30pct"] == value["months"])


def main() -> None:
    candles, sources = source.load_history()
    features, metadata = build_features(candles)
    candidates = []
    frontier = []
    rows_by_key = {}
    for profile, target, stop, hold in itertools.product(
            profiles(), TARGETS, STOPS, HOLDS):
        proposed = select(features, profile)
        routed = source.add_outcomes(
            proposed, candles, metadata, target, stop, hold)
        routed["module"] = "daily_" + profile["family"]
        routed["module_priority"] = {
            "compression_breakout": 0,
            "volatility_release": 1,
            "compression_pressure": 2,
            "inside_pressure": 3,
        }[profile["family"]]
        development = evaluate(routed, features, DEVELOPMENT)
        item = {"profile": profile, "target_pct": target,
                "stop_pct": stop, "hold_sessions": hold,
                "development": development}
        if development["n"] >= 80:
            frontier.append(item)
        if not stage_ok(development, 150):
            continue
        validation = evaluate(routed, features, VALIDATION)
        item["validation"] = validation
        if not stage_ok(validation, 30):
            continue
        item["score"] = round(
            min(development["return_pct"] / len(DEVELOPMENT),
                validation["return_pct"])
            - max(development["dd_pct"], validation["dd_pct"])
            + min(development["literal_target_hit_rate"],
                  validation["literal_target_hit_rate"]) / 10, 6)
        key = json.dumps({"profile": profile, "target": target,
                          "stop": stop, "hold": hold}, sort_keys=True)
        rows_by_key[key] = routed
        item["key"] = key
        candidates.append(item)
    candidates.sort(key=lambda item: item["score"], reverse=True)
    frontier.sort(key=lambda item: (
        item["development"]["return_pct"] - item["development"]["dd_pct"],
        item["development"]["literal_target_hit_rate"]), reverse=True)
    frozen, families = [], set()
    for item in candidates:
        family = item["profile"]["family"]
        if family not in families:
            frozen.append(item)
            families.add(family)
    ensemble = (pd.concat([rows_by_key[item["key"]] for item in frozen],
                          ignore_index=True) if frozen else features.iloc[:0].copy())
    confirmation = evaluate(ensemble, features, CONFIRMATION) if frozen else None
    confirmation_pass = bool(confirmation and objective_ok(confirmation))
    hidden = ({str(year): evaluate(ensemble, features, (year,))
               for year in HIDDEN} if confirmation_pass else None)
    status = ("RESEARCH_CANDIDATE" if hidden and all(
        objective_ok(value) for value in hidden.values())
              else "REJECTED_HIDDEN" if hidden else "REJECTED_CONFIRMATION"
              if frozen else "REJECTED_DEVELOPMENT")
    report = {
        "status": status, "ready_for_live_orders": False,
        "protocol": {
            "hypothesis": "daily compression, inside pressure, and volatility release",
            "venue": "Dukascopy US cash-session CFD proxy",
            "entry": "next session first 15-minute market open",
            "targets_pct": list(TARGETS), "stops_pct": list(STOPS),
            "hold_sessions": list(HOLDS), "ambiguous_bar": "stop first",
            "development": list(DEVELOPMENT), "validation": list(VALIDATION),
            "confirmation": list(CONFIRMATION),
            "hidden_only_after_objective_confirmation": list(HIDDEN),
            "money": "3% current equity margin, fixed 10x gross",
            "orders": "market proxy only; no live orders",
        },
        "feature_rows": len(features), "profiles": len(profiles()),
        "routes_evaluated": len(profiles()) * len(TARGETS) * len(STOPS) * len(HOLDS),
        "qualified_development_validation": len(candidates),
        "frozen_one_per_family": frozen,
        "confirmation_2022": confirmation,
        "confirmation_objective_pass": confirmation_pass,
        "hidden": hidden,
        "frontier_top30": frontier[:30],
        "decision": "No runtime change unless every chronological objective gate passes.",
        "sources": {**sources, str(Path(__file__)): digest(Path(__file__))},
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "status", "feature_rows", "routes_evaluated",
        "qualified_development_validation", "frozen_one_per_family",
        "confirmation_2022", "confirmation_objective_pass", "hidden")},
        indent=2), flush=True)


if __name__ == "__main__":
    main()
