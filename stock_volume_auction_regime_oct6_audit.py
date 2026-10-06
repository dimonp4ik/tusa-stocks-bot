"""Causal opening-volume regimes with literal 0.5%-2% target labels.

The semantic rules are selected only through 2024.  Calendar 2025, observed
2026, and the direct X-Perp sample are revealed after one rule/exit is frozen.
All signals use completed regular-session bars and enter the next market bar.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from stock_5m_family_oct3_audit import regular_groups
from stock_module_cluster_audit import portfolio_gate
import stock_15m_multisession_regime_oct6_audit as base


ROOT = Path("reports/audit_2026_09_08")
OUT = ROOT / "stock_volume_auction_regime_oct6.json"
ROWS_OUT = ROOT / "stock_volume_auction_regime_oct6.csv"
SLOTS = (("open30", .08), ("open60", .16), ("open120", .32))
TARGETS = (.5, 1.0, 2.0)
STOPS = (.5, 1.0)
HOLD_SESSIONS = (1, 2, 3)

# Bounded, economically distinct hypotheses fixed before any test-year result.
SPECS = (
    {"name": "gap_volume_follow", "family": "gap_follow",
     "gap_atr": .30, "move_atr": .12, "rvol": 1.20,
     "location": .65, "market": -.05},
    {"name": "gap_volume_follow_strict", "family": "gap_follow",
     "gap_atr": .55, "move_atr": .20, "rvol": 1.50,
     "location": .72, "market": 0.0},
    {"name": "failed_gap_reversal", "family": "gap_fail",
     "gap_atr": .45, "move_atr": .12, "fill": .35,
     "rvol": 1.20, "market": -.10},
    {"name": "failed_gap_reversal_strict", "family": "gap_fail",
     "gap_atr": .75, "move_atr": .20, "fill": .60,
     "rvol": 1.45, "market": -.05},
    {"name": "idiosyncratic_volume_drive", "family": "idio_drive",
     "residual_atr": .30, "range_atr": .35, "rvol": 1.30,
     "location": .68},
    {"name": "idiosyncratic_volume_drive_strict", "family": "idio_drive",
     "residual_atr": .55, "range_atr": .55, "rvol": 1.70,
     "location": .75},
    {"name": "volume_exhaustion_reversal", "family": "exhaustion",
     "residual_atr": .65, "range_atr": .65, "rvol": 1.70,
     "wick": .18, "market": -.12},
    {"name": "volume_exhaustion_reversal_strict", "family": "exhaustion",
     "residual_atr": .95, "range_atr": .85, "rvol": 2.10,
     "wick": .28, "market": -.08},
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def session_rows(symbol: str, candle: dict) -> tuple[pd.DataFrame, int, np.ndarray]:
    groups = regular_groups(candle)
    lengths = [len(value) for value in groups.values()]
    expected = int(round(float(np.median(lengths))))
    complete = [(day, bars) for day, bars in sorted(groups.items())
                if len(bars) >= math.floor(.9*expected)]
    times = np.asarray(candle["time"], dtype=np.int64)
    rows: list[dict] = []
    regular: list[int] = []
    day_history: list[dict] = []
    for number, (day, bars) in enumerate(complete):
        regular.extend(bars)
        opens = np.asarray([candle["open"][i] for i in bars], dtype=float)
        highs = np.asarray([candle["high"][i] for i in bars], dtype=float)
        lows = np.asarray([candle["low"][i] for i in bars], dtype=float)
        closes = np.asarray([candle["close"][i] for i in bars], dtype=float)
        volumes = np.asarray([candle["volume"][i] for i in bars], dtype=float)
        day_item = {"close": float(closes[-1]), "high": float(highs.max()),
                    "low": float(lows.min()), "volume_curve": np.cumsum(volumes)}
        if number >= 21:
            previous = day_history[-1]
            true_ranges = []
            for position in range(len(day_history)-20, len(day_history)):
                item = day_history[position]
                prior_close = day_history[position-1]["close"]
                true_ranges.append(max(item["high"]-item["low"],
                                       abs(item["high"]-prior_close),
                                       abs(item["low"]-prior_close)))
            atr20 = float(np.mean(true_ranges))
            previous_close = float(previous["close"])
            if atr20 > 0 and previous_close > 0:
                for slot, fraction in SLOTS:
                    position = min(len(bars)-2, max(1, int(round((len(bars)-1)*fraction))))
                    signal_indices = bars[:position+1]
                    entry = bars[position+1]
                    session_open = float(opens[0])
                    close = float(closes[position])
                    high = float(highs[:position+1].max())
                    low = float(lows[:position+1].min())
                    curve_values = []
                    for item in day_history[-20:]:
                        curve = item["volume_curve"]
                        curve_values.append(float(curve[min(position, len(curve)-1)]))
                    expected_volume = float(np.median(curve_values))
                    cumulative_volume = float(volumes[:position+1].sum())
                    width = max(high-low, 1e-12)
                    rows.append({
                        "symbol": symbol, "session_date": day, "slot": slot,
                        "entry_index": entry, "entry_time": int(times[entry]),
                        "year": int(day[:4]), "atr20": atr20,
                        "gap_atr": (session_open-previous_close)/atr20,
                        "move_atr": (close-session_open)/atr20,
                        "range_atr": width/atr20,
                        "close_location": (close-low)/width,
                        "upper_wick_fraction": (high-close)/width,
                        "lower_wick_fraction": (close-low)/width,
                        "rvol": cumulative_volume/max(expected_volume, 1e-12),
                        "gap_fill": (close-session_open)/max(
                            abs(session_open-previous_close), 1e-12),
                    })
        day_history.append(day_item)
    return pd.DataFrame(rows), expected, np.asarray(regular, dtype=np.int64)


def build_features(candles: dict[str, dict]) -> tuple[pd.DataFrame, dict]:
    frames, metadata = [], {}
    for symbol in base.ALL_SYMBOLS:
        frame, bars_per_session, regular = session_rows(symbol, candles[symbol])
        frames.append(frame)
        metadata[symbol] = {"bars_per_session": bars_per_session,
                            "regular": regular}
    raw = pd.concat(frames, ignore_index=True)
    qqq = raw[raw.symbol.eq("QQQUSDT")][
        ["session_date", "slot", "gap_atr", "move_atr", "rvol"]
    ].rename(columns={"gap_atr": "qqq_gap_atr", "move_atr": "qqq_move_atr",
                      "rvol": "qqq_rvol"})
    stocks = raw[raw.symbol.ne("QQQUSDT")].copy()
    stocks = stocks.merge(qqq, on=["session_date", "slot"], how="inner",
                          validate="many_to_one")
    group = stocks.groupby(["session_date", "slot"], sort=False)
    stocks["breadth_up"] = group.move_atr.transform(lambda value: (value > 0).mean())
    stocks["residual_atr"] = stocks.move_atr-stocks.qqq_move_atr
    return stocks.replace([np.inf, -np.inf], np.nan).dropna(), metadata


def directed(frame: pd.DataFrame, direction: np.ndarray) -> pd.DataFrame:
    result = frame.copy()
    result["direction_n"] = direction.astype(np.int8)
    for name in ("gap_atr", "move_atr", "qqq_gap_atr", "qqq_move_atr",
                 "residual_atr"):
        result[f"{name}_dir"] = result[name]*result.direction_n
    result["location_dir"] = np.where(result.direction_n > 0,
                                       result.close_location,
                                       1-result.close_location)
    result["breadth_dir"] = np.where(result.direction_n > 0,
                                      result.breadth_up, 1-result.breadth_up)
    result["rejection_wick"] = np.where(result.direction_n > 0,
                                         result.lower_wick_fraction,
                                         result.upper_wick_fraction)
    return result


def apply_spec(frame: pd.DataFrame, spec: dict) -> pd.DataFrame:
    family = spec["family"]
    if family == "gap_follow":
        found = directed(frame, np.sign(frame.gap_atr.to_numpy(float)))
        mask = (found.gap_atr_dir.ge(spec["gap_atr"])
                & found.move_atr_dir.ge(spec["move_atr"])
                & found.rvol.ge(spec["rvol"])
                & found.location_dir.ge(spec["location"])
                & found.qqq_move_atr_dir.ge(spec["market"]))
    elif family == "gap_fail":
        found = directed(frame, -np.sign(frame.gap_atr.to_numpy(float)))
        mask = ((-found.gap_atr_dir).ge(spec["gap_atr"])
                & found.move_atr_dir.ge(spec["move_atr"])
                & (found.gap_fill*found.direction_n).ge(spec["fill"])
                & found.rvol.ge(spec["rvol"])
                & found.qqq_move_atr_dir.ge(spec["market"]))
    elif family == "idio_drive":
        found = directed(frame, np.sign(frame.residual_atr.to_numpy(float)))
        mask = (found.residual_atr_dir.ge(spec["residual_atr"])
                & found.range_atr.ge(spec["range_atr"])
                & found.rvol.ge(spec["rvol"])
                & found.location_dir.ge(spec["location"]))
    elif family == "exhaustion":
        found = directed(frame, -np.sign(frame.residual_atr.to_numpy(float)))
        mask = ((-found.residual_atr_dir).ge(spec["residual_atr"])
                & found.range_atr.ge(spec["range_atr"])
                & found.rvol.ge(spec["rvol"])
                & found.rejection_wick.ge(spec["wick"])
                & found.qqq_move_atr_dir.ge(spec["market"]))
    else:
        raise ValueError(family)
    output = found.loc[mask & found.direction_n.ne(0)].copy()
    output["entry_spec"] = spec["name"]
    output["score"] = (output.residual_atr.abs()+output.move_atr.abs()
                       +.25*output.range_atr+.1*np.log1p(output.rvol))
    return output


def accepted(frame: pd.DataFrame) -> list[dict]:
    return portfolio_gate(frame.to_dict("records"), None,
                          rank_key=lambda row: (-float(row["score"]),
                                                str(row["symbol"])))


def evaluate(frame: pd.DataFrame, features: pd.DataFrame,
             years: tuple[int, ...]) -> dict:
    subset = frame[frame.year.isin(years)].copy()
    market_days = features[features.year.isin(years)].session_date.nunique()
    rows = accepted(subset)
    value = base.account_metrics(subset, market_days)
    n = len(rows)
    hits = sum(int(row["outcome_code"]) == 1 for row in rows)
    stops = sum(int(row["outcome_code"]) == -1 for row in rows)
    value.update({
        "literal_target_hits": hits,
        "literal_target_hit_rate": round(100*hits/n, 4) if n else 0.0,
        "stops": stops, "timeouts": n-hits-stops,
    })
    return value


def development_ok(value: dict, annual: dict) -> bool:
    return bool(value["n"] >= 250 and value["return_pct"] > 0
                and value["literal_target_hit_rate"] >= 58
                and value["dd_pct"] <= 15
                and value["trades_per_market_day"] >= .5
                and sum(item["return_pct"] > 0 for item in annual.values()) >= 3)


def validation_ok(value: dict, annual: dict) -> bool:
    return bool(value["n"] >= 120 and value["return_pct"] > 0
                and value["literal_target_hit_rate"] >= 58
                and value["dd_pct"] <= 10
                and value["trades_per_market_day"] >= .5
                and all(item["return_pct"] > 0 for item in annual.values()))


def objective_ok(value: dict, minimum: int = 80) -> bool:
    return bool(value["n"] >= minimum and value["return_pct"] > 0
                and 75 <= value["literal_target_hit_rate"] <= 95
                and value["dd_pct"] <= 5 and value["negative_months"] == 0
                and value["trades_per_market_day"] >= 3
                and value["months_ge_30pct"] == value["months"])


def public(item: dict | None) -> dict | None:
    if item is None:
        return None
    return {key: value for key, value in item.items() if key != "_rows"}


def main() -> None:
    historical, historical_sources = base.load_history()
    direct, direct_sources = base.load_direct()
    history_features, history_meta = build_features(historical)
    direct_features, direct_meta = build_features(direct)
    qualified, frontier = [], []
    for spec in SPECS:
        entries = apply_spec(history_features, spec)
        for target, stop, sessions in itertools.product(
                TARGETS, STOPS, HOLD_SESSIONS):
            if len(entries) < 250:
                continue
            rows = base.add_outcomes(entries, historical, history_meta,
                                     target, stop, sessions)
            development = evaluate(rows, history_features, (2017, 2018, 2019, 2020))
            dev_annual = {str(year): evaluate(rows, history_features, (year,))
                          for year in range(2017, 2021)}
            item = {"entry": spec,
                    "exit": {"target_pct": target, "stop_pct": stop,
                             "hold_sessions": sessions},
                    "development_2017_2020": development,
                    "development_years": dev_annual, "_rows": rows}
            frontier.append(item)
            if not development_ok(development, dev_annual):
                continue
            selection = evaluate(rows, history_features, (2021, 2022))
            selection_annual = {str(year): evaluate(rows, history_features, (year,))
                                for year in (2021, 2022)}
            item["selection_2021_2022"] = selection
            item["selection_years"] = selection_annual
            if not validation_ok(selection, selection_annual):
                continue
            validation = evaluate(rows, history_features, (2023, 2024))
            validation_annual = {str(year): evaluate(rows, history_features, (year,))
                                 for year in (2023, 2024)}
            item["validation_2023_2024"] = validation
            item["validation_years"] = validation_annual
            if not validation_ok(validation, validation_annual):
                continue
            item["score"] = round(
                validation["return_pct"]-2*validation["dd_pct"]
                +.2*validation["literal_target_hit_rate"]
                +.1*selection["return_pct"], 8)
            qualified.append(item)
    qualified.sort(key=lambda item: item["score"], reverse=True)
    frontier.sort(key=lambda item: (
        item["development_2017_2020"]["return_pct"],
        item["development_2017_2020"]["literal_target_hit_rate"]), reverse=True)
    frozen = qualified[0] if qualified else None
    results = checks = None
    if frozen:
        rows = frozen["_rows"]
        direct_entries = apply_spec(direct_features, frozen["entry"])
        exit_spec = frozen["exit"]
        direct_rows = base.add_outcomes(
            direct_entries, direct, direct_meta, exit_spec["target_pct"],
            exit_spec["stop_pct"], exit_spec["hold_sessions"])
        rows.to_csv(ROWS_OUT, index=False)
        results = {
            "development_2017_2020": frozen["development_2017_2020"],
            "selection_2021_2022": frozen["selection_2021_2022"],
            "validation_2023_2024": frozen["validation_2023_2024"],
            "test_2025": evaluate(rows, history_features, (2025,)),
            "observed_2026": evaluate(rows, history_features, (2026,)),
            "direct_xperp": evaluate(direct_rows, direct_features, (2026,)),
        }
        checks = {name: objective_ok(value, 30 if name == "direct_xperp" else 80)
                  for name, value in results.items()}
    status = ("RESEARCH_CANDIDATE_AWAITING_NEW_FORWARD"
              if checks and all(checks.values()) else "REJECTED")
    report = {
        "status": status,
        "protocol": {
            "development": "2017-2020 cash proxy",
            "selection": "2021-2022 cash proxy",
            "validation": "2023-2024 cash proxy",
            "frozen_before": ["2025", "2026", "direct X-Perp"],
            "entry": "completed opening checkpoint; next regular-session market bar",
            "hypotheses": [spec["name"] for spec in SPECS],
            "target_pct": TARGETS, "stop_pct": STOPS,
            "hold_sessions": HOLD_SESSIONS,
            "win_definition": "literal target touch; positive timeout is not a win",
            "money": "3% entry-equity margin, fixed 10x gross",
        },
        "feature_rows": {"historical": len(history_features),
                         "direct": len(direct_features)},
        "qualified_before_test": len(qualified),
        "frozen_before_test": public(frozen),
        "top_before_test": [public(item) for item in qualified[:12]],
        "development_frontier": [public(item) for item in frontier[:15]],
        "results": results, "checks": checks,
        "sources": {"historical": historical_sources,
                    "direct": direct_sources,
                    str(Path(__file__)): digest(Path(__file__))},
        "limitations": [
            "Long-run cash volume is a proxy; direct X-Perp volume has different units.",
            "The bounded semantic entry/exit grid creates selection risk.",
            "Direct X-Perp history is short and any pass still needs new forward data.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": status, "feature_rows": report["feature_rows"],
                      "qualified_before_test": len(qualified),
                      "frozen": public(frozen), "results": results,
                      "checks": checks}, indent=2), flush=True)


if __name__ == "__main__":
    main()
