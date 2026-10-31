"""Long-history stock regime audit with fixed 0.5-2% price targets.

The signal is evaluated at three regular-session checkpoints from information
already closed at that instant.  Entry is the next market bar.  Positions may
remain open for one to three regular sessions, allowing the strategy to seek a
material move rather than scalp a few basis points.

2017-2021 is development, 2022-2023 is validation, and one policy is frozen
before 2024, 2025-2026 and direct X-Perp evidence are revealed.
"""
from __future__ import annotations

import collections
from datetime import datetime
import hashlib
import heapq
import itertools
import json
import math
import pickle
from pathlib import Path
from zoneinfo import ZoneInfo

import numba as nb
import numpy as np
import pandas as pd

from stock_module_cluster_audit import portfolio_gate


ROOT = Path("reports/audit_2026_09_08")
MANIFEST = ROOT / "stock_extended_history_manifest.json"
DIRECT_CACHE = ROOT / "stock_xperp_5m_oct3"
OUT = ROOT / "stock_15m_multisession_regime_oct6.json"
ROWS_OUT = ROOT / "stock_15m_multisession_regime_oct6.csv"
SYMBOLS = ("AAPLUSDT", "AMZNUSDT", "METAUSDT", "MRVLUSDT",
           "MSFTUSDT", "MUUSDT", "NVDAUSDT")
ALL_SYMBOLS = (*SYMBOLS, "QQQUSDT")
SLOTS = (("early", .20), ("middle", .50), ("late", .80))
NY = ZoneInfo("America/New_York")
TARGETS = (.5, 1.0, 2.0)
STOPS = (.5, 1.0)
HOLD_SESSIONS = (1, 2, 3)
SPECS = (
    {"name": "trend_loose", "family": "trend", "trend": .01,
     "short": 0.0, "intra": 0.0, "qqq": 0.0, "breadth": .50},
    {"name": "trend_strict", "family": "trend", "trend": .03,
     "short": .005, "intra": .002, "qqq": .01, "breadth": .625},
    {"name": "pullback_resume_loose", "family": "pullback", "trend": .02,
     "pullback": 0.0, "resume": .001, "qqq": 0.0, "breadth": .50},
    {"name": "pullback_resume_strict", "family": "pullback", "trend": .04,
     "pullback": -.005, "resume": .002, "qqq": .01, "breadth": .625},
    {"name": "relative_leader_loose", "family": "relative", "z": 1.0,
     "trend": .01, "intra": 0.0, "breadth": .50},
    {"name": "relative_leader_strict", "family": "relative", "z": 1.5,
     "trend": .03, "intra": .002, "breadth": .625},
    {"name": "gap_followthrough_loose", "family": "gap", "gap": .005,
     "intra": .001, "qqq_intra": 0.0, "breadth": .50},
    {"name": "gap_followthrough_strict", "family": "gap", "gap": .01,
     "intra": .003, "qqq_intra": .001, "breadth": .625},
    {"name": "late_breadth_continuation", "family": "late", "trend": .01,
     "intra": .008, "qqq_intra": .002, "breadth": .75},
)


def regular_groups(candles: dict) -> dict[str, list[int]]:
    """Group causal bars from the regular New York cash session."""
    groups: dict[str, list[int]] = {}
    for index, timestamp in enumerate(candles["time"]):
        local = datetime.fromtimestamp(int(timestamp), NY)
        minute = local.hour*60+local.minute
        if local.weekday() < 5 and 570 <= minute < 960:
            groups.setdefault(local.date().isoformat(), []).append(index)
    return groups


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_history() -> tuple[dict[str, dict], dict[str, str]]:
    raw = MANIFEST.read_bytes()
    manifest = json.loads(raw)
    candles = {}
    sources = {str(MANIFEST): hashlib.sha256(raw).hexdigest()}
    for symbol in ALL_SYMBOLS:
        meta = manifest["symbols"][symbol]
        path = Path(meta["path"])
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != meta["sha256"]:
            raise ValueError(f"historical hash mismatch: {symbol}")
        candles[symbol] = pickle.loads(payload)
        sources[str(path)] = digest
    return candles, sources


def load_direct() -> tuple[dict[str, dict], dict[str, str]]:
    candles, sources = {}, {}
    for symbol in ALL_SYMBOLS:
        path = DIRECT_CACHE / f"{symbol}_5min_current.pkl"
        payload = path.read_bytes()
        candles[symbol] = pickle.loads(payload)
        sources[str(path)] = hashlib.sha256(payload).hexdigest()
    return candles, sources


def session_table(symbol: str, candle: dict) -> tuple[pd.DataFrame, int, np.ndarray]:
    groups = regular_groups(candle)
    lengths = [len(value) for value in groups.values()]
    expected = int(round(float(np.median(lengths))))
    complete = [(day, bars) for day, bars in sorted(groups.items())
                if len(bars) >= math.floor(.9*expected)]
    rows = []
    regular = []
    close_history = []
    for session_number, (day, bars) in enumerate(complete):
        regular.extend(bars)
        first, last = bars[0], bars[-1]
        session_open = float(candle["open"][first])
        session_close = float(candle["close"][last])
        if session_number < 21 or session_open <= 0:
            close_history.append(session_close)
            continue
        previous = close_history[-1]
        ret5 = previous/close_history[-6]-1
        ret20 = previous/close_history[-21]-1
        daily_returns = np.diff(np.log(np.asarray(close_history[-21:], dtype=float)))
        daily_vol20 = float(np.std(daily_returns))
        gap = session_open/previous-1
        timestamps = np.asarray(candle["time"], dtype=np.int64)
        bar_seconds = int(np.median(np.diff(timestamps[bars]))) if len(bars) > 1 else 900
        hour_back = max(1, int(round(3600/bar_seconds)))
        for slot, fraction in SLOTS:
            position = min(len(bars)-2, max(0, int(round((len(bars)-1)*fraction))))
            signal = bars[position]
            entry = bars[position+1]
            reference = bars[max(0, position-hour_back)]
            now_close = float(candle["close"][signal])
            intraday = now_close/session_open-1
            last_hour = now_close/float(candle["close"][reference])-1
            rows.append({
                "symbol": symbol, "session_date": day, "slot": slot,
                "entry_index": entry, "entry_time": int(timestamps[entry]),
                "year": int(day[:4]), "ret5": ret5, "ret20": ret20,
                "daily_vol20": daily_vol20, "gap": gap,
                "intraday": intraday, "last_hour": last_hour,
            })
        close_history.append(session_close)
    return pd.DataFrame(rows), expected, np.asarray(regular, dtype=np.int64)


def build_features(candles: dict[str, dict]) -> tuple[pd.DataFrame, dict]:
    frames, metadata = [], {}
    for symbol in ALL_SYMBOLS:
        frame, bars_per_session, regular = session_table(symbol, candles[symbol])
        frames.append(frame)
        metadata[symbol] = {"bars_per_session": bars_per_session,
                            "regular": regular}
    raw = pd.concat(frames, ignore_index=True)
    qqq = raw[raw.symbol.eq("QQQUSDT")][
        ["session_date", "slot", "ret5", "ret20", "gap", "intraday", "last_hour"]
    ].rename(columns={column: f"qqq_{column}" for column in
                      ("ret5", "ret20", "gap", "intraday", "last_hour")})
    stocks = raw[raw.symbol.ne("QQQUSDT")].copy()
    group = stocks.groupby(["session_date", "slot"], sort=False)
    stocks["breadth_intra"] = group.intraday.transform(lambda value: (value > 0).mean())
    mean5 = group.ret5.transform("mean")
    std5 = group.ret5.transform("std").replace(0, np.nan)
    stocks["cross_z5"] = (stocks.ret5-mean5)/std5
    stocks = stocks.merge(qqq, on=["session_date", "slot"], how="inner",
                          validate="many_to_one")
    return stocks.replace([np.inf, -np.inf], np.nan).dropna(), metadata


def directed(frame: pd.DataFrame, direction: np.ndarray) -> pd.DataFrame:
    output = frame.copy()
    output["direction_n"] = direction.astype(np.int8)
    for column in ("ret5", "ret20", "gap", "intraday", "last_hour",
                   "cross_z5", "qqq_ret20", "qqq_intraday"):
        output[f"{column}_dir"] = output[column]*output.direction_n
    output["breadth_dir"] = np.where(output.direction_n > 0,
                                      output.breadth_intra, 1-output.breadth_intra)
    return output


def apply_spec(frame: pd.DataFrame, spec: dict) -> pd.DataFrame:
    family = spec["family"]
    if family == "trend":
        found = directed(frame, np.sign(frame.ret20.to_numpy(float)))
        mask = (found.direction_n.ne(0)
                & found.ret20_dir.ge(spec["trend"])
                & found.ret5_dir.ge(spec["short"])
                & found.intraday_dir.ge(spec["intra"])
                & found.qqq_ret20_dir.ge(spec["qqq"])
                & found.breadth_dir.ge(spec["breadth"]))
    elif family == "pullback":
        found = directed(frame, np.sign(frame.ret20.to_numpy(float)))
        mask = (found.direction_n.ne(0)
                & found.ret20_dir.ge(spec["trend"])
                & found.ret5_dir.le(spec["pullback"])
                & found.last_hour_dir.ge(spec["resume"])
                & found.qqq_ret20_dir.ge(spec["qqq"])
                & found.breadth_dir.ge(spec["breadth"]))
    elif family == "relative":
        found = directed(frame, np.sign(frame.cross_z5.to_numpy(float)))
        mask = (found.direction_n.ne(0)
                & found.cross_z5_dir.ge(spec["z"])
                & found.ret20_dir.ge(spec["trend"])
                & found.intraday_dir.ge(spec["intra"])
                & found.breadth_dir.ge(spec["breadth"]))
    elif family == "gap":
        found = directed(frame, np.sign(frame.gap.to_numpy(float)))
        mask = (found.direction_n.ne(0)
                & found.gap_dir.ge(spec["gap"])
                & found.intraday_dir.ge(spec["intra"])
                & found.qqq_intraday_dir.ge(spec["qqq_intra"])
                & found.breadth_dir.ge(spec["breadth"]))
    elif family == "late":
        found = directed(frame, np.sign(frame.intraday.to_numpy(float)))
        mask = (found.direction_n.ne(0) & found.slot.eq("late")
                & found.intraday_dir.ge(spec["intra"])
                & found.ret20_dir.ge(spec["trend"])
                & found.qqq_intraday_dir.ge(spec["qqq_intra"])
                & found.breadth_dir.ge(spec["breadth"]))
    else:
        raise ValueError(family)
    output = found.loc[mask].copy()
    output["entry_spec"] = spec["name"]
    output["score"] = (output.ret20_dir.abs() + output.intraday_dir.abs()
                       + .01*output.cross_z5_dir.abs())
    return output


@nb.njit(cache=True)
def resolve(open_, high, low, close, times, regular, lookup, entries,
            directions, target_pct, stop_pct, hold_bars):
    n = len(entries)
    values = np.empty(n, dtype=np.float64)
    exits = np.empty(n, dtype=np.int64)
    outcomes = np.empty(n, dtype=np.int8)
    target_fraction = target_pct/100.0
    stop_fraction = stop_pct/100.0
    for row in range(n):
        entry_index = entries[row]
        position = lookup[entry_index]
        direction = directions[row]
        entry_price = open_[entry_index]
        target = entry_price*(1.0 + direction*target_fraction)
        stop = entry_price*(1.0 - direction*stop_fraction)
        last_position = min(len(regular)-1, position+hold_bars-1)
        exit_index = regular[last_position]
        value = direction*(close[exit_index]/entry_price-1.0)
        outcome = 0
        for cursor in range(position, last_position+1):
            index = regular[cursor]
            bar_open = open_[index]
            if direction > 0:
                if bar_open <= stop:
                    exit_index, value, outcome = index, bar_open/entry_price-1.0, -1
                    break
                if bar_open >= target:
                    exit_index, value, outcome = index, bar_open/entry_price-1.0, 1
                    break
                if low[index] <= stop:
                    exit_index, value, outcome = index, -stop_fraction, -1
                    break
                if high[index] >= target:
                    exit_index, value, outcome = index, target_fraction, 1
                    break
            else:
                if bar_open >= stop:
                    exit_index, value, outcome = index, 1.0-bar_open/entry_price, -1
                    break
                if bar_open <= target:
                    exit_index, value, outcome = index, 1.0-bar_open/entry_price, 1
                    break
                if high[index] >= stop:
                    exit_index, value, outcome = index, -stop_fraction, -1
                    break
                if low[index] <= target:
                    exit_index, value, outcome = index, target_fraction, 1
                    break
        values[row] = value
        exits[row] = times[exit_index]+int(np.median(np.diff(times[max(0, exit_index-2):exit_index+1])))
        outcomes[row] = outcome
    return values, exits, outcomes


def add_outcomes(frame: pd.DataFrame, candles: dict, metadata: dict,
                 target: float, stop: float, sessions: int) -> pd.DataFrame:
    pieces = []
    for symbol, part in frame.groupby("symbol", sort=False):
        candle = candles[symbol]
        arrays = [np.asarray(candle[key], dtype=float)
                  for key in ("open", "high", "low", "close")]
        times = np.asarray(candle["time"], dtype=np.int64)
        regular = metadata[symbol]["regular"]
        lookup = np.full(len(times), -1, dtype=np.int64)
        lookup[regular] = np.arange(len(regular), dtype=np.int64)
        valid = part[part.entry_index.map(lambda value: lookup[int(value)] >= 0)].copy()
        values, exits, outcomes = resolve(
            *arrays, times, regular, lookup,
            valid.entry_index.to_numpy(np.int64),
            valid.direction_n.to_numpy(np.int8), target, stop,
            sessions*int(metadata[symbol]["bars_per_session"]))
        valid["signed_return"] = values
        valid["exit_time"] = exits
        valid["outcome_code"] = outcomes
        valid["target_pct"] = target
        valid["stop_pct"] = stop
        valid["hold_sessions"] = sessions
        valid["direction"] = np.where(valid.direction_n > 0, "LONG", "SHORT")
        valid["outcome"] = np.where(values > 0, "TP", "SL")
        valid["risk_pct"] = stop/100
        valid["net_r"] = values/(stop/100)
        valid["module"] = "multisession_regime"
        valid["module_priority"] = 0
        pieces.append(valid)
    return pd.concat(pieces, ignore_index=True) if pieces else frame.iloc[:0].copy()


def account_metrics(frame: pd.DataFrame, market_days: int) -> dict:
    rows = frame.to_dict("records")
    accepted = portfolio_gate(rows, None, rank_key=lambda row: (-float(row["score"]),))
    if not accepted:
        return {"n": 0, "wr": 0.0, "return_pct": 0.0, "dd_pct": 0.0,
                "negative_months": 0, "months": 0, "monthly_return_pct": {},
                "max_loss_streak": 0, "trades_per_market_day": 0.0}
    equity = peak = 1.0
    maximum_dd = 0.0
    pending = []
    completed = []
    sequence = 0

    def realize(until):
        nonlocal equity, peak, maximum_dd
        while pending and (until is None or pending[0][0] < until):
            exit_time, _, pnl, row = heapq.heappop(pending)
            equity += pnl
            peak = max(peak, equity)
            maximum_dd = max(maximum_dd, (peak-equity)/peak)
            completed.append((exit_time, pnl, row))

    for row in sorted(accepted, key=lambda item: (float(item["entry_time"]),
                                                   -float(item["score"]), item["symbol"])):
        realize(float(row["entry_time"]))
        pnl = equity*.03*10*float(row["signed_return"])
        sequence += 1
        heapq.heappush(pending, (float(row["exit_time"]), sequence, pnl, row))
    realize(None)
    completed.sort(key=lambda item: (item[0], item[2]["entry_time"], item[2]["symbol"]))
    ledger = 1.0
    monthly_start, monthly_end = {}, {}
    wins = streak = maximum_streak = 0
    for exit_time, pnl, row in completed:
        won = pnl > 0
        wins += int(won)
        streak = 0 if won else streak+1
        maximum_streak = max(maximum_streak, streak)
        month = pd.Timestamp(exit_time, unit="s", tz="UTC").strftime("%Y-%m")
        monthly_start.setdefault(month, ledger)
        ledger += pnl
        monthly_end[month] = ledger
    monthly = {key: 100*(monthly_end[key]/monthly_start[key]-1)
               for key in sorted(monthly_start)}
    return {
        "n": len(accepted), "wr": round(100*wins/len(accepted), 2),
        "return_pct": round(100*(equity-1), 4),
        "dd_pct": round(100*maximum_dd, 4),
        "negative_months": sum(value < -1e-12 for value in monthly.values()),
        "months": len(monthly),
        "monthly_return_pct": {key: round(value, 4) for key, value in monthly.items()},
        "months_ge_30pct": sum(value >= 30 for value in monthly.values()),
        "max_loss_streak": maximum_streak,
        "trades_per_market_day": round(len(accepted)/max(1, market_days), 3),
        "mean_signed_move_pct": round(100*float(np.mean(
            [row["signed_return"] for row in accepted])), 5),
    }


def evaluate(frame: pd.DataFrame, feature_frame: pd.DataFrame,
             years: tuple[int, ...] | None = None) -> dict:
    if years is None:
        selected = frame
        market_days = feature_frame.session_date.nunique()
    else:
        selected = frame[frame.year.isin(years)]
        market_days = feature_frame[feature_frame.year.isin(years)].session_date.nunique()
    return account_metrics(selected, market_days)


def development_ok(value: dict, annual: dict) -> bool:
    return bool(value["n"] >= 300 and value["return_pct"] > 0
                and value["dd_pct"] <= 20
                and sum(row["return_pct"] > 0 for row in annual.values()) >= 4
                and value["trades_per_market_day"] >= .5)


def validation_ok(value: dict, annual: dict) -> bool:
    return bool(value["n"] >= 100 and value["return_pct"] > 0
                and value["dd_pct"] <= 10
                and all(row["return_pct"] > 0 for row in annual.values())
                and value["trades_per_market_day"] >= .5)


def objective_ok(value: dict) -> bool:
    return bool(value["n"] >= 100 and value["return_pct"] > 0
                and 75 <= value["wr"] <= 95 and value["dd_pct"] <= 5
                and value["negative_months"] == 0
                and value["trades_per_market_day"] >= 3
                and value["months_ge_30pct"] == value["months"])


def public(item: dict | None) -> dict | None:
    return ({key: value for key, value in item.items() if key != "_rows"}
            if item is not None else None)


def main() -> None:
    historical, historical_sources = load_history()
    direct, direct_sources = load_direct()
    history_features, history_meta = build_features(historical)
    direct_features, direct_meta = build_features(direct)
    frontier, qualified = [], []
    for spec in SPECS:
        entries = apply_spec(history_features, spec)
        if len(entries) < 500:
            continue
        for target, stop, sessions in itertools.product(
                TARGETS, STOPS, HOLD_SESSIONS):
            resolved = add_outcomes(entries, historical, history_meta,
                                    target, stop, sessions)
            dev = evaluate(resolved, history_features, (2017, 2018, 2019, 2020, 2021))
            dev_annual = {str(year): evaluate(resolved, history_features, (year,))
                          for year in range(2017, 2022)}
            item = {"entry": spec, "exit": {"target_pct": target,
                    "stop_pct": stop, "hold_sessions": sessions},
                    "development": dev, "development_years": dev_annual,
                    "_rows": resolved}
            frontier.append(item)
            if not development_ok(dev, dev_annual):
                continue
            validation = evaluate(resolved, history_features, (2022, 2023))
            validation_annual = {str(year): evaluate(
                resolved, history_features, (year,)) for year in (2022, 2023)}
            item["validation"] = validation
            item["validation_years"] = validation_annual
            if not validation_ok(validation, validation_annual):
                continue
            item["score"] = round(
                validation["return_pct"] - 2*validation["dd_pct"]
                - 2*validation["negative_months"]
                + .2*validation["wr"] + .1*dev["return_pct"], 8)
            qualified.append(item)
    qualified.sort(key=lambda item: item["score"], reverse=True)
    frontier.sort(key=lambda item: (item["development"]["return_pct"],
                                     item["development"]["wr"]), reverse=True)
    frozen = qualified[0] if qualified else None
    results = checks = None
    if frozen:
        rows = frozen["_rows"]
        direct_entries = apply_spec(direct_features, frozen["entry"])
        exit_spec = frozen["exit"]
        direct_rows = add_outcomes(direct_entries, direct, direct_meta,
                                   exit_spec["target_pct"], exit_spec["stop_pct"],
                                   exit_spec["hold_sessions"])
        rows.to_csv(ROWS_OUT, index=False)
        results = {
            "development_2017_2021": frozen["development"],
            "validation_2022_2023": frozen["validation"],
            "test_2024": evaluate(rows, history_features, (2024,)),
            "holdout_2025_2026": evaluate(rows, history_features, (2025, 2026)),
            "direct_xperp": evaluate(direct_rows, direct_features, None),
        }
        checks = {name: objective_ok(value) for name, value in results.items()}
    status = ("RESEARCH_CANDIDATE_AWAITING_NEW_FORWARD"
              if checks and all(checks.values()) else "REJECTED")
    report = {
        "status": status,
        "protocol": {
            "development": "2017-2021 cash proxy",
            "validation": "2022-2023 cash proxy",
            "frozen_before": ["2024", "2025-2026", "direct X-Perp"],
            "entry": "three closed regular-session checkpoints; next market bar",
            "exit": "actual 0.5-2% target; 0.5-1% stop; one-three sessions",
            "entry_specs": len(SPECS),
            "exit_specs": len(TARGETS)*len(STOPS)*len(HOLD_SESSIONS),
            "portfolio": "top causal score; cap two per scan; cooldown/exposure/loss pause",
            "money": "3% entry-equity margin, fixed 10x gross",
        },
        "feature_rows": {"historical": len(history_features),
                         "direct": len(direct_features)},
        "qualified_before_hidden": len(qualified),
        "frozen_before_hidden": public(frozen),
        "top_qualified": [public(item) for item in qualified[:12]],
        "development_frontier": [public(item) for item in frontier[:15]],
        "results": results, "checks": checks,
        "sources": {"historical": historical_sources, "direct": direct_sources,
                    str(Path(__file__)): sha256(Path(__file__))},
        "limitations": [
            "Dukascopy cash prices establish long-run structure but are not X-Perp fills.",
            "Direct X-Perp history begins in August 2026 and is a short transfer sample.",
            "A bounded semantic grid still carries selection risk.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": status, "feature_rows": report["feature_rows"],
                      "qualified_before_hidden": len(qualified),
                      "frozen": public(frozen), "results": results,
                      "checks": checks}, indent=2), flush=True)


if __name__ == "__main__":
    main()
