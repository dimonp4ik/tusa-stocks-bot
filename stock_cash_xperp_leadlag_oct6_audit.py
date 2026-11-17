"""Matched-candle cash-to-X-Perp lead/lag audit with material price targets.

The underlying Dukascopy stock candle and the OKX X-Perp candle must both be
closed.  A divergence signal enters the X-Perp at the next 15-minute market
open and seeks a fixed 0.5-2% move.  June-July selects the family, early August
validates it, and late August through September is revealed only after freeze.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import pickle
from pathlib import Path

import numba as nb
import numpy as np
import pandas as pd

from stock_5m_family_oct3_audit import regular_groups
from stock_5m_high_payoff_oct3_audit import accepted, money


ROOT = Path("reports/audit_2026_09_08")
CASH_MANIFEST = ROOT / "stock_extended_history_manifest.json"
XPERP_MANIFEST = ROOT / "stock_xperp_forward_oct2" / "manifest.json"
OUT = ROOT / "stock_cash_xperp_leadlag_oct6.json"
ROWS_OUT = ROOT / "stock_cash_xperp_leadlag_oct6.csv"
SYMBOLS = ("AAPLUSDT", "AMZNUSDT", "GOOGLUSDT", "INTCUSDT", "METAUSDT",
           "MRVLUSDT", "MSFTUSDT", "MUUSDT", "NVDAUSDT", "TSLAUSDT")
LOOKBACKS = (1, 2, 4)
TARGETS = (.5, 1.0, 2.0)
STOPS = (.5, 1.0)
HOLDS = (4, 8, 16)
DEVELOPMENT = ("2026-06-09", "2026-08-01")
VALIDATION = ("2026-08-01", "2026-08-24")
HIDDEN = ("2026-08-24", "2026-09-15")
SPECS = (
    {"name": "catchup_loose", "family": "lead", "lead": .001,
     "cash": .002, "breadth": .50},
    {"name": "catchup_strict", "family": "lead", "lead": .0025,
     "cash": .004, "breadth": .50},
    {"name": "stale_loose", "family": "stale", "lead": .002,
     "cash": .003, "xperp_max": .0015, "breadth": .50},
    {"name": "stale_strict", "family": "stale", "lead": .003,
     "cash": .005, "xperp_max": .001, "breadth": .50},
    {"name": "market_confirmed", "family": "market", "lead": .0015,
     "cash": .002, "qqq": .001, "breadth": .60},
    {"name": "cross_section_leader", "family": "cross", "z": 1.0,
     "lead": .001, "cash": .002, "breadth": .50},
    {"name": "basis_convergence_loose", "family": "basis", "basis": .0015,
     "basis_move": 0.0, "breadth": .50},
    {"name": "basis_convergence_strict", "family": "basis", "basis": .003,
     "basis_move": .0005, "breadth": .50},
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_sources() -> tuple[dict, dict, dict]:
    cash_manifest = json.loads(CASH_MANIFEST.read_text(encoding="utf-8"))
    xperp_manifest = json.loads(XPERP_MANIFEST.read_text(encoding="utf-8"))
    cash, xperp = {}, {}
    sources = {str(CASH_MANIFEST): digest(CASH_MANIFEST),
               str(XPERP_MANIFEST): digest(XPERP_MANIFEST)}
    for symbol in (*SYMBOLS, "QQQUSDT"):
        cash_path = Path(cash_manifest["symbols"][symbol]["path"])
        cash_raw = cash_path.read_bytes()
        if hashlib.sha256(cash_raw).hexdigest() != cash_manifest["symbols"][symbol]["sha256"]:
            raise ValueError(f"cash hash mismatch: {symbol}")
        cash[symbol] = pickle.loads(cash_raw)
        sources[str(cash_path)] = hashlib.sha256(cash_raw).hexdigest()
        xperp_path = Path(xperp_manifest["symbols"][symbol]["path"])
        if not xperp_path.is_absolute():
            xperp_path = Path.cwd() / xperp_path
        xperp_raw = xperp_path.read_bytes()
        if hashlib.sha256(xperp_raw).hexdigest() != xperp_manifest["symbols"][symbol]["sha256"]:
            raise ValueError(f"X-Perp hash mismatch: {symbol}")
        xperp[symbol] = pickle.loads(xperp_raw)
        sources[str(xperp_path)] = hashlib.sha256(xperp_raw).hexdigest()
    return cash, xperp, sources


def raw_features(cash: dict, xperp: dict) -> pd.DataFrame:
    qqq = cash["QQQUSDT"]
    qqq_lookup = {int(value): index for index, value in enumerate(qqq["time"])}
    output = []
    for symbol in SYMBOLS:
        source, venue = cash[symbol], xperp[symbol]
        regular = {index for bars in regular_groups(source).values() for index in bars}
        cash_lookup = {int(value): index for index, value in enumerate(source["time"])}
        venue_lookup = {int(value): index for index, value in enumerate(venue["time"])}
        common = sorted(set(cash_lookup) & set(venue_lookup))
        for timestamp in common:
            cash_index, venue_index = cash_lookup[timestamp], venue_lookup[timestamp]
            if cash_index not in regular or venue_index + 1 >= len(venue["time"]):
                continue
            if int(venue["time"][venue_index+1])-timestamp != 900:
                continue
            for lookback in LOOKBACKS:
                previous = timestamp-lookback*900
                cash_previous = cash_lookup.get(previous)
                venue_previous = venue_lookup.get(previous)
                qqq_now = qqq_lookup.get(timestamp)
                qqq_previous = qqq_lookup.get(previous)
                if None in (cash_previous, venue_previous, qqq_now, qqq_previous):
                    continue
                cash_close = float(source["close"][cash_index])
                venue_close = float(venue["close"][venue_index])
                cash_ret = cash_close/float(source["close"][cash_previous])-1
                venue_ret = venue_close/float(venue["close"][venue_previous])-1
                qqq_ret = (float(qqq["close"][qqq_now])
                           / float(qqq["close"][qqq_previous])-1)
                basis = venue_close/cash_close-1
                previous_basis = (float(venue["close"][venue_previous])
                                  / float(source["close"][cash_previous])-1)
                entry_time = int(venue["time"][venue_index+1])
                output.append({
                    "symbol": symbol, "lookback": lookback,
                    "entry_index": venue_index+1, "entry_time": entry_time,
                    "cash_return": cash_ret, "xperp_return": venue_ret,
                    "lead": cash_ret-venue_ret, "qqq_return": qqq_ret,
                    "basis": basis, "basis_change": basis-previous_basis,
                })
    frame = pd.DataFrame(output)
    stamp = pd.to_datetime(frame.entry_time, unit="s", utc=True)
    frame["year"] = stamp.dt.year
    frame["month"] = stamp.dt.year*100+stamp.dt.month
    grouped = frame.groupby(["entry_time", "lookback"], sort=False)
    frame["breadth_lead"] = grouped.lead.transform(lambda value: (value > 0).mean())
    mean = grouped.lead.transform("mean")
    std = grouped.lead.transform("std").replace(0, np.nan)
    frame["lead_z"] = (frame.lead-mean)/std
    return frame.replace([np.inf, -np.inf], np.nan).dropna()


def directed(frame: pd.DataFrame, direction: np.ndarray) -> pd.DataFrame:
    output = frame.copy()
    output["direction_n"] = direction.astype(np.int8)
    for column in ("cash_return", "xperp_return", "lead", "qqq_return",
                   "basis", "basis_change", "lead_z"):
        output[f"{column}_dir"] = output[column]*output.direction_n
    output["breadth_dir"] = np.where(output.direction_n > 0,
                                      output.breadth_lead, 1-output.breadth_lead)
    return output


def apply_spec(frame: pd.DataFrame, spec: dict) -> pd.DataFrame:
    family = spec["family"]
    if family == "basis":
        # X-Perp above cash implies a short convergence trade.
        found = directed(frame, -np.sign(frame.basis.to_numpy(float)))
        mask = (found.direction_n.ne(0)
                & found.basis_dir.le(-spec["basis"])
                & found.basis_change_dir.le(-spec["basis_move"])
                & found.breadth_dir.ge(spec["breadth"]))
    else:
        found = directed(frame, np.sign(frame.lead.to_numpy(float)))
        mask = (found.direction_n.ne(0)
                & found.lead_dir.ge(spec["lead"])
                & found.cash_return_dir.ge(spec["cash"])
                & found.breadth_dir.ge(spec["breadth"]))
        if family == "stale":
            mask &= found.xperp_return.abs().le(spec["xperp_max"])
        elif family == "market":
            mask &= found.qqq_return_dir.ge(spec["qqq"])
        elif family == "cross":
            mask &= found.lead_z_dir.ge(spec["z"])
    output = found.loc[mask].copy()
    output["entry_spec"] = spec["name"]
    output["score"] = (output.lead_dir.abs() + output.cash_return_dir.abs()
                       + output.basis.abs())
    return output


@nb.njit(cache=True)
def resolve(open_, high, low, close, times, entries, directions,
            target_pct, stop_pct, hold):
    n = len(entries)
    values = np.empty(n, dtype=np.float64)
    exits = np.empty(n, dtype=np.int64)
    outcomes = np.empty(n, dtype=np.int8)
    target_fraction, stop_fraction = target_pct/100.0, stop_pct/100.0
    for row in range(n):
        entry_index = entries[row]
        direction = directions[row]
        entry = open_[entry_index]
        target = entry*(1+direction*target_fraction)
        stop = entry*(1-direction*stop_fraction)
        last = min(len(times)-1, entry_index+hold-1)
        exit_index = last
        value = direction*(close[last]/entry-1)
        outcome = 0
        for index in range(entry_index, last+1):
            if index > entry_index and times[index]-times[index-1] != 900:
                exit_index = index-1
                value = direction*(close[exit_index]/entry-1)
                break
            bar_open = open_[index]
            if direction > 0:
                if bar_open <= stop:
                    exit_index, value, outcome = index, bar_open/entry-1, -1
                    break
                if bar_open >= target:
                    exit_index, value, outcome = index, bar_open/entry-1, 1
                    break
                if low[index] <= stop:
                    exit_index, value, outcome = index, -stop_fraction, -1
                    break
                if high[index] >= target:
                    exit_index, value, outcome = index, target_fraction, 1
                    break
            else:
                if bar_open >= stop:
                    exit_index, value, outcome = index, 1-bar_open/entry, -1
                    break
                if bar_open <= target:
                    exit_index, value, outcome = index, 1-bar_open/entry, 1
                    break
                if high[index] >= stop:
                    exit_index, value, outcome = index, -stop_fraction, -1
                    break
                if low[index] <= target:
                    exit_index, value, outcome = index, target_fraction, 1
                    break
        values[row], exits[row], outcomes[row] = value, times[exit_index]+900, outcome
    return values, exits, outcomes


def add_outcomes(frame: pd.DataFrame, xperp: dict, target: float,
                 stop: float, hold: int) -> pd.DataFrame:
    pieces = []
    for symbol, part in frame.groupby("symbol", sort=False):
        candle = xperp[symbol]
        arrays = [np.asarray(candle[key], dtype=float)
                  for key in ("open", "high", "low", "close")]
        values, exits, outcomes = resolve(
            *arrays, np.asarray(candle["time"], dtype=np.int64),
            part.entry_index.to_numpy(np.int64),
            part.direction_n.to_numpy(np.int8), target, stop, hold)
        found = part.copy()
        found["signed_return"] = values
        found["exit_time"] = exits
        found["outcome_code"] = outcomes
        found["target_pct"] = target
        found["stop_pct"] = stop
        found["hold_bars"] = hold
        found["direction"] = np.where(found.direction_n > 0, "LONG", "SHORT")
        found["outcome"] = np.where(values > 0, "TP", "SL")
        found["risk_pct"] = stop/100
        found["net_r"] = values/(stop/100)
        found["module"] = "cash_xperp_leadlag"
        found["module_priority"] = 0
        pieces.append(found)
    return pd.concat(pieces, ignore_index=True) if pieces else frame.iloc[:0].copy()


def period_rows(frame: pd.DataFrame, period: tuple[str, str]) -> pd.DataFrame:
    start = pd.Timestamp(period[0], tz="UTC").timestamp()
    end = pd.Timestamp(period[1], tz="UTC").timestamp()
    return frame[frame.entry_time.between(start, end-1)]


def evaluate(frame: pd.DataFrame, period: tuple[str, str]) -> dict:
    part = period_rows(frame, period)
    rows = part.to_dict("records")
    result = money(rows)
    accepted_rows = accepted(rows)
    monthly = list(result.get("monthly_return_pct", {}).values())
    result["months"] = len(monthly)
    result["months_ge_30pct"] = sum(value >= 30 for value in monthly)
    result["mean_signed_move_pct"] = (round(100*float(np.mean([
        float(row["risk_pct"])*float(row["net_r"]) for row in accepted_rows])), 5)
        if accepted_rows else 0.0)
    return result


def development_ok(value: dict) -> bool:
    return bool(value["n"] >= 100 and value["return_pct"] > 0
                and value["dd_pct"] <= 10 and value["wr"] >= 40
                and value["trades_per_calendar_day"] >= 1)


def validation_ok(value: dict) -> bool:
    return bool(value["n"] >= 40 and value["return_pct"] > 0
                and value["dd_pct"] <= 5 and value["wr"] >= 40
                and value["trades_per_calendar_day"] >= 1)


def objective_ok(value: dict) -> bool:
    return bool(value["n"] >= 60 and value["return_pct"] > 0
                and 75 <= value["wr"] <= 95 and value["dd_pct"] <= 5
                and value["negative_months"] == 0
                and value["trades_per_calendar_day"] >= 3
                and value["months_ge_30pct"] == value["months"])


def public(item: dict | None) -> dict | None:
    return ({key: value for key, value in item.items() if key != "_rows"}
            if item is not None else None)


def main() -> None:
    cash, xperp, sources = load_sources()
    raw = raw_features(cash, xperp)
    frontier, qualified = [], []
    for spec in SPECS:
        entries = apply_spec(raw, spec)
        for lookback in LOOKBACKS:
            selected = entries[entries.lookback.eq(lookback)]
            if len(selected) < 100:
                continue
            for target, stop, hold in itertools.product(TARGETS, STOPS, HOLDS):
                resolved = add_outcomes(selected, xperp, target, stop, hold)
                dev = evaluate(resolved, DEVELOPMENT)
                item = {"entry": spec, "lookback_bars": lookback,
                        "exit": {"target_pct": target, "stop_pct": stop,
                                 "hold_bars": hold},
                        "development": dev, "_rows": resolved}
                frontier.append(item)
                if not development_ok(dev):
                    continue
                validation = evaluate(resolved, VALIDATION)
                item["validation"] = validation
                if not validation_ok(validation):
                    continue
                item["score"] = round(
                    validation["return_pct"] - 2*validation["dd_pct"]
                    - 2*validation["negative_months"] + .2*validation["wr"]
                    + .1*dev["return_pct"], 8)
                qualified.append(item)
    qualified.sort(key=lambda item: item["score"], reverse=True)
    frontier.sort(key=lambda item: (item["development"]["return_pct"],
                                     item["development"]["wr"]), reverse=True)
    frozen = qualified[0] if qualified else None
    hidden = checks = None
    if frozen:
        frozen["_rows"].to_csv(ROWS_OUT, index=False)
        hidden = evaluate(frozen["_rows"], HIDDEN)
        checks = {"development": objective_ok(frozen["development"]),
                  "validation": objective_ok(frozen["validation"]),
                  "hidden": objective_ok(hidden)}
    status = ("RESEARCH_CANDIDATE_AWAITING_NEW_FORWARD"
              if checks and all(checks.values()) else "REJECTED")
    report = {
        "status": status,
        "protocol": {
            "development": DEVELOPMENT, "validation": VALIDATION,
            "hidden_revealed_after_freeze": HIDDEN,
            "signal": "closed matched cash and X-Perp 15m candles",
            "entry": "next X-Perp 15m market open",
            "exit": "fixed actual 0.5-2% targets, stop-first ambiguity",
            "money": "3% entry-equity margin, fixed 10x gross",
            "entry_specs": len(SPECS)*len(LOOKBACKS),
            "exit_specs": len(TARGETS)*len(STOPS)*len(HOLDS),
        },
        "matched_feature_rows": len(raw),
        "qualified_before_hidden": len(qualified),
        "frozen_before_hidden": public(frozen),
        "top_qualified": [public(item) for item in qualified[:12]],
        "development_frontier": [public(item) for item in frontier[:15]],
        "hidden": hidden, "checks": checks,
        "sources": {**sources, str(Path(__file__)): digest(Path(__file__))},
        "limitations": [
            "Matched per-stock cash data ends on 2026-09-14.",
            "The direct matched history is only about three months.",
            "A bounded semantic grid still creates selection risk.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": status, "matched_feature_rows": len(raw),
                      "qualified_before_hidden": len(qualified),
                      "frozen": public(frozen), "hidden": hidden,
                      "checks": checks}, indent=2), flush=True)


if __name__ == "__main__":
    main()
