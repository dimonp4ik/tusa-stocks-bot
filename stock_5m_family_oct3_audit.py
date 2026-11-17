"""Build and audit causal five-minute stock X-Perp shadow families."""
from __future__ import annotations

from bisect import bisect_left
from datetime import datetime
import hashlib
import json
import math
import pickle
from pathlib import Path
from statistics import median
from zoneinfo import ZoneInfo

import pandas as pd

from stock_forward_monitor import frozen_rows
from stock_xperp_oi_supplement_oct3_audit import money


ROOT = Path("reports/audit_2026_09_08")
CACHE = ROOT / "stock_xperp_5m_oct3"
PRODUCTION_SOURCE = ROOT / "stock_forward_oct3_families.csv"
OUTPUT = ROOT / "stock_5m_family_oct3.json"
ROWS_OUTPUT = ROOT / "stock_5m_family_oct3_rows.csv"
PROFILE = "robust_dynamic_entry_filtered"
NY = ZoneInfo("America/New_York")
SYMBOLS = ("AAPLUSDT", "AMZNUSDT", "METAUSDT", "MRVLUSDT",
           "MSFTUSDT", "MUUSDT", "NVDAUSDT", "QQQUSDT")
TARGETS = (.25, .5, .75, 1.0)
STOP_ATR_MULTS = (1.0, 1.5, 2.0)
DEVELOPMENT = ("2026-08-03", "2026-08-21")
VALIDATION = ("2026-08-21", "2026-09-08")
HIDDEN = ("2026-09-08", "2026-10-03")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean(values) -> float:
    return sum(values) / len(values)


def atr(candles: dict, index: int, length: int = 14) -> float | None:
    if index < length:
        return None
    values = []
    for current in range(index - length + 1, index + 1):
        high = float(candles["high"][current])
        low = float(candles["low"][current])
        previous = float(candles["close"][current - 1])
        values.append(max(high - low, abs(high - previous), abs(low - previous)))
    result = mean(values)
    return result if math.isfinite(result) and result > 0 else None


def family_signals(candles: dict, index: int) -> list[tuple[str, int, dict]]:
    if index < 96:
        return []
    close = [float(value) for value in candles["close"]]
    high = candles["high"]
    low = candles["low"]
    risk_unit = atr(candles, index)
    if not risk_unit:
        return []
    ma20 = mean(close[index - 19:index + 1])
    ma96 = mean(close[index - 95:index + 1])
    recent = close[index - 19:index + 1]
    path = sum(abs(recent[pos] - recent[pos - 1]) for pos in range(1, 20))
    efficiency = abs(recent[-1] - recent[0]) / path if path else 0.0
    std = (mean([(value - ma20) ** 2 for value in recent])) ** .5
    zscore = (close[index] - ma20) / std if std else 0.0
    volume = [float(value) for value in candles["volume"][index - 19:index + 1]]
    volume_ratio = volume[-1] / (median(volume[:-1]) or 1.0)
    common = {
        "atr": risk_unit,
        "efficiency": efficiency,
        "zscore": zscore,
        "volume_ratio": volume_ratio,
        "trend_atr": (ma20 - ma96) / risk_unit,
    }
    result = []
    if efficiency < .3:
        if zscore <= -2:
            result.append(("range_reversion", 1, common))
        elif zscore >= 2:
            result.append(("range_reversion", -1, common))
    if (ma20 > ma96 and close[index] < ma20 - .5 * risk_unit
            and close[index] > ma96 and close[index] > close[index - 1]):
        result.append(("trend_pullback", 1, common))
    elif (ma20 < ma96 and close[index] > ma20 + .5 * risk_unit
          and close[index] < ma96 and close[index] < close[index - 1]):
        result.append(("trend_pullback", -1, common))
    if close[index] > max(float(value) for value in high[index - 20:index]) \
            and close[index] > ma96:
        result.append(("breakout", 1, common))
    elif close[index] < min(float(value) for value in low[index - 20:index]) \
            and close[index] < ma96:
        result.append(("breakout", -1, common))
    return result


def regular_groups(candles: dict) -> dict[str, list[int]]:
    groups = {}
    for index, timestamp in enumerate(candles["time"]):
        local = datetime.fromtimestamp(int(timestamp), NY)
        minute = local.hour * 60 + local.minute
        if local.weekday() < 5 and 570 <= minute < 960:
            groups.setdefault(local.date().isoformat(), []).append(index)
    return groups


def session_bucket(minutes: int) -> str:
    if minutes <= 60:
        return "00_60"
    if minutes <= 150:
        return "65_150"
    return "155_plus"


def market_context(qqq: dict, lookup: dict[int, int], timestamp: int,
                   direction: int) -> dict | None:
    index = lookup.get(timestamp)
    if index is None or index < 96:
        return None
    unit = atr(qqq, index)
    if not unit:
        return None
    closes = [float(value) for value in qqq["close"]]
    ma20 = mean(closes[index - 19:index + 1])
    ma96 = mean(closes[index - 95:index + 1])
    trend = (ma20 - ma96) / unit
    regime = "bull" if trend > .5 else "bear" if trend < -.5 else "range"
    return {
        "qqq_regime_5m": regime,
        "qqq_trend_5m_atr": trend,
        "qqq_trend_dir_5m_atr": direction * trend,
        "qqq_ret12_dir_5m_atr": direction * (
            closes[index] - closes[index - 12]) / unit,
    }


def exit_result(candles: dict, first: int, last: int, direction: int,
                entry: float, risk: float, target_r: float) -> tuple:
    stop = entry - direction * risk
    target = entry + direction * risk * target_r
    for index in range(first, last + 1):
        high = float(candles["high"][index])
        low = float(candles["low"][index])
        stop_hit = low <= stop if direction > 0 else high >= stop
        target_hit = high >= target if direction > 0 else low <= target
        if stop_hit:
            return index, "SL", -1.0
        if target_hit:
            return index, "TP", target_r
    pnl_r = direction * (float(candles["close"][last]) - entry) / risk
    return last, "CLOSE", pnl_r


def build_rows(caches: dict[str, dict]) -> list[dict]:
    qqq = caches["QQQUSDT"]
    qqq_lookup = {int(value): index for index, value in enumerate(qqq["time"])}
    output = []
    for symbol in SYMBOLS:
        candles = caches[symbol]
        groups = regular_groups(candles)
        for bars in groups.values():
            if len(bars) < 3:
                continue
            for offset, index in enumerate(bars[:-1]):
                local = datetime.fromtimestamp(int(candles["time"][index]), NY)
                minutes = local.hour * 60 + local.minute - 570
                if minutes > 380 or index + 1 != bars[offset + 1]:
                    continue
                current = family_signals(candles, index)
                if not current:
                    continue
                previous = {(family, direction)
                            for family, direction, _ in family_signals(candles, index - 1)}
                for family, direction, context in current:
                    if (family, direction) in previous:
                        continue
                    qqq_context = market_context(
                        qqq, qqq_lookup, int(candles["time"][index]), direction)
                    if qqq_context is None:
                        continue
                    entry_index = index + 1
                    entry = float(candles["open"][entry_index])
                    if entry <= 0:
                        continue
                    for stop_mult in STOP_ATR_MULTS:
                        risk = float(context["atr"]) * stop_mult
                        if risk <= 0 or risk >= entry:
                            continue
                        for target_r in TARGETS:
                            exit_index, outcome, net_r = exit_result(
                                candles, entry_index, bars[-1], direction,
                                entry, risk, target_r)
                            output.append({
                                "symbol": symbol,
                                "family": family,
                                "direction": "LONG" if direction > 0 else "SHORT",
                                "target_r": target_r,
                                "stop_atr_mult": stop_mult,
                                "entry_time": int(candles["time"][entry_index]),
                                "exit_time": int(candles["time"][exit_index]) + 300,
                                "outcome": outcome,
                                "net_r": net_r,
                                "risk_pct": risk / entry,
                                "signal_minutes": minutes,
                                "session_bucket_5m": session_bucket(minutes),
                                "efficiency": context["efficiency"],
                                "zscore": context["zscore"],
                                "volume_ratio": context["volume_ratio"],
                                "trend_dir_atr": direction * context["trend_atr"],
                                **qqq_context,
                            })
    return sorted(output, key=lambda row: (
        row["entry_time"], row["symbol"], row["family"],
        row["target_r"], row["stop_atr_mult"]))


def stage_rows(rows: list[dict], period: tuple[str, str]) -> list[dict]:
    start = pd.Timestamp(period[0], tz="UTC").timestamp()
    end = pd.Timestamp(period[1], tz="UTC").timestamp()
    return [row for row in rows if start <= float(row["entry_time"]) < end]


def route_rows(rows: list[dict], route: dict) -> list[dict]:
    return [row for row in rows if all(row[key] == value
                                       for key, value in route.items()
                                       if value != "all")]


def route_metrics(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0, "wr": 0.0, "net_r": 0.0, "mean_r": 0.0,
                "max_loss_streak": 0, "symbols": 0}
    ordered = sorted(rows, key=lambda row: (row["entry_time"], row["symbol"]))
    wins = streak = max_streak = 0
    for row in ordered:
        won = float(row["net_r"]) > 0
        wins += int(won)
        streak = 0 if won else streak + 1
        max_streak = max(max_streak, streak)
    values = [float(row["net_r"]) for row in rows]
    return {
        "n": len(rows), "wr": round(100 * wins / len(rows), 2),
        "net_r": round(sum(values), 6),
        "mean_r": round(mean(values), 6),
        "max_loss_streak": max_streak,
        "symbols": len({row["symbol"] for row in rows}),
    }


def discover(rows: list[dict]) -> list[dict]:
    development = stage_rows(rows, DEVELOPMENT)
    validation = stage_rows(rows, VALIDATION)
    candidates = []
    for family in ("range_reversion", "trend_pullback", "breakout"):
        for direction in ("LONG", "SHORT"):
            for target in TARGETS:
                for stop_mult in STOP_ATR_MULTS:
                    for session in ("all", "00_60", "65_150", "155_plus"):
                        for regime in ("all", "bull", "bear", "range"):
                            route = {
                                "family": family, "direction": direction,
                                "target_r": target, "stop_atr_mult": stop_mult,
                                "session_bucket_5m": session,
                                "qqq_regime_5m": regime,
                            }
                            dev = route_metrics(route_rows(development, route))
                            val = route_metrics(route_rows(validation, route))
                            if not (dev["n"] >= 10 and val["n"] >= 8
                                    and dev["symbols"] >= 3 and val["symbols"] >= 3
                                    and dev["wr"] >= 75 and val["wr"] >= 75
                                    and dev["net_r"] > 0 and val["net_r"] > 0
                                    and dev["max_loss_streak"] <= 3
                                    and val["max_loss_streak"] <= 3):
                                continue
                            score = (min(dev["mean_r"], val["mean_r"])
                                     + .25 * (dev["mean_r"] + val["mean_r"])
                                     + .0001 * min(dev["n"], val["n"]))
                            candidates.append({"route": route,
                                               "development": dev,
                                               "validation": val,
                                               "score": score})
    return sorted(candidates, key=lambda item: item["score"], reverse=True)


def supplement_rows(rows: list[dict], candidates: list[dict]) -> list[dict]:
    selected = []
    for priority, candidate in enumerate(candidates):
        for row in route_rows(rows, candidate["route"]):
            selected.append({**row, "module": f"shadow_stock_5m_{priority + 1}",
                             "module_priority": 100 + priority,
                             "candidate_priority": priority})
    selected.sort(key=lambda row: (
        row["candidate_priority"], row["entry_time"], row["symbol"]))
    unique = {}
    for row in selected:
        unique.setdefault((row["symbol"], row["entry_time"]), row)
    return list(unique.values())


def combine(production: list[dict], supplement: list[dict]) -> list[dict]:
    rows = sorted([*production, *supplement], key=lambda row: (
        int(row["module_priority"]), row["entry_time"], row["symbol"]))
    unique = {}
    for row in rows:
        unique.setdefault((row["symbol"], row["entry_time"]), row)
    return list(unique.values())


def stages(rows: list[dict]) -> dict:
    return {name: money(stage_rows(rows, period)) for name, period in (
        ("development", DEVELOPMENT), ("validation", VALIDATION),
        ("hidden", HIDDEN))}


def improves(value: dict, baseline: dict, added: int) -> bool:
    return (value["n"] >= baseline["n"] + added
            and value["return_pct"] > baseline["return_pct"]
            and value["wr"] >= max(75.0, baseline["wr"] - 5.0)
            and value["dd_pct"] <= baseline["dd_pct"] + .35
            and value["max_loss_streak"] <= baseline["max_loss_streak"] + 1)


def main() -> None:
    caches = {symbol: pickle.loads(
        (CACHE / f"{symbol}_5min_current.pkl").read_bytes()) for symbol in SYMBOLS}
    rows = build_rows(caches)
    pd.DataFrame(rows).to_csv(ROWS_OUTPUT, index=False)
    production = frozen_rows(pd.read_csv(PRODUCTION_SOURCE), PROFILE)
    baseline = stages(production)
    raw_candidates = discover(rows)
    exact = []
    for candidate in raw_candidates:
        supplement = supplement_rows(rows, [candidate])
        values = stages(combine(production, supplement))
        if (improves(values["development"], baseline["development"], 4)
                and improves(values["validation"], baseline["validation"], 4)):
            exact.append({**candidate, "portfolio": {
                "development": values["development"],
                "validation": values["validation"],
            }, "exact_score": (
                values["development"]["return_pct"]
                - baseline["development"]["return_pct"]
                + 2 * (values["validation"]["return_pct"]
                       - baseline["validation"]["return_pct"])
            )})
    exact.sort(key=lambda item: item["exact_score"], reverse=True)
    frozen = []
    used_families = set()
    for candidate in exact:
        key = (candidate["route"]["family"], candidate["route"]["direction"])
        if key in used_families:
            continue
        frozen.append(candidate)
        used_families.add(key)
        if len(frozen) == 3:
            break
    supplement = supplement_rows(rows, frozen)
    candidate_stages = stages(combine(production, supplement))
    hidden_pass = bool(frozen) and improves(
        candidate_stages["hidden"], baseline["hidden"], 5)
    report = {
        "status": ("SHADOW_CANDIDATE_AWAITING_NEW_FORWARD"
                   if hidden_pass else "REJECTED"),
        "protocol": {
            "development": DEVELOPMENT,
            "validation": VALIDATION,
            "hidden_revealed_after_freeze": HIDDEN,
            "signal": "closed 5-minute X-Perp bar during US regular session",
            "entry": "next 5-minute market open proxy",
            "exit": "stop first on ambiguous bar; otherwise session-close mark",
            "portfolio": "production priority, one symbol open, 3h cooldown, cap 2 per 15m scan, loss pause 3",
            "money": "3% of current equity margin, 10x, gross",
        },
        "raw_rows": len(rows),
        "routes_evaluated": 3 * 2 * len(TARGETS) * len(STOP_ATR_MULTS) * 4 * 4,
        "raw_qualified": len(raw_candidates),
        "exact_qualified_before_hidden": len(exact),
        "frozen_before_hidden": frozen,
        "baseline": baseline,
        "candidate": candidate_stages,
        "hidden_pass": hidden_pass,
        "activation_gate": {"new_forward_trades": 30,
                            "positive_new_months": 1, "met": False},
        "sources": {
            str(PRODUCTION_SOURCE): digest(PRODUCTION_SOURCE),
            **{str(CACHE / f"{symbol}_5min_current.pkl"):
               digest(CACHE / f"{symbol}_5min_current.pkl") for symbol in SYMBOLS},
        },
        "limitations": [
            "Direct five-minute X-Perp history spans about sixty-two days.",
            "A finite route search creates multiple-testing risk.",
            "Any passing route remains shadow-only until new forward data pass the gate.",
            "Fees are excluded at the user's request.",
        ],
    }
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "raw_rows": len(rows),
        "raw_qualified": len(raw_candidates),
        "exact_qualified": len(exact),
        "frozen": [candidate["route"] for candidate in frozen],
        "baseline": baseline, "candidate": candidate_stages,
        "hidden_pass": hidden_pass,
    }, indent=2))


if __name__ == "__main__":
    main()
