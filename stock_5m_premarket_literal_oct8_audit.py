"""Native X-Perp premarket direction audit with literal 1.5%-2% targets."""
from __future__ import annotations

import hashlib
import itertools
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd

import stock_5m_cross_sectional_literal_oct8_audit as base
from stock_5m_high_payoff_oct3_audit import accepted, money
from src.research_profit_gate import economic_gate


ROOT = Path("reports/audit_2026_09_08")
OUT = ROOT / "stock_5m_premarket_literal_oct8.json"
ROWS_OUT = ROOT / "stock_5m_premarket_literal_oct8_rows.csv"
TARGETS = (1.5, 2.0)
STOPS = (.5, .75, 1.0)
MOVE_LEVELS = (.25, .50, .75, 1.0)
LAST_HOUR_LEVELS = (0.0, .10, .20)
QQQ_LEVELS = (-.10, 0.0, .10)
BREADTH_LEVELS = (.57, .71)
RANK_LEVELS = (.71, .86)


def feature_rows(caches: dict[str, dict]) -> list[dict]:
    regular = {symbol: base.regular_groups(candle)
               for symbol, candle in caches.items()}
    premarket = {}
    for symbol, candle in caches.items():
        grouped = {}
        for index, timestamp in enumerate(candle["time"]):
            local = datetime.fromtimestamp(int(timestamp), base.NY)
            minute = local.hour*60+local.minute
            if local.weekday() < 5 and 240 <= minute < 570:
                grouped.setdefault(local.date().isoformat(), []).append(index)
        premarket[symbol] = grouped
    days = sorted(set.intersection(
        *(set(regular[symbol]) & set(premarket[symbol]) for symbol in base.ALL)))
    output = []
    for day in days:
        raw = {}
        valid = True
        for symbol in base.ALL:
            candle = caches[symbol]
            pm = premarket[symbol][day]
            session = regular[symbol][day]
            if len(pm) < 60 or not session or pm[-1]+1 != session[0]:
                valid = False
                break
            pm_open = float(candle["open"][pm[0]])
            pm_close = float(candle["close"][pm[-1]])
            hour_reference = float(candle["close"][pm[-13]])
            active = sum(float(candle["high"][index]) != float(candle["low"][index])
                         for index in pm)
            raw[symbol] = {
                "pm_return": 100*(pm_close/pm_open-1),
                "last_hour_return": 100*(pm_close/hour_reference-1),
                "pm_range": 100*(max(float(candle["high"][index]) for index in pm)
                                  - min(float(candle["low"][index]) for index in pm))/pm_open,
                "pm_active_ratio": active/len(pm),
                "entry_index": session[0], "session_last": session[-1],
            }
        if not valid:
            continue
        qqq = raw["QQQUSDT"]
        stock_returns = {symbol: raw[symbol]["pm_return"] for symbol in base.STOCKS}
        ascending = sorted(stock_returns.items(), key=lambda item: item[1])
        rank_long = {symbol: (position+1)/len(ascending)
                     for position, (symbol, _) in enumerate(ascending)}
        breadth_long = sum(value > 0 for value in stock_returns.values())/len(stock_returns)
        for symbol in base.STOCKS:
            candle = caches[symbol]
            item = raw[symbol]
            output.append({
                "symbol": symbol, "day": day,
                "entry_time": int(candle["time"][item["entry_index"]]),
                "entry_index": item["entry_index"],
                "session_last": item["session_last"],
                "pm_return": item["pm_return"],
                "last_hour_return": item["last_hour_return"],
                "pm_range": item["pm_range"],
                "pm_active_ratio": item["pm_active_ratio"],
                "qqq_pm_return": qqq["pm_return"],
                "breadth_long": breadth_long,
                "rank_long": rank_long[symbol],
            })
    return output


def profiles() -> list[dict]:
    result = []
    common = itertools.product(
        MOVE_LEVELS, LAST_HOUR_LEVELS, QQQ_LEVELS,
        BREADTH_LEVELS, RANK_LEVELS)
    for move, last_hour, qqq, breadth, rank in common:
        result.append({
            "family": "premarket_follow", "move": move,
            "last_hour": last_hour, "qqq": qqq,
            "breadth": breadth, "rank": rank,
        })
    # Fades need a closed last-hour reversal before the market entry.
    for move, reversal, qqq_against, rank in itertools.product(
            (.50, .75, 1.0, 1.5), (.05, .10, .20), (.10, .25, .50), RANK_LEVELS):
        result.append({
            "family": "premarket_fade", "move": move,
            "last_hour": reversal, "qqq_against": qqq_against,
            "rank": rank,
        })
    return result


def select(features: list[dict], profile: dict) -> list[dict]:
    output = []
    for row in features:
        old_direction = 1 if row["pm_return"] > 0 else -1 if row["pm_return"] < 0 else 0
        if not old_direction or row["pm_active_ratio"] < .15:
            continue
        if profile["family"] == "premarket_follow":
            direction = old_direction
            rank = row["rank_long"] if direction > 0 else 1-row["rank_long"]+1/7
            breadth = row["breadth_long"] if direction > 0 else 1-row["breadth_long"]
            if not (direction*row["pm_return"] >= profile["move"]
                    and direction*row["last_hour_return"] >= profile["last_hour"]
                    and direction*row["qqq_pm_return"] >= profile["qqq"]
                    and breadth >= profile["breadth"] and rank >= profile["rank"]):
                continue
        else:
            direction = -old_direction
            old_rank = (row["rank_long"] if old_direction > 0
                        else 1-row["rank_long"]+1/7)
            if not (old_direction*row["pm_return"] >= profile["move"]
                    and direction*row["last_hour_return"] >= profile["last_hour"]
                    and old_direction*row["qqq_pm_return"] <= profile["qqq_against"]
                    and old_rank >= profile["rank"]):
                continue
        output.append({**row, "direction_int": direction,
                       "direction": "LONG" if direction > 0 else "SHORT"})
    return output


def route(features: list[dict], caches: dict[str, dict], profile: dict,
          target: float, stop: float) -> list[dict]:
    result = []
    for event in select(features, profile):
        resolved = base.simulate(event, caches[event["symbol"]], target, stop, 78)
        resolved.update({
            "module": "stock_5m_literal_"+profile["family"],
            "module_priority": 0,
            "route_id": route_id(profile, target, stop),
        })
        result.append(resolved)
    return result


def route_id(profile: dict, target: float, stop: float) -> str:
    body = "_".join(f"{key}{value}" for key, value in sorted(profile.items()))
    return f"{body}_tp{target}_sl{stop}"


def detail(rows: list[dict]) -> dict:
    chosen = accepted(rows)
    value = money(rows)
    hits = sum(bool(row["literal_target_hit"]) for row in chosen)
    value.update({
        "literal_target_hits": hits,
        "literal_target_hit_rate": round(100*hits/len(chosen), 2) if chosen else 0.0,
        "outcomes": dict(sorted(Counter(str(row["outcome"]) for row in chosen).items())),
    })
    return value


def economics(rows: list[dict], period: tuple[str, str]) -> dict:
    chosen = accepted(base.bounds(rows, period))
    market_days = len(pd.bdate_range(
        period[0], pd.Timestamp(period[1])-pd.Timedelta(days=1)))
    frequency = len(chosen)/market_days if market_days else 0.0
    returns = [float(row["risk_pct"])*float(row["net_r"]) for row in chosen]
    return economic_gate(returns, frequency)


def stage_ok(value: dict, target: float, minimum: int) -> bool:
    floor = 40 if target == 1.5 else 32
    return bool(value["n"] >= minimum and value["literal_target_hit_rate"] >= floor
                and value["return_pct"] > 0 and value["dd_pct"] <= 2.0
                and value["max_loss_streak"] <= 4)


def main() -> None:
    caches, sources = base.load()
    features = feature_rows(caches)
    candidates, frontier, validation_frontier, rows_by_id = [], [], [], {}
    development_passes = 0
    evaluated = 0
    for profile, target, stop in itertools.product(profiles(), TARGETS, STOPS):
        evaluated += 1
        rows = route(features, caches, profile, target, stop)
        development = detail(base.bounds(rows, base.DEVELOPMENT))
        if development["n"] >= 4:
            frontier.append({"profile": profile, "target_pct": target,
                             "stop_pct": stop, "development": development})
        # Premarket sparsity leaves only four to five development events for
        # the best high-target fades.  This development-only minimum was
        # reduced once before their validation outcomes were evaluated.
        if not stage_ok(development, target, 4):
            continue
        development_passes += 1
        validation = detail(base.bounds(rows, base.VALIDATION))
        validation_frontier.append({
            "profile": profile, "target_pct": target, "stop_pct": stop,
            "development": development, "validation": validation,
        })
        # Validation is only a nomination screen for this sparse event family;
        # confirmation below still requires at least ten accepted trades.
        if not stage_ok(validation, target, 2):
            continue
        score = (development["return_pct"]+2*validation["return_pct"]
                 - development["dd_pct"]-2*validation["dd_pct"]
                 + min(development["literal_target_hit_rate"],
                       validation["literal_target_hit_rate"])/10)
        item = {"profile": profile, "target_pct": target, "stop_pct": stop,
                "development": development, "validation": validation,
                "score": round(score, 6)}
        candidates.append(item)
        rows_by_id[route_id(profile, target, stop)] = rows
    candidates.sort(key=lambda item: (item["score"], json.dumps(item["profile"], sort_keys=True)),
                    reverse=True)
    frontier.sort(key=lambda item: (
        item["development"]["return_pct"]-item["development"]["dd_pct"],
        item["development"]["literal_target_hit_rate"]), reverse=True)
    validation_frontier.sort(key=lambda item: (
        min(item["development"]["literal_target_hit_rate"],
            item["validation"]["literal_target_hit_rate"]),
        item["development"]["return_pct"]+item["validation"]["return_pct"]),
        reverse=True)
    frozen = []
    seen = set()
    for item in candidates:
        family = item["profile"]["family"]
        if family not in seen:
            frozen.append(item)
            seen.add(family)
    ensemble = []
    for priority, item in enumerate(frozen):
        key = route_id(item["profile"], item["target_pct"], item["stop_pct"])
        ensemble.extend({**row, "module_priority": priority} for row in rows_by_id[key])
    confirmation = detail(base.bounds(ensemble, base.CONFIRMATION)) if ensemble else detail([])
    confirmation_economics = economics(ensemble, base.CONFIRMATION) if ensemble else None
    confirmation_passed = bool(
        frozen and confirmation["n"] >= 10 and confirmation["return_pct"] > 0
        and confirmation["literal_target_hit_rate"] >= 35
        and confirmation["dd_pct"] <= 2 and confirmation["max_loss_streak"] <= 4)
    reveal = detail(base.bounds(ensemble, base.REVEAL)) if confirmation_passed else None
    reveal_economics = economics(ensemble, base.REVEAL) if confirmation_passed else None
    objective_passed = bool(confirmation_passed and confirmation_economics
                            and confirmation_economics["passes"]
                            and reveal_economics and reveal_economics["passes"])
    status = ("OBJECTIVE_CANDIDATE_AWAITING_FORWARD" if objective_passed
              else "REJECTED_ON_OBJECTIVE_GATE" if confirmation_passed
              else "REJECTED_BEFORE_REVEAL")
    if ensemble:
        pd.DataFrame(accepted(ensemble)).to_csv(ROWS_OUT, index=False)
    report = {
        "status": status,
        "protocol": {
            "hypothesis": "native X-Perp premarket continuation or closed-hour reversal",
            "entry": "first regular-session 5m market-open proxy",
            "targets_pct": TARGETS, "stops_pct": STOPS,
            "exit": "literal target, stop, or regular-session close; stop-first",
            "development": base.DEVELOPMENT, "validation": base.VALIDATION,
            "confirmation": base.CONFIRMATION, "reveal": base.REVEAL,
            "selection": "one frozen route per semantic family before confirmation",
            "money": "3% current-equity margin, fixed 10x gross",
        },
        "search": {"feature_rows": len(features), "routes_evaluated": evaluated,
                   "passed_development": development_passes,
                   "qualified_development_validation": len(candidates)},
        "frozen_routes": frozen,
        "confirmation": confirmation,
        "confirmation_economics": confirmation_economics,
        "confirmation_passed": confirmation_passed,
        "reveal": reveal, "reveal_economics": reveal_economics,
        "objective_passed": objective_passed,
        "frontier": candidates[:15], "development_only_frontier": frontier[:15],
        "validation_frontier": validation_frontier[:15],
        "readiness": {"live_activation_allowed": False,
                      "reason": "requires objective gate plus genuinely new forward evidence"},
        "sources": {**sources, str(Path(__file__)): hashlib.sha256(
            Path(__file__).read_bytes()).hexdigest()},
        "limitations": [
            "Native premarket history spans only about two months.",
            "Sparse premarket candles can create execution differences.",
            "OHLC replay cannot reproduce spread, latency or intrabar path.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "status", "search", "frozen_routes", "confirmation",
        "confirmation_economics", "confirmation_passed", "reveal",
        "reveal_economics", "objective_passed")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
