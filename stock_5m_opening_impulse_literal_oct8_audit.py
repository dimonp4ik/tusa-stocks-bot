"""Early-session native X-Perp impulse audit with 1.5%-2% targets."""
from __future__ import annotations

import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path

import pandas as pd

import stock_5m_cross_sectional_literal_oct8_audit as base
from stock_5m_high_payoff_oct3_audit import accepted, money
from src.research_profit_gate import economic_gate


ROOT = Path("reports/audit_2026_09_08")
OUT = ROOT / "stock_5m_opening_impulse_literal_oct8.json"
ROWS_OUT = ROOT / "stock_5m_opening_impulse_literal_oct8_rows.csv"
DECISIONS = (2, 5, 11)  # after 15, 30 and 60 minutes
TARGETS = (1.5, 2.0)
STOPS = (.5, .75, 1.0)


def features(caches: dict[str, dict]) -> list[dict]:
    groups = {symbol: base.regular_groups(candle)
              for symbol, candle in caches.items()}
    days = sorted(set.intersection(*(set(groups[symbol]) for symbol in base.ALL)))
    rows = []
    for day in days:
        for decision in DECISIONS:
            if any(len(groups[symbol][day]) <= decision+1 for symbol in base.ALL):
                continue
            raw = {}
            for symbol in base.ALL:
                candle = caches[symbol]
                bars = groups[symbol][day]
                first, signal = bars[0], bars[decision]
                op = float(candle["open"][first])
                close = float(candle["close"][signal])
                high = max(float(candle["high"][index]) for index in bars[:decision+1])
                low = min(float(candle["low"][index]) for index in bars[:decision+1])
                last_open = float(candle["open"][signal])
                last_high = float(candle["high"][signal])
                last_low = float(candle["low"][signal])
                span = max(last_high-last_low, 1e-12)
                raw[symbol] = {
                    "move": 100*(close/op-1),
                    "range": 100*(high-low)/op,
                    "last_bar": 100*(close/last_open-1),
                    "close_pos_long": (close-last_low)/span,
                    "entry_index": bars[decision+1],
                    "session_last": bars[-1],
                }
            stock_moves = {symbol: raw[symbol]["move"] for symbol in base.STOCKS}
            ascending = sorted(stock_moves.items(), key=lambda item: item[1])
            rank_long = {symbol: (position+1)/len(ascending)
                         for position, (symbol, _) in enumerate(ascending)}
            breadth_long = sum(value > 0 for value in stock_moves.values())/len(stock_moves)
            for symbol in base.STOCKS:
                item = raw[symbol]
                candle = caches[symbol]
                rows.append({
                    "symbol": symbol, "day": day, "decision_bar": decision,
                    "entry_time": int(candle["time"][item["entry_index"]]),
                    "entry_index": item["entry_index"],
                    "session_last": item["session_last"],
                    "opening_move": item["move"], "opening_range": item["range"],
                    "last_bar": item["last_bar"],
                    "close_pos_long": item["close_pos_long"],
                    "qqq_move": raw["QQQUSDT"]["move"],
                    "breadth_long": breadth_long, "rank_long": rank_long[symbol],
                })
    return rows


def profiles() -> list[dict]:
    output = []
    for decision, move, last, qqq, breadth, rank, close_pos in itertools.product(
            DECISIONS, (.25, .50, .75, 1.0), (0.0, .05, .10),
            (-.10, 0.0, .10), (.57, .71), (.71, .86), (.60, .75)):
        output.append({
            "family": "opening_impulse_follow", "decision_bar": decision,
            "move": move, "last_bar": last, "qqq": qqq,
            "breadth": breadth, "rank": rank, "close_pos": close_pos,
        })
    for decision, move, reversal, rank, close_pos in itertools.product(
            DECISIONS, (.50, .75, 1.0, 1.5), (.05, .10, .20),
            (.71, .86), (.60, .75)):
        output.append({
            "family": "opening_failed_move", "decision_bar": decision,
            "move": move, "last_bar": reversal,
            "rank": rank, "close_pos": close_pos,
        })
    return output


def select(rows: list[dict], profile: dict) -> list[dict]:
    output = []
    for row in rows:
        if row["decision_bar"] != profile["decision_bar"]:
            continue
        old_direction = 1 if row["opening_move"] > 0 else -1 if row["opening_move"] < 0 else 0
        if not old_direction:
            continue
        if profile["family"] == "opening_impulse_follow":
            direction = old_direction
            rank = row["rank_long"] if direction > 0 else 1-row["rank_long"]+1/7
            breadth = row["breadth_long"] if direction > 0 else 1-row["breadth_long"]
            close_pos = (row["close_pos_long"] if direction > 0
                         else 1-row["close_pos_long"])
            if not (direction*row["opening_move"] >= profile["move"]
                    and direction*row["last_bar"] >= profile["last_bar"]
                    and direction*row["qqq_move"] >= profile["qqq"]
                    and breadth >= profile["breadth"] and rank >= profile["rank"]
                    and close_pos >= profile["close_pos"]):
                continue
        else:
            direction = -old_direction
            old_rank = (row["rank_long"] if old_direction > 0
                        else 1-row["rank_long"]+1/7)
            close_pos = (row["close_pos_long"] if direction > 0
                         else 1-row["close_pos_long"])
            if not (old_direction*row["opening_move"] >= profile["move"]
                    and direction*row["last_bar"] >= profile["last_bar"]
                    and old_rank >= profile["rank"]
                    and close_pos >= profile["close_pos"]):
                continue
        output.append({**row, "direction_int": direction,
                       "direction": "LONG" if direction > 0 else "SHORT"})
    return output


def route(rows: list[dict], caches: dict[str, dict], profile: dict,
          target: float, stop: float) -> list[dict]:
    result = []
    identifier = route_id(profile, target, stop)
    for event in select(rows, profile):
        resolved = base.simulate(event, caches[event["symbol"]], target, stop, 78)
        resolved.update({"module": "stock_5m_literal_"+profile["family"],
                         "module_priority": profile["decision_bar"],
                         "route_id": identifier})
        result.append(resolved)
    return result


def route_id(profile: dict, target: float, stop: float) -> str:
    body = "_".join(f"{key}{value}" for key, value in sorted(profile.items()))
    return f"{body}_tp{target}_sl{stop}"


def detail(rows: list[dict]) -> dict:
    chosen = accepted(rows)
    value = money(rows)
    hits = sum(bool(row["literal_target_hit"]) for row in chosen)
    value.update({"literal_target_hits": hits,
                  "literal_target_hit_rate": round(100*hits/len(chosen), 2) if chosen else 0,
                  "outcomes": dict(sorted(Counter(str(row["outcome"]) for row in chosen).items()))})
    return value


def economics(rows: list[dict], period: tuple[str, str]) -> dict:
    chosen = accepted(base.bounds(rows, period))
    market_days = len(pd.bdate_range(period[0], pd.Timestamp(period[1])-pd.Timedelta(days=1)))
    frequency = len(chosen)/market_days if market_days else 0
    signed = [float(row["risk_pct"])*float(row["net_r"]) for row in chosen]
    return economic_gate(signed, frequency)


def stage_ok(value: dict, target: float, minimum: int) -> bool:
    floor = 40 if target == 1.5 else 32
    return bool(value["n"] >= minimum and value["literal_target_hit_rate"] >= floor
                and value["return_pct"] > 0 and value["dd_pct"] <= 2
                and value["max_loss_streak"] <= 4)


def main() -> None:
    caches, sources = base.load()
    rows = features(caches)
    candidates, frontier, validation_frontier, rows_by_id = [], [], [], {}
    development_passes = 0
    evaluated = 0
    for profile, target, stop in itertools.product(profiles(), TARGETS, STOPS):
        evaluated += 1
        trades = route(rows, caches, profile, target, stop)
        development = detail(base.bounds(trades, base.DEVELOPMENT))
        if development["n"] >= 5:
            frontier.append({"profile": profile, "target_pct": target,
                             "stop_pct": stop, "development": development})
        if not stage_ok(development, target, 8):
            continue
        development_passes += 1
        validation = detail(base.bounds(trades, base.VALIDATION))
        validation_frontier.append({"profile": profile, "target_pct": target,
                                    "stop_pct": stop, "development": development,
                                    "validation": validation})
        if not stage_ok(validation, target, 6):
            continue
        score = (development["return_pct"]+2*validation["return_pct"]
                 - development["dd_pct"]-2*validation["dd_pct"]
                 + min(development["literal_target_hit_rate"],
                       validation["literal_target_hit_rate"])/10)
        item = {"profile": profile, "target_pct": target, "stop_pct": stop,
                "development": development, "validation": validation,
                "score": round(score, 6)}
        candidates.append(item)
        rows_by_id[route_id(profile, target, stop)] = trades
    candidates.sort(key=lambda item: (item["score"], json.dumps(item["profile"], sort_keys=True)), reverse=True)
    frontier.sort(key=lambda item: (item["development"]["return_pct"]-item["development"]["dd_pct"], item["development"]["literal_target_hit_rate"]), reverse=True)
    validation_frontier.sort(key=lambda item: (
        min(item["development"]["literal_target_hit_rate"],
            item["validation"]["literal_target_hit_rate"]),
        item["development"]["return_pct"]+item["validation"]["return_pct"]), reverse=True)
    frozen, seen = [], set()
    for item in candidates:
        family = item["profile"]["family"]
        if family not in seen:
            frozen.append(item); seen.add(family)
    ensemble = []
    for priority, item in enumerate(frozen):
        key = route_id(item["profile"], item["target_pct"], item["stop_pct"])
        ensemble.extend({**row, "module_priority": priority} for row in rows_by_id[key])
    confirmation = detail(base.bounds(ensemble, base.CONFIRMATION)) if ensemble else detail([])
    confirmation_economics = economics(ensemble, base.CONFIRMATION) if ensemble else None
    confirmation_passed = bool(frozen and confirmation["n"] >= 12
                               and confirmation["literal_target_hit_rate"] >= 35
                               and confirmation["return_pct"] > 0
                               and confirmation["dd_pct"] <= 2
                               and confirmation["max_loss_streak"] <= 4)
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
        "protocol": {"hypothesis": "closed early-session impulse continuation or failed move",
                     "decision_bars": DECISIONS, "entry": "next native X-Perp 5m market open",
                     "targets_pct": TARGETS, "stops_pct": STOPS,
                     "exit": "target, stop or regular-session close; stop-first",
                     "development": base.DEVELOPMENT, "validation": base.VALIDATION,
                     "confirmation": base.CONFIRMATION, "reveal": base.REVEAL,
                     "money": "3% current-equity margin, fixed 10x gross"},
        "search": {"feature_rows": len(rows), "routes_evaluated": evaluated,
                   "passed_development": development_passes,
                   "qualified_development_validation": len(candidates)},
        "frozen_routes": frozen, "confirmation": confirmation,
        "confirmation_economics": confirmation_economics,
        "confirmation_passed": confirmation_passed,
        "reveal": reveal, "reveal_economics": reveal_economics,
        "objective_passed": objective_passed,
        "frontier": candidates[:15], "development_only_frontier": frontier[:15],
        "validation_frontier": validation_frontier[:15],
        "readiness": {"live_activation_allowed": False,
                      "reason": "requires objective gate and genuinely new forward evidence"},
        "sources": {**sources, str(Path(__file__)): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "limitations": ["Native five-minute history spans only about two months.",
                        "The bounded semantic grid creates selection risk.",
                        "OHLC replay cannot reproduce spread, latency or intrabar path."],
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "status", "search", "frozen_routes", "confirmation",
        "confirmation_economics", "confirmation_passed", "reveal",
        "reveal_economics", "objective_passed")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
