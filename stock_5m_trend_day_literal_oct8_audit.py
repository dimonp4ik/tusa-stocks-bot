"""Fixed-time trend-day stock X-Perp audit with literal price targets."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from statistics import median

import pandas as pd

import stock_5m_cross_sectional_literal_oct8_audit as base
from stock_5m_high_payoff_oct3_audit import accepted, money


ROOT = Path("reports/audit_2026_09_08")
OUT = ROOT / "stock_5m_trend_day_literal_oct8.json"
ROWS_OUT = ROOT / "stock_5m_trend_day_literal_oct8_rows.csv"
DECISIONS = (12, 24, 36)  # 10:30, 11:30 and 12:30 New York.
QQQ_MIN = (.15, .30, .50)
BREADTH_MIN = (.57, .71, .86)
RANK_MIN = (.71, .86)
EXCESS_MIN = (-.10, .10, .25)
TARGETS = (.5, 1.0, 1.5, 2.0)
STOPS = (.5, .75, 1.0)
HIT_FLOORS = {.5: 55.0, 1.0: 40.0, 1.5: 28.0, 2.0: 22.0}


def build_events(caches: dict[str, dict]) -> list[dict]:
    sessions = {symbol: base.regular_groups(candle)
                for symbol, candle in caches.items()}
    events = []
    common_days = sorted(set.intersection(
        *(set(sessions[symbol]) for symbol in base.ALL)))
    for day in common_days:
        for decision in DECISIONS:
            if any(len(sessions[symbol][day]) <= decision
                   for symbol in base.ALL):
                continue
            qbars = sessions["QQQUSDT"][day]
            qi = qbars[decision]
            qqq = caches["QQQUSDT"]
            qopen = float(qqq["open"][qbars[0]])
            qclose = float(qqq["close"][qi])
            qret = 100*(qclose/qopen-1)
            if not qret:
                continue
            direction = 1 if qret > 0 else -1
            stock_returns = {}
            stock_context = {}
            for symbol in base.STOCKS:
                candle = caches[symbol]
                bars = sessions[symbol][day]
                index = bars[decision]
                session_open = float(candle["open"][bars[0]])
                close = float(candle["close"][index])
                ret = 100*(close/session_open-1)
                stock_returns[symbol] = ret
                start = max(1, decision-12)
                path = sum(abs(float(candle["close"][bars[pos]])
                               - float(candle["close"][bars[pos-1]]))
                           for pos in range(start, decision+1))
                displacement = abs(close-float(candle["close"][bars[start-1]]))
                volumes = [float(candle["volume"][bars[pos]])
                           for pos in range(max(0, decision-20), decision)]
                volume_ratio = (float(candle["volume"][index])
                                /(median(volumes) or 1))
                stock_context[symbol] = {
                    "entry_index": index+1, "session_last": bars[-1],
                    "efficiency": displacement/path if path else 0.0,
                    "volume_ratio": volume_ratio,
                }
            directed = {symbol: direction*value
                        for symbol, value in stock_returns.items()}
            ordered = sorted(directed.items(), key=lambda item: item[1])
            ranks = {symbol: (position+1)/len(ordered)
                     for position, (symbol, _) in enumerate(ordered)}
            breadth = sum(value > 0 for value in directed.values())/len(directed)
            for symbol in base.STOCKS:
                context = stock_context[symbol]
                events.append({
                    "symbol": symbol, "day": day,
                    "signal_time": int(caches[symbol]["time"][context["entry_index"]-1]),
                    "entry_time": int(caches[symbol]["time"][context["entry_index"]]),
                    "entry_index": context["entry_index"],
                    "session_last": context["session_last"],
                    "decision_bar": decision, "direction_int": direction,
                    "direction": "LONG" if direction > 0 else "SHORT",
                    "qqq_move_dir_pct": abs(qret), "breadth_dir": breadth,
                    "rank_dir": ranks[symbol],
                    "stock_move_dir_pct": directed[symbol],
                    "excess_dir_pct": directed[symbol]-abs(qret),
                    "efficiency": context["efficiency"],
                    "volume_ratio": context["volume_ratio"],
                })
    return events


def selected(events: list[dict], config: dict) -> list[dict]:
    return [row for row in events
            if row["decision_bar"] == config["decision_bar"]
            and row["qqq_move_dir_pct"] >= config["qqq_min"]
            and row["breadth_dir"] >= config["breadth_min"]
            and row["rank_dir"] >= config["rank_min"]
            and row["excess_dir_pct"] >= config["excess_min"]
            and row["efficiency"] >= config["efficiency_min"]
            and row["volume_ratio"] <= 10]


def route(events: list[dict], caches: dict[str, dict], config: dict) -> list[dict]:
    rows = []
    for event in selected(events, config):
        row = base.simulate(event, caches[event["symbol"]],
                            config["target_pct"], config["stop_pct"], 78)
        row.update({
            "module": "stock_5m_literal_trend_day_leader",
            "module_priority": int(config["decision_bar"]),
            "route_id": route_id(config),
        })
        rows.append(row)
    return rows


def route_id(config: dict) -> str:
    return (f"trend_day_d{config['decision_bar']}_q{config['qqq_min']}"
            f"_b{config['breadth_min']}_rank{config['rank_min']}"
            f"_x{config['excess_min']}_e{config['efficiency_min']}"
            f"_tp{config['target_pct']}_sl{config['stop_pct']}")


def detail(rows: list[dict]) -> dict:
    chosen = accepted(rows)
    result = money(rows)
    hits = sum(bool(row["literal_target_hit"]) for row in chosen)
    result.update({
        "literal_target_hits": hits,
        "literal_target_hit_rate": round(100*hits/len(chosen), 2) if chosen else 0.0,
    })
    return result


def period(rows: list[dict], bounds: tuple[str, str]) -> list[dict]:
    return base.bounds(rows, bounds)


def stage_ok(value: dict, target: float, minimum: int) -> bool:
    return bool(value["n"] >= minimum
                and value["literal_target_hit_rate"] >= HIT_FLOORS[target]
                and value["return_pct"] > 0 and value["dd_pct"] <= 2.0
                and value["max_loss_streak"] <= 4)


def ensemble_ok(value: dict) -> bool:
    return bool(value["n"] >= 12 and value["return_pct"] > 0
                and value["dd_pct"] <= 2.0 and value["max_loss_streak"] <= 4
                and value["literal_target_hit_rate"] >= 35)


def main() -> None:
    caches, sources = base.load()
    events = build_events(caches)
    configs = [dict(zip(
        ("decision_bar", "qqq_min", "breadth_min", "rank_min", "excess_min",
         "efficiency_min", "target_pct", "stop_pct"), values))
        for values in itertools.product(
            DECISIONS, QQQ_MIN, BREADTH_MIN, RANK_MIN, EXCESS_MIN,
            (.15, .30, .50), TARGETS, STOPS)]
    candidates = []
    frontiers = []
    rows_by_id = {}
    for config in configs:
        rows = route(events, caches, config)
        development = detail(period(rows, base.DEVELOPMENT))
        if development["n"] >= 4:
            frontiers.append({"config": config, "development": development})
        if not stage_ok(development, config["target_pct"], 6):
            continue
        validation = detail(period(rows, base.VALIDATION))
        if not stage_ok(validation, config["target_pct"], 5):
            continue
        score = (development["return_pct"]+2*validation["return_pct"]
                 - development["dd_pct"]-2*validation["dd_pct"]
                 + min(development["literal_target_hit_rate"],
                       validation["literal_target_hit_rate"])/10)
        item = {"config": config, "development": development,
                "validation": validation, "score": round(score, 6)}
        candidates.append(item)
        rows_by_id[route_id(config)] = rows
    candidates.sort(key=lambda item: (item["score"], route_id(item["config"])),
                    reverse=True)
    frontiers.sort(key=lambda item: (
        item["development"]["return_pct"]-item["development"]["dd_pct"],
        item["development"]["literal_target_hit_rate"]), reverse=True)

    # Freeze one independently selected route per decision time.
    frozen = []
    used = set()
    for item in candidates:
        decision = item["config"]["decision_bar"]
        if decision not in used:
            frozen.append(item)
            used.add(decision)
    ensemble = []
    for item in frozen:
        ensemble.extend(rows_by_id[route_id(item["config"])])
    confirmation = detail(period(ensemble, base.CONFIRMATION)) if ensemble else detail([])
    confirmation_passed = bool(frozen and ensemble_ok(confirmation))
    reveal = detail(period(ensemble, base.REVEAL)) if confirmation_passed else None
    reveal_passed = bool(reveal and reveal["n"] >= 3
                         and reveal["return_pct"] > 0
                         and reveal["max_loss_streak"] <= 3)
    status = ("RESEARCH_CANDIDATE_AWAITING_MORE_FORWARD" if reveal_passed
              else "REJECTED_ON_INITIAL_REVEAL" if confirmation_passed
              else "REJECTED_BEFORE_REVEAL")
    if confirmation_passed:
        pd.DataFrame(accepted(ensemble)).to_csv(ROWS_OUT, index=False)
    report = {
        "status": status,
        "protocol": {
            "hypothesis": "fixed-time QQQ breadth trend plus cross-sectional leader",
            "decision_bars": {"12": "10:30 NY", "24": "11:30 NY", "36": "12:30 NY"},
            "entry": "next native X-Perp 5m market-open proxy",
            "targets_pct": TARGETS, "stops_pct": STOPS,
            "exit": "target, stop, or cash-session close; stop-first",
            "selection": "development then validation; at most one route per decision time",
            "reveal": "Oct3-7 only after frozen ensemble passes Sep8-Oct2 confirmation",
            "money": "3% current-equity margin, fixed 10x gross",
            "orders": "market proxies only; no live orders",
        },
        "periods": {"development": base.DEVELOPMENT,
                    "validation": base.VALIDATION,
                    "confirmation": base.CONFIRMATION, "reveal": base.REVEAL},
        "search": {"base_events": len(events), "routes_evaluated": len(configs),
                   "qualified_development_validation": len(candidates)},
        "frozen_routes": frozen,
        "confirmation": confirmation,
        "confirmation_passed": confirmation_passed,
        "reveal": reveal, "initial_reveal_passed": reveal_passed,
        "frontier": candidates[:15],
        "development_only_frontier": frontiers[:15],
        "readiness": {"live_activation_allowed": False,
                      "reason": "historical research is insufficient for live activation"},
        "sources": {**sources, str(Path(__file__)): hashlib.sha256(
            Path(__file__).read_bytes()).hexdigest()},
        "limitations": [
            "Native five-minute history spans only about two months.",
            "Fixed-time signals cannot guarantee three trades every market day.",
            "OHLC bars do not reproduce spread, latency or intrabar path.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "status", "search", "frozen_routes", "confirmation",
        "confirmation_passed", "reveal")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
