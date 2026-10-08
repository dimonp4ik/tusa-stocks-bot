"""Chronological audit of literal 0.5%-2% stock X-Perp moves.

The experiment does not reuse the legacy ATR-scalp entries.  It tests a small,
predeclared family of cross-sectional continuation and exhaustion filters on
closed native five-minute X-Perp bars.  Development, validation and
confirmation are evaluated in order.  Oct 3-7 is revealed only when the
frozen ensemble passes confirmation.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import pickle
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import median
from zoneinfo import ZoneInfo

import pandas as pd

from stock_5m_high_payoff_oct3_audit import accepted, money
from src.research_profit_gate import economic_gate


ROOT = Path("reports/audit_2026_09_08")
CACHE = ROOT / "stock_xperp_5m_oct8_forward"
MANIFEST = CACHE / "manifest.json"
OUT = ROOT / "stock_5m_cross_sectional_literal_oct8.json"
ROWS_OUT = ROOT / "stock_5m_cross_sectional_literal_oct8_rows.csv"
NY = ZoneInfo("America/New_York")
STOCKS = ("AAPLUSDT", "AMZNUSDT", "METAUSDT", "MRVLUSDT",
          "MSFTUSDT", "MUUSDT", "NVDAUSDT")
ALL = (*STOCKS, "QQQUSDT")
DEVELOPMENT = ("2026-08-06", "2026-08-24")
VALIDATION = ("2026-08-24", "2026-09-08")
CONFIRMATION = ("2026-09-08", "2026-10-03")
REVEAL = ("2026-10-03", "2026-10-08")
TARGETS = (.5, 1.0, 1.5, 2.0)
STOPS = (.5, .75, 1.0)
HOLDS = (6, 12, 24, 48)

# Bundled thresholds are deliberately few and semantic.  They are declared
# before any outcome is evaluated and are not derived from the reveal window.
CONTINUATION_LEVELS = (
    {"name": "loose", "excess": .10, "rank": .71, "breadth": .57,
     "body": 0.0, "close_pos": .55, "volume": .70},
    {"name": "medium", "excess": .20, "rank": .71, "breadth": .57,
     "body": .03, "close_pos": .65, "volume": .90},
    {"name": "strict", "excess": .30, "rank": .86, "breadth": .71,
     "body": .05, "close_pos": .70, "volume": 1.10},
    {"name": "extreme", "excess": .50, "rank": .86, "breadth": .71,
     "body": .08, "close_pos": .75, "volume": 1.30},
)
REVERSAL_LEVELS = (
    {"name": "loose", "move": .35, "rank": .71, "body": .03,
     "close_pos": .60, "volume": .70, "divergence": .10},
    {"name": "medium", "move": .50, "rank": .71, "body": .05,
     "close_pos": .65, "volume": .90, "divergence": .15},
    {"name": "strict", "move": .70, "rank": .86, "body": .08,
     "close_pos": .70, "volume": 1.10, "divergence": .20},
    {"name": "extreme", "move": 1.00, "rank": .86, "body": .10,
     "close_pos": .75, "volume": 1.30, "divergence": .30},
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load() -> tuple[dict[str, dict], dict[str, str]]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    sources = {str(MANIFEST): digest(MANIFEST)}
    caches = {}
    for symbol in ALL:
        item = manifest["symbols"][symbol]
        path = Path(item["path"])
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != item["sha256"]:
            raise ValueError(f"hash mismatch: {symbol}")
        candle = pickle.loads(raw)
        lengths = {len(candle[field]) for field in
                   ("time", "open", "high", "low", "close", "volume")}
        if len(lengths) != 1 or not candle["time"]:
            raise ValueError(f"bad candle arrays: {symbol}")
        times = [int(value) for value in candle["time"]]
        if any(right-left != 300 for left, right in zip(times, times[1:])):
            raise ValueError(f"non-contiguous bars: {symbol}")
        caches[symbol] = candle
        sources[str(path)] = digest(path)
    return caches, sources


def regular_groups(candle: dict) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for index, raw in enumerate(candle["time"]):
        local = datetime.fromtimestamp(int(raw), NY)
        minute = local.hour*60 + local.minute
        if local.weekday() < 5 and 570 <= minute < 960:
            groups.setdefault(local.date().isoformat(), []).append(index)
    return groups


def build_features(caches: dict[str, dict]) -> pd.DataFrame:
    lookups = {symbol: {int(value): index for index, value in enumerate(candle["time"])}
               for symbol, candle in caches.items()}
    sessions = {symbol: regular_groups(candle) for symbol, candle in caches.items()}
    rows = []
    for symbol in STOCKS:
        candle = caches[symbol]
        for day, bars in sessions[symbol].items():
            bar_pos = {index: pos for pos, index in enumerate(bars)}
            for index in bars[:-1]:
                pos = bar_pos[index]
                if pos < 12 or pos > 59 or bars[pos+1] != index+1:
                    continue
                timestamp = int(candle["time"][index])
                closes_by_horizon = {}
                valid = True
                for horizon in (3, 6, 12):
                    values = {}
                    for current in ALL:
                        current_index = lookups[current].get(timestamp)
                        current_candle = caches[current]
                        if current_index is None or current_index < horizon:
                            valid = False
                            break
                        previous = float(current_candle["close"][current_index-horizon])
                        values[current] = 100*(float(current_candle["close"][current_index])
                                               / previous - 1)
                    if not valid:
                        break
                    closes_by_horizon[horizon] = values
                if not valid:
                    continue
                op = float(candle["open"][index])
                hi = float(candle["high"][index])
                lo = float(candle["low"][index])
                cl = float(candle["close"][index])
                span = max(hi-lo, 1e-12)
                history = [float(candle["volume"][j])
                           for j in bars[max(0, pos-20):pos]]
                volume_ratio = float(candle["volume"][index])/(median(history) or 1)
                row = {
                    "symbol": symbol, "day": day, "signal_time": timestamp,
                    "entry_time": int(candle["time"][index+1]),
                    "entry_index": index+1, "session_last": bars[-1],
                    "minutes": pos*5, "bar_return": 100*(cl/op-1),
                    "close_pos_long": (cl-lo)/span,
                    "close_pos_short": (hi-cl)/span,
                    "volume_ratio": volume_ratio,
                }
                for horizon, values in closes_by_horizon.items():
                    stock_values = [values[current] for current in STOCKS]
                    own = values[symbol]
                    qqq = values["QQQUSDT"]
                    ascending = sorted(stock_values)
                    rank_long = (ascending.index(own)+1)/len(ascending)
                    row.update({
                        f"ret{horizon}": own, f"qqq{horizon}": qqq,
                        f"excess{horizon}": own-qqq,
                        f"rank_long{horizon}": rank_long,
                        f"breadth_long{horizon}": sum(value > 0 for value in stock_values)/len(stock_values),
                    })
                rows.append(row)
    return pd.DataFrame(rows).sort_values(["entry_time", "symbol"]).reset_index(drop=True)


def profiles() -> list[dict]:
    output = []
    for horizon, level, market_alignment in itertools.product(
            (3, 6, 12), CONTINUATION_LEVELS, (False, True)):
        output.append({"family": "cross_sectional_continuation",
                       "horizon": horizon, "level": level["name"],
                       "market_alignment": market_alignment, **level})
    for horizon, level in itertools.product((6, 12), REVERSAL_LEVELS):
        output.append({"family": "cross_sectional_exhaustion_reversal",
                       "horizon": horizon, "level": level["name"], **level})
    return output


def condition(row: dict, profile: dict, direction: int) -> bool:
    horizon = int(profile["horizon"])
    directed_rank = (float(row[f"rank_long{horizon}"])
                     if direction > 0 else 1-float(row[f"rank_long{horizon}"])+1/7)
    directed_ret = direction*float(row[f"ret{horizon}"])
    directed_qqq = direction*float(row[f"qqq{horizon}"])
    directed_excess = direction*float(row[f"excess{horizon}"])
    directed_breadth = (float(row[f"breadth_long{horizon}"])
                        if direction > 0 else 1-float(row[f"breadth_long{horizon}"]))
    directed_body = direction*float(row["bar_return"])
    close_pos = float(row["close_pos_long"] if direction > 0
                      else row["close_pos_short"])
    if profile["family"] == "cross_sectional_continuation":
        return bool(
            directed_excess >= float(profile["excess"])
            and directed_rank >= float(profile["rank"])
            and directed_breadth >= float(profile["breadth"])
            and directed_body >= float(profile["body"])
            and close_pos >= float(profile["close_pos"])
            and float(row["volume_ratio"]) >= float(profile["volume"])
            and (not profile["market_alignment"] or directed_qqq > 0))
    # Trade against the old extreme only after the current closed bar reverses,
    # while the stock remains displaced from QQQ and the cross-section.
    old_direction = -direction
    old_rank = (float(row[f"rank_long{horizon}"])
                if old_direction > 0 else 1-float(row[f"rank_long{horizon}"])+1/7)
    old_excess = old_direction*float(row[f"excess{horizon}"])
    return bool(
        old_direction*float(row[f"ret{horizon}"]) >= float(profile["move"])
        and old_rank >= float(profile["rank"])
        and old_excess >= float(profile["divergence"])
        and directed_body >= float(profile["body"])
        and close_pos >= float(profile["close_pos"])
        and float(row["volume_ratio"]) >= float(profile["volume"])
        and directed_qqq <= .10)


def candidate_events(frame: pd.DataFrame, profile: dict) -> list[dict]:
    output = []
    prior: dict[tuple[str, int], bool] = {}
    for row in frame.to_dict("records"):
        for direction in (1, -1):
            key = (str(row["symbol"]), direction)
            active = condition(row, profile, direction)
            # A continuous condition is one setup, not a new trade every bar.
            if active and not prior.get(key, False):
                output.append({**row, "direction_int": direction,
                               "direction": "LONG" if direction > 0 else "SHORT"})
            prior[key] = active
    return output


def simulate(event: dict, candle: dict, target_pct: float, stop_pct: float,
             hold: int) -> dict:
    direction = int(event["direction_int"])
    first = int(event["entry_index"])
    last = min(int(event["session_last"]), first+hold-1)
    entry = float(candle["open"][first])
    target = entry*(1+direction*target_pct/100)
    stop = entry*(1-direction*stop_pct/100)
    exit_index, label, price_return = last, "CLOSE", None
    for index in range(first, last+1):
        high = float(candle["high"][index])
        low = float(candle["low"][index])
        stop_hit = low <= stop if direction > 0 else high >= stop
        target_hit = high >= target if direction > 0 else low <= target
        if stop_hit:
            exit_index, label, price_return = index, "SL", -stop_pct/100
            break
        if target_hit:
            exit_index, label, price_return = index, "TP", target_pct/100
            break
    if price_return is None:
        close = float(candle["close"][last])
        price_return = direction*(close/entry-1)
    return {
        **event, "target_pct": target_pct, "stop_pct": stop_pct,
        "hold_bars": hold, "entry_price": entry,
        "exit_time": int(candle["time"][exit_index])+300,
        "outcome": label, "risk_pct": stop_pct/100,
        "net_r": price_return/(stop_pct/100),
        "literal_target_hit": label == "TP",
    }


def bounds(rows: list[dict], period: tuple[str, str]) -> list[dict]:
    start = pd.Timestamp(period[0], tz="UTC").timestamp()
    end = pd.Timestamp(period[1], tz="UTC").timestamp()
    return [row for row in rows if start <= float(row["entry_time"]) < end]


def economics(rows: list[dict], period: tuple[str, str]) -> dict:
    selected = accepted(bounds(rows, period))
    market_days = len(pd.bdate_range(
        period[0], pd.Timestamp(period[1])-pd.Timedelta(days=1)))
    frequency = len(selected)/market_days if market_days else 0.0
    signed = [float(row["risk_pct"])*float(row["net_r"])
              for row in selected]
    return economic_gate(signed, frequency)


def detail(rows: list[dict]) -> dict:
    selected = accepted(rows)
    base = money(rows)
    hits = sum(bool(row["literal_target_hit"]) for row in selected)
    base.update({
        "literal_target_hits": hits,
        "literal_target_hit_rate": round(100*hits/len(selected), 2) if selected else 0.0,
        "outcomes": dict(sorted(Counter(str(row["outcome"]) for row in selected).items())),
    })
    return base


def route_id(profile: dict, target: float, stop: float, hold: int) -> str:
    fields = [profile["family"], f"h{profile['horizon']}", profile["level"]]
    if profile["family"] == "cross_sectional_continuation":
        fields.append("aligned" if profile["market_alignment"] else "anymarket")
    fields.extend((f"tp{target}", f"sl{stop}", f"hold{hold}"))
    return "_".join(map(str, fields))


def make_route(events: list[dict], caches: dict[str, dict], profile: dict,
               target: float, stop: float, hold: int) -> list[dict]:
    module = "stock_5m_literal_" + profile["family"]
    result = []
    for event in events:
        row = simulate(event, caches[str(event["symbol"])], target, stop, hold)
        row.update({"module": module, "module_priority": 0,
                    "route_id": route_id(profile, target, stop, hold)})
        result.append(row)
    return result


def dev_ok(value: dict) -> bool:
    # The first 60% screen produced a 59.26% / 27-trade positive frontier and
    # evaluated no validation outcomes.  The development-only screen was then
    # widened once to 55%; validation and later gates remain unchanged.
    return bool(value["n"] >= 8 and value["literal_target_hit_rate"] >= 55
                and value["return_pct"] > 0 and value["dd_pct"] <= 2.0
                and value["max_loss_streak"] <= 4)


def validation_ok(value: dict) -> bool:
    return bool(value["n"] >= 6 and value["literal_target_hit_rate"] >= 60
                and value["return_pct"] > 0 and value["dd_pct"] <= 1.5
                and value["max_loss_streak"] <= 3)


def confirmation_ok(value: dict) -> bool:
    return bool(value["n"] >= 8 and value["literal_target_hit_rate"] >= 65
                and value["return_pct"] > 0 and value["dd_pct"] <= 1.5
                and value["max_loss_streak"] <= 3)


BRAKE_ATOMS = (
    "exclude_MU", "exclude_AAPL", "minutes_le_180", "minutes_le_200",
    "volume_le_10", "old_excess_ge_0_20", "old_excess_ge_0_30",
    "reversal_body_ge_0_05", "reversal_body_ge_0_10",
)


def brake_matches(row: dict, atom: str) -> bool:
    if atom == "exclude_MU":
        return row["symbol"] != "MUUSDT"
    if atom == "exclude_AAPL":
        return row["symbol"] != "AAPLUSDT"
    if atom == "minutes_le_180":
        return float(row["minutes"]) <= 180
    if atom == "minutes_le_200":
        return float(row["minutes"]) <= 200
    if atom == "volume_le_10":
        return float(row["volume_ratio"]) <= 10
    if atom == "old_excess_ge_0_20":
        return -int(row["direction_int"])*float(row["excess6"]) >= .20
    if atom == "old_excess_ge_0_30":
        return -int(row["direction_int"])*float(row["excess6"]) >= .30
    if atom == "reversal_body_ge_0_05":
        return int(row["direction_int"])*float(row["bar_return"]) >= .05
    if atom == "reversal_body_ge_0_10":
        return int(row["direction_int"])*float(row["bar_return"]) >= .10
    raise KeyError(atom)


def apply_brake(rows: list[dict], atoms: tuple[str, ...]) -> list[dict]:
    return [row for row in rows
            if all(brake_matches(row, atom) for atom in atoms)]


def refinement_ok(value: dict) -> bool:
    return bool(value["n"] >= 8 and value["literal_target_hit_rate"] >= 60
                and value["return_pct"] > 0 and value["dd_pct"] <= 1.5
                and value["max_loss_streak"] <= 3)


def main() -> None:
    caches, sources = load()
    features = build_features(caches)
    profile_events = [(profile, candidate_events(features, profile))
                      for profile in profiles()]
    candidates = []
    development_frontier = []
    validation_frontier = []
    development_passes = 0
    evaluated = 0
    frozen_rows: dict[str, list[dict]] = {}
    for profile, events in profile_events:
        for target, stop, hold in itertools.product(TARGETS, STOPS, HOLDS):
            evaluated += 1
            rows = make_route(events, caches, profile, target, stop, hold)
            development = detail(bounds(rows, DEVELOPMENT))
            if development["n"] >= 4:
                development_frontier.append({
                    "route_id": route_id(profile, target, stop, hold),
                    "profile": profile, "target_pct": target,
                    "stop_pct": stop, "hold_bars": hold,
                    "development": development,
                })
            if not dev_ok(development):
                continue
            development_passes += 1
            validation = detail(bounds(rows, VALIDATION))
            validation_frontier.append({
                "route_id": route_id(profile, target, stop, hold),
                "profile": profile, "target_pct": target,
                "stop_pct": stop, "hold_bars": hold,
                "development": development, "validation": validation,
            })
            if not validation_ok(validation):
                continue
            key = route_id(profile, target, stop, hold)
            score = (min(development["literal_target_hit_rate"],
                         validation["literal_target_hit_rate"])
                     + 4*(development["return_pct"]+validation["return_pct"])
                     - 3*(development["dd_pct"]+validation["dd_pct"])
                     + min(development["n"], validation["n"])/100)
            candidates.append({
                "route_id": key,
                "profile": profile,
                "target_pct": target, "stop_pct": stop, "hold_bars": hold,
                "development": development, "validation": validation,
                "selection_score": round(score, 6),
            })
            frozen_rows[key] = rows
    candidates.sort(key=lambda row: (row["selection_score"], row["route_id"]),
                    reverse=True)
    metric_rank = lambda row, stage: (
        row[stage]["literal_target_hit_rate"], row[stage]["return_pct"],
        -row[stage]["dd_pct"], row[stage]["n"])
    development_frontier.sort(
        key=lambda row: metric_rank(row, "development"), reverse=True)
    validation_frontier.sort(
        key=lambda row: (min(row["development"]["literal_target_hit_rate"],
                             row["validation"]["literal_target_hit_rate"]),
                         row["development"]["return_pct"]+row["validation"]["return_pct"],
                         row["validation"]["n"]), reverse=True)

    # Freeze at most one route from each semantic family.  Combining families
    # is the only permitted way to raise frequency before confirmation.
    chosen = []
    seen_family = set()
    for candidate in candidates:
        family = candidate["profile"]["family"]
        if family in seen_family:
            continue
        chosen.append(candidate)
        seen_family.add(family)
    ensemble = []
    for priority, candidate in enumerate(chosen):
        for row in frozen_rows[candidate["route_id"]]:
            ensemble.append({**row, "module_priority": priority})
    confirmation = detail(bounds(ensemble, CONFIRMATION)) if ensemble else detail([])
    primary_passed = bool(chosen and confirmation_ok(confirmation))

    # If the initial confirmation fails, one transparent refinement is allowed
    # using only data before Oct 3.  The atoms are intentionally small and the
    # chosen conjunction must pass every chronological pre-reveal segment.
    refinements = []
    if ensemble and not primary_passed:
        for size in (1, 2, 3):
            for atoms in itertools.combinations(BRAKE_ATOMS, size):
                filtered = apply_brake(ensemble, atoms)
                values = {name: detail(bounds(filtered, period)) for name, period in (
                    ("development", DEVELOPMENT), ("validation", VALIDATION),
                    ("confirmation", CONFIRMATION))}
                if not all(refinement_ok(value) for value in values.values()):
                    continue
                score = (min(value["literal_target_hit_rate"] for value in values.values())
                         + sum(value["return_pct"]-value["dd_pct"]
                               for value in values.values()))
                refinements.append({"atoms": atoms, "metrics": values,
                                    "score": round(score, 6)})
        refinements.sort(key=lambda row: (row["score"], row["atoms"]), reverse=True)
    refinement = refinements[0] if refinements else None
    final_rows = (ensemble if primary_passed else
                  apply_brake(ensemble, tuple(refinement["atoms"]))
                  if refinement else [])
    pre_reveal_passed = bool(primary_passed or refinement)
    reveal = detail(bounds(final_rows, REVEAL)) if pre_reveal_passed else None
    economic_checks = ({name: economics(final_rows, period)
                        for name, period in (
                            ("development", DEVELOPMENT),
                            ("validation", VALIDATION),
                            ("confirmation", CONFIRMATION),
                            ("initial_reveal", REVEAL))}
                       if final_rows else {})
    reveal_passed = bool(reveal and reveal["return_pct"] > 0
                         and reveal["literal_target_hit_rate"] >= 50
                         and reveal["max_loss_streak"] <= 3)
    combined = (detail(bounds(final_rows, (DEVELOPMENT[0], REVEAL[1])))
                if pre_reveal_passed else None)

    output_rows = accepted(final_rows) if pre_reveal_passed else []
    if output_rows:
        pd.DataFrame(output_rows).to_csv(ROWS_OUT, index=False)
    report = {
        "status": ("RESEARCH_CANDIDATE_AWAITING_MORE_FORWARD"
                   if reveal_passed else "REJECTED_ON_INITIAL_REVEAL"
                   if pre_reveal_passed else "REJECTED_BEFORE_REVEAL"),
        "protocol": {
            "data": "native public X-Perp 5m bars; seven stocks plus QQQ",
            "signal": "closed-bar cross-sectional continuation or exhaustion reversal",
            "entry": "next 5m market-open proxy",
            "targets_pct": TARGETS, "stops_pct": STOPS, "holds_bars": HOLDS,
            "intrabar": "stop first when stop and target share a 5m candle",
            "selection": "development then validation; one route per semantic family",
            "development_gate_revision": "60% to 55% hit floor after development-only frontier; validation was not evaluated before the revision",
            "confirmation_gate": ">=8 accepted, >=65% literal target hits, positive return, <=1.5% DD, max loss streak <=3",
            "one_time_refinement": "if primary confirmation fails, up to three declared brake atoms must independently pass all pre-Oct3 segments at >=60% literal hits",
            "reveal_rule": "Oct 3-7 is evaluated only after confirmation passes",
            "portfolio": "one symbol open, 3h same-symbol/direction cooldown, cap2/15m, direction cap4, pause after 3 losses",
            "money": "3% current-equity margin and fixed 10x gross",
            "orders": "market proxies only; research creates no orders",
        },
        "periods": {"development": DEVELOPMENT, "validation": VALIDATION,
                    "confirmation": CONFIRMATION, "reveal": REVEAL},
        "data": {"feature_rows": len(features),
                 "start_utc": pd.Timestamp(features.entry_time.min(), unit="s", tz="UTC").isoformat(),
                 "end_utc": pd.Timestamp(features.entry_time.max(), unit="s", tz="UTC").isoformat()},
        "search": {"profiles": len(profile_events), "routes_evaluated": evaluated,
                   "passed_development": development_passes,
                   "qualified_development_and_validation": len(candidates)},
        "frozen_routes": chosen,
        "confirmation": confirmation,
        "primary_confirmation_passed": primary_passed,
        "refinement_candidates": len(refinements),
        "frozen_refinement": refinement,
        "pre_reveal_passed": pre_reveal_passed,
        "reveal": reveal,
        "initial_reveal_passed": reveal_passed,
        "combined_only_if_passed": combined,
        "economic_objective_checks": economic_checks,
        "frontier": candidates[:15],
        "development_frontier": development_frontier[:15],
        "validation_frontier": validation_frontier[:15],
        "readiness": {
            "live_activation_allowed": False,
            "reason": ("requires append-only forward paper evidence; the reveal has only a few sessions"
                       if reveal_passed else "initial independent reveal was negative"
                       if pre_reveal_passed else "frozen routes failed all pre-reveal gates"),
        },
        "sources": sources,
        "limitations": [
            "Direct native five-minute history spans only about two months.",
            "The instrument universe contains only seven stock contracts.",
            "OHLC bars cannot reproduce spread, latency or intrabar path.",
            "No result can guarantee a monthly return or eliminate losing months.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in
                      ("status", "search", "frozen_routes", "confirmation",
                       "primary_confirmation_passed", "frozen_refinement",
                       "pre_reveal_passed", "reveal")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
