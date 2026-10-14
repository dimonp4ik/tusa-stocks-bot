"""Causal audit of intraday VWAP reclaim and pullback-hold structures.

The protocol and thresholds are deliberately declared in this file.  Signals
use closed 15-minute bars, enter at the next bar open, and treat a bar touching
both stop and target as a stop.  Development/validation must pass before later
years are revealed.
"""
from __future__ import annotations

import collections
import hashlib
import json
import math
import pickle
from datetime import datetime
from pathlib import Path
from statistics import median
from zoneinfo import ZoneInfo

import pandas as pd


ROOT = Path("reports/audit_2026_09_08")
MANIFEST = ROOT / "stock_extended_history_manifest.json"
OUT = ROOT / "stock_vwap_structure_oct15.json"
NY = ZoneInfo("America/New_York")
TRADED = ("AAPLUSDT", "AMZNUSDT", "GOOGLUSDT", "METAUSDT", "MSFTUSDT",
          "NVDAUSDT", "TSLAUSDT", "INTCUSDT", "MRVLUSDT", "MUUSDT")
FAMILIES = ("vwap_reclaim", "vwap_pullback_hold")
TARGETS = (.5, 1.0, 1.5, 2.0)
STOPS = (.5, .75, 1.0)
DEVIATIONS = (.15, .30)
RVOLS = (.8, 1.2)
MARKET_FILTERS = (False, True)
DEVELOPMENT = (2017, 2018, 2019, 2020, 2021)
VALIDATION = (2022, 2023)
SEALED = (2024, 2025, 2026)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def groups(candles: dict) -> dict[str, list[int]]:
    value: dict[str, list[int]] = {}
    for i, timestamp in enumerate(candles["time"]):
        local = datetime.fromtimestamp(timestamp, NY)
        minute = local.hour * 60 + local.minute
        if local.weekday() < 5 and 570 <= minute < 960:
            value.setdefault(local.date().isoformat(), []).append(i)
    return value


def atr(candles: dict, index: int, period: int = 14) -> float | None:
    if index <= period:
        return None
    values = []
    for j in range(index - period, index):
        previous = float(candles["close"][j - 1])
        values.append(max(float(candles["high"][j])-float(candles["low"][j]),
                          abs(float(candles["high"][j])-previous),
                          abs(float(candles["low"][j])-previous)))
    result = sum(values) / len(values)
    return result if math.isfinite(result) and result > 0 else None


def session_features(candles: dict) -> tuple[dict[int, dict], dict[str, list[int]]]:
    """Features available at each bar close, including causal same-slot RVOL."""
    by_day = groups(candles)
    history: dict[int, collections.deque] = collections.defaultdict(
        lambda: collections.deque(maxlen=20))
    output: dict[int, dict] = {}
    days = sorted(by_day)
    for day_position, day in enumerate(days):
        bars = by_day[day]
        if len(bars) < 20:
            continue
        cumulative_pv = cumulative_volume = 0.0
        day_rows = []
        for slot, j in enumerate(bars):
            volume = max(0.0, float(candles["volume"][j]))
            typical = (float(candles["high"][j])+float(candles["low"][j])
                       +float(candles["close"][j])) / 3
            cumulative_pv += typical*volume
            cumulative_volume += volume
            vwap = cumulative_pv/cumulative_volume if cumulative_volume else typical
            baseline = median(history[slot]) if len(history[slot]) >= 10 else None
            output[j] = {
                "slot": slot, "vwap": vwap,
                "rvol": volume/baseline if baseline and baseline > 0 else None,
            }
            day_rows.append((slot, volume))
        for slot, volume in day_rows:
            history[slot].append(volume)
    return output, by_day


def qqq_context(candles: dict) -> dict[int, dict]:
    features, by_day = session_features(candles)
    result = {}
    for day in sorted(by_day):
        bars = by_day[day]
        if not bars:
            continue
        day_open = float(candles["open"][bars[0]])
        for j in bars:
            feature = features.get(j)
            scale = atr(candles, j)
            if not feature or not scale:
                continue
            close = float(candles["close"][j])
            result[int(candles["time"][j])] = {
                "move_atr": (close-day_open)/scale,
                "vwap_side": (close-feature["vwap"])/scale,
            }
    return result


def candidates(symbol: str, candles: dict, market: dict[int, dict]) -> list[dict]:
    features, by_day = session_features(candles)
    rows = []
    days = sorted(by_day)
    for day_position, day in enumerate(days):
        bars = by_day[day]
        if len(bars) < 20:
            continue
        used: set[tuple[str, int]] = set()
        for offset in range(3, min(len(bars)-1, 18)):
            j, previous = bars[offset], bars[offset-1]
            current_feature, prior_feature = features.get(j), features.get(previous)
            scale = atr(candles, j)
            context = market.get(int(candles["time"][j]))
            if (not current_feature or not prior_feature or not scale or not context
                    or current_feature["rvol"] is None):
                continue
            close = float(candles["close"][j])
            open_ = float(candles["open"][j])
            high = float(candles["high"][j])
            low = float(candles["low"][j])
            prior_close = float(candles["close"][previous])
            vwap = float(current_feature["vwap"])
            prior_vwap = float(prior_feature["vwap"])
            slope = (vwap-prior_vwap)/scale
            body = (close-open_)/scale
            definitions = []

            # A closed bar crosses VWAP after the prior close was meaningfully
            # displaced.  Direction is the side reclaimed at this close.
            if prior_close < prior_vwap and close > vwap and body > 0:
                definitions.append(("vwap_reclaim", 1,
                                    (prior_vwap-prior_close)/scale))
            elif prior_close > prior_vwap and close < vwap and body < 0:
                definitions.append(("vwap_reclaim", -1,
                                    (prior_close-prior_vwap)/scale))

            # VWAP is sloping and price probes it, then closes back on the
            # trend side.  The probe depth is measured from the bar extreme.
            if slope > .01 and low <= vwap and close > vwap and body > 0:
                definitions.append(("vwap_pullback_hold", 1,
                                    max(0.0, (vwap-low)/scale)))
            elif slope < -.01 and high >= vwap and close < vwap and body < 0:
                definitions.append(("vwap_pullback_hold", -1,
                                    max(0.0, (high-vwap)/scale)))

            for family, direction, deviation in definitions:
                key = (family, direction)
                if key in used:
                    continue
                used.add(key)
                rows.append({
                    "symbol": symbol, "family": family,
                    "direction_n": direction,
                    "direction": "LONG" if direction > 0 else "SHORT",
                    "signal_time": int(candles["time"][j]),
                    "entry_time": int(candles["time"][bars[offset+1]]),
                    "entry_index": bars[offset+1], "session_end": bars[-1],
                    "_execution_1": tuple(bars[offset+1:]),
                    "_execution_2": tuple(
                        bars[offset+1:] +
                        (by_day[days[day_position+1]]
                         if day_position+1 < len(days) else [])),
                    "year": datetime.fromtimestamp(candles["time"][j], NY).year,
                    "session_date": day, "deviation_atr": deviation,
                    "body_atr": direction*body,
                    "vwap_slope_dir_atr": direction*slope,
                    "rvol": float(current_feature["rvol"]),
                    "market_move_dir_atr": direction*float(context["move_atr"]),
                    "market_vwap_side_dir": direction*float(context["vwap_side"]),
                })
    return rows


def resolve(row: dict, candles: dict, target_pct: float,
            stop_pct: float, hold_sessions: int = 1) -> dict:
    sign = int(row["direction_n"])
    entry = float(candles["open"][int(row["entry_index"])])
    target = entry*(1+sign*target_pct/100)
    stop = entry*(1-sign*stop_pct/100)
    outcome = "TIMEOUT"
    execution = row[f"_execution_{hold_sessions}"]
    exit_price = float(candles["close"][execution[-1]])
    exit_index = execution[-1]
    mfe_pct = mae_pct = 0.0
    for j in execution:
        open_ = float(candles["open"][j])
        high = float(candles["high"][j])
        low = float(candles["low"][j])
        mfe_pct = max(mfe_pct, 100*(high-entry)/entry if sign > 0
                      else 100*(entry-low)/entry)
        mae_pct = max(mae_pct, 100*(entry-low)/entry if sign > 0
                      else 100*(high-entry)/entry)
        stop_hit = low <= stop if sign > 0 else high >= stop
        target_hit = high >= target if sign > 0 else low <= target
        if stop_hit:  # conservative ordering when both are touched
            outcome, exit_index = "SL", j
            exit_price = min(open_, stop) if sign > 0 else max(open_, stop)
            break
        if target_hit:
            outcome, exit_index, exit_price = "TP", j, target
            break
    signed_return = sign*(exit_price-entry)/entry
    return {
        **row, "exit_time": int(candles["time"][exit_index]),
        "outcome": outcome, "signed_return": signed_return,
        "mfe_pct": mfe_pct, "mae_pct": mae_pct,
        "risk_pct": stop_pct/100, "net_r": signed_return/(stop_pct/100),
        "target_pct": target_pct, "stop_pct": stop_pct,
        "hold_sessions": hold_sessions,
        "module": family_module(row), "score": score(row),
    }


def family_module(row: dict) -> str:
    return "vwap_" + str(row["family"])


def score(row: dict) -> float:
    return (float(row["body_atr"])+.5*float(row["deviation_atr"])
            +.25*float(row["market_move_dir_atr"])
            +.1*min(float(row["rvol"]), 4.0))


def select(rows: list[dict], config: dict) -> list[dict]:
    chosen = [row for row in rows
              if row["family"] == config["family"]
              and row["deviation_atr"] >= config["deviation"]
              and row["rvol"] >= config["rvol"]
              and row["body_atr"] >= .05
              and (not config["market"] or (
                  row["market_move_dir_atr"] >= 0
                  and row["market_vwap_side_dir"] >= 0))]
    # Causal portfolio rules: at most two entries per scan, no overlapping
    # position in one symbol, and no more than four same-direction positions.
    accepted, opened = [], {}
    per_scan = collections.Counter()
    for row in sorted(chosen, key=lambda value: (
            value["entry_time"], -value["score"], value["symbol"])):
        now = row["entry_time"]
        opened = {symbol: active for symbol, active in opened.items()
                  if active["exit_time"] >= now}
        if row["symbol"] in opened or per_scan[now] >= 2:
            continue
        if sum(active["direction"] == row["direction"]
               for active in opened.values()) >= 4:
            continue
        accepted.append(row)
        opened[row["symbol"]] = row
        per_scan[now] += 1
    return accepted


def metrics(rows: list[dict], market_days: int) -> dict:
    if not rows:
        return {"n": 0, "wr": 0.0, "literal_target_hit_rate": 0.0,
                "return_pct": 0.0, "dd_pct": 0.0, "negative_months": 0,
                "months": 0, "months_ge_30pct": 0, "max_loss_streak": 0,
                "trades_per_market_day": 0.0, "cluster_share_pct": 0.0}
    equity = peak = 1.0
    maximum_dd = 0.0
    monthly_start, monthly_end = {}, {}
    losses = maximum_losses = wins = 0
    clusters = collections.Counter()
    for row in sorted(rows, key=lambda value: (
            value["exit_time"], value["entry_time"], value["symbol"])):
        month = datetime.fromtimestamp(row["exit_time"], NY).strftime("%Y-%m")
        monthly_start.setdefault(month, equity)
        equity += equity*.03*10*float(row["signed_return"])
        peak = max(peak, equity)
        maximum_dd = max(maximum_dd, (peak-equity)/peak)
        monthly_end[month] = equity
        won = row["outcome"] == "TP"
        wins += int(won)
        losses = 0 if won else losses+1
        maximum_losses = max(maximum_losses, losses)
        clusters[int(row["entry_time"]//3600)] += 1
    monthly = {key: 100*(monthly_end[key]/monthly_start[key]-1)
               for key in sorted(monthly_start)}
    outcomes = collections.Counter(row["outcome"] for row in rows)
    profitable = sum(float(row["signed_return"]) > 0 for row in rows)
    signed = [100*float(row["signed_return"]) for row in rows]
    timeouts = [100*float(row["signed_return"]) for row in rows
                if row["outcome"] == "TIMEOUT"]
    return {
        "n": len(rows), "wr": round(100*wins/len(rows), 3),
        "literal_target_hit_rate": round(100*wins/len(rows), 3),
        "profitable_trade_rate": round(100*profitable/len(rows), 3),
        "outcomes": dict(sorted(outcomes.items())),
        "mean_signed_return_pct": round(sum(signed)/len(signed), 5),
        "median_signed_return_pct": round(float(median(signed)), 5),
        "mean_mfe_pct": round(sum(float(row["mfe_pct"]) for row in rows)
                              / len(rows), 5),
        "mean_mae_pct": round(sum(float(row["mae_pct"]) for row in rows)
                              / len(rows), 5),
        "timeout_mean_return_pct": (round(sum(timeouts)/len(timeouts), 5)
                                    if timeouts else None),
        "return_pct": round(100*(equity-1), 4),
        "dd_pct": round(100*maximum_dd, 4),
        "negative_months": sum(value < 0 for value in monthly.values()),
        "months": len(monthly),
        "months_ge_30pct": sum(value >= 30 for value in monthly.values()),
        "max_loss_streak": maximum_losses,
        "trades_per_market_day": round(len(rows)/max(1, market_days), 3),
        "cluster_share_pct": round(100*max(clusters.values())/len(rows), 3),
        "monthly_return_pct": {key: round(value, 4)
                               for key, value in monthly.items()},
    }


def evaluate(rows: list[dict], config: dict, years: tuple[int, ...],
             market_days: dict[int, set[str]]) -> dict:
    subset = [row for row in rows if row["year"] in years
              and row["target_pct"] == config["target"]
              and row["stop_pct"] == config["stop"]]
    days = len(set().union(*(market_days.get(year, set()) for year in years)))
    return metrics(select(subset, config), days)


def period_metrics(accepted: list[dict], years: tuple[int, ...],
                   market_days: dict[int, set[str]]) -> dict:
    subset = [row for row in accepted if row["year"] in years]
    days = len(set().union(*(market_days.get(year, set()) for year in years)))
    return metrics(subset, days)


def development_ok(value: dict, annual: dict[str, dict]) -> bool:
    return bool(value["n"] >= 600 and value["return_pct"] > 0
                and value["literal_target_hit_rate"] >= 60
                and value["dd_pct"] <= 12
                and value["trades_per_market_day"] >= 1.0
                and sum(item["return_pct"] > 0 for item in annual.values()) >= 4)


def validation_ok(value: dict, annual: dict[str, dict]) -> bool:
    return bool(value["n"] >= 240 and value["return_pct"] > 0
                and value["literal_target_hit_rate"] >= 60
                and value["dd_pct"] <= 8
                and value["trades_per_market_day"] >= 1.0
                and all(item["return_pct"] > 0 for item in annual.values()))


def main() -> None:
    manifest_raw = MANIFEST.read_bytes()
    manifest = json.loads(manifest_raw)
    candles, sources = {}, {}
    for symbol in (*TRADED, "QQQUSDT"):
        meta = manifest["symbols"][symbol]
        path = Path(meta["path"])
        digest = sha(path)
        if digest != meta["sha256"]:
            raise ValueError(f"{symbol}: source hash mismatch")
        candles[symbol] = pickle.loads(path.read_bytes())
        sources[symbol] = {"path": str(path), "sha256": digest,
                           "rows": len(candles[symbol]["time"])}

    market = qqq_context(candles["QQQUSDT"])
    raw, market_days = [], collections.defaultdict(set)
    for symbol in TRADED:
        found = candidates(symbol, candles[symbol], market)
        raw.extend(found)
        for row in found:
            market_days[row["year"]].add(row["session_date"])

    resolved = []
    for row in raw:
        for target in TARGETS:
            for stop in STOPS:
                resolved.append(resolve(row, candles[row["symbol"]], target, stop))

    frontier, qualified = [], []
    for family in FAMILIES:
        for deviation in DEVIATIONS:
            for rvol in RVOLS:
                for market_filter in MARKET_FILTERS:
                    for target in TARGETS:
                        for stop in STOPS:
                            config = {"family": family, "deviation": deviation,
                                      "rvol": rvol, "market": market_filter,
                                      "target": target, "stop": stop}
                            matching = [row for row in resolved
                                        if row["target_pct"] == target
                                        and row["stop_pct"] == stop]
                            accepted = select(matching, config)
                            development = period_metrics(
                                accepted, DEVELOPMENT, market_days)
                            annual = {str(year): period_metrics(
                                          accepted, (year,), market_days)
                                      for year in DEVELOPMENT}
                            item = {"config": config,
                                    "development_2017_2021": development,
                                    "development_annual": annual}
                            frontier.append(item)
                            if development_ok(development, annual):
                                validation = period_metrics(
                                    accepted, VALIDATION, market_days)
                                validation_annual = {
                                    str(year): period_metrics(
                                        accepted, (year,), market_days)
                                    for year in VALIDATION}
                                item["validation_2022_2023"] = validation
                                item["validation_annual"] = validation_annual
                                if validation_ok(validation, validation_annual):
                                    qualified.append(item)

    frontier.sort(key=lambda item: (
        item["development_2017_2021"]["return_pct"],
        item["development_2017_2021"]["literal_target_hit_rate"],
        -item["development_2017_2021"]["dd_pct"]), reverse=True)
    qualified.sort(key=lambda item: (
        item["validation_2022_2023"]["return_pct"],
        item["validation_2022_2023"]["literal_target_hit_rate"]), reverse=True)
    frozen = qualified[0] if qualified else None
    sealed = None
    objective = None
    if frozen:
        config = frozen["config"]
        matching = [row for row in resolved
                    if row["target_pct"] == config["target"]
                    and row["stop_pct"] == config["stop"]]
        accepted = select(matching, config)
        sealed = {
            str(year): period_metrics(accepted, (year,), market_days)
            for year in SEALED}
        combined = period_metrics(accepted, SEALED, market_days)
        sealed["combined_2024_2026"] = combined
        objective = bool(
            combined["literal_target_hit_rate"] >= 75
            and combined["dd_pct"] <= 5
            and combined["negative_months"] == 0
            and combined["trades_per_market_day"] >= 3
            and combined["months_ge_30pct"] == combined["months"])

    report = {
        "status": ("RESEARCH_CANDIDATE_AWAITING_DIRECT_TRANSFER"
                   if objective else "REJECTED"),
        "decision": ("No runtime change. The route failed its causal gates."
                     if not objective else
                     "Do not enable. First verify on direct OKX X-Perp data."),
        "protocol": {
            "hypothesis": [
                "VWAP reclaim after material displacement",
                "VWAP pullback-hold with causal VWAP slope"],
            "development": "2017-2021", "validation": "2022-2023",
            "sealed_test": "2024-2026; revealed only after both earlier gates",
            "entry": "next 15-minute regular-session bar open (market proxy)",
            "targets_pct": TARGETS, "stops_pct": STOPS,
            "ambiguity": "stop first if one bar touches stop and target",
            "position_model": "3% current equity margin, fixed 10x gross",
            "fees": "excluded at user request",
        },
        "counts": {"raw_candidates": len(raw), "resolved_rows": len(resolved),
                   "grid": len(frontier),
                   "development_qualified": sum(
                       "validation_2022_2023" in item for item in frontier),
                   "validation_qualified": len(qualified)},
        "frozen": frozen, "sealed_results": sealed,
        "objective_passed": objective,
        "top_development_frontier": frontier[:20],
        "top_development_hit_rate": sorted(
            frontier, key=lambda item: (
                item["development_2017_2021"]["literal_target_hit_rate"],
                item["development_2017_2021"]["return_pct"]),
            reverse=True)[:20],
        "source_manifest": {"path": str(MANIFEST),
                            "sha256": hashlib.sha256(manifest_raw).hexdigest()},
        "sources": sources,
        "limitations": [
            "Dukascopy cash-session tick volume is a causal activity proxy, not OKX X-Perp volume.",
            "Intrabar ordering is unavailable; stop-first handling is conservative.",
            "A historical pass would still require an untouched direct X-Perp transfer test.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "status", "counts", "frozen", "sealed_results")}, indent=2))


if __name__ == "__main__":
    main()
