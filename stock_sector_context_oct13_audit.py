"""Audit causal sector context on the exact material stock router.

The selection protocol only sees 2018-2020.  Later eras are opened in order and
only when the same frozen rule passes the previous era.  All sector features use
the ETF candle closed before the market entry.
"""
from __future__ import annotations

import bisect
import hashlib
import json
import math
import pickle
from collections import defaultdict
from pathlib import Path

import pandas as pd

import stock_stop_path_cross_era_oct12_audit as base
from src.stock_venue_router import MIN_FIRST_TARGET_MOVE_PCT
from stock_module_cluster_audit import portfolio_gate


ROOT = base.ROOT
SECTOR_MANIFEST = ROOT / "stock_sector_history_oct13_manifest.json"
OUT = ROOT / "stock_sector_context_oct13.json"
ROWS_OUT = ROOT / "stock_sector_context_oct13_rows.csv"

# XLC is unavailable in this feed.  QQQ is retained only for the two
# communication-services names; the new sector test is XLK/XLY for the other
# eight names.  SMH is deliberately excluded because its feed begins in 2024.
SECTOR_BY_STOCK = {
    "AAPLUSDT": "XLKUSDT", "MSFTUSDT": "XLKUSDT",
    "NVDAUSDT": "XLKUSDT", "INTCUSDT": "XLKUSDT",
    "MRVLUSDT": "XLKUSDT", "MUUSDT": "XLKUSDT",
    "AMZNUSDT": "XLYUSDT", "TSLAUSDT": "XLYUSDT",
    "GOOGLUSDT": "QQQUSDT", "METAUSDT": "QQQUSDT",
    "QQQUSDT": "QQQUSDT", "SPYUSDT": "SPYUSDT",
}

RULES = (
    "sector_4_align",
    "sector_16_align",
    "sector_4_and_16_align",
    "sector_impulse_align",
    "sector_breakout_align",
    "leader_confirmed",
    "laggard_catchup",
    "aligned_not_extended",
    "sector_4_and_16_align_direct_only",
    "avoid_sector_16_extreme_1pct",
    "avoid_sector_16_extreme_1_5pct",
    "sector_16_neutral_0_3pct",
    "sector_16_mild_opposition",
    "low_sector_volatility",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_sector_history(stock_caches: dict[str, dict]) -> tuple[dict[str, dict], dict[str, str]]:
    manifest = json.loads(SECTOR_MANIFEST.read_text(encoding="utf-8"))
    sectors: dict[str, dict] = {}
    sources = {str(SECTOR_MANIFEST): digest(SECTOR_MANIFEST)}
    for symbol, item in manifest["symbols"].items():
        path = Path(item["path"])
        raw = path.read_bytes()
        found = hashlib.sha256(raw).hexdigest()
        if found != item["sha256"]:
            raise ValueError(f"sector source hash mismatch: {symbol}")
        candle = pickle.loads(raw)
        if len(candle["time"]) != item["rows"]:
            raise ValueError(f"sector row count mismatch: {symbol}")
        sectors[symbol] = candle
        sources[str(path)] = found
    # QQQ already belongs to the independently hashed stock-history manifest.
    sectors["QQQUSDT"] = stock_caches["QQQUSDT"]
    sectors["SPYUSDT"] = stock_caches["SPYUSDT"]
    return sectors, sources


def prior_index(candle: dict, entry_time: int) -> int:
    """Return the final fully closed 15-minute candle before entry."""
    times = candle["time"]
    index = bisect.bisect_right(times, entry_time - 900) - 1
    if index < 64:
        raise ValueError(f"insufficient causal sector history before {entry_time}")
    used = int(times[index])
    if used > entry_time - 900 or entry_time - 900 - used > 3600:
        raise ValueError(f"stale or future sector candle: {used=} {entry_time=}")
    return index


def signed_return(close: list[float], index: int, bars: int, direction: int) -> float:
    return direction * (float(close[index]) / float(close[index - bars]) - 1.0)


def enrich(row: dict, stocks: dict[str, dict], sectors: dict[str, dict]) -> dict:
    found = dict(row)
    symbol = str(row["symbol"])
    sector_symbol = SECTOR_BY_STOCK[symbol]
    entry_time = int(float(row["entry_time"]))
    direction = 1 if str(row["direction"]) == "LONG" else -1
    sector = sectors[sector_symbol]
    stock = stocks[symbol]
    si = prior_index(sector, entry_time)
    xi = prior_index(stock, entry_time)
    sc = sector["close"]
    xc = stock["close"]
    sector_4 = signed_return(sc, si, 4, direction)
    sector_16 = signed_return(sc, si, 16, direction)
    sector_32 = signed_return(sc, si, 32, direction)
    stock_4 = signed_return(xc, xi, 4, direction)
    stock_16 = signed_return(xc, xi, 16, direction)
    excess_16 = stock_16 - sector_16
    previous = [float(value) for value in sc[si - 16:si]]
    breakout = (float(sc[si]) > max(previous) if direction > 0
                else float(sc[si]) < min(previous))
    one_bar = [abs(float(sc[i]) / float(sc[i - 1]) - 1.0)
               for i in range(si - 31, si + 1)]
    volatility_32 = math.sqrt(sum(value * value for value in one_bar) / len(one_bar))
    found.update({
        "sector_symbol": sector_symbol,
        "sector_last_closed_time": int(sector["time"][si]),
        "stock_last_closed_time": int(stock["time"][xi]),
        "sector_4_signed": sector_4,
        "sector_16_signed": sector_16,
        "sector_32_signed": sector_32,
        "stock_4_signed": stock_4,
        "stock_16_signed": stock_16,
        "excess_16_signed": excess_16,
        "sector_breakout_align": bool(breakout),
        "sector_volatility_32": volatility_32,
        "direct_sector_proxy": sector_symbol in {"XLKUSDT", "XLYUSDT"},
    })
    return found


def keep(row: dict, rule: str) -> bool:
    s4 = float(row["sector_4_signed"])
    s16 = float(row["sector_16_signed"])
    x4 = float(row["stock_4_signed"])
    excess = float(row["excess_16_signed"])
    aligned = s4 > 0 and s16 > 0
    if rule == "sector_4_align":
        return s4 > 0
    if rule == "sector_16_align":
        return s16 > 0
    if rule == "sector_4_and_16_align":
        return aligned
    if rule == "sector_impulse_align":
        return aligned and s16 >= .0015
    if rule == "sector_breakout_align":
        return bool(row["sector_breakout_align"])
    if rule == "leader_confirmed":
        return aligned and 0 <= excess <= .0075
    if rule == "laggard_catchup":
        return s16 >= .0015 and -.0075 <= excess < 0 and x4 > 0
    if rule == "aligned_not_extended":
        return aligned and excess <= .0075
    if rule == "sector_4_and_16_align_direct_only":
        return aligned and bool(row["direct_sector_proxy"])
    if rule == "avoid_sector_16_extreme_1pct":
        return abs(s16) < .01
    if rule == "avoid_sector_16_extreme_1_5pct":
        return abs(s16) < .015
    if rule == "sector_16_neutral_0_3pct":
        return abs(s16) < .003
    if rule == "sector_16_mild_opposition":
        return -.003 <= s16 < 0
    if rule == "low_sector_volatility":
        return float(row["sector_volatility_32"]) < .0042
    raise KeyError(rule)


def candidate_rows(rows: list[dict], rule: str, years: tuple[int, ...]) -> list[dict]:
    selected = [row for row in rows if base.year_of(row) in set(years) and keep(row, rule)]
    return portfolio_gate(selected, None)


def selection_gate(candidate: dict, baseline: dict) -> bool:
    """Require a broad improvement without hiding the frequency loss."""
    return bool(
        candidate["n"] >= .45 * baseline["n"]
        and candidate["trades_per_market_day"] >= 1.0
        and candidate["wr"] >= baseline["wr"]
        and candidate["literal_target_hit_rate"] >= baseline["literal_target_hit_rate"]
        and candidate["return_3pct_margin_pct"] >= baseline["return_3pct_margin_pct"]
        and candidate["dd_pct"] <= baseline["dd_pct"]
        and candidate["negative_months"] <= baseline["negative_months"]
        and candidate["minimum_month_pct"] >= baseline["minimum_month_pct"]
        and candidate["max_loss_streak"] <= baseline["max_loss_streak"]
    )


def satellite_gate(candidate: dict, baseline: dict) -> bool:
    """Allow a sparse route only when it is independently profitable and calmer."""
    return bool(
        candidate["n"] >= .10 * baseline["n"]
        and candidate["trades_per_market_day"] >= 1.0
        and candidate["wr"] >= 55.0
        and candidate["return_3pct_margin_pct"] > 0
        and candidate["mean_signed_price_return_pct"] > 0
        and candidate["dd_pct"] <= .75 * baseline["dd_pct"]
        and candidate["minimum_month_pct"] >= -2.0
        and candidate["max_loss_streak"] <= baseline["max_loss_streak"]
    )


def evaluate_frozen(rows: list[dict], rule: str, gate) -> dict:
    output = {"rule": rule, "status": "REJECTED_VALIDATION"}
    for name, years in (("validation", base.VALIDATION), ("test", base.TEST),
                        ("confirmation", base.CONFIRMATION)):
        score = base.metrics(candidate_rows(rows, rule, years))
        baseline_score = base.metrics(portfolio_gate(base.period(rows, years), None))
        output[name] = score
        output[f"{name}_pass"] = gate(score, baseline_score)
        if not output[f"{name}_pass"]:
            output["status"] = f"REJECTED_{name.upper()}"
            break
    else:
        output["status"] = "SECTOR_SATELLITE_RESEARCH_CANDIDATE"
    return output


def bucket_summary(rows: list[dict], field: str, edges: tuple[float, ...]) -> dict:
    output = {}
    for left, right in zip(edges, edges[1:]):
        part = [row for row in rows if left <= float(row[field]) < right]
        key = f"[{left:.4f},{right:.4f})"
        output[key] = base.metrics(part)
    return output


def stop_lift(rows: list[dict], rule: str) -> dict:
    exposed = [row for row in rows if keep(row, rule)]
    rejected = [row for row in rows if not keep(row, rule)]
    def rate(part: list[dict]) -> float:
        return (100 * sum(str(row["outcome"]) == "SL" for row in part) / len(part)
                if part else 0.0)
    return {"kept_n": len(exposed), "kept_stop_rate_pct": round(rate(exposed), 4),
            "rejected_n": len(rejected), "rejected_stop_rate_pct": round(rate(rejected), 4)}


def main() -> None:
    stocks, stock_sources = base.load_history()
    sectors, sector_sources = load_sector_history(stocks)
    frame = pd.read_csv(base.SOURCE)
    raw_all = base.current.vector_frozen_rows(frame)
    audit_years = set(base.DEVELOPMENT + base.VALIDATION + base.TEST + base.CONFIRMATION)
    raw = [row for row in raw_all if base.year_of(row) in audit_years
           and 100 * float(row["risk_pct"]) * float(row["target_r"])
           + 1e-12 >= MIN_FIRST_TARGET_MOVE_PCT]
    enriched = []
    missing_context = []
    for row in raw:
        try:
            enriched.append(enrich(row, stocks, sectors))
        except ValueError as error:
            missing_context.append({"symbol": str(row["symbol"]),
                                    "entry_time": int(float(row["entry_time"])),
                                    "reason": str(error)})

    periods = {"development": base.DEVELOPMENT, "validation": base.VALIDATION,
               "test": base.TEST, "confirmation": base.CONFIRMATION}
    baseline_rows = {name: portfolio_gate(base.period(enriched, years), None)
                     for name, years in periods.items()}
    baseline = {name: base.metrics(rows) for name, rows in baseline_rows.items()}
    frontier = []
    eligible = []
    satellite_eligible = []
    for rule in RULES:
        score = base.metrics(candidate_rows(enriched, rule, base.DEVELOPMENT))
        passed = selection_gate(score, baseline["development"])
        satellite_passed = satellite_gate(score, baseline["development"])
        frontier.append({"rule": rule, "development": score,
                         "development_pass": passed,
                         "satellite_development_pass": satellite_passed})
        if passed:
            rank = (baseline["development"]["negative_months"] - score["negative_months"],
                    score["minimum_month_pct"], score["return_3pct_margin_pct"],
                    score["literal_target_hit_rate"], -score["dd_pct"])
            eligible.append((rank, rule, score))
        if satellite_passed:
            satellite_rank = (score["minimum_month_pct"],
                              score["return_3pct_margin_pct"], score["wr"],
                              -score["dd_pct"])
            satellite_eligible.append((satellite_rank, rule, score))

    frozen = validation = test = confirmation = None
    status = "REJECTED_DEVELOPMENT"
    if eligible:
        _, rule, development = max(eligible, key=lambda item: item[0])
        frozen = {"rule": rule, "development": development,
                  "development_baseline": baseline["development"],
                  "eligible_rules": len(eligible)}
        validation = base.metrics(candidate_rows(enriched, rule, base.VALIDATION))
        status = "REJECTED_VALIDATION"
        if selection_gate(validation, baseline["validation"]):
            test = base.metrics(candidate_rows(enriched, rule, base.TEST))
            status = "REJECTED_TEST"
            if selection_gate(test, baseline["test"]):
                confirmation = base.metrics(candidate_rows(enriched, rule, base.CONFIRMATION))
                status = ("SECTOR_CONTEXT_RESEARCH_CANDIDATE"
                          if selection_gate(confirmation, baseline["confirmation"])
                          else "REJECTED_CONFIRMATION")

    satellite = None
    if satellite_eligible:
        _, satellite_rule, satellite_development = max(
            satellite_eligible, key=lambda item: item[0])
        satellite = evaluate_frozen(enriched, satellite_rule, satellite_gate)
        satellite["development"] = satellite_development
        satellite["eligible_rules"] = len(satellite_eligible)

    accepted = portfolio_gate(enriched, None)
    causal_violations = sum(
        int(row["sector_last_closed_time"]) > int(float(row["entry_time"])) - 900
        or int(row["stock_last_closed_time"]) > int(float(row["entry_time"])) - 900
        for row in enriched)
    if causal_violations:
        raise ValueError(f"causal timestamp violations: {causal_violations}")
    by_year = {str(year): {rule: stop_lift(base.period(accepted, (year,)), rule)
                           for rule in RULES}
               for year in range(2018, 2027)}
    report = {
        "status": status, "ready_for_live_orders": False,
        "protocol": {
            "router": "exact robust_dynamic_entry_filtered material route",
            "development": list(base.DEVELOPMENT),
            "validation": list(base.VALIDATION), "test": list(base.TEST),
            "confirmation_only": list(base.CONFIRMATION),
            "sector_mapping": SECTOR_BY_STOCK,
            "causal_cutoff": "ETF and stock features end at entry_time minus 900 seconds",
            "money": "3% current-equity margin, fixed 10x gross",
            "market_orders_only": True,
        },
        "data_quality": {
            "causal_timestamp_violations": causal_violations,
            "raw_rows": len(raw), "rows_with_sector_context": len(enriched),
            "rows_excluded_for_missing_or_stale_context": len(missing_context),
            "excluded_context_examples": missing_context[:10],
            "smh_excluded_from_selection": True,
            "smh_reason": "Dukascopy SMH history begins 2024-10 and cannot support the cross-era protocol",
        },
        "baseline": baseline, "frontier": frontier,
        "frozen_before_validation": frozen, "validation": validation,
        "test": test, "confirmation": confirmation,
        "satellite_frozen_before_validation": satellite,
        "stop_attribution_by_year": by_year,
        "development_factor_buckets": {
            "sector_16_signed": bucket_summary(
                base.period(accepted, base.DEVELOPMENT), "sector_16_signed",
                (-math.inf, -.003, 0, .003, math.inf)),
            "excess_16_signed": bucket_summary(
                base.period(accepted, base.DEVELOPMENT), "excess_16_signed",
                (-math.inf, -.0075, 0, .0075, math.inf)),
        },
        "economic_objective": {
            "required_constant_signed_price_return_per_trade_pct":
                base.REQUIRED_SIGNED_PRICE_RETURN_PCT,
            "baseline_all_periods": base.metrics(accepted),
        },
        "sources": {**stock_sources, **sector_sources,
                    str(base.SOURCE): digest(base.SOURCE),
                    str(Path(base.__file__)): digest(Path(base.__file__)),
                    str(Path(__file__)): digest(Path(__file__))},
        "limitations": [
            "Dukascopy cash/ETF bars are research proxies rather than OKX X-Perp fills.",
            "QQQ is only a fallback for GOOGL and META because this feed lacks XLC.",
            "Sector context is an entry filter; it cannot increase trade frequency by itself.",
            "Previously published later eras are sequential robustness evidence, not pristine holdouts.",
        ],
    }
    pd.DataFrame(accepted).to_csv(ROWS_OUT, index=False)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": status, "causal_violations": causal_violations,
                      "frozen": frozen, "validation": validation,
                      "test": test, "confirmation": confirmation,
                      "frontier": frontier}, indent=2), flush=True)


if __name__ == "__main__":
    main()
