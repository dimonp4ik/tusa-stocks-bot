"""Audit causally validated stock order blocks on the exact material router."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from statistics import median

import pandas as pd

import stock_stop_path_cross_era_oct12_audit as base
from src.stock_venue_router import MIN_FIRST_TARGET_MOVE_PCT
from stock_module_cluster_audit import portfolio_gate


OUT = base.ROOT / "stock_valid_order_block_oct13.json"
RULES = (
    "reject_opposing_displacement_ob",
    "reject_opposing_bos_ob",
    "reject_opposing_bos_volume_ob",
    "require_supportive_displacement_ob",
    "require_supportive_bos_ob",
    "require_supportive_bos_volume_ob",
)
FIELDS = tuple(rule.replace("reject_", "").replace("require_", "") for rule in RULES)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def overlaps(low: float, high: float, left: float, right: float) -> bool:
    return min(high, max(left, right)) >= max(low, min(left, right))


def valid_ob_features(row: dict, candle: dict) -> dict[str, bool]:
    lookup = {int(value): index for index, value in enumerate(candle["time"])}
    index = lookup[int(float(row["entry_time"]))]
    signal = index - 1
    entry = float(candle["open"][index])
    direction = 1 if str(row["direction"]) == "LONG" else -1
    risk = entry * float(row["risk_pct"])
    target = entry + direction * risk * float(row["target_r"])
    stop = entry - direction * risk
    flags = {field: False for field in FIELDS}
    start = max(21, signal - 96)
    for origin in range(start, signal - 2):
        opening = float(candle["open"][origin])
        close_origin = float(candle["close"][origin])
        high_origin = float(candle["high"][origin])
        low_origin = float(candle["low"][origin])
        bullish = close_origin > opening
        bearish = close_origin < opening
        supply_zone = (min(opening, close_origin), high_origin)
        demand_zone = (low_origin, max(opening, close_origin))
        prior_low = min(float(value) for value in candle["low"][origin - 8:origin])
        prior_high = max(float(value) for value in candle["high"][origin - 8:origin])
        for displaced in range(origin + 1, min(signal, origin + 3) + 1):
            close_displaced = float(candle["close"][displaced])
            volumes = [float(value) for value in candle["volume"][displaced - 20:displaced]]
            median_volume = float(median(volumes)) if volumes else 0.0
            volume_confirmed = (median_volume > 0
                                and float(candle["volume"][displaced]) >= 1.2 * median_volume)
            down = ((close_origin - close_displaced) / risk >= .75
                    and close_displaced < low_origin)
            up = ((close_displaced - close_origin) / risk >= .75
                  and close_displaced > high_origin)
            supply_valid = (down and max(float(value) for value in
                                         candle["close"][displaced:signal + 1])
                            <= high_origin * 1.001)
            demand_valid = (up and min(float(value) for value in
                                       candle["close"][displaced:signal + 1])
                            >= low_origin * .999)
            supply_bos = supply_valid and close_displaced < prior_low
            demand_bos = demand_valid and close_displaced > prior_high
            supply_opposing = direction > 0 and overlaps(*supply_zone, entry, target)
            supply_supportive = direction < 0 and overlaps(*supply_zone, entry, stop)
            demand_opposing = direction < 0 and overlaps(*demand_zone, entry, target)
            demand_supportive = direction > 0 and overlaps(*demand_zone, entry, stop)
            opposing_valid = ((supply_valid and supply_opposing)
                              or (demand_valid and demand_opposing))
            opposing_bos = ((supply_bos and supply_opposing)
                            or (demand_bos and demand_opposing))
            supportive_valid = ((supply_valid and supply_supportive)
                                or (demand_valid and demand_supportive))
            supportive_bos = ((supply_bos and supply_supportive)
                              or (demand_bos and demand_supportive))
            flags["opposing_displacement_ob"] |= opposing_valid
            flags["opposing_bos_ob"] |= opposing_bos
            flags["opposing_bos_volume_ob"] |= opposing_bos and volume_confirmed
            flags["supportive_displacement_ob"] |= supportive_valid
            flags["supportive_bos_ob"] |= supportive_bos
            flags["supportive_bos_volume_ob"] |= supportive_bos and volume_confirmed
    return flags


def apply_rule(rows: list[dict], name: str) -> list[dict]:
    field = name.replace("reject_", "").replace("require_", "")
    required = name.startswith("require_")
    return [row for row in rows if bool(row[field]) == required]


def factor_summary(rows: list[dict]) -> dict:
    baseline = (sum(str(row["outcome"]) == "SL" for row in rows) / len(rows)
                if rows else 0.0)
    output = {}
    for field in FIELDS:
        exposed = [row for row in rows if bool(row[field])]
        clear = [row for row in rows if not bool(row[field])]
        exposed_rate = (sum(str(row["outcome"]) == "SL" for row in exposed)
                        / len(exposed) if exposed else 0.0)
        clear_rate = (sum(str(row["outcome"]) == "SL" for row in clear)
                      / len(clear) if clear else 0.0)
        output[field] = {
            "exposed_n": len(exposed),
            "exposed_stop_rate_pct": round(100 * exposed_rate, 4),
            "clear_stop_rate_pct": round(100 * clear_rate, 4),
            "stop_rate_lift": round(exposed_rate / baseline, 4) if baseline else None,
        }
    return output


def main() -> None:
    candles, sources = base.load_history()
    frame = pd.read_csv(base.SOURCE)
    raw_all = base.current.vector_frozen_rows(frame)
    raw = [row for row in raw_all if 100 * float(row["risk_pct"])
           * float(row["target_r"]) + 1e-12 >= MIN_FIRST_TARGET_MOVE_PCT]
    enriched = []
    for row in raw:
        found = dict(row)
        found.update(valid_ob_features(row, candles[str(row["symbol"])]))
        enriched.append(found)

    periods = {
        "development": base.DEVELOPMENT,
        "validation": base.VALIDATION,
        "test": base.TEST,
        "confirmation": base.CONFIRMATION,
    }
    baseline = {name: base.metrics(portfolio_gate(base.period(enriched, years), None))
                for name, years in periods.items()}
    frontier = []
    eligible = []
    for rule in RULES:
        development = base.metrics(portfolio_gate(
            base.period(apply_rule(enriched, rule), base.DEVELOPMENT), None))
        passed = base.dominates(development, baseline["development"], .60)
        frontier.append({"rule": rule, "development": development,
                         "development_pass": passed})
        if passed:
            rank = (baseline["development"]["negative_months"]
                    - development["negative_months"],
                    development["minimum_month_pct"],
                    development["return_3pct_margin_pct"],
                    development["literal_target_hit_rate"], -development["dd_pct"])
            eligible.append((rank, rule, development))

    frozen = validation = test = confirmation = None
    status = "REJECTED_DEVELOPMENT"
    if eligible:
        _, rule, development = max(eligible, key=lambda item: item[0])
        frozen = {"rule": rule, "development": development,
                  "development_baseline": baseline["development"],
                  "eligible_rules": len(eligible)}
        validation = base.metrics(portfolio_gate(
            base.period(apply_rule(enriched, rule), base.VALIDATION), None))
        status = "REJECTED_VALIDATION"
        if base.dominates(validation, baseline["validation"], .60):
            test = base.metrics(portfolio_gate(
                base.period(apply_rule(enriched, rule), base.TEST), None))
            status = "REJECTED_TEST"
            if base.dominates(test, baseline["test"], .60):
                confirmation = base.metrics(portfolio_gate(
                    base.period(apply_rule(enriched, rule), base.CONFIRMATION), None))
                status = ("VALID_ORDER_BLOCK_RESEARCH_CANDIDATE"
                          if base.dominates(confirmation, baseline["confirmation"], .60)
                          else "REJECTED_CONFIRMATION")

    accepted = portfolio_gate(enriched, None)
    attribution = {str(year): factor_summary(base.period(accepted, (year,)))
                   for year in range(2018, 2027)}
    report = {
        "status": status, "ready_for_live_orders": False,
        "protocol": {
            "router": "exact robust_dynamic_entry_filtered material route",
            "development": list(base.DEVELOPMENT),
            "validation": list(base.VALIDATION), "test": list(base.TEST),
            "confirmation_only": list(base.CONFIRMATION),
            "order_block": "opposite candle plus >=0.75R displacement within 3 bars",
            "bos_tier": "displacement close also breaks the prior 8-bar extreme",
            "volume_tier": "BOS displacement volume >=1.2x trailing 20-bar median",
            "validity": "no later pre-entry close crosses the block far edge",
            "zone": "block overlaps entry-to-target or entry-to-stop path",
            "money": "3% current-equity margin, fixed 10x gross",
            "market_orders_only": True,
        },
        "baseline": baseline, "factor_attribution_by_year": attribution,
        "frontier": frontier, "frozen_before_validation": frozen,
        "validation": validation, "test": test, "confirmation": confirmation,
        "sources": {**sources, str(base.SOURCE): digest(base.SOURCE),
                    str(Path(base.__file__)): digest(Path(base.__file__)),
                    str(Path(__file__)): digest(Path(__file__))},
        "limitations": [
            "Order blocks are reproducible OHLCV definitions, not historical order-book depth.",
            "Dukascopy volume is a proxy and does not represent consolidated US equity volume.",
            "Previously published later years are attribution evidence rather than pristine holdouts.",
            "No historical filter may change runtime without fresh forward validation.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": status, "eligible": len(eligible),
                      "frozen": frozen, "validation": validation,
                      "test": test, "confirmation": confirmation},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
