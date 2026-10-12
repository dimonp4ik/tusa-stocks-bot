"""Chronological stock audit of structure-conditioned close-confirmed stops."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

import stock_stop_path_cross_era_oct12_audit as base
from src.backtest_integrity import simulate_exit
from src.stock_venue_router import MIN_FIRST_TARGET_MOVE_PCT
from stock_module_cluster_audit import portfolio_gate


OUT = base.ROOT / "stock_conditional_close_cross_era_oct12.json"
NY = ZoneInfo("America/New_York")
MASKS = (
    "stop_zone_swing",
    "aligned_stop_zone_swing",
    "safe_stop_zone_swing",
    "aligned_safe_stop_zone_swing",
    "clear_path",
)
POLICIES = (
    {"name": "close_1_backstop_1_5", "kind": "close", "backstop_r": 1.5},
    {"name": "close_1_backstop_2", "kind": "close", "backstop_r": 2.0},
    {"name": "hard_1_25", "kind": "hard", "stop_r": 1.25},
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cash_session_indices(candle: dict, entry_index: int) -> range:
    entry_day = pd.Timestamp(int(candle["time"][entry_index]), unit="s", tz="UTC").tz_convert(NY).date()
    last = entry_index
    for index in range(entry_index + 1, len(candle["time"])):
        stamp = pd.Timestamp(int(candle["time"][index]), unit="s", tz="UTC").tz_convert(NY)
        if stamp.date() != entry_day or stamp.hour >= 16:
            break
        last = index
    return range(entry_index, last + 1)


def resolve(row: dict, candle: dict, policy: dict) -> dict:
    lookup = {int(value): index for index, value in enumerate(candle["time"])}
    entry_time = int(float(row["entry_time"]))
    entry_index = lookup[entry_time]
    indices = cash_session_indices(candle, entry_index)
    entry = float(candle["open"][entry_index])
    direction = 1 if str(row["direction"]) == "LONG" else -1
    risk = entry * float(row["risk_pct"])
    target = entry + direction * risk * float(row["target_r"])
    if policy["kind"] == "close":
        stop = entry - direction * risk
        result = simulate_exit(
            candle, indices, direction=str(row["direction"]), entry=entry,
            sl=stop, tp1=target, tp2=target, atr=risk, tp1_fraction=1,
            trail=False, trail_mult=0, stop_on_close=True,
            backstop_r=float(policy["backstop_r"]), choose_trail=lambda *_: 0)
    else:
        stop = entry - direction * risk * float(policy["stop_r"])
        result = simulate_exit(
            candle, indices, direction=str(row["direction"]), entry=entry,
            sl=stop, tp1=target, tp2=target, atr=risk, tp1_fraction=1,
            trail=False, trail_mult=0, stop_on_close=False, backstop_r=1,
            choose_trail=lambda *_: 0)
    net_r = direction * (float(result.price) - entry) / risk
    return {
        "outcome": result.outcome,
        "net_r": net_r,
        "exit_time": int(candle["time"][result.bar]) + 900,
    }


def active_mask(row: dict, name: str) -> bool:
    stop_zone = bool(row["stop_zone_has_prior_swing"])
    aligned = not bool(row["both_indices_against"])
    safe = not (bool(row["adverse_failed_breakout"])
                or bool(row["opposing_order_block_proxy"])
                or bool(row["target_path_has_prior_swing"]))
    values = {
        "stop_zone_swing": stop_zone,
        "aligned_stop_zone_swing": aligned and stop_zone,
        "safe_stop_zone_swing": safe and stop_zone,
        "aligned_safe_stop_zone_swing": aligned and safe and stop_zone,
        "clear_path": safe,
    }
    return values[name]


def routed(rows: list[dict], candles: dict[str, dict], mask: str,
           policy: dict) -> list[dict]:
    output = []
    for row in rows:
        found = dict(row)
        if active_mask(row, mask):
            found.update(resolve(row, candles[str(row["symbol"])], policy))
        output.append(found)
    return output


def parity(rows: list[dict], candles: dict[str, dict]) -> dict:
    policy = {"kind": "hard", "stop_r": 1.0}
    mismatches = []
    for row in rows:
        value = resolve(row, candles[str(row["symbol"])], policy)
        if (value["outcome"] != row["outcome"]
                or abs(float(value["net_r"]) - float(row["net_r"])) > 1e-8
                or int(value["exit_time"]) != int(float(row["exit_time"]))):
            mismatches.append({
                "symbol": row["symbol"], "entry_time": int(row["entry_time"]),
                "expected": {key: row[key] for key in ("outcome", "net_r", "exit_time")},
                "actual": value,
            })
    return {"rows": len(rows), "mismatches": len(mismatches),
            "examples": mismatches[:10], "exact": not mismatches}


def dominates(candidate: dict, baseline: dict, retention: float) -> bool:
    return bool(
        candidate["n"] >= retention * baseline["n"]
        and candidate["trades_per_market_day"] >= max(
            1.5, .8 * baseline["trades_per_market_day"])
        and candidate["wr"] >= baseline["wr"]
        and candidate["literal_target_hit_rate"] >= baseline["literal_target_hit_rate"]
        and candidate["return_3pct_margin_pct"] >= baseline["return_3pct_margin_pct"]
        and candidate["dd_pct"] <= baseline["dd_pct"]
        and candidate["negative_months"] <= baseline["negative_months"]
        and candidate["minimum_month_pct"] >= baseline["minimum_month_pct"]
        and candidate["max_loss_streak"] <= baseline["max_loss_streak"])


def evaluate(rows: list[dict], years: tuple[int, ...]) -> dict:
    return base.metrics(portfolio_gate(base.period(rows, years), None))


def main() -> None:
    candles, sources = base.load_history()
    frame = pd.read_csv(base.SOURCE)
    raw_all = base.current.vector_frozen_rows(frame)
    raw = [row for row in raw_all if 100 * float(row["risk_pct"])
           * float(row["target_r"]) + 1e-12 >= MIN_FIRST_TARGET_MOVE_PCT]
    raw_cluster = base.Counter((int(float(row["entry_time"])), str(row["direction"]))
                               for row in raw)
    accepted = portfolio_gate(raw, None)
    accepted_cluster = base.Counter(
        (int(float(row["entry_time"])), str(row["direction"])) for row in accepted)
    enriched = [base.direct.causal_features(
        row, candles[str(row["symbol"])], raw_cluster, accepted_cluster)
        for row in raw]
    exact = parity(enriched, candles)
    if not exact["exact"]:
        raise ValueError(f"baseline resolver parity failed: {exact}")

    baseline = {
        "development": evaluate(enriched, base.DEVELOPMENT),
        "validation": evaluate(enriched, base.VALIDATION),
        "test": evaluate(enriched, base.TEST),
        "confirmation": evaluate(enriched, base.CONFIRMATION),
    }
    frontier = []
    eligible = []
    routed_cache = {}
    for mask, policy in itertools.product(MASKS, POLICIES):
        candidate = routed(enriched, candles, mask, policy)
        routed_cache[(mask, policy["name"])] = candidate
        score = evaluate(candidate, base.DEVELOPMENT)
        passed = dominates(score, baseline["development"], .60)
        frontier.append({"mask": mask, "policy": policy,
                         "development": score, "development_pass": passed})
        if passed:
            rank = (baseline["development"]["negative_months"] - score["negative_months"],
                    score["minimum_month_pct"], score["return_3pct_margin_pct"],
                    score["literal_target_hit_rate"], -score["dd_pct"])
            eligible.append((rank, mask, policy, score))

    frozen = validation = test = confirmation = None
    status = "REJECTED_DEVELOPMENT"
    if eligible:
        _, mask, policy, development = max(eligible, key=lambda item: item[0])
        candidate = routed_cache[(mask, policy["name"])]
        frozen = {"mask": mask, "policy": policy,
                  "development": development,
                  "development_baseline": baseline["development"],
                  "eligible_candidates": len(eligible)}
        validation = evaluate(candidate, base.VALIDATION)
        status = "REJECTED_VALIDATION"
        if dominates(validation, baseline["validation"], .60):
            test = evaluate(candidate, base.TEST)
            status = "REJECTED_TEST"
            if dominates(test, baseline["test"], .60):
                confirmation = evaluate(candidate, base.CONFIRMATION)
                status = ("CONDITIONAL_CLOSE_RESEARCH_CANDIDATE"
                          if dominates(confirmation, baseline["confirmation"], .60)
                          else "REJECTED_CONFIRMATION")

    report = {
        "status": status, "ready_for_live_orders": False,
        "protocol": {
            "router": "exact robust_dynamic_entry_filtered material route",
            "entry_and_exit": "market proxies",
            "development": list(base.DEVELOPMENT),
            "validation": list(base.VALIDATION), "test": list(base.TEST),
            "confirmation_only": list(base.CONFIRMATION),
            "default_stop": "hard 1R intrabar",
            "conditional_stop": "close beyond 1R with hard 1.5R/2R backstop",
            "same_bar_policy": "resting backstop before close stop before target",
            "money": "3% current-equity margin, fixed 10x gross",
        },
        "baseline_resolver_parity": exact,
        "mask_definitions": {
            "stop_zone_swing": "confirmed prior swing lies between entry and original stop",
            "aligned": "QQQ and SPY are not both moving against the trade",
            "safe": "no failed breakout, opposing block or prior swing before target",
            "clear_path": "safe without requiring a protective stop-side swing",
        },
        "baseline": baseline, "frontier": frontier,
        "frozen_before_validation": frozen,
        "validation": validation, "test": test, "confirmation": confirmation,
        "sources": {**sources, str(base.SOURCE): digest(base.SOURCE),
                    str(Path(base.__file__)): digest(Path(base.__file__)),
                    str(Path(__file__)): digest(Path(__file__))},
        "limitations": [
            "Cash-proxy candles are not X-Perp fills.",
            "Close-confirmed stops can realize a larger loss than the original hard stop.",
            "OHLC bars cannot reveal tick order when stop and target share a candle.",
            "No historical candidate may change runtime without fresh forward evidence.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": status, "parity": exact,
                      "eligible": len(eligible), "frozen": frozen,
                      "validation": validation, "test": test,
                      "confirmation": confirmation}, indent=2), flush=True)


if __name__ == "__main__":
    main()
