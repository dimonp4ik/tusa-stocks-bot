"""Cross-era causal stop-path audit for the exact frozen material stock router."""
from __future__ import annotations

import hashlib
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

import stock_current_router_cross_era_oct11_audit as current
import stock_stop_cause_oct11_audit as direct
from src.stock_venue_router import MIN_FIRST_TARGET_MOVE_PCT
from stock_module_cluster_audit import portfolio_gate


ROOT = Path("reports/audit_2026_09_08")
MANIFEST = ROOT / "stock_extended_history_manifest.json"
SOURCE = ROOT / "stock_session_family_market_context_2017_2026.csv"
OUT = ROOT / "stock_stop_path_cross_era_oct12.json"
ROWS_OUT = ROOT / "stock_stop_path_cross_era_oct12_rows.csv"
RULES = (
    "reject_adverse_failed_breakout",
    "reject_index_opposition",
    "reject_opposing_ob",
    "reject_target_path_swing",
    "require_stop_zone_swing",
    "require_supportive_sweep",
    "reject_liquidity_closer_to_stop",
    "reject_structure_conflict",
    "single_directional_signal_per_scan",
)
DEVELOPMENT = (2018, 2019, 2020)
VALIDATION = (2021, 2022)
TEST = (2023, 2024)
CONFIRMATION = (2025, 2026)
REQUIRED_SIGNED_PRICE_RETURN_PCT = 1.327709


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_history() -> tuple[dict[str, dict], dict[str, str]]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    caches = {}
    sources = {str(MANIFEST): digest(MANIFEST)}
    for symbol, item in sorted(manifest["symbols"].items()):
        path = Path(item["path"])
        raw = path.read_bytes()
        found = hashlib.sha256(raw).hexdigest()
        if found != item["sha256"]:
            raise ValueError(f"source hash mismatch: {symbol}")
        candle = pickle.loads(raw)
        lengths = {key: len(candle.get(key, []))
                   for key in ("time", "open", "high", "low", "close", "volume")}
        if len(set(lengths.values())) != 1 or not lengths["time"]:
            raise ValueError(f"bad candle lengths: {symbol} {lengths}")
        times = [int(value) for value in candle["time"]]
        if any(right <= left for left, right in zip(times, times[1:])):
            raise ValueError(f"non-increasing candle time: {symbol}")
        caches[symbol] = candle
        sources[str(path)] = found
    return caches, sources


def year_of(row: dict) -> int:
    return pd.Timestamp(float(row["entry_time"]), unit="s", tz="UTC").year


def period(rows: list[dict], years: tuple[int, ...]) -> list[dict]:
    allowed = set(years)
    return [row for row in rows if year_of(row) in allowed]


def metrics(rows: list[dict]) -> dict:
    if not rows:
        return {
            "n": 0, "wr": 0.0, "literal_target_hit_rate": 0.0,
            "return_3pct_margin_pct": 0.0, "dd_pct": 0.0,
            "negative_months": 0, "minimum_month_pct": 0.0,
            "max_loss_streak": 0, "trades_per_market_day": 0.0,
            "mean_signed_price_return_pct": 0.0,
            "projected_22_market_day_pct": 0.0,
        }
    ordered = sorted(rows, key=lambda row: (
        float(row["exit_time"]), float(row["entry_time"]), str(row["symbol"])))
    equity = peak = 1.0
    max_dd = 0.0
    streak = max_streak = wins = literal_hits = 0
    monthly_factors: dict[str, float] = defaultdict(lambda: 1.0)
    days = set()
    signed_returns = []
    for row in ordered:
        signed = float(row["risk_pct"]) * float(row["net_r"])
        signed_returns.append(signed)
        pnl_factor = 1 + .03 * 10 * signed
        equity *= pnl_factor
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak)
        won = float(row["net_r"]) > 0
        wins += int(won)
        literal_hits += int(str(row["outcome"]) == "TP2")
        streak = 0 if won else streak + 1
        max_streak = max(max_streak, streak)
        stamp = pd.Timestamp(float(row["entry_time"]), unit="s", tz="UTC")
        month = stamp.strftime("%Y-%m")
        monthly_factors[month] *= pnl_factor
        days.add(stamp.strftime("%Y-%m-%d"))
    monthly = {month: 100 * (factor - 1)
               for month, factor in sorted(monthly_factors.items())}
    trades_per_day = len(rows) / len(days)
    mean_signed = sum(signed_returns) / len(signed_returns)
    monthly_trade_count = 22 * trades_per_day
    projected = 100 * ((1 + .03 * 10 * mean_signed) ** monthly_trade_count - 1)
    return {
        "n": len(rows),
        "wr": round(100 * wins / len(rows), 4),
        "literal_target_hit_rate": round(100 * literal_hits / len(rows), 4),
        "return_3pct_margin_pct": round(100 * (equity - 1), 6),
        "dd_pct": round(100 * max_dd, 6),
        "negative_months": sum(value < 0 for value in monthly.values()),
        "minimum_month_pct": round(min(monthly.values()), 6),
        "median_month_pct": round(float(pd.Series(monthly.values()).median()), 6),
        "max_loss_streak": max_streak,
        "trades_per_market_day": round(trades_per_day, 4),
        "mean_signed_price_return_pct": round(100 * mean_signed, 6),
        "projected_22_market_day_pct": round(projected, 6),
        "months": len(monthly),
        "months_ge_30pct": sum(value >= 30 for value in monthly.values()),
    }


def linear_negative_months(rows: list[dict]) -> int:
    monthly: dict[str, float] = defaultdict(float)
    for row in rows:
        month = pd.Timestamp(float(row["entry_time"]), unit="s", tz="UTC").strftime("%Y-%m")
        monthly[month] += float(row["risk_pct"]) * float(row["net_r"])
    return sum(value < 0 for value in monthly.values())


def path_summary(rows: list[dict]) -> dict:
    stopped = [row for row in rows if str(row["outcome"]) == "SL"]
    fields = (
        "stop_candle_closed_back_inside", "stop_and_target_same_bar",
        "target_after_stop_same_session", "entry_after_stop_same_session",
        "target_after_stop_13h", "hard_reversal_proxy",
    )
    return {
        "stops": len(stopped),
        **{field: {
            "count": sum(bool(row.get(field)) for row in stopped),
            "pct": round(100 * sum(bool(row.get(field)) for row in stopped)
                         / len(stopped), 4) if stopped else 0.0,
        } for field in fields},
    }


def historical_post_stop_path(row: dict, candle: dict) -> tuple[dict, bool]:
    """Map replay's bar-close exit timestamp to its 15m candle start."""
    if str(row["outcome"]) != "SL":
        return {}, False
    times = {int(value) for value in candle["time"]}
    exit_time = int(float(row["exit_time"]))
    if exit_time - 900 not in times:
        raise ValueError(f"stop candle missing: {row['symbol']} {exit_time}")
    normalized = dict(row)
    normalized["exit_time"] = exit_time - 900
    return direct.post_stop_path(normalized, candle), True


def factor_summary(rows: list[dict]) -> dict:
    base = sum(str(row["outcome"]) == "SL" for row in rows) / len(rows) if rows else 0.0
    fields = (
        "target_path_has_prior_swing", "stop_zone_has_prior_swing",
        "liquidity_closer_to_stop", "opposing_order_block_proxy",
        "adverse_failed_breakout", "supportive_sweep", "both_indices_against",
        "index_divergence",
    )
    output = {}
    for field in fields:
        exposed = [row for row in rows if bool(row[field])]
        clear = [row for row in rows if not bool(row[field])]
        rate = (sum(str(row["outcome"]) == "SL" for row in exposed) / len(exposed)
                if exposed else 0.0)
        output[field] = {
            "exposed_n": len(exposed),
            "exposed_stop_rate_pct": round(100 * rate, 4),
            "clear_stop_rate_pct": round(
                100 * sum(str(row["outcome"]) == "SL" for row in clear) / len(clear), 4)
            if clear else 0.0,
            "stop_rate_lift": round(rate / base, 4) if base else None,
        }
    return output


def apply_rule(rows: list[dict], name: str) -> list[dict]:
    def keep(row: dict) -> bool:
        if name == "reject_adverse_failed_breakout":
            return not row["adverse_failed_breakout"]
        if name == "reject_index_opposition":
            return not row["both_indices_against"]
        if name == "reject_opposing_ob":
            return not row["opposing_order_block_proxy"]
        if name == "reject_target_path_swing":
            return not row["target_path_has_prior_swing"]
        if name == "require_stop_zone_swing":
            return row["stop_zone_has_prior_swing"]
        if name == "require_supportive_sweep":
            return row["supportive_sweep"]
        if name == "reject_liquidity_closer_to_stop":
            return not row["liquidity_closer_to_stop"]
        if name == "reject_structure_conflict":
            return not (row["adverse_failed_breakout"]
                        or row["opposing_order_block_proxy"]
                        or row["target_path_has_prior_swing"])
        if name == "single_directional_signal_per_scan":
            return int(row["raw_same_scan"]) <= 1
        raise KeyError(name)
    return [row for row in rows if keep(row)]


def dominates(candidate: dict, baseline: dict, minimum_retention: float) -> bool:
    return bool(
        candidate["n"] >= minimum_retention * baseline["n"]
        and candidate["trades_per_market_day"] >= max(
            1.5, .8 * baseline["trades_per_market_day"])
        and candidate["wr"] >= baseline["wr"]
        and candidate["literal_target_hit_rate"] >= baseline["literal_target_hit_rate"]
        and candidate["return_3pct_margin_pct"] >= baseline["return_3pct_margin_pct"]
        and candidate["dd_pct"] <= baseline["dd_pct"]
        and candidate["negative_months"] <= baseline["negative_months"]
        and candidate["minimum_month_pct"] >= baseline["minimum_month_pct"]
        and candidate["max_loss_streak"] <= baseline["max_loss_streak"]
    )


def candidate_audit(raw: list[dict]) -> dict:
    base = {"development": portfolio_gate(period(raw, DEVELOPMENT), None),
            "validation": portfolio_gate(period(raw, VALIDATION), None),
            "test": portfolio_gate(period(raw, TEST), None),
            "confirmation": portfolio_gate(period(raw, CONFIRMATION), None)}
    baseline = {name: metrics(rows) for name, rows in base.items()}
    frontier = []
    eligible = []
    for name in RULES:
        development = portfolio_gate(period(apply_rule(raw, name), DEVELOPMENT), None)
        score = metrics(development)
        passed = dominates(score, baseline["development"], .60)
        frontier.append({"rule": name, "development": score,
                         "development_pass": passed})
        if passed:
            rank = (baseline["development"]["negative_months"] - score["negative_months"],
                    score["minimum_month_pct"], score["return_3pct_margin_pct"],
                    score["literal_target_hit_rate"], -score["dd_pct"])
            eligible.append((rank, name, score))
    frozen = validation = test = confirmation = None
    status = "REJECTED_DEVELOPMENT"
    if eligible:
        _, name, development = max(eligible, key=lambda item: item[0])
        frozen = {"rule": name, "development": development,
                  "development_baseline": baseline["development"],
                  "eligible_rules": len(eligible)}
        validation = metrics(portfolio_gate(
            period(apply_rule(raw, name), VALIDATION), None))
        status = "REJECTED_VALIDATION"
        if dominates(validation, baseline["validation"], .60):
            test = metrics(portfolio_gate(period(apply_rule(raw, name), TEST), None))
            status = "REJECTED_TEST"
            if dominates(test, baseline["test"], .60):
                confirmation = metrics(portfolio_gate(
                    period(apply_rule(raw, name), CONFIRMATION), None))
                status = ("STRUCTURE_FILTER_RESEARCH_CANDIDATE"
                          if dominates(confirmation, baseline["confirmation"], .60)
                          else "REJECTED_CONFIRMATION")
    return {
        "status": status,
        "periods": {"development": list(DEVELOPMENT),
                    "validation": list(VALIDATION), "test": list(TEST),
                    "confirmation_only": list(CONFIRMATION)},
        "baseline": baseline,
        "frontier": frontier,
        "frozen_before_validation": frozen,
        "validation": validation, "test": test, "confirmation": confirmation,
    }


def cluster_summary(rows: list[dict]) -> dict:
    output = {}
    for size in sorted({int(row["accepted_same_scan"]) for row in rows}):
        part = [row for row in rows if int(row["accepted_same_scan"]) == size]
        output[str(size)] = metrics(part)
    return output


def main() -> None:
    caches, sources = load_history()
    frame = pd.read_csv(SOURCE)
    raw_all = current.vector_frozen_rows(frame)
    raw = [row for row in raw_all if 100 * float(row["risk_pct"])
           * float(row["target_r"]) + 1e-12 >= MIN_FIRST_TARGET_MOVE_PCT]
    accepted = portfolio_gate(raw, None)
    raw_cluster = Counter((int(float(row["entry_time"])), str(row["direction"]))
                          for row in raw)
    accepted_cluster = Counter((int(float(row["entry_time"])), str(row["direction"]))
                               for row in accepted)
    enriched = [direct.causal_features(row, caches[str(row["symbol"])],
                                       raw_cluster, accepted_cluster)
                for row in raw]
    by_key = {(str(row["symbol"]), int(float(row["entry_time"]))): row
              for row in enriched}
    accepted_enriched = [by_key[(str(row["symbol"]), int(float(row["entry_time"])))]
                         for row in accepted]
    mapped_bar_close_exit_timestamps = 0
    for row in accepted_enriched:
        path, normalized = historical_post_stop_path(
            row, caches[str(row["symbol"])])
        row.update(path)
        mapped_bar_close_exit_timestamps += int(normalized)

    exact_baseline = current.metrics(accepted)
    rebuilt_baseline = metrics(accepted_enriched)
    parity = {
        "n": rebuilt_baseline["n"] == exact_baseline["n"],
        "wr": abs(rebuilt_baseline["wr"] - exact_baseline["wr"]) < 1e-9,
        "return": abs(rebuilt_baseline["return_3pct_margin_pct"]
                      - exact_baseline["return_pct"]) < 1e-9,
        "dd": abs(rebuilt_baseline["dd_pct"] - exact_baseline["dd_pct"]) < 1e-9,
        "negative_months": (linear_negative_months(accepted_enriched)
                            == exact_baseline["negative_months"]),
    }
    if not all(parity.values()):
        raise ValueError(f"exact-router baseline parity failed: {parity}")

    attribution = {}
    paths = {}
    for year in range(2018, 2027):
        part = period(accepted_enriched, (year,))
        attribution[str(year)] = factor_summary(part)
        paths[str(year)] = path_summary(part)
    report = {
        "status": "DIAGNOSTIC_COMPLETE_NO_RUNTIME_CHANGE",
        "ready_for_live_orders": False,
        "protocol": {
            "router": "exact robust_dynamic_entry_filtered material route",
            "entry": "next 15m cash-proxy bar after closed signal",
            "material_target_floor_pct": MIN_FIRST_TARGET_MOVE_PCT,
            "causal_features": "all structure and index fields known before entry",
            "post_stop_use": "diagnostic only; never an entry feature",
            "money": "3% current-equity margin, fixed 10x gross",
            "market_orders_only": True,
        },
        "baseline_parity": parity,
        "mapped_bar_close_exit_timestamps": mapped_bar_close_exit_timestamps,
        "baseline": rebuilt_baseline,
        "stop_path_by_year": paths,
        "factor_attribution_by_year": attribution,
        "cluster_by_accepted_same_scan": cluster_summary(accepted_enriched),
        "candidate_filter_audit": candidate_audit(enriched),
        "economic_objective": {
            "required_constant_signed_price_return_per_trade_pct":
                REQUIRED_SIGNED_PRICE_RETURN_PCT,
            "actual_mean_signed_price_return_per_trade_pct":
                rebuilt_baseline["mean_signed_price_return_pct"],
            "actual_projected_22_market_day_pct":
                rebuilt_baseline["projected_22_market_day_pct"],
            "passes_30pct_monthly_gate": bool(
                rebuilt_baseline["projected_22_market_day_pct"] >= 30
                and rebuilt_baseline["negative_months"] == 0
                and rebuilt_baseline["trades_per_market_day"] >= 3),
        },
        "sources": {
            str(SOURCE): digest(SOURCE), str(MANIFEST): digest(MANIFEST),
            str(Path(current.__file__)): digest(Path(current.__file__)),
            str(Path(direct.__file__)): digest(Path(direct.__file__)),
            str(Path(__file__)): digest(Path(__file__)), **sources,
        },
        "limitations": [
            "Cash-proxy bars diagnose cross-era structure but are not X-Perp fills.",
            "Swing and order-block fields are fixed OHLCV proxies, not historical book depth.",
            "Post-stop recovery cannot be known at entry and is not used to select filters.",
            "Previously published later years are diagnostic evidence rather than untouched holdouts.",
        ],
    }
    pd.DataFrame(accepted_enriched).to_csv(ROWS_OUT, index=False)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "parity": parity,
        "baseline": rebuilt_baseline,
        "economic_objective": report["economic_objective"],
        "candidate": report["candidate_filter_audit"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
