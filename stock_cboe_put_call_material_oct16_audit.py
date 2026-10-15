"""Causal material stock routes from prior-session Cboe put/call flow."""
from __future__ import annotations

import hashlib
import itertools
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import stock_15m_multisession_regime_oct6_audit as base
from src.research_profit_gate import economic_gate, minimum_binary_hit_rate_pct


ROOT = Path("reports/audit_2026_09_08")
FLOW_ROOT = ROOT / "stock_cboe_put_call_oct16"
OUT = ROOT / "stock_cboe_put_call_material_oct16.json"
EXITS = (
    {"name": "balanced", "target": 1.0, "stop": .75, "sessions": 1},
    {"name": "swing", "target": 1.5, "stop": .75, "sessions": 2},
    {"name": "runner", "target": 2.0, "stop": 1.0, "sessions": 3},
)
FAMILIES = (
    "equity_fear_reversal", "index_hedge_riskoff",
    "equity_complacency_breakdown", "hedge_unwind_momentum",
    "vix_call_fear", "equity_call_chase",
)
PROFILES = ("broad", "strict")
DEV_YEARS = (2020, 2021)
VALIDATION_YEARS = (2022, 2023)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_flow() -> tuple[pd.DataFrame, dict, dict[str, str]]:
    manifest_path = FLOW_ROOT / "manifest.json"
    manifest_raw = manifest_path.read_bytes()
    manifest = json.loads(manifest_raw)
    data_path = Path(manifest["data_path"])
    found = digest(data_path)
    if found != manifest["data_sha256"]:
        raise ValueError("Cboe put/call normalized source hash mismatch")
    frame = pd.DataFrame(pickle.loads(data_path.read_bytes()))
    frame["date"] = pd.to_datetime(frame.date, utc=True)
    frame["date_epoch"] = frame.date.map(lambda value: int(value.timestamp()))
    frame = frame.sort_values("date_epoch")
    if frame.date_epoch.duplicated().any() or not frame.date_epoch.is_monotonic_increasing:
        raise ValueError("invalid Cboe put/call chronology")
    return frame, manifest, {
        str(manifest_path): hashlib.sha256(manifest_raw).hexdigest(),
        str(data_path): found,
    }


def flow_features(frame: pd.DataFrame) -> pd.DataFrame:
    value = frame.copy()
    ratio_names = ("total_pc", "index_pc", "etp_pc", "equity_pc", "vix_pc", "spx_pc")
    for name in ratio_names:
        # A 63-session baseline preserves a complete 2020 development year;
        # the official source begins only in October 2019.
        mean = value[name].rolling(63, min_periods=40).mean()
        std = value[name].rolling(63, min_periods=40).std().replace(0, np.nan)
        value[f"{name}_z63"] = (value[name]-mean)/std
        value[f"{name}_delta5"] = value[name]-value[name].shift(5)
        value[f"{name}_delta20"] = value[name]-value[name].shift(20)
    value["index_equity_spread"] = value.index_pc-value.equity_pc
    spread_mean = value.index_equity_spread.rolling(63, min_periods=40).mean()
    spread_std = value.index_equity_spread.rolling(63, min_periods=40).std().replace(0, np.nan)
    value["index_equity_spread_z"] = (value.index_equity_spread-spread_mean)/spread_std
    total_volume = value.total_volume_total.replace(0, np.nan)
    log_volume = np.log(total_volume)
    volume_mean = log_volume.rolling(63, min_periods=40).mean()
    volume_std = log_volume.rolling(63, min_periods=40).std().replace(0, np.nan)
    value["option_volume_z63"] = (log_volume-volume_mean)/volume_std
    value["option_volume_change5"] = total_volume/total_volume.shift(5)-1
    needed = [
        *[f"{name}_z63" for name in ratio_names],
        *[f"{name}_delta5" for name in ratio_names],
        "index_equity_spread_z", "option_volume_z63", "option_volume_change5",
    ]
    return value.replace([np.inf, -np.inf], np.nan).dropna(subset=needed)


def attach_flow(features: pd.DataFrame, flow: pd.DataFrame) -> pd.DataFrame:
    value = features.copy()
    value["session_dt"] = pd.to_datetime(value.session_date, utc=True)
    value["session_epoch"] = value.session_dt.map(lambda x: int(x.timestamp())).astype(np.int64)
    joined = pd.merge_asof(
        value.sort_values("session_epoch"), flow.sort_values("date_epoch"),
        left_on="session_epoch", right_on="date_epoch", direction="backward",
        allow_exact_matches=False, tolerance=7*86400)
    joined = joined.dropna(subset=["date_epoch"])
    if (joined.date_epoch >= joined.session_epoch).any():
        raise ValueError("same-session or future Cboe put/call row attached")
    return joined


def directed(frame: pd.DataFrame, direction: int) -> pd.DataFrame:
    found = frame.copy()
    found["direction_n"] = np.int8(direction)
    for name in ("ret5", "ret20", "intraday", "last_hour", "cross_z5",
                 "qqq_ret20", "qqq_intraday"):
        found[f"{name}_dir"] = found[name]*direction
    found["breadth_dir"] = (found.breadth_intra if direction > 0
                            else 1-found.breadth_intra)
    return found


def entries(frame: pd.DataFrame, family: str, profile: str) -> pd.DataFrame:
    strict = profile == "strict"
    if family == "equity_fear_reversal":
        found = directed(frame, 1)
        mask = (found.equity_pc_z63.ge(1.5 if strict else .8)
                & found.total_pc_z63.ge(1.0 if strict else .5)
                & found.qqq_intraday_dir.ge(.004 if strict else 0)
                & found.last_hour_dir.ge(.002 if strict else 0)
                & found.breadth_dir.ge(.625 if strict else .5))
        score = found.equity_pc_z63+found.total_pc_z63+100*found.last_hour_dir
    elif family == "index_hedge_riskoff":
        found = directed(frame, -1)
        mask = (found.index_pc_z63.ge(1.5 if strict else .8)
                & found.index_equity_spread_z.ge(1.0 if strict else .4)
                & found.qqq_intraday_dir.ge(.004 if strict else 0)
                & found.last_hour_dir.ge(.002 if strict else 0)
                & found.breadth_dir.ge(.625 if strict else .5))
        score = found.index_pc_z63+found.index_equity_spread_z+100*found.last_hour_dir
    elif family == "equity_complacency_breakdown":
        found = directed(frame, -1)
        mask = (found.equity_pc_z63.le(-1.5 if strict else -.8)
                & found.total_pc_z63.le(-1.0 if strict else -.4)
                & found.qqq_intraday_dir.ge(.004 if strict else 0)
                & found.last_hour_dir.ge(.002 if strict else 0))
        score = -found.equity_pc_z63-found.total_pc_z63+100*found.last_hour_dir
    elif family == "hedge_unwind_momentum":
        found = directed(frame, 1)
        mask = (found.total_pc_delta5.le(-.25 if strict else -.12)
                & found.index_pc_delta5.le(-.3 if strict else -.12)
                & found.qqq_ret20_dir.ge(.03 if strict else .01)
                & found.last_hour_dir.ge(.002 if strict else 0)
                & found.breadth_dir.ge(.625 if strict else .5))
        score = -found.total_pc_delta5-found.index_pc_delta5+found.qqq_ret20_dir
    elif family == "vix_call_fear":
        found = directed(frame, -1)
        mask = (found.vix_pc_z63.le(-1.5 if strict else -.8)
                & found.index_pc_z63.ge(.75 if strict else 0)
                & found.qqq_intraday_dir.ge(.004 if strict else 0)
                & found.last_hour_dir.ge(.002 if strict else 0))
        score = -found.vix_pc_z63+found.index_pc_z63+100*found.last_hour_dir
    elif family == "equity_call_chase":
        found = directed(frame, 1)
        mask = (found.equity_pc_z63.le(-1.5 if strict else -.8)
                & found.qqq_ret20_dir.ge(.04 if strict else .02)
                & found.qqq_intraday_dir.ge(.004 if strict else 0)
                & found.last_hour_dir.ge(.002 if strict else 0)
                & found.breadth_dir.ge(.625 if strict else .5))
        score = -found.equity_pc_z63+found.qqq_ret20_dir+100*found.last_hour_dir
    else:
        raise ValueError(family)
    output = found.loc[mask].copy()
    output["family"], output["profile"] = family, profile
    output["score"] = score.loc[mask].astype(float)
    return output


def accepted_rows(frame: pd.DataFrame) -> list[dict]:
    return base.portfolio_gate(
        frame.to_dict("records"), None,
        rank_key=lambda row: (-float(row["score"]), str(row["symbol"])))


def evaluate(frame: pd.DataFrame, features: pd.DataFrame, years: tuple[int, ...]) -> dict:
    selected = frame[frame.year.isin(years)].copy()
    market_days = features[features.year.isin(years)].session_date.nunique()
    result = base.account_metrics(selected, market_days)
    rows = accepted_rows(selected)
    hits = sum(int(row["outcome_code"]) == 1 for row in rows)
    result["literal_target_hits"] = hits
    result["literal_target_hit_rate"] = round(100*hits/len(rows), 4) if rows else 0.0
    months = list(result.get("monthly_return_pct", {}).values())
    result["minimum_month_pct"] = round(min(months), 4) if months else 0.0
    result["median_month_pct"] = round(float(np.median(months)), 4) if months else 0.0
    result["economic_gate"] = (economic_gate(
        [float(row["signed_return"]) for row in rows],
        float(result["trades_per_market_day"])) if rows else {
            "n": 0, "observed_trades_per_day": 0.0,
            "projected_monthly_return_pct": 0.0,
            "frequency_pass": False, "return_pass": False, "passes": False})
    return result


def development_ok(value: dict, annual: dict) -> bool:
    return bool(value["n"] >= 150 and value["return_pct"] > 0
                and value["dd_pct"] <= 12
                and all(item["return_pct"] > 0 for item in annual.values())
                and value["trades_per_market_day"] >= .3)


def validation_ok(value: dict, annual: dict) -> bool:
    return bool(value["n"] >= 100 and value["return_pct"] > 0
                and value["dd_pct"] <= 8
                and all(item["return_pct"] > 0 for item in annual.values())
                and value["economic_gate"]["passes"])


def hidden_ok(value: dict) -> bool:
    return bool(value["n"] >= 100 and value["return_pct"] > 0
                and value["dd_pct"] <= 5 and value["negative_months"] == 0
                and 75 <= value["literal_target_hit_rate"] <= 95
                and value["economic_gate"]["passes"])


def public(item: dict | None) -> dict | None:
    return ({key: value for key, value in item.items() if key != "_rows"}
            if item else None)


def required_hit_rate(target: float, stop: float) -> float | None:
    value = minimum_binary_hit_rate_pct(target, stop)
    return round(value, 6) if value is not None else None


def main() -> None:
    candles, price_sources = base.load_history()
    raw_features, metadata = base.build_features(candles)
    raw_flow, manifest, flow_sources = load_flow()
    flow = flow_features(raw_flow)
    features = attach_flow(raw_features, flow)
    frontier, survivors, validation_failures = [], [], []
    for family, profile, exit_spec in itertools.product(FAMILIES, PROFILES, EXITS):
        proposed = entries(features, family, profile)
        if proposed.empty:
            continue
        resolved = base.add_outcomes(
            proposed, candles, metadata, exit_spec["target"],
            exit_spec["stop"], exit_spec["sessions"])
        resolved["module"] = f"stock_cboe_pc_{family}"
        development = evaluate(resolved, features, DEV_YEARS)
        annual_dev = {str(year): evaluate(resolved, features, (year,))
                      for year in DEV_YEARS}
        item = {
            "route_id": f"{family}_{profile}_{exit_spec['name']}",
            "family": family, "profile": profile, "exit": exit_spec,
            "development_2020_2021": development,
            "development_years": annual_dev,
        }
        frontier.append(item)
        if not development_ok(development, annual_dev):
            continue
        validation = evaluate(resolved, features, VALIDATION_YEARS)
        annual_validation = {str(year): evaluate(resolved, features, (year,))
                             for year in VALIDATION_YEARS}
        item["validation_2022_2023"] = validation
        item["validation_years"] = annual_validation
        if not validation_ok(validation, annual_validation):
            validation_failures.append(item)
            continue
        item["score"] = round(
            validation["economic_gate"]["projected_monthly_return_pct"]
            -2*validation["dd_pct"], 8)
        item["_rows"] = resolved
        survivors.append(item)
    survivors.sort(key=lambda item: item["score"], reverse=True)
    frozen = survivors[0] if survivors else None
    test_2024 = confirmation_2025 = reveal_2026 = None
    if frozen:
        test_2024 = evaluate(frozen["_rows"], features, (2024,))
        if hidden_ok(test_2024):
            confirmation_2025 = evaluate(frozen["_rows"], features, (2025,))
        if confirmation_2025 and hidden_ok(confirmation_2025):
            reveal_2026 = evaluate(frozen["_rows"], features, (2026,))
    frontier.sort(key=lambda item: (
        item["development_2020_2021"]["economic_gate"]["projected_monthly_return_pct"],
        item["development_2020_2021"]["literal_target_hit_rate"],
        item["development_2020_2021"]["return_pct"]), reverse=True)
    validation_failures.sort(key=lambda item: (
        item["validation_2022_2023"]["economic_gate"]["projected_monthly_return_pct"],
        item["validation_2022_2023"]["return_pct"]), reverse=True)
    status = ("RESEARCH_CANDIDATE_AWAITING_XPERP_FORWARD"
              if reveal_2026 and hidden_ok(reveal_2026) else "REJECTED")
    report = {
        "status": status, "ready_for_live_orders": False,
        "protocol": {
            "source": "official Cboe daily put/call ratios, volume and open interest",
            "causal_availability": manifest["causal_availability"],
            "families": list(FAMILIES), "profiles": list(PROFILES),
            "exits": list(EXITS), "entry": "next regular-session market bar",
            "development": "2020-2021", "validation": "2022-2023",
            "test": "2024", "confirmation": "2025", "untouched_reveal": "2026",
            "money": "3% current-equity margin, fixed 10x leverage",
            "orders": "market entries only", "fees": "excluded by user instruction",
        },
        "source_quality": manifest["quality"],
        "rows": {"base_features": len(raw_features), "put_call_days": len(flow),
                 "enriched": len(features)},
        "minimum_ideal_hit_rate_for_30pct_month_at_3_trades_day": {
            spec["name"]: required_hit_rate(spec["target"], spec["stop"])
            for spec in EXITS},
        "qualified_before_2024": len(survivors),
        "frozen_before_2024": public(frozen), "test_2024": test_2024,
        "confirmation_2025": confirmation_2025, "untouched_2026": reveal_2026,
        "validation_failures": [public(item) for item in validation_failures[:12]],
        "development_frontier": [public(item) for item in frontier[:18]],
        "sources": {**price_sources, **flow_sources,
                    str(Path(__file__)): digest(Path(__file__))},
        "limitations": [
            "Cboe end-of-day flow gates only later sessions and cannot time intraday entries alone.",
            "Put/call volume does not distinguish opening from closing option trades.",
            "Dukascopy cash CFDs establish structural history, not X-Perp fills.",
            "Any historical survivor still requires untouched direct X-Perp forward evidence.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": status, "rows": report["rows"],
        "qualified_before_2024": len(survivors),
        "frozen_before_2024": public(frozen), "test_2024": test_2024,
        "confirmation_2025": confirmation_2025, "untouched_2026": reveal_2026,
        "best_development": report["development_frontier"][:4],
        "best_validation_failures": report["validation_failures"][:3],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
