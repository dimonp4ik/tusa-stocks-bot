"""Causal stock routes from prior official Cboe implied correlation."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

import stock_15m_multisession_regime_oct6_audit as base
from src.research_profit_gate import economic_gate, minimum_binary_hit_rate_pct


ROOT = Path("reports/audit_2026_09_08")
COR_ROOT = ROOT / "stock_cboe_correlation_oct26"
OUT = ROOT / "stock_cboe_correlation_material_oct26.json"
EXITS = (
    {"name": "balanced", "target": 1.0, "stop": .75, "sessions": 1},
    {"name": "swing", "target": 1.5, "stop": .75, "sessions": 2},
    {"name": "runner", "target": 2.0, "stop": 1.0, "sessions": 3},
)
FAMILIES = (
    "systemic_shock_continuation", "correlation_relief_reversal",
    "low_correlation_stock_trend", "curve_inversion_warning",
    "synchronized_breakout",
)
PROFILES = ("broad", "strict")
DEVELOPMENT = (2017, 2018, 2019, 2020, 2021)
VALIDATION = (2022, 2023)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_correlation() -> tuple[pd.DataFrame, dict, dict[str, str]]:
    manifest_path = COR_ROOT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data_path = Path(manifest["data_path"])
    if digest(data_path) != manifest["data_sha256"]:
        raise ValueError("correlation data hash mismatch")
    frame = pd.read_csv(data_path).sort_values("date_epoch")
    if frame.date_epoch.duplicated().any() or not frame.date_epoch.is_monotonic_increasing:
        raise ValueError("invalid correlation chronology")
    sources = {str(manifest_path): digest(manifest_path),
               str(data_path): digest(data_path)}
    for item in manifest["sources"].values():
        path = Path(item["path"])
        if digest(path) != item["sha256"]:
            raise ValueError(f"raw correlation hash mismatch: {path}")
        sources[str(path)] = digest(path)
    return frame, manifest, sources


def correlation_features(frame: pd.DataFrame) -> pd.DataFrame:
    value = frame.sort_values("date_epoch").copy()
    value["correlation_date"] = pd.to_datetime(value.date_epoch, unit="s", utc=True)
    needed = []
    for name in ("cor1m_close", "cor3m_close", "cor6m_close"):
        value[f"{name}_change5"] = value[name].pct_change(5)
        mean = value[name].rolling(252, min_periods=160).mean()
        std = value[name].rolling(252, min_periods=160).std().replace(0, np.nan)
        value[f"{name}_z252"] = (value[name]-mean)/std
        value[f"{name}_rank252"] = value[name].rolling(
            252, min_periods=160).rank(pct=True)
        needed += [f"{name}_change5", f"{name}_z252", f"{name}_rank252"]
    value["cor_curve_1m_6m"] = value.cor1m_close-value.cor6m_close
    mean = value.cor_curve_1m_6m.rolling(252, min_periods=160).mean()
    std = value.cor_curve_1m_6m.rolling(252, min_periods=160).std().replace(0, np.nan)
    value["cor_curve_z252"] = (value.cor_curve_1m_6m-mean)/std
    needed += ["cor_curve_1m_6m", "cor_curve_z252"]
    return value.replace([np.inf, -np.inf], np.nan).dropna(subset=needed)[
        ["date_epoch", "correlation_date", "cor1m_close", "cor3m_close",
         "cor6m_close", *needed]]


def attach(features: pd.DataFrame, correlation: pd.DataFrame) -> pd.DataFrame:
    value = features.copy()
    value["session_dt"] = pd.to_datetime(value.session_date, utc=True)
    value["session_epoch"] = value.session_dt.map(
        lambda stamp: int(stamp.timestamp())).astype(np.int64)
    joined = pd.merge_asof(
        value.sort_values("session_epoch"), correlation.sort_values("date_epoch"),
        left_on="session_epoch", right_on="date_epoch", direction="backward",
        allow_exact_matches=False, tolerance=7*86400,
    ).dropna(subset=["date_epoch"])
    if (joined.date_epoch >= joined.session_epoch).any():
        raise ValueError("same-session or future correlation close attached")
    joined["correlation_lag_days"] = (
        (joined.session_epoch-joined.date_epoch)/86400)
    return joined


def directed(frame: pd.DataFrame, direction: pd.Series) -> pd.DataFrame:
    found = frame.copy()
    found["direction_n"] = direction.astype(np.int8)
    for name in ("ret5", "ret20", "intraday", "last_hour", "cross_z5",
                 "qqq_ret20", "qqq_intraday", "qqq_last_hour"):
        found[f"{name}_dir"] = found[name]*found.direction_n
    found["breadth_dir"] = np.where(
        found.direction_n > 0, found.breadth_intra, 1-found.breadth_intra)
    return found


def entries(frame: pd.DataFrame, family: str, profile: str) -> pd.DataFrame:
    strict = profile == "strict"
    if family == "systemic_shock_continuation":
        found = directed(frame, pd.Series(-1, index=frame.index))
        mask = (found.cor1m_close_z252.ge(1.5 if strict else .75)
                & found.cor_curve_z252.ge(.75 if strict else 0)
                & found.qqq_intraday_dir.ge(.004 if strict else 0)
                & found.last_hour_dir.ge(.002 if strict else 0)
                & found.breadth_dir.ge(.625 if strict else .50))
        score = (found.cor1m_close_z252+found.cor_curve_z252
                 +found.qqq_intraday_dir+found.last_hour_dir)
    elif family == "correlation_relief_reversal":
        found = directed(frame, pd.Series(1, index=frame.index))
        mask = (found.cor1m_close_z252.ge(1.0 if strict else .5)
                & found.cor1m_close_change5.le(-.15 if strict else -.07)
                & found.qqq_last_hour_dir.ge(.002 if strict else 0)
                & found.breadth_dir.ge(.625 if strict else .50))
        score = (found.cor1m_close_z252-found.cor1m_close_change5
                 +found.qqq_last_hour_dir)
    elif family == "low_correlation_stock_trend":
        direction = np.sign(frame.ret20).replace(0, np.nan)
        found = directed(frame, direction.fillna(0))
        mask = (found.direction_n.ne(0)
                & found.cor1m_close_rank252.le(.20 if strict else .35)
                & found.ret20_dir.ge(.03 if strict else .01)
                & found.last_hour_dir.ge(.002 if strict else 0)
                & found.cross_z5_dir.ge(.5 if strict else 0))
        score = (-found.cor1m_close_rank252
                 +found.ret20_dir+found.last_hour_dir)
    elif family == "curve_inversion_warning":
        found = directed(frame, pd.Series(-1, index=frame.index))
        mask = (found.cor_curve_z252.ge(1.5 if strict else .75)
                & found.cor1m_close_change5.ge(.10 if strict else .03)
                & found.qqq_intraday_dir.ge(.003 if strict else 0)
                & found.last_hour_dir.ge(.002 if strict else 0))
        score = (found.cor_curve_z252+found.cor1m_close_change5
                 +found.qqq_intraday_dir+found.last_hour_dir)
    elif family == "synchronized_breakout":
        direction = np.sign(frame.qqq_intraday).replace(0, np.nan)
        found = directed(frame, direction.fillna(0))
        mask = (found.direction_n.ne(0)
                & found.cor1m_close_rank252.ge(.80 if strict else .65)
                & found.qqq_intraday_dir.ge(.004 if strict else .002)
                & found.last_hour_dir.ge(.002 if strict else 0)
                & found.breadth_dir.ge(.625 if strict else .50))
        score = (found.cor1m_close_rank252
                 +found.qqq_intraday_dir+found.last_hour_dir)
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


def evaluate(frame: pd.DataFrame, features: pd.DataFrame,
             years: tuple[int, ...]) -> dict:
    selected = frame[frame.year.isin(years)].copy()
    days = features[features.year.isin(years)].session_date.nunique()
    value = base.account_metrics(selected, days)
    rows = accepted_rows(selected)
    hits = sum(int(row["outcome_code"]) == 1 for row in rows)
    value["literal_target_hits"] = hits
    value["literal_target_hit_rate"] = round(
        100*hits/len(rows), 4) if rows else 0.0
    values = list(value.get("monthly_return_pct", {}).values())
    value["minimum_month_pct"] = round(min(values), 4) if values else 0.0
    value["median_month_pct"] = round(float(np.median(values)), 4) if values else 0.0
    value["economic_gate"] = (economic_gate(
        [float(row["signed_return"]) for row in rows],
        float(value["trades_per_market_day"])) if rows else {
            "n": 0, "observed_trades_per_day": 0.0,
            "projected_monthly_return_pct": 0.0,
            "frequency_pass": False, "return_pass": False, "passes": False})
    return value


def development_ok(value: dict, annual: dict) -> bool:
    return bool(value["n"] >= 250 and value["return_pct"] > 0
                and value["dd_pct"] <= 12
                and sum(row["return_pct"] > 0 for row in annual.values()) >= 4
                and value["trades_per_market_day"] >= .5)


def validation_ok(value: dict, annual: dict) -> bool:
    return bool(value["n"] >= 100 and value["return_pct"] > 0
                and value["dd_pct"] <= 8
                and all(row["return_pct"] > 0 for row in annual.values())
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
    raw_correlation, manifest, correlation_sources = load_correlation()
    correlation = correlation_features(raw_correlation)
    features = attach(raw_features, correlation)
    frontier, survivors, validation_failures, resolved_routes = [], [], [], {}
    for family, profile, exit_spec in itertools.product(
            FAMILIES, PROFILES, EXITS):
        proposed = entries(features, family, profile)
        if proposed.empty:
            continue
        resolved = base.add_outcomes(
            proposed, candles, metadata, exit_spec["target"],
            exit_spec["stop"], exit_spec["sessions"])
        resolved["module"] = f"stock_cboe_correlation_{family}"
        route_id = f"{family}_{profile}_{exit_spec['name']}"
        resolved_routes[route_id] = resolved
        development = evaluate(resolved, features, DEVELOPMENT)
        annual_dev = {str(year): evaluate(resolved, features, (year,))
                      for year in DEVELOPMENT}
        item = {
            "route_id": route_id,
            "family": family, "profile": profile, "exit": exit_spec,
            "development_2017_2021": development,
            "development_years": annual_dev,
        }
        frontier.append(item)
        if not development_ok(development, annual_dev):
            continue
        validation = evaluate(resolved, features, VALIDATION)
        annual_validation = {str(year): evaluate(
            resolved, features, (year,)) for year in VALIDATION}
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
        item["development_2017_2021"]["economic_gate"][
            "projected_monthly_return_pct"],
        item["development_2017_2021"]["return_pct"]), reverse=True)
    validation_failures.sort(key=lambda item: (
        item["validation_2022_2023"]["economic_gate"][
            "projected_monthly_return_pct"],
        item["validation_2022_2023"]["return_pct"]), reverse=True)
    status = ("RESEARCH_CANDIDATE_AWAITING_XPERP_FORWARD"
              if reveal_2026 and hidden_ok(reveal_2026) else "REJECTED")
    union_ids = [f"{family}_broad_runner" for family in FAMILIES]
    union = pd.concat([resolved_routes[name] for name in union_ids],
                      ignore_index=True)
    union = union.sort_values(
        ["entry_time", "symbol", "score"], ascending=[True, True, False]
    ).drop_duplicates(["entry_time", "symbol"], keep="first")
    union_dev = evaluate(union, features, DEVELOPMENT)
    union_validation = evaluate(union, features, VALIDATION)
    union_test_2024 = None
    if (union_dev["economic_gate"]["passes"]
            and union_validation["economic_gate"]["passes"]):
        union_test_2024 = evaluate(union, features, (2024,))
    report = {
        "status": status, "ready_for_live_orders": False,
        "protocol": {
            "source": "official Cboe 1M/3M/6M implied-correlation daily OHLC",
            "causal_availability": manifest["causal_availability"],
            "families": list(FAMILIES), "profiles": list(PROFILES),
            "exits": list(EXITS), "entry": "next regular-session market bar",
            "development": list(DEVELOPMENT), "validation": list(VALIDATION),
            "conditional_test": 2024, "conditional_confirmation": 2025,
            "conditional_reveal": 2026,
            "money": "3% current-equity margin, fixed 10x",
            "orders": "market entries only", "fees": "excluded by user instruction",
        },
        "source_quality": manifest["quality"],
        "rows": {"base_features": len(raw_features),
                 "correlation_days": len(correlation), "enriched": len(features)},
        "correlation_lag_days": {
            "minimum": float(features.correlation_lag_days.min()),
            "median": float(features.correlation_lag_days.median()),
            "maximum": float(features.correlation_lag_days.max()),
        },
        "minimum_ideal_hit_rate_for_30pct_month_at_3_trades_day": {
            spec["name"]: required_hit_rate(
                spec["target"], spec["stop"]) for spec in EXITS},
        "qualified_before_2024": len(survivors),
        "predeclared_broad_runner_union": {
            "routes": union_ids, "development": union_dev,
            "validation": union_validation,
            "test_2024": union_test_2024,
        },
        "frozen_before_2024": public(frozen), "test_2024": test_2024,
        "confirmation_2025": confirmation_2025, "untouched_2026": reveal_2026,
        "validation_failures": [public(item) for item in validation_failures[:12]],
        "development_frontier": [public(item) for item in frontier[:18]],
        "sources": {**price_sources, **correlation_sources,
                    str(Path(__file__).resolve()): digest(Path(__file__).resolve())},
        "limitations": [
            "Prior implied correlation is a daily regime input and cannot time entries alone.",
            "Dukascopy cash prices establish long-run structure, not X-Perp fills.",
            "Only seven technology stocks are common across all evaluation years.",
            "Any survivor still requires untouched direct X-Perp forward evidence.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": status, "rows": report["rows"],
        "correlation_lag_days": report["correlation_lag_days"],
        "qualified_before_2024": len(survivors),
        "frozen_before_2024": public(frozen), "test_2024": test_2024,
        "best_development": report["development_frontier"][:4],
        "best_validation_failures": report["validation_failures"][:3],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
