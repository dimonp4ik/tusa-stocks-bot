"""Causal material stock routes from the prior official Cboe VXN close."""
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
VXN_ROOT = ROOT / "stock_cboe_vxn_oct26"
OUT = ROOT / "stock_cboe_vxn_material_oct26.json"
EXITS = (
    {"name": "balanced", "target": 1.0, "stop": .75, "sessions": 1},
    {"name": "swing", "target": 1.5, "stop": .75, "sessions": 2},
    {"name": "runner", "target": 2.0, "stop": 1.0, "sessions": 3},
)
FAMILIES = (
    "shock_continuation", "shock_relief_reversal", "low_vol_trend",
    "vol_price_divergence", "compression_release",
)
PROFILES = ("broad", "strict")
DEVELOPMENT = (2017, 2018, 2019, 2020, 2021)
VALIDATION = (2022, 2023)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_vxn() -> tuple[pd.DataFrame, dict, dict[str, str]]:
    manifest_path = VXN_ROOT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_path = Path(manifest["source"]["path"])
    data_path = Path(manifest["data_path"])
    if digest(raw_path) != manifest["source"]["sha256"]:
        raise ValueError("raw VXN hash mismatch")
    if digest(data_path) != manifest["data_sha256"]:
        raise ValueError("normalized VXN hash mismatch")
    frame = pd.DataFrame(pickle.loads(data_path.read_bytes())).sort_values(
        "date_epoch")
    if frame.date_epoch.duplicated().any() or not frame.date_epoch.is_monotonic_increasing:
        raise ValueError("invalid normalized VXN chronology")
    return frame, manifest, {
        str(manifest_path): digest(manifest_path),
        str(raw_path): digest(raw_path), str(data_path): digest(data_path)}


def vxn_features(frame: pd.DataFrame) -> pd.DataFrame:
    value = frame.sort_values("date_epoch").copy()
    value["vxn_date"] = pd.to_datetime(value.date_epoch, unit="s", utc=True)
    value["vxn_change1"] = value.vxn_close.pct_change()
    value["vxn_change5"] = value.vxn_close.pct_change(5)
    value["vxn_change20"] = value.vxn_close.pct_change(20)
    value["vxn_close_location"] = (
        (value.vxn_close-value.vxn_low)
        /(value.vxn_high-value.vxn_low).replace(0, np.nan))
    value["vxn_range_pct"] = (
        (value.vxn_high-value.vxn_low)/value.vxn_close)
    mean = value.vxn_close.rolling(252, min_periods=160).mean()
    std = value.vxn_close.rolling(252, min_periods=160).std().replace(0, np.nan)
    value["vxn_z252"] = (value.vxn_close-mean)/std
    value["vxn_realized20"] = np.log(value.vxn_close).diff().rolling(
        20, min_periods=15).std()
    value["vxn_realized_rank252"] = value.vxn_realized20.rolling(
        252, min_periods=160).rank(pct=True)
    needed = [
        "vxn_change1", "vxn_change5", "vxn_change20",
        "vxn_close_location", "vxn_range_pct", "vxn_z252",
        "vxn_realized20", "vxn_realized_rank252",
    ]
    return value.replace([np.inf, -np.inf], np.nan).dropna(subset=needed)[
        ["date_epoch", "vxn_date", "vxn_open", "vxn_high", "vxn_low",
         "vxn_close", *needed]]


def attach(features: pd.DataFrame, vxn: pd.DataFrame) -> pd.DataFrame:
    value = features.copy()
    value["session_dt"] = pd.to_datetime(value.session_date, utc=True)
    value["session_epoch"] = value.session_dt.map(
        lambda stamp: int(stamp.timestamp())).astype(np.int64)
    joined = pd.merge_asof(
        value.sort_values("session_epoch"), vxn.sort_values("date_epoch"),
        left_on="session_epoch", right_on="date_epoch", direction="backward",
        allow_exact_matches=False, tolerance=7*86400,
    ).dropna(subset=["date_epoch"])
    if (joined.date_epoch >= joined.session_epoch).any():
        raise ValueError("same-session or future VXN close attached")
    joined["vxn_lag_days"] = (
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
    if family == "shock_continuation":
        found = directed(frame, pd.Series(-1, index=frame.index))
        mask = (found.vxn_change5.ge(.20 if strict else .10)
                & found.vxn_z252.ge(1.0 if strict else .5)
                & found.qqq_intraday_dir.ge(.004 if strict else 0)
                & found.last_hour_dir.ge(.002 if strict else 0)
                & found.breadth_dir.ge(.625 if strict else .50))
        score = (found.vxn_change5+found.vxn_z252
                 +found.qqq_intraday_dir+found.last_hour_dir)
    elif family == "shock_relief_reversal":
        found = directed(frame, pd.Series(1, index=frame.index))
        mask = (found.vxn_z252.ge(1.5 if strict else 1.0)
                & found.vxn_change1.le(-.07 if strict else -.03)
                & found.vxn_close_location.le(.25 if strict else .45)
                & found.qqq_last_hour_dir.ge(.002 if strict else 0)
                & found.breadth_dir.ge(.625 if strict else .50))
        score = (found.vxn_z252-found.vxn_change1
                 -found.vxn_close_location+found.qqq_last_hour_dir)
    elif family == "low_vol_trend":
        found = directed(frame, pd.Series(1, index=frame.index))
        mask = (found.vxn_z252.le(-.5 if strict else 0)
                & found.vxn_change20.le(-.10 if strict else 0)
                & found.ret20_dir.ge(.03 if strict else .01)
                & found.qqq_intraday_dir.ge(.003 if strict else 0)
                & found.last_hour_dir.ge(.002 if strict else 0)
                & found.breadth_dir.ge(.625 if strict else .50))
        score = (-found.vxn_z252-found.vxn_change20
                 +found.ret20_dir+found.last_hour_dir)
    elif family == "vol_price_divergence":
        found = directed(frame, pd.Series(-1, index=frame.index))
        mask = (found.vxn_change5.ge(.10 if strict else .04)
                & found.ret20.ge(.03 if strict else .01)
                & found.qqq_intraday_dir.ge(.003 if strict else 0)
                & found.last_hour_dir.ge(.002 if strict else 0))
        score = (found.vxn_change5+found.ret20
                 +found.qqq_intraday_dir+found.last_hour_dir)
    elif family == "compression_release":
        direction = np.sign(frame.qqq_intraday).replace(0, np.nan)
        found = directed(frame, direction.fillna(0))
        mask = (found.direction_n.ne(0)
                & found.vxn_realized_rank252.le(.20 if strict else .35)
                & found.vxn_change1.abs().ge(.05 if strict else .025)
                & found.qqq_intraday_dir.ge(.004 if strict else .002)
                & found.last_hour_dir.ge(.002 if strict else 0)
                & found.breadth_dir.ge(.625 if strict else .50))
        score = (-found.vxn_realized_rank252+found.vxn_change1.abs()
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
    raw_vxn, manifest, vxn_sources = load_vxn()
    vxn = vxn_features(raw_vxn)
    features = attach(raw_features, vxn)
    frontier, survivors, validation_failures = [], [], []
    for family, profile, exit_spec in itertools.product(
            FAMILIES, PROFILES, EXITS):
        proposed = entries(features, family, profile)
        if proposed.empty:
            continue
        resolved = base.add_outcomes(
            proposed, candles, metadata, exit_spec["target"],
            exit_spec["stop"], exit_spec["sessions"])
        resolved["module"] = f"stock_cboe_vxn_{family}"
        development = evaluate(resolved, features, DEVELOPMENT)
        annual_dev = {str(year): evaluate(resolved, features, (year,))
                      for year in DEVELOPMENT}
        item = {
            "route_id": f"{family}_{profile}_{exit_spec['name']}",
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
    report = {
        "status": status, "ready_for_live_orders": False,
        "protocol": {
            "source": "official Cboe Nasdaq-100 Volatility Index daily OHLC",
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
                 "vxn_days": len(vxn), "enriched": len(features)},
        "vxn_lag_days": {
            "minimum": float(features.vxn_lag_days.min()),
            "median": float(features.vxn_lag_days.median()),
            "maximum": float(features.vxn_lag_days.max()),
        },
        "minimum_ideal_hit_rate_for_30pct_month_at_3_trades_day": {
            spec["name"]: required_hit_rate(
                spec["target"], spec["stop"]) for spec in EXITS},
        "qualified_before_2024": len(survivors),
        "frozen_before_2024": public(frozen), "test_2024": test_2024,
        "confirmation_2025": confirmation_2025, "untouched_2026": reveal_2026,
        "validation_failures": [public(item) for item in validation_failures[:12]],
        "development_frontier": [public(item) for item in frontier[:18]],
        "sources": {**price_sources, **vxn_sources,
                    str(Path(__file__).resolve()): digest(Path(__file__).resolve())},
        "limitations": [
            "The prior VXN close is a daily regime input and cannot time entries alone.",
            "Dukascopy cash prices establish long-run structure, not X-Perp fills.",
            "Only seven technology stocks are common across all evaluation years.",
            "Any survivor still requires untouched direct X-Perp forward evidence.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": status, "rows": report["rows"],
        "vxn_lag_days": report["vxn_lag_days"],
        "qualified_before_2024": len(survivors),
        "frozen_before_2024": public(frozen), "test_2024": test_2024,
        "best_development": report["development_frontier"][:4],
        "best_validation_failures": report["validation_failures"][:3],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
