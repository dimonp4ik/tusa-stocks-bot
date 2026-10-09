"""Causal material stock moves from the prior Cboe volatility surface close."""
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
SURFACE_ROOT = ROOT / "stock_cboe_vol_surface_oct9"
OUT = ROOT / "stock_cboe_vol_surface_material_oct9.json"
EXITS = (
    {"name": "balanced", "target": 1.0, "stop": .75, "sessions": 1},
    {"name": "swing", "target": 1.5, "stop": .75, "sessions": 2},
    {"name": "runner", "target": 2.0, "stop": 1.0, "sessions": 3},
)
FAMILIES = (
    "risk_off_inversion", "vol_relief_reversal", "tail_warning",
    "vol_of_vol_capitulation", "contango_risk_on",
)
PROFILES = ("broad", "strict")
DEV_YEARS = (2017, 2018, 2019, 2020, 2021)
VALIDATION_YEARS = (2022, 2023)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_surface() -> tuple[pd.DataFrame, dict, dict[str, str]]:
    manifest_path = SURFACE_ROOT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sources = {str(manifest_path): digest(manifest_path)}
    for item in manifest["sources"].values():
        path = Path(item["path"])
        found = digest(path)
        if found != item["sha256"]:
            raise ValueError(f"Cboe source hash mismatch: {path}")
        sources[str(path)] = found
    data_path = Path(manifest["data_path"])
    if digest(data_path) != manifest["data_sha256"]:
        raise ValueError("Cboe normalized surface hash mismatch")
    sources[str(data_path)] = digest(data_path)
    frame = pd.DataFrame(pickle.loads(data_path.read_bytes())).sort_values("date_epoch")
    if frame.date_epoch.duplicated().any() or not frame.date_epoch.is_monotonic_increasing:
        raise ValueError("invalid Cboe normalized chronology")
    return frame, manifest, sources


def surface_features(frame: pd.DataFrame) -> pd.DataFrame:
    value = frame.sort_values("date_epoch").copy()
    value["surface_date"] = pd.to_datetime(value.date_epoch, unit="s", utc=True)
    value["term_9d_30d"] = value.vix9d/value.vix-1
    value["term_30d_3m"] = value.vix/value.vix3m-1
    value["term_3m_6m"] = value.vix3m/value.vix6m-1
    value["term_6m_1y"] = value.vix6m/value.vix1y-1
    value["curve_9d_6m"] = value.vix9d/value.vix6m-1
    value["vix_change5"] = value.vix/value.vix.shift(5)-1
    value["vix_change20"] = value.vix/value.vix.shift(20)-1
    for name in ("vix", "skew", "vvix"):
        mean = value[name].rolling(252, min_periods=160).mean()
        std = value[name].rolling(252, min_periods=160).std().replace(0, np.nan)
        value[f"{name}_z252"] = (value[name]-mean)/std
    needed = [
        "term_9d_30d", "term_30d_3m", "term_3m_6m", "term_6m_1y",
        "curve_9d_6m", "vix_change5", "vix_change20", "vix_z252",
        "skew_z252", "vvix_z252",
    ]
    return value.replace([np.inf, -np.inf], np.nan).dropna(subset=needed)[
        ["date_epoch", "surface_date", "vix", "skew", "vvix", *needed]]


def attach_surface(features: pd.DataFrame, surface: pd.DataFrame) -> pd.DataFrame:
    value = features.copy()
    value["session_dt"] = pd.to_datetime(value.session_date, utc=True)
    value["session_epoch"] = value.session_dt.map(
        lambda stamp: int(stamp.timestamp())).astype(np.int64)
    value = pd.merge_asof(
        value.sort_values("session_epoch"), surface.sort_values("date_epoch"),
        left_on="session_epoch", right_on="date_epoch", direction="backward",
        allow_exact_matches=False, tolerance=7*86400,
    ).dropna(subset=["date_epoch"])
    if (value.date_epoch >= value.session_epoch).any():
        raise ValueError("same-session or future Cboe close attached")
    return value


def directed(frame: pd.DataFrame, direction: pd.Series) -> pd.DataFrame:
    found = frame.copy()
    found["direction_n"] = direction.astype(np.int8)
    for name in ("ret5", "ret20", "intraday", "last_hour", "cross_z5",
                 "qqq_ret20", "qqq_intraday"):
        found[f"{name}_dir"] = found[name]*found.direction_n
    found["breadth_dir"] = np.where(
        found.direction_n > 0, found.breadth_intra, 1-found.breadth_intra)
    return found


def entries(frame: pd.DataFrame, family: str, profile: str) -> pd.DataFrame:
    strict = profile == "strict"
    if family == "risk_off_inversion":
        direction = pd.Series(-1, index=frame.index)
        found = directed(frame, direction)
        mask = (found.term_9d_30d.ge(.05 if strict else 0)
                & found.term_30d_3m.ge(.02 if strict else -.02)
                & found.vix_change5.ge(.15 if strict else .05)
                & found.qqq_intraday_dir.ge(.004 if strict else 0)
                & found.last_hour_dir.ge(.002 if strict else 0)
                & found.breadth_dir.ge(.625 if strict else .50))
        score = (found.term_9d_30d+found.term_30d_3m
                 +found.vix_change5+found.last_hour_dir)
    elif family == "vol_relief_reversal":
        direction = pd.Series(1, index=frame.index)
        found = directed(frame, direction)
        mask = (found.vix_z252.ge(1.5 if strict else 1.0)
                & found.vix_change5.le(-.10 if strict else -.05)
                & found.qqq_intraday_dir.ge(.004 if strict else 0)
                & found.last_hour_dir.ge(.002 if strict else 0)
                & found.breadth_dir.ge(.625 if strict else .50))
        score = found.vix_z252-found.vix_change5+found.last_hour_dir
    elif family == "tail_warning":
        direction = pd.Series(-1, index=frame.index)
        found = directed(frame, direction)
        mask = (found.skew_z252.ge(1.5 if strict else 1.0)
                & found.term_9d_30d.le(-.05 if strict else 0)
                & found.ret20_dir.le(-.02 if strict else -.01)
                & found.last_hour_dir.ge(.002 if strict else 0))
        score = found.skew_z252-found.term_9d_30d+found.last_hour_dir
    elif family == "vol_of_vol_capitulation":
        direction = pd.Series(1, index=frame.index)
        found = directed(frame, direction)
        mask = (found.vvix_z252.ge(2.0 if strict else 1.25)
                & found.vix_z252.ge(1.5 if strict else 1.0)
                & found.vix_change5.ge(.20 if strict else .10)
                & found.intraday_dir.le(-.005 if strict else 0)
                & found.last_hour_dir.ge(.003 if strict else .001))
        score = found.vvix_z252+found.vix_z252+found.vix_change5+found.last_hour_dir
    elif family == "contango_risk_on":
        direction = pd.Series(1, index=frame.index)
        found = directed(frame, direction)
        mask = (found.term_9d_30d.le(-.15 if strict else -.08)
                & found.term_30d_3m.le(-.08 if strict else -.03)
                & found.vix_change5.le(-.05 if strict else 0)
                & found.ret20_dir.ge(.03 if strict else .01)
                & found.last_hour_dir.ge(.002 if strict else 0)
                & found.breadth_dir.ge(.625 if strict else .50))
        score = (-found.term_9d_30d-found.term_30d_3m-found.vix_change5
                 +found.ret20_dir+found.last_hour_dir)
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
    market_days = features[features.year.isin(years)].session_date.nunique()
    result = base.account_metrics(selected, market_days)
    rows = accepted_rows(selected)
    hits = sum(int(row["outcome_code"]) == 1 for row in rows)
    result["literal_target_hits"] = hits
    result["literal_target_hit_rate"] = round(
        100*hits/len(rows), 4) if rows else 0.0
    values = list(result.get("monthly_return_pct", {}).values())
    result["minimum_month_pct"] = round(min(values), 4) if values else 0.0
    result["median_month_pct"] = round(float(np.median(values)), 4) if values else 0.0
    result["economic_gate"] = (economic_gate(
        [float(row["signed_return"]) for row in rows],
        float(result["trades_per_market_day"])) if rows else {
            "n": 0, "observed_trades_per_day": 0.0,
            "projected_monthly_return_pct": 0.0,
            "frequency_pass": False, "return_pass": False, "passes": False})
    return result


def development_ok(value: dict, annual: dict) -> bool:
    return bool(value["n"] >= 250 and value["return_pct"] > 0
                and value["dd_pct"] <= 12
                and sum(item["return_pct"] > 0 for item in annual.values()) >= 4
                and value["trades_per_market_day"] >= .5)


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
    raw_surface, manifest, surface_sources = load_surface()
    surface = surface_features(raw_surface)
    features = attach_surface(raw_features, surface)
    frontier, survivors, validation_failures = [], [], []
    for family, profile, exit_spec in itertools.product(FAMILIES, PROFILES, EXITS):
        proposed = entries(features, family, profile)
        if proposed.empty:
            continue
        resolved = base.add_outcomes(
            proposed, candles, metadata, exit_spec["target"],
            exit_spec["stop"], exit_spec["sessions"])
        resolved["module"] = f"stock_cboe_{family}"
        development = evaluate(resolved, features, DEV_YEARS)
        annual_dev = {str(year): evaluate(resolved, features, (year,))
                      for year in DEV_YEARS}
        item = {
            "route_id": f"{family}_{profile}_{exit_spec['name']}",
            "family": family, "profile": profile, "exit": exit_spec,
            "development_2017_2021": development,
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
        item["development_2017_2021"]["economic_gate"][
            "projected_monthly_return_pct"],
        item["development_2017_2021"]["literal_target_hit_rate"],
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
            "source": "Cboe VIX horizon indices, SKEW and VVIX daily closes",
            "causal_availability": manifest["causal_availability"],
            "families": list(FAMILIES), "profiles": list(PROFILES),
            "exits": list(EXITS), "entry": "next regular-session market bar",
            "development": "2017-2021", "validation": "2022-2023",
            "test": "2024", "confirmation": "2025",
            "untouched_reveal": "2026",
            "money": "3% current-equity margin, fixed 10x leverage",
            "orders": "market entries only", "fees": "excluded by user instruction",
        },
        "source_quality": manifest["quality"],
        "rows": {"base_features": len(raw_features),
                 "surface_days": len(surface), "enriched": len(features)},
        "minimum_ideal_hit_rate_for_30pct_month_at_3_trades_day": {
            spec["name"]: required_hit_rate(spec["target"], spec["stop"])
            for spec in EXITS},
        "qualified_before_2024": len(survivors),
        "frozen_before_2024": public(frozen), "test_2024": test_2024,
        "confirmation_2025": confirmation_2025, "untouched_2026": reveal_2026,
        "validation_failures": [public(item) for item in validation_failures[:12]],
        "development_frontier": [public(item) for item in frontier[:18]],
        "sources": {**price_sources, **surface_sources,
                    str(Path(__file__)): digest(Path(__file__))},
        "limitations": [
            "Cboe closes gate the next stock session and do not time intraday entries alone.",
            "Dukascopy cash CFDs establish structural history, not X-Perp fills.",
            "Only seven technology stocks are available in every evaluation year.",
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
