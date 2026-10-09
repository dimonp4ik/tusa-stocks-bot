"""Causal material stock-move audit of prior-session FINRA short volume."""
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
SHORT_ROOT = ROOT / "stock_finra_short_volume_oct9"
OUT = ROOT / "stock_finra_short_volume_material_oct9.json"
EXITS = (
    {"name": "balanced", "target": 1.0, "stop": .75, "sessions": 1},
    {"name": "swing", "target": 1.5, "stop": .75, "sessions": 2},
    {"name": "runner", "target": 2.0, "stop": 1.0, "sessions": 3},
)
BASE_FAMILIES = (
    "short_pressure_continuation", "short_covering_squeeze",
    "low_short_momentum", "short_price_divergence",
    "cross_sectional_pressure",
)
FAMILIES = (*BASE_FAMILIES, "short_volume_router")
PROFILES = ("broad", "strict")
DEV_YEARS = (2019, 2020, 2021)
VALIDATION_YEARS = (2022, 2023)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_short() -> tuple[pd.DataFrame, dict, dict[str, str]]:
    manifest_path = SHORT_ROOT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data_path = Path(manifest["data_path"])
    found = digest(data_path)
    if found != manifest["data_sha256"]:
        raise ValueError("FINRA normalized archive hash mismatch")
    frame = pd.DataFrame(pickle.loads(data_path.read_bytes())).sort_values(
        ["asset", "date_epoch"])
    if frame.duplicated(["asset", "date_epoch"]).any():
        raise ValueError("duplicate FINRA asset/day")
    return frame, manifest, {
        str(manifest_path): digest(manifest_path), str(data_path): found}


def short_features(frame: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for asset, value in frame.groupby("asset", sort=True):
        value = value.sort_values("date_epoch").copy()
        for raw, label in (("short_ratio", "short"),
                           ("short_exempt_ratio", "exempt")):
            mean = value[raw].rolling(60, min_periods=40).mean()
            std = value[raw].rolling(60, min_periods=40).std().replace(0, np.nan)
            value[f"{label}_z60"] = (value[raw]-mean)/std
        logged_volume = np.log1p(value.total_volume.clip(lower=0))
        mean = logged_volume.rolling(60, min_periods=40).mean()
        std = logged_volume.rolling(60, min_periods=40).std().replace(0, np.nan)
        value["volume_z60"] = (logged_volume-mean)/std
        value["short_delta1"] = value.short_ratio-value.short_ratio.shift(1)
        value["short_delta5"] = value.short_ratio-value.short_ratio.shift(5)
        value["short_delta20"] = value.short_ratio-value.short_ratio.shift(20)
        value["asset"] = asset
        pieces.append(value)
    value = pd.concat(pieces, ignore_index=True)
    stocks = value[value.asset.ne("QQQ")].copy()
    group = stocks.groupby("date_epoch", sort=False)
    mean = group.short_ratio.transform("mean")
    std = group.short_ratio.transform("std").replace(0, np.nan)
    stocks["short_cross_z"] = (stocks.short_ratio-mean)/std
    qqq = value[value.asset.eq("QQQ")][
        ["date_epoch", "short_ratio", "short_z60", "short_delta5"]
    ].rename(columns={
        "short_ratio": "qqq_short_ratio", "short_z60": "qqq_short_z60",
        "short_delta5": "qqq_short_delta5",
    })
    stocks = stocks.merge(qqq, on="date_epoch", how="inner", validate="many_to_one")
    stocks["short_premium_qqq"] = stocks.short_ratio-stocks.qqq_short_ratio
    stocks["source_date"] = pd.to_datetime(stocks.date_epoch, unit="s", utc=True)
    needed = [
        "short_z60", "exempt_z60", "volume_z60", "short_delta1",
        "short_delta5", "short_delta20", "short_cross_z",
        "qqq_short_ratio", "qqq_short_z60", "qqq_short_delta5",
        "short_premium_qqq",
    ]
    return stocks.replace([np.inf, -np.inf], np.nan).dropna(subset=needed)[
        ["asset", "date_epoch", "source_date", "short_ratio", *needed]]


def attach_short(features: pd.DataFrame, short: pd.DataFrame) -> pd.DataFrame:
    sessions = features[["symbol", "session_date"]].drop_duplicates().copy()
    sessions["asset"] = sessions.symbol.str.removesuffix("USDT")
    sessions["session_epoch"] = pd.to_datetime(
        sessions.session_date, utc=True).map(lambda value: int(value.timestamp()))
    attached = []
    for asset, left in sessions.groupby("asset", sort=True):
        right = short[short.asset.eq(asset)]
        found = pd.merge_asof(
            left.sort_values("session_epoch"), right.sort_values("date_epoch"),
            left_on="session_epoch", right_on="date_epoch", direction="backward",
            allow_exact_matches=False, tolerance=7*86400,
            suffixes=("", "_source"),
        ).dropna(subset=["date_epoch"])
        attached.append(found)
    lookup = pd.concat(attached, ignore_index=True)
    if (lookup.date_epoch >= lookup.session_epoch).any():
        raise ValueError("same-session or future FINRA aggregate attached")
    columns = [name for name in lookup.columns
               if name not in ("symbol", "asset_source", "session_epoch")]
    result = features.merge(lookup[["symbol", *columns]],
                            on=["symbol", "session_date"], how="inner",
                            validate="many_to_one")
    return result


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
    if family == "short_volume_router":
        pieces = [entries(frame, name, profile) for name in BASE_FAMILIES]
        output = pd.concat([value for value in pieces if not value.empty],
                           ignore_index=True)
        consensus = output.groupby(
            ["symbol", "entry_time"]).direction_n.transform("nunique").eq(1)
        output = output.loc[consensus].sort_values(
            ["score", "symbol"], ascending=[False, True]).drop_duplicates(
                ["symbol", "entry_time"], keep="first")
        output["family"] = family
        return output
    strict = profile == "strict"
    if family == "short_pressure_continuation":
        found = directed(frame, pd.Series(-1, index=frame.index))
        mask = (found.short_z60.ge(1.5 if strict else .75)
                & found.short_delta5.ge(.04 if strict else .02)
                & found.ret5_dir.ge(.015 if strict else .005)
                & found.intraday_dir.ge(.004 if strict else 0)
                & found.last_hour_dir.ge(.002 if strict else 0)
                & found.breadth_dir.ge(.625 if strict else .50))
        score = found.short_z60+10*found.short_delta5+found.ret5_dir+found.last_hour_dir
    elif family == "short_covering_squeeze":
        found = directed(frame, pd.Series(1, index=frame.index))
        mask = (found.short_z60.ge(1.0 if strict else .5)
                & found.short_delta5.le(-.04 if strict else -.02)
                & found.ret5_dir.ge(.015 if strict else .005)
                & found.intraday_dir.ge(.004 if strict else 0)
                & found.last_hour_dir.ge(.002 if strict else 0)
                & found.breadth_dir.ge(.625 if strict else .50))
        score = found.short_z60-10*found.short_delta5+found.ret5_dir+found.last_hour_dir
    elif family == "low_short_momentum":
        found = directed(frame, pd.Series(1, index=frame.index))
        mask = (found.short_z60.le(-1.5 if strict else -.75)
                & found.short_delta20.le(-.04 if strict else -.02)
                & found.ret20_dir.ge(.03 if strict else .01)
                & found.last_hour_dir.ge(.002 if strict else 0)
                & found.breadth_dir.ge(.625 if strict else .50))
        score = -found.short_z60-10*found.short_delta20+found.ret20_dir+found.last_hour_dir
    elif family == "short_price_divergence":
        found = directed(frame, pd.Series(1, index=frame.index))
        mask = (found.short_z60.ge(1.25 if strict else .5)
                & found.short_delta5.ge(.03 if strict else .015)
                & found.ret5_dir.ge(.02 if strict else .01)
                & found.intraday_dir.ge(.004 if strict else 0)
                & found.last_hour_dir.ge(.002 if strict else 0))
        score = found.short_z60+10*found.short_delta5+found.ret5_dir+found.last_hour_dir
    elif family == "cross_sectional_pressure":
        direction = -np.sign(frame.short_cross_z)
        found = directed(frame, direction)
        mask = (found.short_cross_z.abs().ge(1.5 if strict else 1.0)
                & found.short_delta5.mul(-found.direction_n).ge(
                    .03 if strict else .01)
                & found.ret5_dir.ge(.01 if strict else 0)
                & found.last_hour_dir.ge(.002 if strict else 0)
                & found.breadth_dir.ge(.625 if strict else .50))
        score = found.short_cross_z.abs()+10*found.short_delta5.abs()+found.ret5_dir
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
    result["literal_target_hit_rate"] = round(100*hits/len(rows), 4) if rows else 0.0
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
    return bool(value["n"] >= 200 and value["return_pct"] > 0
                and value["dd_pct"] <= 12
                and sum(item["return_pct"] > 0 for item in annual.values()) >= 2
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
    raw_short, manifest, short_sources = load_short()
    daily = short_features(raw_short)
    features = attach_short(raw_features, daily)
    frontier, survivors, validation_failures = [], [], []
    for family, profile, exit_spec in itertools.product(FAMILIES, PROFILES, EXITS):
        proposed = entries(features, family, profile)
        if proposed.empty:
            continue
        resolved = base.add_outcomes(
            proposed, candles, metadata, exit_spec["target"],
            exit_spec["stop"], exit_spec["sessions"])
        resolved["module"] = f"stock_finra_{family}"
        development = evaluate(resolved, features, DEV_YEARS)
        annual_dev = {str(year): evaluate(resolved, features, (year,))
                      for year in DEV_YEARS}
        item = {
            "route_id": f"{family}_{profile}_{exit_spec['name']}",
            "family": family, "profile": profile, "exit": exit_spec,
            "development_2019_2021": development,
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
    test_2024 = confirmation_2025 = None
    if frozen:
        test_2024 = evaluate(frozen["_rows"], features, (2024,))
        if hidden_ok(test_2024):
            confirmation_2025 = evaluate(frozen["_rows"], features, (2025,))
    frontier.sort(key=lambda item: (
        item["development_2019_2021"]["economic_gate"][
            "projected_monthly_return_pct"],
        item["development_2019_2021"]["literal_target_hit_rate"],
        item["development_2019_2021"]["return_pct"]), reverse=True)
    validation_failures.sort(key=lambda item: (
        item["validation_2022_2023"]["economic_gate"][
            "projected_monthly_return_pct"],
        item["validation_2022_2023"]["return_pct"]), reverse=True)
    status = ("RESEARCH_CANDIDATE_AWAITING_XPERP_FORWARD"
              if confirmation_2025 and hidden_ok(confirmation_2025)
              else "REJECTED")
    report = {
        "status": status, "ready_for_live_orders": False,
        "protocol": {
            "source": "FINRA Reg SHO daily short-sale volume",
            "semantic_warning": manifest["semantic_warning"],
            "causal_availability": manifest["causal_availability"],
            "families": list(FAMILIES), "profiles": list(PROFILES),
            "exits": list(EXITS), "entry": "next regular-session market bar",
            "development": "2019-2021", "validation": "2022-2023",
            "test": "2024", "confirmation": "2025",
            "money": "3% current-equity margin, fixed 10x leverage",
            "orders": "market entries only", "fees": "excluded by user instruction",
        },
        "source_quality": {key: value for key, value in manifest["quality"].items()
                           if key != "daily"},
        "rows": {"base_features": len(raw_features),
                 "short_feature_rows": len(daily), "enriched": len(features)},
        "minimum_ideal_hit_rate_for_30pct_month_at_3_trades_day": {
            spec["name"]: required_hit_rate(spec["target"], spec["stop"])
            for spec in EXITS},
        "qualified_before_2024": len(survivors),
        "frozen_before_2024": public(frozen), "test_2024": test_2024,
        "confirmation_2025": confirmation_2025,
        "validation_failures": [public(item) for item in validation_failures[:12]],
        "development_frontier": [public(item) for item in frontier[:18]],
        "sources": {**price_sources, **short_sources,
                    str(Path(__file__)): digest(Path(__file__))},
        "limitations": [
            "FINRA daily short-sale volume is not short interest.",
            "FINRA data excludes short-sale volume executed on exchanges.",
            "Daily aggregates gate the next stock session and do not time entries alone.",
            "Dukascopy cash CFDs establish structural history, not X-Perp fills.",
            "Any historical survivor still requires untouched direct X-Perp forward evidence.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": status, "rows": report["rows"],
        "qualified_before_2024": len(survivors),
        "frozen_before_2024": public(frozen), "test_2024": test_2024,
        "confirmation_2025": confirmation_2025,
        "best_development": report["development_frontier"][:4],
        "best_validation_failures": report["validation_failures"][:3],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
