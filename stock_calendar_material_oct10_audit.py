"""Causal stock calendar-regime audit with literal 1-2% price targets."""
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
OUT = ROOT / "stock_calendar_material_oct10.json"
EXITS = (
    {"name": "balanced", "target": 1.0, "stop": .75, "sessions": 1},
    {"name": "swing", "target": 1.5, "stop": .75, "sessions": 2},
    {"name": "runner", "target": 2.0, "stop": 1.0, "sessions": 3},
)
BASE_FAMILIES = (
    "turn_month_momentum", "quarter_end_rebalance",
    "opex_week_continuation", "post_opex_reversal",
    "midmonth_breakout",
)
FAMILIES = (*BASE_FAMILIES, "calendar_router")
PROFILES = ("broad", "strict")
DEV_YEARS = (2017, 2018, 2019, 2020, 2021)
VALIDATION_YEARS = (2022, 2023)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def third_friday(value: pd.Timestamp) -> pd.Timestamp:
    first = value.replace(day=1)
    first_friday = first + pd.Timedelta(days=(4-first.weekday()) % 7)
    return first_friday + pd.Timedelta(days=14)


def add_calendar(features: pd.DataFrame) -> pd.DataFrame:
    sessions = pd.DataFrame({
        "session_date": sorted(features.session_date.unique())})
    sessions["session_dt"] = pd.to_datetime(sessions.session_date, utc=True)
    sessions["month_key"] = sessions.session_dt.dt.strftime("%Y-%m")
    sessions["quarter_key"] = (
        sessions.session_dt.dt.year.astype(str) + "-Q"
        + sessions.session_dt.dt.quarter.astype(str))
    sessions["month_pos"] = sessions.groupby("month_key").cumcount()+1
    sessions["month_size"] = sessions.groupby("month_key").session_date.transform("size")
    sessions["month_pos_rev"] = sessions.month_size-sessions.month_pos+1
    sessions["quarter_pos"] = sessions.groupby("quarter_key").cumcount()+1
    sessions["quarter_size"] = sessions.groupby("quarter_key").session_date.transform("size")
    sessions["quarter_pos_rev"] = sessions.quarter_size-sessions.quarter_pos+1
    sessions["opex_dt"] = sessions.session_dt.map(third_friday)
    sessions["days_to_opex"] = (sessions.opex_dt-sessions.session_dt).dt.days
    sessions["is_opex_week"] = sessions.days_to_opex.between(0, 4)
    sessions["post_opex_pos"] = 0
    for _, part in sessions.groupby("month_key", sort=False):
        after = part.index[part.session_dt.gt(part.opex_dt)].tolist()[:2]
        for position, index in enumerate(after, 1):
            sessions.loc[index, "post_opex_pos"] = position
    columns = [
        "session_date", "month_pos", "month_pos_rev", "quarter_pos_rev",
        "days_to_opex", "is_opex_week", "post_opex_pos",
    ]
    return features.merge(sessions[columns], on="session_date", how="left",
                          validate="many_to_one")


def directed(frame: pd.DataFrame, direction: pd.Series) -> pd.DataFrame:
    found = frame.copy()
    found["direction_n"] = direction.astype(np.int8)
    for name in ("ret5", "ret20", "intraday", "last_hour", "cross_z5",
                 "qqq_ret5", "qqq_ret20", "qqq_intraday"):
        found[f"{name}_dir"] = found[name]*found.direction_n
    found["breadth_dir"] = np.where(
        found.direction_n > 0, found.breadth_intra, 1-found.breadth_intra)
    return found


def entries(frame: pd.DataFrame, family: str, profile: str) -> pd.DataFrame:
    if family == "calendar_router":
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
    if family == "turn_month_momentum":
        direction = np.sign(frame.ret20).replace(0, np.nan).fillna(0)
        found = directed(frame, direction)
        mask = ((found.month_pos.le(2) | found.month_pos_rev.le(2))
                & found.ret20_dir.ge(.025 if strict else .01)
                & found.qqq_ret20_dir.ge(.015 if strict else 0)
                & found.last_hour_dir.ge(.002 if strict else 0)
                & found.breadth_dir.ge(.625 if strict else .50))
        score = found.ret20_dir+found.qqq_ret20_dir+found.last_hour_dir
    elif family == "quarter_end_rebalance":
        direction = -np.sign(frame.cross_z5).replace(0, np.nan).fillna(0)
        found = directed(frame, direction)
        mask = (found.quarter_pos_rev.le(3)
                & found.cross_z5_dir.le(-1.5 if strict else -.75)
                & found.ret5_dir.le(-.015 if strict else -.005)
                & found.last_hour_dir.ge(.002 if strict else 0))
        score = -found.cross_z5_dir-found.ret5_dir+found.last_hour_dir
    elif family == "opex_week_continuation":
        direction = np.sign(frame.qqq_ret20).replace(0, np.nan).fillna(0)
        found = directed(frame, direction)
        mask = (found.is_opex_week
                & found.qqq_ret20_dir.ge(.02 if strict else .01)
                & found.ret20_dir.ge(.025 if strict else .01)
                & found.last_hour_dir.ge(.002 if strict else 0)
                & found.breadth_dir.ge(.625 if strict else .50))
        score = found.qqq_ret20_dir+found.ret20_dir+found.last_hour_dir
    elif family == "post_opex_reversal":
        direction = -np.sign(frame.ret5).replace(0, np.nan).fillna(0)
        found = directed(frame, direction)
        mask = (found.post_opex_pos.between(1, 2)
                & found.ret5_dir.le(-.025 if strict else -.0125)
                & found.intraday_dir.le(-.005 if strict else 0)
                & found.last_hour_dir.ge(.003 if strict else .001))
        score = -found.ret5_dir-found.intraday_dir+found.last_hour_dir
    elif family == "midmonth_breakout":
        direction = np.sign(frame.ret20).replace(0, np.nan).fillna(0)
        found = directed(frame, direction)
        mask = (found.month_pos.between(8, 12)
                & found.ret20_dir.ge(.03 if strict else .015)
                & found.ret5_dir.ge(.01 if strict else 0)
                & found.last_hour_dir.ge(.002 if strict else 0)
                & found.breadth_dir.ge(.625 if strict else .50))
        score = found.ret20_dir+found.ret5_dir+found.last_hour_dir
    else:
        raise ValueError(family)
    output = found.loc[mask & found.direction_n.ne(0)].copy()
    output["family"], output["profile"] = family, profile
    output["score"] = score.loc[output.index].astype(float)
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
    candles, sources = base.load_history()
    raw_features, metadata = base.build_features(candles)
    features = add_calendar(raw_features)
    frontier, survivors, validation_failures = [], [], []
    for family, profile, exit_spec in itertools.product(FAMILIES, PROFILES, EXITS):
        proposed = entries(features, family, profile)
        if proposed.empty:
            continue
        resolved = base.add_outcomes(
            proposed, candles, metadata, exit_spec["target"],
            exit_spec["stop"], exit_spec["sessions"])
        resolved["module"] = f"stock_calendar_{family}"
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
            - 2*validation["dd_pct"], 8)
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
        item["development_2017_2021"]["economic_gate"]["projected_monthly_return_pct"],
        item["development_2017_2021"]["return_pct"]), reverse=True)
    validation_failures.sort(key=lambda item: (
        item["validation_2022_2023"]["economic_gate"]["projected_monthly_return_pct"],
        item["validation_2022_2023"]["return_pct"]), reverse=True)
    status = ("RESEARCH_CANDIDATE_AWAITING_XPERP_FORWARD"
              if confirmation_2025 and hidden_ok(confirmation_2025)
              else "REJECTED")
    report = {
        "status": status, "ready_for_live_orders": False,
        "protocol": {
            "mechanism": "preknown stock-session calendar regimes",
            "families": list(FAMILIES), "profiles": list(PROFILES),
            "exits": list(EXITS), "entry": "next regular-session market bar",
            "development": "2017-2021", "validation": "2022-2023",
            "test": "2024", "confirmation": "2025",
            "money": "3% current-equity margin, fixed 10x leverage",
            "orders": "market entries only", "fees": "excluded by user instruction",
        },
        "rows": {"features": len(features)},
        "minimum_ideal_hit_rate_for_30pct_month_at_3_trades_day": {
            spec["name"]: required_hit_rate(spec["target"], spec["stop"])
            for spec in EXITS},
        "qualified_before_2024": len(survivors),
        "frozen_before_2024": public(frozen), "test_2024": test_2024,
        "confirmation_2025": confirmation_2025,
        "validation_failures": [public(item) for item in validation_failures[:12]],
        "development_frontier": [public(item) for item in frontier[:18]],
        "sources": {**sources, str(Path(base.__file__)): digest(Path(base.__file__)),
                    str(Path(__file__)): digest(Path(__file__))},
        "limitations": [
            "Calendar regimes are known in advance but do not imply directional edge.",
            "Holiday-adjusted monthly expiration is represented by its calendar week.",
            "Dukascopy cash CFDs establish structural history, not X-Perp fills.",
            "Any historical survivor still requires untouched direct X-Perp evidence.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": status, "qualified_before_2024": len(survivors),
        "frozen_before_2024": public(frozen), "test_2024": test_2024,
        "confirmation_2025": confirmation_2025,
        "best_development": report["development_frontier"][:5],
        "best_validation_failures": report["validation_failures"][:3],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
