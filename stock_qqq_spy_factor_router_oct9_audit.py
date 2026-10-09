"""Long-history QQQ-vs-SPY factor-regime router for material stock moves.

The technology factor spread is computed only from completed regular-session
bars.  Four predeclared semantic families are evaluated with fixed 1-2% price
targets.  Entry is the next market bar.  2024, 2025 and 2026 are revealed in
sequence only when every earlier gate passes.
"""
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
OUT = ROOT / "stock_qqq_spy_factor_router_oct9.json"
EXITS = (
    {"name": "balanced", "target": 1.0, "stop": .75, "sessions": 1},
    {"name": "swing", "target": 1.5, "stop": .75, "sessions": 2},
    {"name": "runner", "target": 2.0, "stop": 1.0, "sessions": 3},
)
PROFILES = (
    {"family": "factor_continuation", "level": "loose",
     "factor20": .003, "stock20": .005, "factor5": -.002,
     "last": 0.0, "breadth": .50},
    {"family": "factor_continuation", "level": "medium",
     "factor20": .007, "stock20": .01, "factor5": 0.0,
     "last": .001, "breadth": .50},
    {"family": "factor_continuation", "level": "strict",
     "factor20": .012, "stock20": .02, "factor5": .002,
     "last": .002, "breadth": .625},
    {"family": "factor_pullback_resume", "level": "loose",
     "factor20": .005, "stock20": .01, "stock5_max": .002,
     "last": 0.0, "factor_last": 0.0, "breadth": .50},
    {"family": "factor_pullback_resume", "level": "medium",
     "factor20": .01, "stock20": .02, "stock5_max": 0.0,
     "last": .001, "factor_last": 0.0, "breadth": .50},
    {"family": "factor_pullback_resume", "level": "strict",
     "factor20": .015, "stock20": .03, "stock5_max": -.003,
     "last": .002, "factor_last": .001, "breadth": .625},
    {"family": "factor_catchup", "level": "loose",
     "factor20": .005, "stock20": .005, "lag": -.001,
     "last": 0.0, "factor_last": 0.0, "breadth": .50},
    {"family": "factor_catchup", "level": "medium",
     "factor20": .01, "stock20": .01, "lag": -.002,
     "last": .001, "factor_last": 0.0, "breadth": .50},
    {"family": "factor_catchup", "level": "strict",
     "factor20": .015, "stock20": .02, "lag": -.004,
     "last": .002, "factor_last": .001, "breadth": .625},
    {"family": "factor_exhaustion", "level": "loose",
     "factor5": .005, "stock5": .005, "last": 0.0,
     "factor_last": 0.0, "breadth_max": .50},
    {"family": "factor_exhaustion", "level": "medium",
     "factor5": .01, "stock5": .01, "last": .001,
     "factor_last": 0.0, "breadth_max": .50},
    {"family": "factor_exhaustion", "level": "strict",
     "factor5": .015, "stock5": .02, "last": .002,
     "factor_last": .001, "breadth_max": .375},
)
DEV_YEARS = (2017, 2018, 2019, 2020, 2021)
VALIDATION_YEARS = (2022, 2023)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load() -> tuple[dict, dict]:
    candles, sources = base.load_history()
    manifest = json.loads(base.MANIFEST.read_text(encoding="utf-8"))
    item = manifest["symbols"]["SPYUSDT"]
    path = Path(item["path"])
    raw = path.read_bytes()
    found = hashlib.sha256(raw).hexdigest()
    if found != item["sha256"]:
        raise ValueError("SPY history hash mismatch")
    candles["SPYUSDT"] = pickle.loads(raw)
    sources[str(path)] = found
    return candles, sources


def build_features(candles: dict) -> tuple[pd.DataFrame, dict]:
    core = {symbol: candles[symbol] for symbol in base.ALL_SYMBOLS}
    frame, metadata = base.build_features(core)
    spy, bars, regular = base.session_table("SPYUSDT", candles["SPYUSDT"])
    metadata["SPYUSDT"] = {"bars_per_session": bars, "regular": regular}
    spy = spy[["session_date", "slot", "ret5", "ret20", "gap",
               "intraday", "last_hour"]].rename(columns={
                   name: f"spy_{name}" for name in
                   ("ret5", "ret20", "gap", "intraday", "last_hour")})
    result = frame.merge(spy, on=["session_date", "slot"], how="inner",
                         validate="many_to_one")
    for name in ("ret5", "ret20", "gap", "intraday", "last_hour"):
        result[f"factor_{name}"] = result[f"qqq_{name}"] - result[f"spy_{name}"]
    return result.replace([np.inf, -np.inf], np.nan).dropna(), metadata


def directed(frame: pd.DataFrame, direction: np.ndarray) -> pd.DataFrame:
    found = frame.copy()
    found["direction_n"] = direction.astype(np.int8)
    for name in ("ret5", "ret20", "intraday", "last_hour", "cross_z5",
                 "factor_ret5", "factor_ret20", "factor_intraday",
                 "factor_last_hour"):
        found[f"{name}_dir"] = found[name] * found.direction_n
    found["breadth_dir"] = np.where(
        found.direction_n > 0, found.breadth_intra, 1 - found.breadth_intra)
    return found


def entries(frame: pd.DataFrame, profile: dict) -> pd.DataFrame:
    family = profile["family"]
    if family == "factor_exhaustion":
        factor_direction = np.sign(frame.factor_ret5.to_numpy(float))
        found = directed(frame, -factor_direction)
        mask = (found.direction_n.ne(0)
                & found.factor_ret5_dir.le(-profile["factor5"])
                & found.ret5_dir.le(-profile["stock5"])
                & found.last_hour_dir.ge(profile["last"])
                & found.factor_last_hour_dir.ge(profile["factor_last"])
                & found.breadth_dir.le(profile["breadth_max"]))
    else:
        factor_direction = np.sign(frame.factor_ret20.to_numpy(float))
        found = directed(frame, factor_direction)
        if family == "factor_continuation":
            mask = (found.direction_n.ne(0)
                    & found.factor_ret20_dir.ge(profile["factor20"])
                    & found.ret20_dir.ge(profile["stock20"])
                    & found.factor_ret5_dir.ge(profile["factor5"])
                    & found.last_hour_dir.ge(profile["last"])
                    & found.breadth_dir.ge(profile["breadth"]))
        elif family == "factor_pullback_resume":
            mask = (found.direction_n.ne(0)
                    & found.factor_ret20_dir.ge(profile["factor20"])
                    & found.ret20_dir.ge(profile["stock20"])
                    & found.ret5_dir.le(profile["stock5_max"])
                    & found.last_hour_dir.ge(profile["last"])
                    & found.factor_last_hour_dir.ge(profile["factor_last"])
                    & found.breadth_dir.ge(profile["breadth"]))
        elif family == "factor_catchup":
            relative_intraday = found.intraday_dir - found.qqq_intraday * found.direction_n
            mask = (found.direction_n.ne(0)
                    & found.factor_ret20_dir.ge(profile["factor20"])
                    & found.ret20_dir.ge(profile["stock20"])
                    & relative_intraday.le(profile["lag"])
                    & found.last_hour_dir.ge(profile["last"])
                    & found.factor_last_hour_dir.ge(profile["factor_last"])
                    & found.breadth_dir.ge(profile["breadth"]))
        else:
            raise ValueError(family)
    output = found.loc[mask].copy()
    output["family"] = family
    output["profile"] = profile["level"]
    output["score"] = (
        output.factor_ret20_dir.abs() + output.ret20_dir.abs()
        + output.last_hour_dir.clip(lower=0)
        + .01 * output.cross_z5_dir.abs())
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
        100 * hits / len(rows), 4) if rows else 0.0
    result["economic_gate"] = (economic_gate(
        [float(row["signed_return"]) for row in rows],
        float(result["trades_per_market_day"])) if rows else {
            "n": 0, "observed_trades_per_day": 0.0,
            "projected_monthly_return_pct": 0.0,
            "frequency_pass": False, "return_pass": False, "passes": False,
        })
    return result


def dev_ok(value: dict, annual: dict) -> bool:
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


def required_hit_rate(target: float, stop: float) -> float | None:
    value = minimum_binary_hit_rate_pct(target, stop)
    return round(value, 6) if value is not None else None


def main() -> None:
    candles, sources = load()
    features, metadata = build_features(candles)
    routes, development_frontier, validation_failures = [], [], []
    for profile, exit_spec in itertools.product(PROFILES, EXITS):
        proposed = entries(features, profile)
        if proposed.empty:
            continue
        resolved = base.add_outcomes(
            proposed, candles, metadata, exit_spec["target"],
            exit_spec["stop"], exit_spec["sessions"])
        resolved["module"] = f"stock_factor_{profile['family']}"
        route_id = f"{profile['family']}_{profile['level']}_{exit_spec['name']}"
        development = evaluate(resolved, features, DEV_YEARS)
        annual_dev = {str(year): evaluate(resolved, features, (year,))
                      for year in DEV_YEARS}
        item = {
            "route_id": route_id, "profile": profile, "exit": exit_spec,
            "development_2017_2021": development,
            "development_years": annual_dev,
        }
        development_frontier.append(item)
        if not dev_ok(development, annual_dev):
            continue
        validation = evaluate(resolved, features, VALIDATION_YEARS)
        annual_validation = {
            str(year): evaluate(resolved, features, (year,))
            for year in VALIDATION_YEARS}
        item["validation_2022_2023"] = validation
        item["validation_years"] = annual_validation
        if not validation_ok(validation, annual_validation):
            validation_failures.append(item)
            continue
        item["score"] = round(
            validation["economic_gate"]["projected_monthly_return_pct"]
            - 2 * validation["dd_pct"], 8)
        item["_rows"] = resolved
        routes.append(item)
    routes.sort(key=lambda item: item["score"], reverse=True)
    frozen = routes[0] if routes else None
    test_2024 = confirmation_2025 = reveal_2026 = None
    if frozen:
        test_2024 = evaluate(frozen["_rows"], features, (2024,))
        if hidden_ok(test_2024):
            confirmation_2025 = evaluate(frozen["_rows"], features, (2025,))
        if confirmation_2025 and hidden_ok(confirmation_2025):
            reveal_2026 = evaluate(frozen["_rows"], features, (2026,))
    development_frontier.sort(key=lambda item: (
        item["development_2017_2021"]["economic_gate"]["projected_monthly_return_pct"],
        item["development_2017_2021"]["return_pct"]), reverse=True)
    validation_failures.sort(key=lambda item: (
        item["validation_2022_2023"]["economic_gate"]["projected_monthly_return_pct"],
        item["validation_2022_2023"]["return_pct"]), reverse=True)

    def public(item):
        return ({key: value for key, value in item.items() if key != "_rows"}
                if item else None)

    status = ("RESEARCH_CANDIDATE_AWAITING_XPERP_FORWARD"
              if reveal_2026 and hidden_ok(reveal_2026) else "REJECTED")
    report = {
        "status": status,
        "protocol": {
            "mechanism": "QQQ minus SPY technology-factor regime",
            "families": sorted({profile["family"] for profile in PROFILES}),
            "profiles": len(PROFILES), "exits": EXITS,
            "entry": "next regular-session market bar",
            "fit_or_model": "none; predeclared causal rules",
            "development": "2017-2021", "validation": "2022-2023",
            "test": "2024", "confirmation": "2025",
            "untouched_reveal": "2026",
            "money": "3% current-equity margin, fixed 10x leverage",
            "fees": "excluded by user instruction",
        },
        "rows": {"features": len(features)},
        "minimum_ideal_hit_rate_for_30pct_month_at_3_trades_day": {
            spec["name"]: required_hit_rate(spec["target"], spec["stop"])
            for spec in EXITS
        },
        "qualified_before_2024": len(routes),
        "frozen_before_2024": public(frozen),
        "test_2024": test_2024,
        "confirmation_2025": confirmation_2025,
        "untouched_2026": reveal_2026,
        "validation_failures": [public(item) for item in validation_failures[:12]],
        "development_frontier": [public(item) for item in development_frontier[:12]],
        "sources": {**sources, str(Path(__file__)): digest(Path(__file__))},
        "limitations": [
            "Dukascopy cash CFDs establish structural history, not X-Perp fills.",
            "Only seven technology stocks are available in every evaluation year.",
            "Any historical survivor still requires untouched direct X-Perp forward evidence.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": status, "rows": report["rows"],
        "qualified_before_2024": len(routes),
        "frozen_before_2024": public(frozen),
        "test_2024": test_2024,
        "confirmation_2025": confirmation_2025,
        "untouched_2026": reveal_2026,
        "best_development": report["development_frontier"][:3],
        "best_validation_failures": report["validation_failures"][:3],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
