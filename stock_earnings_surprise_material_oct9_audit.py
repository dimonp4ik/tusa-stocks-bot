"""Literal stock moves after independently matched EPS surprises."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

import stock_15m_multisession_regime_oct6_audit as base
import stock_earnings_postrelease_material_oct9_audit as event_source
from src.research_profit_gate import economic_gate, minimum_binary_hit_rate_pct


ROOT = Path("reports/audit_2026_09_08")
SURPRISE_PATH = ROOT / "stock_yahoo_earnings_surprise_history_oct9.json"
OUT = ROOT / "stock_earnings_surprise_material_oct9.json"
FAMILIES = ("surprise_continuation", "surprise_underreaction",
            "contradiction_reclaim", "overreaction_reversal",
            "next_session_surprise_drift")
PROFILES = ("broad", "strict")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def attach_surprises(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    report = json.loads(SURPRISE_PATH.read_text(encoding="utf-8"))
    rows = pd.DataFrame(report["earnings"])
    rows = rows.dropna(subset=["surprise_pct"]).copy()
    rows["symbol"] = rows.symbol + "USDT"
    rows = rows.rename(columns={"announcement_date": "yahoo_date"})
    rows = rows[["symbol", "yahoo_date", "surprise_pct", "eps_estimate",
                 "eps_actual", "event_time_utc"]].drop_duplicates(
                     ["symbol", "yahoo_date"])
    merged = frame.merge(
        rows, left_on=["symbol", "announcement_date"],
        right_on=["symbol", "yahoo_date"], how="inner", validate="many_to_one")
    diagnostics = {
        "yahoo_records_with_surprise": len(rows),
        "matched_announcements": merged[
            ["symbol", "announcement_date"]].drop_duplicates().shape[0],
        "matched_by_symbol": merged[
            ["symbol", "announcement_date"]].drop_duplicates().groupby(
                "symbol").size().to_dict(),
    }
    return merged, diagnostics


def build_proposals(frame: pd.DataFrame) -> pd.DataFrame:
    pieces = []

    def add(family: str, profile: str, direction: pd.Series,
            condition: pd.Series, score: pd.Series) -> None:
        chosen = condition & direction.ne(0)
        part = frame.loc[chosen].copy()
        part["direction_n"] = direction.loc[chosen].astype(np.int8)
        part["family"], part["profile"] = family, profile
        part["score"] = score.loc[chosen].astype(float)
        pieces.append(part)

    surprise_direction = np.sign(frame.surprise_pct)
    for profile in PROFILES:
        strict = profile == "strict"
        direction = surprise_direction
        breadth = frame.breadth_intra.where(direction > 0, 1-frame.breadth_intra)
        add("surprise_continuation", profile, direction,
            frame.earnings_offset.eq(0) & frame.slot.isin(("early", "middle"))
            & frame.surprise_pct.abs().ge(10 if strict else 5)
            & (frame.gap*direction).ge(.01 if strict else 0)
            & (frame.intraday*direction).ge(.005 if strict else .002)
            & breadth.ge(.625 if strict else .50),
            frame.surprise_pct.abs()/100 + frame.gap*direction
            + frame.intraday*direction + breadth/100)

        direction = surprise_direction
        add("surprise_underreaction", profile, direction,
            frame.earnings_offset.eq(0) & frame.slot.isin(("middle", "late"))
            & frame.surprise_pct.abs().ge(15 if strict else 8)
            & frame.gap.abs().le(.01 if strict else .015)
            & (frame.intraday*direction).ge(.008 if strict else .004)
            & (frame.last_hour*direction).gt(0),
            frame.surprise_pct.abs()/100 + frame.intraday*direction
            + frame.last_hour*direction)

        direction = surprise_direction
        add("contradiction_reclaim", profile, direction,
            frame.earnings_offset.eq(0) & frame.slot.isin(("middle", "late"))
            & frame.surprise_pct.abs().ge(15 if strict else 8)
            & (frame.gap*direction).le(-.02 if strict else -.01)
            & (frame.intraday*direction).ge(.012 if strict else .006)
            & (frame.last_hour*direction).gt(0),
            frame.surprise_pct.abs()/100 - frame.gap*direction
            + frame.intraday*direction)

        direction = -surprise_direction
        add("overreaction_reversal", profile, direction,
            frame.earnings_offset.eq(0) & frame.slot.isin(("middle", "late"))
            & frame.surprise_pct.abs().ge(10 if strict else 5)
            & (frame.gap*surprise_direction).ge(.06 if strict else .04)
            & (frame.intraday*direction).ge(.015 if strict else .008)
            & (frame.last_hour*direction).gt(0),
            frame.surprise_pct.abs()/100 + frame.gap*surprise_direction
            + frame.intraday*direction)

        direction = surprise_direction
        breadth = frame.breadth_intra.where(direction > 0, 1-frame.breadth_intra)
        add("next_session_surprise_drift", profile, direction,
            frame.earnings_offset.eq(1) & frame.slot.isin(("early", "middle"))
            & frame.surprise_pct.abs().ge(10 if strict else 5)
            & (frame.event_total*direction).ge(.04 if strict else .02)
            & (frame.intraday*direction).ge(.003 if strict else 0)
            & (frame.last_hour*direction).gt(0)
            & breadth.ge(.625 if strict else .50),
            frame.surprise_pct.abs()/100 + frame.event_total*direction
            + frame.intraday*direction + breadth/100)
    return pd.concat(pieces, ignore_index=True) if pieces else frame.iloc[:0].copy()


def route(proposals: pd.DataFrame, family: str, profile: str) -> pd.DataFrame:
    selected = proposals[proposals.profile.eq(profile)]
    if family != "all_regimes":
        selected = selected[selected.family.eq(family)]
    return selected.sort_values("score", ascending=False).drop_duplicates(
        ["symbol", "entry_time", "direction_n"])


def evaluate(frame: pd.DataFrame, features: pd.DataFrame,
             years: tuple[int, ...]) -> dict:
    selected = frame[frame.year.isin(years)].copy()
    market_days = features[features.year.isin(years)].session_date.nunique()
    result = base.account_metrics(selected, market_days)
    accepted = base.portfolio_gate(
        selected.to_dict("records"), None,
        rank_key=lambda row: (-float(row["score"]), str(row["symbol"])))
    hits = sum(int(row["outcome_code"]) == 1 for row in accepted)
    result["literal_target_hits"] = hits
    result["literal_target_hit_rate"] = round(
        100*hits/len(accepted), 4) if accepted else 0.0
    result["mean_signed_move_pct"] = (
        round(100*float(np.mean([row["signed_return"] for row in accepted])), 6)
        if accepted else 0.0)
    result["economic_gate"] = economic_gate(
        [float(row["signed_return"]) for row in accepted],
        float(result["trades_per_market_day"])) if accepted else {
            "n": 0, "observed_trades_per_day": 0.0,
            "projected_monthly_return_pct": 0.0,
            "frequency_pass": False, "return_pass": False, "passes": False}
    return result


def development_ok(value: dict, annual: dict) -> bool:
    return bool(value["n"] >= 15 and value["return_pct"] > 0
                and value["dd_pct"] <= 6
                and value["literal_target_hit_rate"] >= 55
                and sum(item["return_pct"] > 0 for item in annual.values()) >= 4)


def validation_ok(value: dict, annual: dict) -> bool:
    return bool(value["n"] >= 8 and value["return_pct"] > 0
                and value["dd_pct"] <= 5
                and value["literal_target_hit_rate"] >= 55
                and all(item["return_pct"] > 0 for item in annual.values()))


def hidden_ok(value: dict) -> bool:
    return bool(value["n"] >= 4 and value["return_pct"] > 0
                and value["dd_pct"] <= 4 and value["negative_months"] == 0
                and value["literal_target_hit_rate"] >= 60)


def public(item: dict | None) -> dict | None:
    return ({key: value for key, value in item.items() if key != "_rows"}
            if item else None)


def main() -> None:
    candles, sources = base.load_history()
    raw_features, metadata = base.build_features(candles)
    features = raw_features[raw_features.symbol.isin(event_source.SYMBOLS)].copy()
    announcements, _ = event_source.load_events()
    event_frame, event_diagnostics = event_source.attach_events(
        features, candles, announcements)
    enriched, surprise_diagnostics = attach_surprises(event_frame)
    proposals = build_proposals(enriched)
    families = (*FAMILIES, "all_regimes")
    frontier, survivors, validation_failures = [], [], []
    for (family, profile), exit_spec in itertools.product(
            itertools.product(families, PROFILES), event_source.EXITS):
        proposed = route(proposals, family, profile)
        if proposed.empty:
            continue
        resolved = base.add_outcomes(
            proposed, candles, metadata, exit_spec["target"],
            exit_spec["stop"], exit_spec["sessions"])
        resolved["module"] = f"earnings_surprise_{family}"
        development = evaluate(resolved, features, event_source.DEV_YEARS)
        annual_dev = {str(year): evaluate(resolved, features, (year,))
                      for year in event_source.DEV_YEARS}
        item = {"family": family, "profile": profile, "exit": exit_spec,
                "development_2017_2021": development,
                "development_years": annual_dev}
        frontier.append(item)
        if not development_ok(development, annual_dev):
            continue
        validation = evaluate(resolved, features, event_source.VALIDATION_YEARS)
        annual_validation = {str(year): evaluate(resolved, features, (year,))
                             for year in event_source.VALIDATION_YEARS}
        item["validation_2022_2023"] = validation
        item["validation_years"] = annual_validation
        if not validation_ok(validation, annual_validation):
            validation_failures.append(item)
            continue
        item["score"] = round(validation["return_pct"]-2*validation["dd_pct"], 8)
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
        item["development_2017_2021"]["literal_target_hit_rate"],
        item["development_2017_2021"]["return_pct"]), reverse=True)
    validation_failures.sort(key=lambda item: (
        item["validation_2022_2023"]["return_pct"],
        -item["validation_2022_2023"]["dd_pct"]), reverse=True)
    status = ("RESEARCH_SUPPLEMENT_AWAITING_2025_UPDATE"
              if confirmation_2025 and hidden_ok(confirmation_2025)
              else "REJECTED")
    report = {
        "status": status,
        "protocol": {
            "mechanism": "Yahoo EPS surprise exactly matched to SEC 8-K announcement date",
            "causal_availability": "first complete regular session strictly after SEC announcement_date",
            "families": list(FAMILIES), "profiles": list(PROFILES),
            "exits": list(event_source.EXITS), "development": "2017-2021",
            "validation": "2022-2023", "test": "2024",
            "confirmation": "2025 only if test passes; Yahoo endpoint is stale after May",
            "money": "3% current-equity margin, fixed 10x leverage",
            "fees": "excluded by user instruction",
        },
        "event_coverage": event_diagnostics,
        "surprise_coverage": surprise_diagnostics,
        "rows": {"base_features": len(features), "matched_event_rows": len(enriched),
                 "semantic_proposals": len(proposals)},
        "minimum_ideal_hit_rate_for_30pct_month_at_3_trades_day": {
            item["name"]: round(minimum_binary_hit_rate_pct(
                item["target"], item["stop"]), 6)
            if minimum_binary_hit_rate_pct(item["target"], item["stop"]) is not None
            else None for item in event_source.EXITS},
        "qualified_supplements_before_2024": len(survivors),
        "frozen_before_2024": public(frozen), "test_2024": test_2024,
        "confirmation_2025": confirmation_2025,
        "validation_failures": [public(item) for item in validation_failures[:12]],
        "development_frontier": [public(item) for item in frontier[:18]],
        "sources": {**sources, str(event_source.EVENTS_PATH): digest(
            event_source.EVENTS_PATH), str(SURPRISE_PATH): digest(SURPRISE_PATH),
                    str(Path(__file__)): digest(Path(__file__))},
        "limitations": [
            "Yahoo visualization earnings stopped updating during 2025.",
            "Exact-date matching excludes inconsistent or extra SEC filing dates.",
            "A stable sparse supplement cannot satisfy whole-bot frequency alone.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": status, "surprise_coverage": surprise_diagnostics,
        "rows": report["rows"],
        "qualified_supplements_before_2024": len(survivors),
        "frozen_before_2024": report["frozen_before_2024"],
        "test_2024": test_2024, "confirmation_2025": confirmation_2025,
        "best_development": report["development_frontier"][:6],
        "best_validation_failures": report["validation_failures"][:4],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
