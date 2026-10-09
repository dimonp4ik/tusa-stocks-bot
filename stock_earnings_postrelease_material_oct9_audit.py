"""Causal post-earnings regimes for literal 1-2% stock moves.

Announcement date D is intentionally made available only on the first complete
regular session strictly after D.  Provisional SEC filing-time/session fields
are ignored.  MRVL is excluded because the source begins only in 2021 for it;
the remaining six symbols have comparable 2017-2026 coverage.
"""
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
EVENTS_PATH = ROOT / "stock_sec_earnings_history_oct9.json"
OUT = ROOT / "stock_earnings_postrelease_material_oct9.json"
SYMBOLS = ("AAPLUSDT", "AMZNUSDT", "METAUSDT", "MSFTUSDT", "MUUSDT",
           "NVDAUSDT")
FAMILIES = ("gap_continuation", "gap_absorption", "next_session_drift",
            "next_session_reversal", "muted_open_breakout")
PROFILES = ("broad", "strict")
EXITS = (
    {"name": "one_percent", "target": 1.0, "stop": .75, "sessions": 1},
    {"name": "material", "target": 1.5, "stop": .75, "sessions": 2},
    {"name": "runner", "target": 2.0, "stop": 1.0, "sessions": 3},
)
DEV_YEARS = (2017, 2018, 2019, 2020, 2021)
VALIDATION_YEARS = (2022, 2023)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_events() -> tuple[dict[str, list[str]], dict]:
    report = json.loads(EVENTS_PATH.read_text(encoding="utf-8"))
    output = {}
    for symbol in SYMBOLS:
        ticker = symbol.removesuffix("USDT")
        dates = sorted({row["announcement_date"] for row in report["earnings"]
                        if row["ticker"].upper() == ticker
                        and row.get("date_uncertain") != "yes"})
        output[symbol] = dates
    return output, report


def complete_session_dates(symbol: str, candle: dict) -> list[str]:
    groups = base.regular_groups(candle)
    lengths = [len(value) for value in groups.values()]
    expected = int(round(float(np.median(lengths))))
    return [day for day, bars in sorted(groups.items())
            if len(bars) >= int(np.floor(.9 * expected))]


def attach_events(features: pd.DataFrame, candles: dict,
                  announcements: dict[str, list[str]]) -> tuple[pd.DataFrame, dict]:
    pieces, diagnostics = [], {}
    for symbol in SYMBOLS:
        sessions = complete_session_dates(symbol, candles[symbol])
        session_set = set(sessions)
        mappings = []
        for announcement in announcements[symbol]:
            future = [day for day in sessions if day > announcement]
            if len(future) < 3:
                continue
            event_session = future[0]
            mappings.extend([
                {"session_date": future[offset], "earnings_offset": offset,
                 "announcement_date": announcement,
                 "event_session": event_session}
                for offset in range(3)
            ])
        mapping = pd.DataFrame(mappings).drop_duplicates(
            ["session_date", "earnings_offset"])
        part = features[features.symbol.eq(symbol)].merge(
            mapping, on="session_date", how="inner", validate="many_to_one")
        event_rows = part[part.earnings_offset.eq(0)][
            ["event_session", "slot", "gap", "intraday", "last_hour"]
        ].rename(columns={
            "gap": "event_gap", "intraday": "event_intraday",
            "last_hour": "event_last_hour"})
        # The late checkpoint approximates the completed event-session move and
        # is attached only to later sessions; same-session rows keep their live
        # checkpoint values.
        late = event_rows[event_rows.slot.eq("late")].drop(columns="slot")
        part = part.merge(late, on="event_session", how="left",
                          validate="many_to_one")
        part["event_total"] = part.event_gap + part.event_intraday
        same = part.earnings_offset.eq(0)
        part.loc[same, "event_gap"] = part.loc[same, "gap"]
        part.loc[same, "event_intraday"] = part.loc[same, "intraday"]
        part.loc[same, "event_last_hour"] = part.loc[same, "last_hour"]
        part.loc[same, "event_total"] = (
            part.loc[same, "gap"] + part.loc[same, "intraday"])
        pieces.append(part)
        diagnostics[symbol] = {
            "announcements": len(announcements[symbol]),
            "mapped_events": mapping.event_session.nunique(),
            "first": min(announcements[symbol]), "last": max(announcements[symbol]),
            "complete_sessions": len(session_set),
        }
    result = pd.concat(pieces, ignore_index=True)
    return result.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["event_gap", "event_intraday", "event_total"]), diagnostics


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

    for profile in PROFILES:
        strict = profile == "strict"
        direction = np.sign(frame.event_gap)
        breadth = frame.breadth_intra.where(direction > 0, 1-frame.breadth_intra)
        qqq = frame.qqq_intraday * direction
        add("gap_continuation", profile, direction,
            frame.earnings_offset.eq(0) & frame.slot.isin(("early", "middle"))
            & frame.event_gap.abs().ge(.03 if strict else .015)
            & (frame.intraday*direction).ge(.006 if strict else .002)
            & qqq.ge(0 if strict else -.003)
            & breadth.ge(.625 if strict else .50),
            frame.event_gap.abs() + (frame.intraday*direction) + breadth/100)

        direction = -np.sign(frame.event_gap)
        add("gap_absorption", profile, direction,
            frame.earnings_offset.eq(0) & frame.slot.isin(("middle", "late"))
            & frame.event_gap.abs().ge(.04 if strict else .02)
            & (frame.intraday*direction).ge(.012 if strict else .005)
            & (frame.last_hour*direction).gt(0),
            frame.event_gap.abs() + (frame.intraday*direction)
            + (frame.last_hour*direction))

        direction = np.sign(frame.event_total)
        breadth = frame.breadth_intra.where(direction > 0, 1-frame.breadth_intra)
        add("next_session_drift", profile, direction,
            frame.earnings_offset.eq(1) & frame.slot.isin(("early", "middle"))
            & frame.event_total.abs().ge(.04 if strict else .02)
            & (frame.intraday*direction).ge(.003 if strict else 0)
            & (frame.last_hour*direction).gt(0)
            & breadth.ge(.625 if strict else .50),
            frame.event_total.abs() + (frame.intraday*direction) + breadth/100)

        direction = -np.sign(frame.event_total)
        add("next_session_reversal", profile, direction,
            frame.earnings_offset.eq(1) & frame.slot.isin(("middle", "late"))
            & frame.event_total.abs().ge(.06 if strict else .04)
            & (frame.gap*direction).ge(.008 if strict else 0)
            & (frame.intraday*direction).ge(.006 if strict else .002),
            frame.event_total.abs() + (frame.gap*direction)
            + (frame.intraday*direction))

        direction = np.sign(frame.intraday)
        breadth = frame.breadth_intra.where(direction > 0, 1-frame.breadth_intra)
        add("muted_open_breakout", profile, direction,
            frame.earnings_offset.eq(0) & frame.slot.isin(("middle", "late"))
            & frame.event_gap.abs().le(.008 if strict else .012)
            & frame.intraday.abs().ge(.015 if strict else .008)
            & (frame.last_hour*direction).gt(0)
            & breadth.ge(.625 if strict else .50),
            frame.intraday.abs() + (frame.last_hour*direction) + breadth/100)

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
        round(100*float(np.mean(
            [row["signed_return"] for row in accepted])), 6)
        if accepted else 0.0)
    result["economic_gate"] = economic_gate(
        [float(row["signed_return"]) for row in accepted],
        float(result["trades_per_market_day"])) if accepted else {
            "n": 0, "observed_trades_per_day": 0.0,
            "projected_monthly_return_pct": 0.0,
            "frequency_pass": False, "return_pass": False, "passes": False}
    return result


def dev_ok(value: dict, annual: dict) -> bool:
    return bool(value["n"] >= 30 and value["return_pct"] > 0
                and value["dd_pct"] <= 8
                and value["literal_target_hit_rate"] >= 50
                and sum(item["return_pct"] > 0 for item in annual.values()) >= 4)


def validation_ok(value: dict, annual: dict) -> bool:
    return bool(value["n"] >= 15 and value["return_pct"] > 0
                and value["dd_pct"] <= 6
                and value["literal_target_hit_rate"] >= 50
                and all(item["return_pct"] > 0 for item in annual.values()))


def hidden_ok(value: dict) -> bool:
    return bool(value["n"] >= 8 and value["return_pct"] > 0
                and value["dd_pct"] <= 5 and value["negative_months"] == 0
                and value["literal_target_hit_rate"] >= 60)


def public(item: dict | None) -> dict | None:
    return ({key: value for key, value in item.items() if key != "_rows"}
            if item else None)


def main() -> None:
    candles, sources = base.load_history()
    raw_features, metadata = base.build_features(candles)
    features = raw_features[raw_features.symbol.isin(SYMBOLS)].copy()
    announcements, event_report = load_events()
    enriched, diagnostics = attach_events(features, candles, announcements)
    proposals = build_proposals(enriched)
    families = (*FAMILIES, "all_regimes")
    frontier, survivors, validation_failures = [], [], []
    for (family, profile), exit_spec in itertools.product(
            itertools.product(families, PROFILES), EXITS):
        proposed = route(proposals, family, profile)
        if proposed.empty:
            continue
        resolved = base.add_outcomes(
            proposed, candles, metadata, exit_spec["target"],
            exit_spec["stop"], exit_spec["sessions"])
        resolved["module"] = f"earnings_{family}"
        development = evaluate(resolved, features, DEV_YEARS)
        annual_dev = {str(year): evaluate(resolved, features, (year,))
                      for year in DEV_YEARS}
        item = {"family": family, "profile": profile, "exit": exit_spec,
                "development_2017_2021": development,
                "development_years": annual_dev}
        frontier.append(item)
        if not dev_ok(development, annual_dev):
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
            validation["return_pct"] - 2*validation["dd_pct"], 8)
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
        item["development_2017_2021"]["literal_target_hit_rate"],
        item["development_2017_2021"]["return_pct"]), reverse=True)
    validation_failures.sort(key=lambda item: (
        item["validation_2022_2023"]["return_pct"],
        -item["validation_2022_2023"]["dd_pct"]), reverse=True)
    status = ("RESEARCH_SUPPLEMENT_AWAITING_DIRECT_FORWARD"
              if reveal_2026 and hidden_ok(reveal_2026) else "REJECTED")
    report = {
        "status": status,
        "protocol": {
            "mechanism": "post-earnings reaction from SEC 8-K Item 2.02 dates",
            "causal_availability": "first complete regular session strictly after announcement_date",
            "ignored_source_fields": ["session", "time_et"],
            "families": list(FAMILIES), "profiles": list(PROFILES),
            "exits": list(EXITS), "development": "2017-2021",
            "validation": "2022-2023", "test": "2024",
            "confirmation": "2025", "untouched_reveal": "2026",
            "money": "3% current-equity margin, fixed 10x leverage",
            "fees": "excluded by user instruction",
        },
        "coverage": diagnostics,
        "excluded_incomplete_symbol": "MRVLUSDT: SEC compilation begins in 2021",
        "rows": {"base_features": len(features), "earnings_enriched": len(enriched),
                 "semantic_proposals": len(proposals)},
        "minimum_ideal_hit_rate_for_30pct_month_at_3_trades_day": {
            item["name"]: round(minimum_binary_hit_rate_pct(
                item["target"], item["stop"], market_days=22), 6)
            if minimum_binary_hit_rate_pct(
                item["target"], item["stop"], market_days=22) is not None else None
            for item in EXITS},
        "qualified_supplements_before_2024": len(survivors),
        "frozen_before_2024": public(frozen), "test_2024": test_2024,
        "confirmation_2025": confirmation_2025, "untouched_2026": reveal_2026,
        "validation_failures": [public(item) for item in validation_failures[:12]],
        "development_frontier": [public(item) for item in frontier[:18]],
        "sources": {**sources, str(EVENTS_PATH): digest(EVENTS_PATH),
                    str(Path(__file__)): digest(Path(__file__))},
        "source_archive": {key: event_report[key] for key in (
            "provider", "source", "source_sha256", "license", "records")},
        "limitations": [
            "SEC filing dates identify releases but contain no historical EPS surprise value.",
            "The conservative next-session rule may miss the largest initial reaction.",
            "A stable sparse supplement cannot satisfy the whole-bot frequency target alone.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": status, "coverage": diagnostics, "rows": report["rows"],
        "qualified_supplements_before_2024": len(survivors),
        "frozen_before_2024": report["frozen_before_2024"],
        "test_2024": test_2024, "confirmation_2025": confirmation_2025,
        "untouched_2026": reveal_2026,
        "best_development": report["development_frontier"][:6],
        "best_validation_failures": report["validation_failures"][:4],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
