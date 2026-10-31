"""Chronological audit of non-earnings SEC 8-K event regimes.

Only official SEC acceptance timestamps are used.  Every event is delayed to
the next complete regular session, so neither filing-day price action nor a
future event can leak into an entry.  Item 2.02 earnings filings are excluded
because they were tested by the existing earnings audit.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

import stock_earnings_postrelease_material_oct9_audit as old
from src.research_profit_gate import economic_gate, minimum_binary_hit_rate_pct


ROOT = Path("reports/audit_2026_09_08")
EVENTS_PATH = ROOT / "stock_sec_8k_history_oct31.json"
OUT = ROOT / "stock_sec_8k_material_oct31.json"
SYMBOLS = old.SYMBOLS
CATEGORIES = (
    "material_transaction", "governance", "reg_fd_or_other",
    "distress_or_accounting", "all_events",
)
FAMILIES = (
    "reaction_continuation", "reaction_absorption",
    "next_session_drift", "next_session_reversal", "muted_breakout",
)
PROFILES = ("broad", "strict")
EXITS = old.EXITS
DEV_YEARS = old.DEV_YEARS
VALIDATION_YEARS = old.VALIDATION_YEARS


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_events() -> tuple[dict[str, list[dict]], dict]:
    report = json.loads(EVENTS_PATH.read_text(encoding="utf-8"))
    output = {symbol: [] for symbol in SYMBOLS}
    for row in report["events"]:
        symbol = f"{row['symbol']}USDT"
        if symbol in output:
            output[symbol].append(row)
    for rows in output.values():
        rows.sort(key=lambda row: (
            row["acceptance_datetime_utc"], row["accession"]))
    return output, report


def attach_events(features: pd.DataFrame, candles: dict,
                  events: dict[str, list[dict]]) -> tuple[pd.DataFrame, dict]:
    pieces, diagnostics = [], {}
    for symbol in SYMBOLS:
        sessions = old.complete_session_dates(symbol, candles[symbol])
        mappings = []
        for event in events[symbol]:
            # This intentionally ignores the filing session even when a filing
            # arrived before the close.  Only a later complete session is used.
            accepted_day = event["acceptance_datetime_utc"][:10]
            future = [day for day in sessions if day > accepted_day]
            if len(future) < 2:
                continue
            for offset in range(2):
                mappings.append({
                    "session_date": future[offset],
                    "event_session": future[0],
                    "event_offset": offset,
                    "event_category": event["category"],
                    "acceptance_datetime_utc": event[
                        "acceptance_datetime_utc"],
                    "accession": event["accession"],
                })
        mapping = pd.DataFrame(mappings)
        if mapping.empty:
            continue
        # Several filings of the same category may arrive before one session.
        # They represent one observable regime, not duplicate trades.
        mapping = mapping.sort_values("acceptance_datetime_utc").drop_duplicates(
            ["session_date", "event_offset", "event_category"], keep="last")
        part = features[features.symbol.eq(symbol)].merge(
            mapping, on="session_date", how="inner", validate="many_to_many")
        event_rows = part[
            part.event_offset.eq(0) & part.slot.eq("late")
        ][["event_session", "event_category", "gap", "intraday", "last_hour"]]
        event_rows = event_rows.rename(columns={
            "gap": "event_gap", "intraday": "event_intraday",
            "last_hour": "event_last_hour",
        }).drop_duplicates(["event_session", "event_category"])
        part = part.merge(
            event_rows, on=["event_session", "event_category"], how="left",
            validate="many_to_one")
        same = part.event_offset.eq(0)
        part.loc[same, "event_gap"] = part.loc[same, "gap"]
        part.loc[same, "event_intraday"] = part.loc[same, "intraday"]
        part.loc[same, "event_last_hour"] = part.loc[same, "last_hour"]
        part["event_total"] = part.event_gap + part.event_intraday
        pieces.append(part)
        diagnostics[symbol] = {
            "official_filings": len(events[symbol]),
            "mapped_event_sessions": int(mapping.event_session.nunique()),
            "first_acceptance": events[symbol][0][
                "acceptance_datetime_utc"] if events[symbol] else None,
            "last_acceptance": events[symbol][-1][
                "acceptance_datetime_utc"] if events[symbol] else None,
        }
    result = pd.concat(pieces, ignore_index=True)
    result = result.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["event_gap", "event_intraday", "event_total"])
    return result, diagnostics


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
        add("reaction_continuation", profile, direction,
            frame.event_offset.eq(0) & frame.slot.isin(("early", "middle"))
            & frame.event_gap.abs().ge(.025 if strict else .012)
            & (frame.intraday*direction).ge(.005 if strict else .0015)
            & breadth.ge(.625 if strict else .50),
            frame.event_gap.abs() + frame.intraday*direction + breadth/100)

        direction = -np.sign(frame.event_gap)
        add("reaction_absorption", profile, direction,
            frame.event_offset.eq(0) & frame.slot.isin(("middle", "late"))
            & frame.event_gap.abs().ge(.03 if strict else .018)
            & (frame.intraday*direction).ge(.01 if strict else .004)
            & (frame.last_hour*direction).gt(0),
            frame.event_gap.abs() + frame.intraday*direction
            + frame.last_hour*direction)

        direction = np.sign(frame.event_total)
        breadth = frame.breadth_intra.where(direction > 0, 1-frame.breadth_intra)
        add("next_session_drift", profile, direction,
            frame.event_offset.eq(1) & frame.slot.isin(("early", "middle"))
            & frame.event_total.abs().ge(.035 if strict else .018)
            & (frame.intraday*direction).ge(.003 if strict else 0)
            & breadth.ge(.625 if strict else .50),
            frame.event_total.abs() + frame.intraday*direction + breadth/100)

        direction = -np.sign(frame.event_total)
        add("next_session_reversal", profile, direction,
            frame.event_offset.eq(1) & frame.slot.isin(("middle", "late"))
            & frame.event_total.abs().ge(.05 if strict else .03)
            & (frame.gap*direction).ge(.006 if strict else 0)
            & (frame.intraday*direction).ge(.005 if strict else .0015),
            frame.event_total.abs() + frame.gap*direction
            + frame.intraday*direction)

        direction = np.sign(frame.intraday)
        breadth = frame.breadth_intra.where(direction > 0, 1-frame.breadth_intra)
        add("muted_breakout", profile, direction,
            frame.event_offset.eq(0) & frame.slot.isin(("middle", "late"))
            & frame.event_gap.abs().le(.007 if strict else .012)
            & frame.intraday.abs().ge(.014 if strict else .007)
            & frame.last_hour.mul(direction).gt(0)
            & breadth.ge(.625 if strict else .50),
            frame.intraday.abs() + frame.last_hour*direction + breadth/100)
    return pd.concat(pieces, ignore_index=True) if pieces else frame.iloc[:0].copy()


def route(proposals: pd.DataFrame, category: str, family: str,
          profile: str) -> pd.DataFrame:
    selected = proposals[proposals.profile.eq(profile)]
    if category != "all_events":
        selected = selected[selected.event_category.eq(category)]
    if family != "all_regimes":
        selected = selected[selected.family.eq(family)]
    return selected.sort_values("score", ascending=False).drop_duplicates(
        ["symbol", "entry_time", "direction_n"])


def evaluate(frame: pd.DataFrame, features: pd.DataFrame,
             years: tuple[int, ...]) -> dict:
    value = old.evaluate(frame, features, years)
    selected = frame[frame.year.isin(years)].copy()
    accepted = old.base.portfolio_gate(
        selected.to_dict("records"), None,
        rank_key=lambda row: (-float(row["score"]), str(row["symbol"])))
    if accepted:
        hours = pd.Series([int(float(row["entry_time"])//3600)
                           for row in accepted]).value_counts()
        symbols = pd.Series([str(row["symbol"]) for row in accepted]).value_counts()
        value["max_entries_same_rolling_hour"] = int(hours.iloc[0])
        value["largest_symbol_share_pct"] = round(
            100*float(symbols.iloc[0])/len(accepted), 4)
        value["economic_gate"] = economic_gate(
            [float(row["signed_return"]) for row in accepted],
            float(value["trades_per_market_day"]))
    else:
        value["max_entries_same_rolling_hour"] = 0
        value["largest_symbol_share_pct"] = 0.0
        value["economic_gate"] = {
            "n": 0, "observed_trades_per_day": 0.0,
            "projected_monthly_return_pct": 0.0,
            "frequency_pass": False, "return_pass": False, "passes": False,
        }
    return value


def public(item: dict | None) -> dict | None:
    return ({key: value for key, value in item.items() if key != "_rows"}
            if item else None)


def main() -> None:
    candles, sources = old.base.load_history()
    raw_features, metadata = old.base.build_features(candles)
    features = raw_features[raw_features.symbol.isin(SYMBOLS)].copy()
    events, event_report = load_events()
    enriched, diagnostics = attach_events(features, candles, events)
    proposals = build_proposals(enriched)
    frontier, survivors, validation_failures = [], [], []
    for category, family, profile, exit_spec in itertools.product(
            CATEGORIES, (*FAMILIES, "all_regimes"), PROFILES, EXITS):
        proposed = route(proposals, category, family, profile)
        if proposed.empty:
            continue
        resolved = old.base.add_outcomes(
            proposed, candles, metadata, exit_spec["target"],
            exit_spec["stop"], exit_spec["sessions"])
        resolved["module"] = f"sec_8k_{category}_{family}"
        development = evaluate(resolved, features, DEV_YEARS)
        annual_dev = {str(year): evaluate(resolved, features, (year,))
                      for year in DEV_YEARS}
        item = {
            "category": category, "family": family, "profile": profile,
            "exit": exit_spec, "development_2017_2021": development,
            "development_years": annual_dev,
        }
        frontier.append(item)
        if not old.dev_ok(development, annual_dev):
            continue
        validation = evaluate(resolved, features, VALIDATION_YEARS)
        annual_validation = {
            str(year): evaluate(resolved, features, (year,))
            for year in VALIDATION_YEARS}
        item["validation_2022_2023"] = validation
        item["validation_years"] = annual_validation
        if not old.validation_ok(validation, annual_validation):
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
        if old.hidden_ok(test_2024):
            confirmation_2025 = evaluate(frozen["_rows"], features, (2025,))
        if confirmation_2025 and old.hidden_ok(confirmation_2025):
            reveal_2026 = evaluate(frozen["_rows"], features, (2026,))
    frontier.sort(key=lambda item: (
        item["development_2017_2021"]["economic_gate"][
            "projected_monthly_return_pct"],
        item["development_2017_2021"]["return_pct"]), reverse=True)
    validation_failures.sort(key=lambda item: (
        item["validation_2022_2023"]["return_pct"],
        -item["validation_2022_2023"]["dd_pct"]), reverse=True)
    status = ("RESEARCH_SUPPLEMENT_AWAITING_DIRECT_FORWARD"
              if reveal_2026 and old.hidden_ok(reveal_2026) else "REJECTED")
    report = {
        "status": status,
        "ready_for_live_orders": False,
        "orders_possible": False,
        "runtime_change_allowed": False,
        "protocol": {
            "mechanism": "official non-earnings SEC 8-K event reaction",
            "causal_availability": (
                "first complete regular session strictly after official SEC "
                "acceptance_datetime_utc date"),
            "excluded_items": ["2.02 earnings", "9.01 attachment index"],
            "categories": list(CATEGORIES), "families": list(FAMILIES),
            "profiles": list(PROFILES), "exits": list(EXITS),
            "development": "2017-2021", "validation": "2022-2023",
            "test": "2024", "confirmation": "2025",
            "untouched_reveal": "2026",
            "money": "3% current-equity margin, fixed 10x leverage",
            "orders": "market only", "fees": "excluded by user instruction",
        },
        "coverage": diagnostics,
        "rows": {"base_features": len(features),
                 "event_enriched": len(enriched),
                 "semantic_proposals": len(proposals)},
        "minimum_ideal_hit_rate_for_30pct_month_at_3_trades_day": {
            item["name"]: round(minimum_binary_hit_rate_pct(
                item["target"], item["stop"], market_days=22), 6)
            if minimum_binary_hit_rate_pct(
                item["target"], item["stop"], market_days=22) is not None
            else None for item in EXITS},
        "qualified_supplements_before_2024": len(survivors),
        "frozen_before_2024": public(frozen),
        "test_2024": test_2024,
        "confirmation_2025": confirmation_2025,
        "untouched_2026": reveal_2026,
        "validation_failures": [public(item)
                                for item in validation_failures[:12]],
        "development_frontier": [public(item) for item in frontier[:20]],
        "sources": {**sources, str(EVENTS_PATH): digest(EVENTS_PATH),
                    str(Path(__file__)): digest(Path(__file__))},
        "source_archive": {
            "provider": event_report["provider"],
            "source_policy": event_report["source_policy"],
            "records": event_report["records"],
            "records_by_category": event_report["records_by_category"],
            "source_hashes": {url: value["sha256"] for url, value
                              in event_report["sources"].items()},
        },
        "limitations": [
            "SEC item codes describe event type but not sentiment or surprise magnitude.",
            "The next-session delay is conservative and misses filing-day reactions.",
            "An event supplement cannot satisfy the whole-bot frequency target alone.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True),
                   encoding="utf-8")
    print(json.dumps({
        "status": status, "coverage": diagnostics, "rows": report["rows"],
        "qualified_supplements_before_2024": len(survivors),
        "frozen_before_2024": report["frozen_before_2024"],
        "test_2024": test_2024, "confirmation_2025": confirmation_2025,
        "untouched_2026": reveal_2026,
        "best_development": report["development_frontier"][:5],
        "best_validation_failures": report["validation_failures"][:3],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
