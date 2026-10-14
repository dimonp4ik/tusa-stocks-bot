"""Prespecified two-session refinement of the causal VWAP structure audit."""
from __future__ import annotations

import collections
import hashlib
import json
import pickle
from pathlib import Path

import stock_vwap_structure_oct15_audit as base


OUT = base.ROOT / "stock_vwap_multisession_oct15.json"
TARGETS = (.5, 1.0)
STOPS = (.75, 1.0)


def main() -> None:
    manifest_raw = base.MANIFEST.read_bytes()
    manifest = json.loads(manifest_raw)
    candles, sources = {}, {}
    for symbol in (*base.TRADED, "QQQUSDT"):
        meta = manifest["symbols"][symbol]
        path = Path(meta["path"])
        digest = base.sha(path)
        if digest != meta["sha256"]:
            raise ValueError(f"{symbol}: source hash mismatch")
        candles[symbol] = pickle.loads(path.read_bytes())
        sources[symbol] = {"path": str(path), "sha256": digest,
                           "rows": len(candles[symbol]["time"])}

    market = base.qqq_context(candles["QQQUSDT"])
    raw, market_days = [], collections.defaultdict(set)
    for symbol in base.TRADED:
        found = base.candidates(symbol, candles[symbol], market)
        raw.extend(found)
        for row in found:
            market_days[row["year"]].add(row["session_date"])
    resolved = [base.resolve(row, candles[row["symbol"]], target, stop, 2)
                for row in raw for target in TARGETS for stop in STOPS]

    frontier, qualified = [], []
    for family in base.FAMILIES:
        for deviation in base.DEVIATIONS:
            for rvol in base.RVOLS:
                for market_filter in base.MARKET_FILTERS:
                    for target in TARGETS:
                        for stop in STOPS:
                            config = {"family": family, "deviation": deviation,
                                      "rvol": rvol, "market": market_filter,
                                      "target": target, "stop": stop,
                                      "hold_sessions": 2}
                            matching = [row for row in resolved
                                        if row["target_pct"] == target
                                        and row["stop_pct"] == stop]
                            accepted = base.select(matching, config)
                            development = base.period_metrics(
                                accepted, base.DEVELOPMENT, market_days)
                            annual = {str(year): base.period_metrics(
                                          accepted, (year,), market_days)
                                      for year in base.DEVELOPMENT}
                            item = {"config": config,
                                    "development_2017_2021": development,
                                    "development_annual": annual}
                            frontier.append(item)
                            if base.development_ok(development, annual):
                                validation = base.period_metrics(
                                    accepted, base.VALIDATION, market_days)
                                validation_annual = {
                                    str(year): base.period_metrics(
                                        accepted, (year,), market_days)
                                    for year in base.VALIDATION}
                                item["validation_2022_2023"] = validation
                                item["validation_annual"] = validation_annual
                                if base.validation_ok(validation,
                                                      validation_annual):
                                    item["_accepted"] = accepted
                                    qualified.append(item)

    frontier.sort(key=lambda item: (
        item["development_2017_2021"]["return_pct"],
        item["development_2017_2021"]["literal_target_hit_rate"],
        -item["development_2017_2021"]["dd_pct"]), reverse=True)
    qualified.sort(key=lambda item: (
        item["validation_2022_2023"]["return_pct"],
        item["validation_2022_2023"]["literal_target_hit_rate"]), reverse=True)
    frozen = qualified[0] if qualified else None
    frozen_accepted = frozen.get("_accepted") if frozen else None
    for item in qualified:
        item.pop("_accepted", None)
    sealed = objective = None
    if frozen:
        accepted = frozen_accepted
        sealed = {str(year): base.period_metrics(
                      accepted, (year,), market_days)
                  for year in base.SEALED}
        combined = base.period_metrics(accepted, base.SEALED, market_days)
        sealed["combined_2024_2026"] = combined
        objective = bool(
            combined["literal_target_hit_rate"] >= 75
            and combined["dd_pct"] <= 5
            and combined["negative_months"] == 0
            and combined["trades_per_market_day"] >= 3
            and combined["months_ge_30pct"] == combined["months"])

    hit_frontier = sorted(frontier, key=lambda item: (
        item["development_2017_2021"]["literal_target_hit_rate"],
        item["development_2017_2021"]["return_pct"]), reverse=True)
    report = {
        "status": ("RESEARCH_CANDIDATE_AWAITING_DIRECT_TRANSFER"
                   if objective else "REJECTED"),
        "decision": ("No runtime change. The two-session refinement failed."
                     if not objective else
                     "Do not enable. First verify on direct OKX X-Perp data."),
        "protocol": {
            "parent": str(base.OUT),
            "reason": "One-session MFE was near 0.5%; test slower continuation.",
            "development": "2017-2021", "validation": "2022-2023",
            "sealed_test": "2024-2026; revealed only after both earlier gates",
            "entry": "next 15-minute regular-session bar open (market proxy)",
            "maximum_hold": "remainder of signal day plus next regular session",
            "targets_pct": TARGETS, "stops_pct": STOPS,
            "ambiguity": "stop first if one bar touches stop and target",
            "position_model": "3% current equity margin, fixed 10x gross",
            "fees": "excluded at user request",
        },
        "counts": {"raw_candidates": len(raw), "resolved_rows": len(resolved),
                   "grid": len(frontier),
                   "development_qualified": sum(
                       "validation_2022_2023" in item for item in frontier),
                   "validation_qualified": len(qualified)},
        "frozen": frozen, "sealed_results": sealed,
        "objective_passed": objective,
        "top_development_frontier": frontier[:20],
        "top_development_hit_rate": hit_frontier[:20],
        "source_manifest": {
            "path": str(base.MANIFEST),
            "sha256": hashlib.sha256(manifest_raw).hexdigest()},
        "sources": sources,
        "limitations": [
            "Dukascopy cash-session tick volume is not OKX X-Perp volume.",
            "Overnight bars are excluded; gaps enter the next session at its open.",
            "A historical pass would still require untouched direct X-Perp transfer.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "status", "counts", "frozen", "sealed_results")}, indent=2))


if __name__ == "__main__":
    main()
