"""Mine an interpretable stock regime router on 2020-24 and reveal 2025-26.

Every route fixes an entry family, direction, prior-day QQQ regime and cash
session segment.  At most two causal thresholds may refine a route.  The tested
years and recent X-Perp execution feed never participate in rule selection.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from filter_lab import gate, metrics
from stock_session_family_lab import LEVELS
from stock_walk_forward_filter_lab import load_external


DEVELOPMENT = (2020, 2021, 2022, 2023, 2024)
HOLDOUT = (2025, 2026)
KEYS = ("family", "direction", "qqq_regime", "session_bucket")
CONDITION_FIELDS = tuple(field for field in LEVELS
                         if field not in {"signal_minutes"})


def add_context(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["year"] = pd.to_datetime(frame["entry_time"], unit="s", utc=True).dt.year
    frame["session_bucket"] = pd.cut(
        frame["signal_minutes"], bins=(-1, 30, 90, 150, np.inf),
        labels=("00_30", "45_90", "105_150", "165_plus"),
    ).astype(str)
    return frame


def compact(rows: list[dict] | pd.DataFrame) -> dict:
    records = rows.to_dict("records") if isinstance(rows, pd.DataFrame) else rows
    value = metrics(records)
    return {key: value[key] for key in (
        "n", "wr", "wr_lower_95", "net_r", "mean_r", "pf", "dd_r",
        "active_days", "symbols")}


def fast(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {"n": 0, "wr": 0.0, "net_r": 0.0, "mean_r": 0.0,
                "pf": None, "dd_r": 0.0, "symbols": 0}
    ordered = frame.sort_values(["exit_time", "symbol"])
    values = ordered["net_r"].to_numpy(float)
    gains = values[values > 0].sum()
    losses = -values[values < 0].sum()
    curve = values.cumsum()
    peaks = np.maximum.accumulate(np.maximum(curve, 0))
    return {
        "n": int(len(values)),
        "wr": float((values > 0).mean() * 100),
        "net_r": float(values.sum()),
        "mean_r": float(values.mean()),
        "pf": float(gains / losses) if losses else None,
        "dd_r": float(np.max(peaks - curve)),
        "symbols": int(frame["symbol"].nunique()),
    }


def atoms() -> list[tuple[str, str, float]]:
    return [(field, operator, float(value))
            for field in CONDITION_FIELDS
            for value in LEVELS[field]
            for operator in ("ge", "le")]


def condition_mask(frame: pd.DataFrame, conditions: tuple[tuple, ...]) -> np.ndarray:
    mask = np.ones(len(frame), dtype=bool)
    for field, operator, value in conditions:
        actual = pd.to_numeric(frame[field], errors="coerce").to_numpy(float)
        valid = np.isfinite(actual)
        mask &= valid & (actual >= value if operator == "ge" else actual <= value)
    return mask


def rule_frame(frame: pd.DataFrame, candidate: dict) -> pd.DataFrame:
    mask = np.ones(len(frame), dtype=bool)
    for key, value in candidate["route"].items():
        mask &= frame[key].astype(str).to_numpy() == str(value)
    mask &= condition_mask(frame, tuple(tuple(value) for value in candidate["conditions"]))
    return frame.loc[mask]


def score(frame: pd.DataFrame) -> dict | None:
    aggregate = fast(frame)
    yearly = {year: fast(frame[frame["year"] == year]) for year in DEVELOPMENT}
    positive = sum(value["net_r"] > 0 for value in yearly.values())
    worst = min(value["net_r"] for value in yearly.values())
    if (aggregate["n"] < 100 or aggregate["symbols"] < 7
            or aggregate["wr"] < 82 or aggregate["mean_r"] < .025
            or (aggregate["pf"] or 0) < 1.15
            or any(value["n"] < 10 for value in yearly.values())
            or positive < 4 or worst < -2.0):
        return None
    means = [value["mean_r"] for value in yearly.values()]
    robust_score = (min(means) * 8 + float(np.quantile(means, .25)) * 5
                    + aggregate["mean_r"] * 3
                    + min(aggregate["n"], 1000) / 25_000)
    return {"aggregate": aggregate, "development_years": yearly,
            "positive_years": positive, "worst_year_r": worst,
            "robust_score": robust_score}


def mine(development: pd.DataFrame, *, strict: bool = False) -> list[dict]:
    public_atoms = atoms()
    candidates = []
    for key, base in development.groupby(list(KEYS), observed=True, sort=True):
        values = base["net_r"].to_numpy(float)
        years = base["year"].to_numpy(int)
        symbols = pd.factorize(base["symbol"])[0]

        def fast_mask(mask: np.ndarray) -> dict:
            chosen = values[mask]
            if not len(chosen):
                return {"n": 0, "wr": 0.0, "net_r": 0.0, "mean_r": 0.0,
                        "pf": None, "dd_r": 0.0, "symbols": 0}
            gains = chosen[chosen > 0].sum()
            losses = -chosen[chosen < 0].sum()
            curve = chosen.cumsum()
            peaks = np.maximum.accumulate(np.maximum(curve, 0))
            return {
                "n": int(len(chosen)), "wr": float((chosen > 0).mean() * 100),
                "net_r": float(chosen.sum()), "mean_r": float(chosen.mean()),
                "pf": float(gains / losses) if losses else None,
                "dd_r": float(np.max(peaks - curve)),
                "symbols": int(len(np.unique(symbols[mask]))),
            }

        def score_mask(mask: np.ndarray) -> dict | None:
            aggregate = fast_mask(mask)
            yearly = {year: fast_mask(mask & (years == year)) for year in DEVELOPMENT}
            positive = sum(value["net_r"] > 0 for value in yearly.values())
            worst = min(value["net_r"] for value in yearly.values())
            minimum_wr = 83 if strict else 82
            minimum_mean = .035 if strict else .025
            required_positive = 5 if strict else 4
            minimum_worst = 0 if strict else -2.0
            if (aggregate["n"] < 100 or aggregate["symbols"] < 7
                    or aggregate["wr"] < minimum_wr or aggregate["mean_r"] < minimum_mean
                    or (aggregate["pf"] or 0) < 1.15
                    or any(value["n"] < 10 for value in yearly.values())
                    or positive < required_positive or worst <= minimum_worst):
                return None
            means = [value["mean_r"] for value in yearly.values()]
            robust_score = (min(means) * 8 + float(np.quantile(means, .25)) * 5
                            + aggregate["mean_r"] * 3
                            + min(aggregate["n"], 1000) / 25_000)
            return {"aggregate": aggregate, "development_years": yearly,
                    "positive_years": positive, "worst_year_r": worst,
                    "robust_score": robust_score}

        atom_masks = {atom: condition_mask(base, (atom,)) for atom in public_atoms}
        ranked = []
        for atom, mask in atom_masks.items():
            if int(mask.sum()) < 80:
                continue
            result = fast_mask(mask)
            yearly_means = [fast_mask(mask & (years == year))["mean_r"]
                            for year in DEVELOPMENT]
            rank = min(yearly_means) * 3 + result["mean_r"]
            ranked.append((rank, atom))
        strongest = [atom for _, atom in sorted(ranked, reverse=True)[:20]]
        rules = [tuple()] + [(atom,) for atom in public_atoms]
        rules += [pair for pair in itertools.combinations(strongest, 2)
                  if pair[0][0] != pair[1][0]]
        seen = set()
        for conditions in rules:
            if not conditions:
                mask = np.ones(len(base), dtype=bool)
            elif len(conditions) == 1:
                mask = atom_masks[conditions[0]]
            else:
                mask = atom_masks[conditions[0]] & atom_masks[conditions[1]]
            signature = np.packbits(mask).tobytes()
            if signature in seen:
                continue
            seen.add(signature)
            result = score_mask(mask)
            if result:
                candidates.append({
                    "route": dict(zip(KEYS, key)),
                    "conditions": [list(value) for value in conditions],
                    "_development_row_ids": base.index.to_numpy()[mask].astype(int).tolist(),
                    **result,
                })
    return sorted(candidates,
                  key=lambda value: (value["robust_score"], value["aggregate"]["net_r"]),
                  reverse=True)


def routed_candidates(frame: pd.DataFrame, selected: list[dict]) -> pd.DataFrame:
    parts = []
    for priority, candidate in enumerate(selected):
        part = rule_frame(frame, candidate).copy()
        part["route_priority"] = priority
        part["route_id"] = priority
        parts.append(part)
    if not parts:
        return frame.iloc[0:0]
    combined = pd.concat(parts).sort_values(["route_priority", "family"])
    return combined.drop_duplicates(["symbol", "entry_time"])


def accepted(frame: pd.DataFrame, selected: list[dict], cap: int = 4) -> pd.DataFrame:
    routed = routed_candidates(frame, selected)
    if routed.empty:
        return routed
    rows = gate(routed.sort_values(["entry_time", "symbol", "route_priority"])
                .to_dict("records"), cap)
    return pd.DataFrame(rows) if rows else routed.iloc[0:0]


def portfolio_report(frame: pd.DataFrame, selected: list[dict], cap: int = 4) -> dict:
    chosen = accepted(frame, selected, cap)
    yearly = {year: fast(chosen[chosen["year"] == year]) for year in DEVELOPMENT}
    aggregate = fast(chosen)
    score_value = (sum(value["net_r"] for value in yearly.values())
                   + min(value["net_r"] for value in yearly.values()) * 4
                   + aggregate["mean_r"] * 100
                   - aggregate["dd_r"] * .25)
    return {"score": score_value, "all": aggregate, "years": yearly}


def select_portfolio(development: pd.DataFrame, candidates: list[dict], *,
                     strict: bool = False) -> list[dict]:
    pool, route_counts = [], {}
    for candidate in candidates:
        key = tuple(candidate["route"].items())
        if route_counts.get(key, 0) >= 4:
            continue
        route_counts[key] = route_counts.get(key, 0) + 1
        pool.append(candidate)
        if len(pool) >= 300:
            break
    selected = []
    current = development.iloc[0:0]
    best_score = float("-inf")
    for _ in range(16):
        choice = None
        for candidate in pool:
            if candidate in selected:
                continue
            addition = development.loc[candidate["_development_row_ids"]].copy()
            addition["route_priority"] = len(selected)
            trial_frame = pd.concat([current, addition])
            trial_frame = trial_frame.drop_duplicates(["symbol", "entry_time"], keep="first")
            aggregate = fast(trial_frame)
            years = {year: fast(trial_frame[trial_frame["year"] == year])
                     for year in DEVELOPMENT}
            score_value = (sum(value["net_r"] for value in years.values())
                           + min(value["net_r"] for value in years.values()) * 4
                           + aggregate["mean_r"] * 100 - aggregate["dd_r"] * .25)
            result = {"score": score_value, "all": aggregate, "years": years}
            years = result["years"]
            if (result["all"]["wr"] < (84 if strict else 83)
                    or min(value["wr"] for value in years.values()) < (81 if strict else 80)
                    or sum(value["net_r"] > 0 for value in years.values()) < (5 if strict else 4)
                    or min(value["net_r"] for value in years.values()) <= (0 if strict else -1.5)):
                continue
            if choice is None or result["score"] > choice[0]:
                choice = (result["score"], candidate)
        if choice is None or (selected and choice[0] <= best_score + .25):
            break
        best_score = choice[0]
        selected.append(choice[1])
        addition = development.loc[choice[1]["_development_row_ids"]].copy()
        addition["route_priority"] = len(selected) - 1
        current = pd.concat([current, addition]).drop_duplicates(
            ["symbol", "entry_time"], keep="first")
    return selected


def detailed(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {"all": compact(frame), "years": {}, "months": {},
                "negative_months": {}}
    months = pd.to_datetime(frame["entry_time"], unit="s", utc=True).dt.strftime("%Y-%m")
    monthly = {key: compact(part) for key, part in frame.groupby(months)}
    return {
        "all": compact(frame),
        "years": {str(year): compact(part) for year, part in frame.groupby("year")},
        "months": monthly,
        "negative_months": {key: value for key, value in monthly.items()
                            if value["net_r"] < -.01},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("deep_csv", type=Path)
    parser.add_argument("--external-index-dir", type=Path, required=True)
    parser.add_argument("--external-execution-dir", type=Path, required=True)
    parser.add_argument("--symbols", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    source_raw = args.deep_csv.read_bytes()
    frame = pd.read_csv(args.deep_csv)
    frame = add_context(frame[frame["target_r"] == .25])
    development = frame[frame["year"].isin(DEVELOPMENT)].copy()
    candidates = mine(development, strict=args.strict)
    print(json.dumps({"phase": "mined", "candidate_count": len(candidates)}), flush=True)
    selected = select_portfolio(development, candidates, strict=args.strict)
    print(json.dumps({"phase": "selected", "selected_count": len(selected)}), flush=True)
    historical = accepted(frame, selected)

    symbols = tuple(value.strip() for value in args.symbols.split(",") if value.strip())
    external, external_sources = load_external(
        args.external_index_dir, args.external_execution_dir, symbols)
    external = add_context(external[external["target_r"] == .25])
    external_rows = accepted(external, selected)

    public_selected = [{key: value for key, value in item.items()
                        if key in {"route", "conditions", "aggregate",
                                   "development_years", "positive_years",
                                   "worst_year_r", "robust_score"}}
                       for item in selected]
    report = {
        "status": "RESEARCH_ONLY",
        "target_r": .25,
        "stop_atr": 1,
        "strict_all_development_years_positive": args.strict,
        "development_years": list(DEVELOPMENT),
        "unseen_holdout_years": list(HOLDOUT),
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "selected": public_selected,
        "historical": detailed(historical),
        "development": detailed(historical[historical["year"].isin(DEVELOPMENT)]),
        "holdout": detailed(historical[historical["year"].isin(HOLDOUT)]),
        "external_xperp": detailed(external_rows),
        "sources": {
            str(args.deep_csv): hashlib.sha256(source_raw).hexdigest(),
            **external_sources,
        },
        "limitations": [
            "Rules and portfolio composition are selected on 2020-2024 only.",
            "The 2025-2026 deep-feed rows and recent X-Perp rows are revealed after selection.",
            "Many simple candidates are tested; future paper-forward confirmation is required.",
            "Gross R excludes fees and client-selected position sizing at user request.",
            "Historical results cannot guarantee future profit or eliminate losing months.",
        ],
    }
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    detail = args.out.with_suffix(".csv")
    historical.to_csv(detail, index=False)
    report["detail"] = {"path": str(detail),
                        "sha256": hashlib.sha256(detail.read_bytes()).hexdigest()}
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "development": report["development"]["all"],
        "development_years": report["development"]["years"],
        "holdout": report["holdout"]["all"],
        "holdout_years": report["holdout"]["years"],
        "external_xperp": report["external_xperp"]["all"],
    }, indent=2))


if __name__ == "__main__":
    main()
