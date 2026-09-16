"""Sequentially select causal stock-session modules across 2017-2026."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from filter_lab import gate
from stock_robust_regime_router import compact, fast
from stock_session_family_lab import LEVELS


DISCOVERY = (2017, 2018, 2019, 2020, 2021, 2022, 2023)
VALIDATION = 2024
CONFIRMATION = 2025
HOLDOUT = 2026
KEYS = ("family", "direction", "qqq_regime", "session_bucket", "target_r")

EXTRA_LEVELS = {
    field: (-2, -1, -.5, 0, .5, 1, 2)
    for field in (
        "qqq_intraday_move_dir_atr", "qqq_signal_body_dir_atr",
        "qqq_opening_position_dir_atr", "spy_intraday_move_dir_atr",
        "spy_signal_body_dir_atr", "spy_opening_position_dir_atr",
    )
}
EXTRA_LEVELS.update({
    field: (.5, .75, 1, 1.5, 2, 3)
    for field in (
        "qqq_signal_range_atr", "qqq_opening_range_atr",
        "spy_signal_range_atr", "spy_opening_range_atr",
    )
})
SEARCH_LEVELS = {**{key: value for key, value in LEVELS.items()
                     if key != "signal_minutes"}, **EXTRA_LEVELS}


def add_context(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["year"] = pd.to_datetime(frame["entry_time"], unit="s", utc=True).dt.year
    frame["month"] = pd.to_datetime(frame["entry_time"], unit="s", utc=True).dt.strftime("%Y-%m")
    frame["session_bucket"] = pd.cut(
        frame["signal_minutes"], bins=(-1, 30, 90, 150, np.inf),
        labels=("00_30", "45_90", "105_150", "165_plus"),
    ).astype(str)
    return frame


def mask_for(frame: pd.DataFrame, conditions: tuple[tuple, ...]) -> np.ndarray:
    mask = np.ones(len(frame), dtype=bool)
    for field, operator, value in conditions:
        actual = pd.to_numeric(frame[field], errors="coerce").to_numpy(float)
        mask &= np.isfinite(actual) & (actual >= value if operator == "ge" else actual <= value)
    return mask


def atom_masks_for(frame: pd.DataFrame, atoms: list[tuple]) -> dict[tuple, np.ndarray]:
    arrays = {field: pd.to_numeric(frame[field], errors="coerce").to_numpy(float)
              for field in {atom[0] for atom in atoms}}
    result = {}
    for atom in atoms:
        field, operator, value = atom
        actual = arrays[field]
        result[atom] = np.isfinite(actual) & (
            actual >= value if operator == "ge" else actual <= value)
    return result


def quick_arrays(values: np.ndarray, symbols: np.ndarray | None = None) -> dict:
    if not len(values):
        return {"n": 0, "wr": 0.0, "net_r": 0.0, "mean_r": 0.0,
                "pf": None, "dd_r": 0.0, "symbols": 0}
    gains = values[values > 0].sum()
    losses = -values[values < 0].sum()
    curve = values.cumsum()
    peaks = np.maximum.accumulate(np.maximum(curve, 0))
    return {"n": int(len(values)), "wr": float((values > 0).mean() * 100),
            "net_r": float(values.sum()), "mean_r": float(values.mean()),
            "pf": float(gains / losses) if losses else None,
            "dd_r": float(np.max(peaks - curve)),
            "symbols": int(len(np.unique(symbols))) if symbols is not None else 0}


def apply(frame: pd.DataFrame, candidate: dict) -> pd.DataFrame:
    mask = np.ones(len(frame), dtype=bool)
    for field, value in candidate["route"].items():
        mask &= frame[field].astype(str).to_numpy() == str(value)
    return frame.loc[mask & mask_for(frame, tuple(tuple(item)
                                                  for item in candidate["conditions"]))]


def minimum_wr(target: float, stage: str) -> float:
    values = {.25: (83, 82), .5: (72, 70), .75: (63, 61), 1.0: (56, 54)}
    return values[target][0 if stage == "discovery" else 1]


def mine(frame: pd.DataFrame) -> list[dict]:
    discovery = frame[frame["year"].isin(DISCOVERY)]
    validation = frame[frame["year"] == VALIDATION]
    confirmation = frame[frame["year"] == CONFIRMATION]
    atoms = [(field, operator, float(value))
             for field, levels in SEARCH_LEVELS.items()
             for value in levels for operator in ("ge", "le")]
    candidates = []
    for key, base in discovery.groupby(list(KEYS), observed=True, sort=True):
        target = float(key[-1])
        values = base["net_r"].to_numpy(float)
        years = base["year"].to_numpy(int)
        symbols = pd.factorize(base["symbol"])[0]

        route = dict(zip(KEYS, key))
        def route_subset(source: pd.DataFrame) -> pd.DataFrame:
            route_mask = np.ones(len(source), dtype=bool)
            for field, value in route.items():
                route_mask &= source[field].astype(str).to_numpy() == str(value)
            return source.loc[route_mask]

        validation_base = route_subset(validation)
        confirmation_base = route_subset(confirmation)

        def metrics(mask: np.ndarray) -> dict:
            chosen = values[mask]
            return quick_arrays(chosen, symbols[mask])

        atom_masks = atom_masks_for(base, atoms)
        validation_atom_masks = atom_masks_for(validation_base, atoms)
        confirmation_atom_masks = atom_masks_for(confirmation_base, atoms)

        def stage_metrics(stage: pd.DataFrame, stage_masks: dict,
                          conditions: tuple[tuple, ...]) -> dict:
            if not conditions:
                mask = np.ones(len(stage), dtype=bool)
            elif len(conditions) == 1:
                mask = stage_masks[conditions[0]]
            else:
                mask = stage_masks[conditions[0]] & stage_masks[conditions[1]]
            return quick_arrays(stage["net_r"].to_numpy(float)[mask])
        ranked = []
        for atom, mask in atom_masks.items():
            if int(mask.sum()) < 100:
                continue
            overall = metrics(mask)
            yearly = [metrics(mask & (years == year))["mean_r"] for year in DISCOVERY]
            ranked.append((float(np.quantile(yearly, .25)) * 3 + overall["mean_r"], atom))
        ranked_atoms = [atom for _, atom in sorted(ranked, reverse=True)]
        strongest = ranked_atoms[:20]
        rules = [tuple()] + [(atom,) for atom in ranked_atoms[:40]]
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
            overall = metrics(mask)
            if (overall["n"] < 140 or overall["symbols"] < 7
                    or overall["wr"] < minimum_wr(target, "discovery")
                    or overall["net_r"] <= 0 or overall["mean_r"] < .02):
                continue
            by_year = {year: metrics(mask & (years == year)) for year in DISCOVERY}
            present = [item for item in by_year.values() if item["n"] >= 8]
            if len(present) < 6 or sum(item["net_r"] > 0 for item in present) < 5:
                continue
            candidate = {"route": route,
                         "conditions": [list(item) for item in conditions]}
            val = stage_metrics(validation_base, validation_atom_masks, conditions)
            confirm = stage_metrics(confirmation_base, confirmation_atom_masks, conditions)
            threshold = minimum_wr(target, "validation")
            if (val["n"] < 12 or confirm["n"] < 12
                    or val["wr"] < threshold or confirm["wr"] < threshold
                    or val["net_r"] <= 0 or confirm["net_r"] <= 0):
                continue
            means = [item["mean_r"] for item in present]
            score = (min(val["mean_r"], confirm["mean_r"]) * 10
                     + float(np.quantile(means, .25)) * 5 + overall["mean_r"]
                     + min(overall["n"], 800) / 20000)
            candidates.append({**candidate, "discovery": overall,
                               "discovery_years": by_year,
                               "validation_2024": val, "confirmation_2025": confirm,
                               "score": score})
    return sorted(candidates, key=lambda item: (item["score"],
                                                 item["discovery"]["net_r"]), reverse=True)


def select(candidates: list[dict], limit: int = 16) -> list[dict]:
    selected = []
    broad_seen = set()
    for candidate in candidates:
        route = candidate["route"]
        broad = tuple((key, route[key]) for key in KEYS if key != "target_r")
        if broad in broad_seen:
            continue
        broad_seen.add(broad)
        selected.append(candidate)
        if len(selected) >= limit:
            break
    return selected


def calibration_pool(candidates: list[dict], per_route: int = 3,
                     limit: int = 300) -> list[dict]:
    selected = []
    counts = {}
    for candidate in candidates:
        route = candidate["route"]
        broad = tuple((key, route[key]) for key in KEYS if key != "target_r")
        if counts.get(broad, 0) >= per_route:
            continue
        counts[broad] = counts.get(broad, 0) + 1
        selected.append(candidate)
        if len(selected) >= limit:
            break
    return selected


def union(frame: pd.DataFrame, selected: list[dict], *, use_gate: bool) -> pd.DataFrame:
    parts = []
    for priority, candidate in enumerate(selected):
        part = apply(frame, candidate).copy()
        part["module_priority"] = priority
        parts.append(part)
    if not parts:
        return frame.iloc[0:0]
    combined = pd.concat(parts).sort_values(["module_priority", "family"])
    combined = combined.drop_duplicates(["symbol", "entry_time"], keep="first")
    if not use_gate:
        return combined
    rows = gate(combined.sort_values(["entry_time", "symbol", "module_priority"])
                .to_dict("records"), 4)
    return pd.DataFrame(rows) if rows else combined.iloc[0:0]


def detail(frame: pd.DataFrame) -> dict:
    months = {str(key): compact(part) for key, part in frame.groupby("month")}
    return {"all": compact(frame),
            "years": {str(key): compact(part) for key, part in frame.groupby("year")},
            "months": months,
            "negative_months": {key: value for key, value in months.items()
                                if value["net_r"] < -.01}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    raw = args.dataset.read_bytes()
    frame = add_context(pd.read_csv(args.dataset))
    candidates = mine(frame)
    print(json.dumps({"phase": "mined", "candidates": len(candidates)}), flush=True)
    selected = select(candidates)
    pool = calibration_pool(candidates)
    filter_output = union(frame, selected, use_gate=False)
    gated = union(frame, selected, use_gate=True)
    holdout = filter_output[filter_output["year"] == HOLDOUT]
    report = {
        "status": "SEQUENTIAL_RESEARCH_ONLY",
        "selection_sequence": ["2017-2023 discovery", "2024 validation",
                               "2025 confirmation", "2026 holdout"],
        "candidate_count": len(candidates), "selected_count": len(selected),
        "selected": selected, "calibration_pool_count": len(pool),
        "calibration_pool": pool,
        "filter_output": detail(filter_output),
        "with_existing_portfolio_gate": detail(gated),
        "holdout_2026": detail(holdout),
        "holdout_2026_h1": detail(holdout[holdout["entry_time"] <
            pd.Timestamp("2026-07-01", tz="UTC").timestamp()]),
        "holdout_2026_h2": detail(holdout[holdout["entry_time"] >=
            pd.Timestamp("2026-07-01", tz="UTC").timestamp()]),
        "source": {"path": str(args.dataset), "sha256": hashlib.sha256(raw).hexdigest()},
        "limitations": [
            "The dataset uses causal cash-session features and next-bar market entries.",
            "The underlying cash feed is not exact OKX X-Perp execution; venue replay is required.",
            "Gross outcomes exclude fees and client-selected sizing at user request.",
        ],
    }
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    filter_output.to_csv(args.out.with_suffix(".csv"), index=False)
    print(json.dumps({"candidate_count": len(candidates), "selected_count": len(selected),
                      "filter_output": report["filter_output"]["all"],
                      "holdout": report["holdout_2026"]["all"],
                      "holdout_h2": report["holdout_2026_h2"]["all"]}, indent=2))


if __name__ == "__main__":
    main()
