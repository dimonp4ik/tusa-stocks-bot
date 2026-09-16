"""Select simple stock X-Perp modules on July only, then reveal August/September."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from stock_2026_regime_calibration import detail, threshold
from stock_strict_causal_modules import (
    KEYS, SEARCH_LEVELS, add_context, atom_masks_for, mask_for, quick_arrays,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--block-cv", action="store_true",
                        help="require stability in both chronological July halves")
    args = parser.parse_args()
    raw = args.dataset.read_bytes()
    frame = add_context(pd.read_csv(args.dataset))
    july = frame[frame["month"] == "2026-07"]
    august = frame[frame["month"] == "2026-08"]
    september = frame[frame["month"] == "2026-09"]
    atoms = [(field, operator, float(value))
             for field, levels in SEARCH_LEVELS.items()
             for value in levels for operator in ("ge", "le")]
    candidates = []
    july_dates = pd.to_datetime(
        july["entry_time"], unit="s", utc=True
    ).dt.tz_convert("America/New_York").dt.date
    ordered_days = sorted(july_dates.unique())
    first_half_days = set(ordered_days[:len(ordered_days) // 2])
    july_first_half = pd.Series(
        july_dates.isin(first_half_days).to_numpy(), index=july.index
    )

    for key, base in july.groupby(list(KEYS), observed=True, sort=True):
        route = dict(zip(KEYS, key))
        masks = atom_masks_for(base, atoms)
        values = base["net_r"].to_numpy(float)
        symbols = base["symbol"].to_numpy()
        ranked = []
        for atom, selected in masks.items():
            if int(selected.sum()) < 10:
                continue
            stats = quick_arrays(values[selected], symbols[selected])
            ranked.append((stats["mean_r"] + min(stats["n"], 50) / 1000, atom))
        strongest = [atom for _, atom in sorted(ranked, reverse=True)[:30]]
        rules = [tuple()] + [(atom,) for atom in strongest]
        rules += [pair for pair in itertools.combinations(strongest[:20], 2)
                  if pair[0][0] != pair[1][0]]
        seen = set()
        for conditions in rules:
            if not conditions:
                selected = np.ones(len(base), dtype=bool)
            elif len(conditions) == 1:
                selected = masks[conditions[0]]
            else:
                selected = masks[conditions[0]] & masks[conditions[1]]
            signature = np.packbits(selected).tobytes()
            if signature in seen:
                continue
            seen.add(signature)
            stats = quick_arrays(values[selected], symbols[selected])
            if (stats["n"] < 12 or stats["symbols"] < 5
                    or stats["wr"] < threshold(float(route["target_r"]))
                    or stats["net_r"] <= 0):
                continue
            folds = None
            if args.block_cv:
                first = july_first_half.loc[base.index].to_numpy(bool)
                fold_stats = [
                    quick_arrays(values[selected & fold], symbols[selected & fold])
                    for fold in (first, ~first)
                ]
                required = threshold(float(route["target_r"]))
                if any(item["n"] < 5 or item["symbols"] < 3
                       or item["wr"] < required or item["net_r"] <= 0
                       for item in fold_stats):
                    continue
                folds = fold_stats
                score = (min(item["mean_r"] for item in folds) * 10
                         + sum(item["net_r"] for item in folds) / 20)
            else:
                score = stats["mean_r"] * 10 + stats["net_r"] / 20
            candidates.append({
                "route": route, "conditions": [list(item) for item in conditions],
                "july": stats, "july_folds": folds, "score": score,
            })

    candidates.sort(key=lambda item: (item["score"], item["july"]["n"]),
                    reverse=True)
    selected, broad_seen = [], set()
    for candidate in candidates:
        broad = tuple((key, candidate["route"][key])
                      for key in KEYS if key != "target_r")
        if broad in broad_seen:
            continue
        broad_seen.add(broad)
        selected.append(candidate)
        if len(selected) >= 6:
            break

    def apply(source: pd.DataFrame, candidate: dict) -> pd.DataFrame:
        route = candidate["route"]
        selected_rows = np.ones(len(source), dtype=bool)
        for field, value in route.items():
            selected_rows &= source[field].astype(str).to_numpy() == str(value)
        conditions = tuple(tuple(item) for item in candidate["conditions"])
        return source.loc[selected_rows & mask_for(source, conditions)]

    def union(source: pd.DataFrame) -> pd.DataFrame:
        parts = []
        for priority, candidate in enumerate(selected):
            part = apply(source, candidate).copy()
            part["module_priority"] = priority
            parts.append(part)
        if not parts:
            return source.iloc[0:0]
        combined = pd.concat(parts).sort_values(["module_priority", "family"])
        return combined.drop_duplicates(["symbol", "entry_time"], keep="first")

    outputs = {month: union(source) for month, source in (
        ("2026-07", july), ("2026-08", august), ("2026-09", september))}
    combined = pd.concat(outputs.values()).sort_values("entry_time")
    report = {
        "status": "STRICT_FORWARD_DIAGNOSTIC",
        "selection_month": "2026-07",
        "selection_policy": ("two chronological July folds must each pass"
                             if args.block_cv else "single July aggregate"),
        "forward_months_never_used_for_selection": ["2026-08", "2026-09"],
        "candidate_count": len(candidates),
        "selected": selected,
        "july_selection": detail(outputs["2026-07"]),
        "august_forward": detail(outputs["2026-08"]),
        "september_forward": detail(outputs["2026-09"]),
        "all": detail(combined),
        "source": {"path": str(args.dataset), "sha256": hashlib.sha256(raw).hexdigest()},
        "limitations": [
            "One selection month permits substantial data-mining bias.",
            "This diagnostic cannot replace paper-forward evidence.",
        ],
    }
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    combined.to_csv(args.out.with_suffix(".csv"), index=False)
    print(json.dumps({
        "candidates": len(candidates),
        "selected": len(selected),
        "july": report["july_selection"]["all"],
        "august": report["august_forward"]["all"],
        "september": report["september_forward"]["all"],
    }, indent=2))


if __name__ == "__main__":
    main()
