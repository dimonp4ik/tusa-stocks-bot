"""Calibrate simple July/August X-Perp regimes, reveal September."""
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
    KEYS,
    SEARCH_LEVELS,
    add_context,
    atom_masks_for,
    mask_for,
    quick_arrays,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xperp_csv", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    raw = args.xperp_csv.read_bytes()
    frame = add_context(pd.read_csv(args.xperp_csv))
    july = frame[frame["month"] == "2026-07"]
    august = frame[frame["month"] == "2026-08"]
    september = frame[frame["month"] == "2026-09"]
    atoms = [(field, operator, float(value))
             for field, levels in SEARCH_LEVELS.items()
             for value in levels for operator in ("ge", "le")]
    candidates = []
    for key, base in july.groupby(list(KEYS), observed=True, sort=True):
        route = dict(zip(KEYS, key))
        target = float(route["target_r"])
        def subset(source):
            mask = np.ones(len(source), dtype=bool)
            for field, value in route.items():
                mask &= source[field].astype(str).to_numpy() == str(value)
            return source.loc[mask]
        aug_base = subset(august)
        july_masks = atom_masks_for(base, atoms)
        aug_masks = atom_masks_for(aug_base, atoms)
        july_values = base["net_r"].to_numpy(float)
        aug_values = aug_base["net_r"].to_numpy(float)
        july_symbols = base["symbol"].to_numpy()
        ranked = []
        for atom, mask in july_masks.items():
            if int(mask.sum()) < 10:
                continue
            metric = quick_arrays(july_values[mask], july_symbols[mask])
            ranked.append((metric["mean_r"] + min(metric["n"], 50) / 1000, atom))
        strongest = [atom for _, atom in sorted(ranked, reverse=True)[:30]]
        rules = [tuple()] + [(atom,) for atom in strongest]
        rules += [pair for pair in itertools.combinations(strongest[:20], 2)
                  if pair[0][0] != pair[1][0]]
        seen = set()
        for conditions in rules:
            if not conditions:
                mask_j = np.ones(len(base), dtype=bool)
                mask_a = np.ones(len(aug_base), dtype=bool)
            elif len(conditions) == 1:
                mask_j, mask_a = july_masks[conditions[0]], aug_masks[conditions[0]]
            else:
                mask_j = july_masks[conditions[0]] & july_masks[conditions[1]]
                mask_a = aug_masks[conditions[0]] & aug_masks[conditions[1]]
            signature = np.packbits(mask_j).tobytes()
            if signature in seen:
                continue
            seen.add(signature)
            mj = quick_arrays(july_values[mask_j], july_symbols[mask_j])
            ma = quick_arrays(aug_values[mask_a], aug_base["symbol"].to_numpy()[mask_a])
            required = threshold(target)
            if (mj["n"] < 12 or ma["n"] < 10 or mj["symbols"] < 5 or ma["symbols"] < 5
                    or mj["wr"] < required or ma["wr"] < required
                    or mj["net_r"] <= 0 or ma["net_r"] <= 0):
                continue
            score = min(mj["mean_r"], ma["mean_r"]) * 10 + (mj["net_r"] + ma["net_r"]) / 20
            candidates.append({"route": route,
                               "conditions": [list(item) for item in conditions],
                               "july": mj, "august": ma, "score": score})

    candidates.sort(key=lambda item: (item["score"], item["july"]["n"]), reverse=True)
    selected, broad_seen = [], set()
    for candidate in candidates:
        route = candidate["route"]
        broad = tuple((key, route[key]) for key in KEYS if key != "target_r")
        if broad in broad_seen:
            continue
        broad_seen.add(broad)
        selected.append(candidate)
        if len(selected) >= 16:
            break

    def apply_rule(source: pd.DataFrame, candidate: dict) -> pd.DataFrame:
        route = candidate["route"]
        mask = np.ones(len(source), dtype=bool)
        for field, value in route.items():
            mask &= source[field].astype(str).to_numpy() == str(value)
        conditions = tuple(tuple(item) for item in candidate["conditions"])
        return source.loc[mask & mask_for(source, conditions)]

    def union(source: pd.DataFrame) -> pd.DataFrame:
        parts = []
        for priority, candidate in enumerate(selected):
            part = apply_rule(source, candidate).copy()
            part["module_priority"] = priority
            parts.append(part)
        if not parts:
            return source.iloc[0:0]
        combined = pd.concat(parts).sort_values(["module_priority", "family"])
        return combined.drop_duplicates(["symbol", "entry_time"], keep="first")

    july_output, august_output, september_output = map(union, (july, august, september))
    combined = pd.concat([july_output, august_output, september_output]).sort_values("entry_time")
    report = {
        "status": "PAPER_CANDIDATE" if selected else "NO_RECENT_XPERP_RULES",
        "candidate_count": len(candidates), "selected_count": len(selected),
        "candidate_pool": candidates, "selected": selected,
        "july_discovery": detail(july_output),
        "august_calibration": detail(august_output),
        "september_holdout": detail(september_output),
        "all": detail(combined),
        "source": {"path": str(args.xperp_csv), "sha256": hashlib.sha256(raw).hexdigest()},
        "limitations": [
            "Only two calibration months are available on direct X-Perp.",
            "Many simple candidates are tested; future paper-forward evidence is mandatory.",
            "Gross outcomes exclude fees and client-selected sizing at user request.",
        ],
    }
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    combined.to_csv(args.out.with_suffix(".csv"), index=False)
    print(json.dumps({"candidate_count": len(candidates), "selected_count": len(selected),
                      "july": report["july_discovery"]["all"],
                      "august": report["august_calibration"]["all"],
                      "september": report["september_holdout"]["all"],
                      "all": report["all"]["all"]}, indent=2))


if __name__ == "__main__":
    main()
