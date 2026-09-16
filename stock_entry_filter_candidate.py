"""Select simple entry refinements for the frozen robust-dynamic stock profile.

Discovery uses July, August validates one rule per module, and September is
revealed only after the rule is frozen.  A module refinement is retained in the
combined shadow candidate only when it also preserves September net R and win
rate.  The script changes entries only; targets remain those already frozen by
``robust_dynamic``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import pandas as pd

from stock_dynamic_target_candidate import metric
from stock_robust_regime_router import atoms, condition_mask


DISCOVERY = "2026-07"
VALIDATION = "2026-08"
HOLDOUT = "2026-09"


def month_metric(frame: pd.DataFrame, month: str) -> dict:
    return metric(frame[frame["month"].eq(month)])


def select_rule(rows: pd.DataFrame) -> dict:
    base = {month: month_metric(rows, month)
            for month in (DISCOVERY, VALIDATION, HOLDOUT)}
    discovery = base[DISCOVERY]
    seen = set()
    candidates = []
    for atom in atoms():
        mask = condition_mask(rows, (atom,))
        key = tuple(rows.index[mask])
        if key in seen or len(key) == len(rows):
            continue
        seen.add(key)
        kept = rows.loc[list(key)]
        found = month_metric(kept, DISCOVERY)
        if (found["n"] < math.ceil(.65 * discovery["n"])
                or found["wr"] + 1e-9 < discovery["wr"]
                or found["net_r"] + 1e-9 < discovery["net_r"]):
            continue
        candidates.append({
            "rule": list(atom), "kept": kept,
            "discovery": found,
            "discovery_delta_r": found["net_r"] - discovery["net_r"],
            "discovery_delta_wr": found["wr"] - discovery["wr"],
        })
    candidates.sort(key=lambda item: (
        item["discovery_delta_r"], item["discovery_delta_wr"],
        item["discovery"]["n"],
    ), reverse=True)

    qualified = []
    validation = base[VALIDATION]
    for item in candidates[:20]:
        found = month_metric(item["kept"], VALIDATION)
        if (found["n"] < math.ceil(.60 * validation["n"])
                or found["wr"] + 1e-9 < validation["wr"]
                or found["net_r"] + 1e-9 < validation["net_r"]):
            continue
        item = {**item, "validation": found,
                "validation_delta_r": found["net_r"] - validation["net_r"],
                "validation_delta_wr": found["wr"] - validation["wr"]}
        qualified.append(item)
    qualified.sort(key=lambda item: (
        item["validation_delta_r"], item["validation_delta_wr"],
        item["discovery_delta_r"], item["discovery_delta_wr"],
        item["validation"]["n"] + item["discovery"]["n"],
    ), reverse=True)

    if not qualified:
        return {"base": base, "selected": None, "accepted": False,
                "reason": "no July rule confirmed in August"}

    chosen = qualified[0]
    holdout = month_metric(chosen["kept"], HOLDOUT)
    calibration_r_gain = chosen["discovery_delta_r"] + chosen["validation_delta_r"]
    holdout_pass = (
        holdout["n"] >= math.ceil(.60 * base[HOLDOUT]["n"])
        and holdout["wr"] + 1e-9 >= base[HOLDOUT]["wr"]
        and holdout["net_r"] + 1e-9 >= base[HOLDOUT]["net_r"]
    )
    accepted = calibration_r_gain > 1e-9 and holdout_pass
    selected = {key: value for key, value in chosen.items() if key != "kept"}
    selected["holdout"] = holdout
    selected["holdout_pass"] = holdout_pass
    selected["calibration_net_r_gain"] = calibration_r_gain
    return {
        "base": base, "discovery_candidates": len(candidates),
        "validation_qualified": len(qualified), "selected": selected,
        "accepted": accepted,
        "reason": ("positive July-August R gain and September preserved"
                   if accepted else
                   "rejected: no strict calibration R gain or September degraded"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("signals", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    raw = args.signals.read_bytes()
    frame = pd.read_csv(args.signals)
    if set(frame["month"].unique()) != {DISCOVERY, VALIDATION, HOLDOUT}:
        raise ValueError("expected exactly July, August and September 2026")

    modules = {}
    output_parts = []
    accepted_rules = []
    for priority, rows in frame.groupby("module_priority", sort=True):
        result = select_rule(rows)
        modules[str(int(priority))] = {
            key: value for key, value in result.items() if key != "kept"
        }
        selected = result.get("selected")
        if result["accepted"] and selected:
            rule = tuple(selected["rule"])
            kept = rows.loc[condition_mask(rows, (rule,))].copy()
            output_parts.append(kept)
            accepted_rules.append({
                "module_priority": int(priority), "module": str(rows["module"].iloc[0]),
                "rule": list(rule),
            })
        else:
            output_parts.append(rows.copy())

    output = pd.concat(output_parts, ignore_index=True).sort_values(
        ["entry_time", "symbol"]
    ).reset_index(drop=True)
    base_keys = set(zip(frame["symbol"], frame["entry_time"].astype(int)))
    output_keys = set(zip(output["symbol"], output["entry_time"].astype(int)))
    if not output_keys.issubset(base_keys):
        raise AssertionError("entry filter created a trade")

    local_dates = pd.to_datetime(
        output["entry_time"], unit="s", utc=True
    ).dt.tz_convert("America/New_York").dt.date
    daily = output.assign(local_date=local_dates).groupby("local_date")["net_r"].sum()
    months = {month: metric(output[output["month"].eq(month)])
              for month in (DISCOVERY, VALIDATION, HOLDOUT)}
    sessions = {
        month: int(len(pd.bdate_range(
            pd.Period(month).start_time.date(),
            pd.to_datetime(output.loc[output["month"].eq(month), "entry_time"].max(),
                           unit="s", utc=True).date(),
        ))) for month in (DISCOVERY, VALIDATION, HOLDOUT)
    }
    report = {
        "status": "SHADOW_CANDIDATE_AWAITING_UNSEEN_FORWARD",
        "profile": "robust_dynamic_entry_filtered",
        "active_in_live_router": False,
        "selection_policy": {
            "discovery": DISCOVERY,
            "validation": VALIDATION,
            "holdout": HOLDOUT,
            "discovery_rule": (
                "one causal atom, retain >=65%, do not reduce WR or net R; "
                "rank by R gain, WR gain, retention"
            ),
            "validation_rule": (
                "retain >=60%, do not reduce WR or net R; rank the July top-20 "
                "by August R gain then WR gain"
            ),
            "holdout_rule": (
                "retain >=60%, preserve WR and net R; also require a strict "
                "July-August net-R gain"
            ),
            "holdout_used_to_choose_alternative_threshold": False,
        },
        "accepted_entry_filters": accepted_rules,
        "modules": modules,
        "entry_set": {
            "base": int(len(frame)), "candidate": int(len(output)),
            "removed": int(len(frame) - len(output)),
            "candidate_is_subset": output_keys.issubset(base_keys),
        },
        "base": {
            "all": metric(frame),
            "months": {month: metric(frame[frame["month"].eq(month)])
                       for month in (DISCOVERY, VALIDATION, HOLDOUT)},
        },
        "candidate": {
            "all": metric(output), "months": months,
            "frequency": {
                month: {"sessions": sessions[month],
                        "trades_per_session": months[month]["n"] / sessions[month]}
                for month in sessions
            },
            "days": {
                "active": int(len(daily)), "positive": int((daily > 0).sum()),
                "flat": int((daily == 0).sum()), "negative": int((daily < 0).sum()),
                "worst_r": float(daily.min()),
            },
        },
        "sources": {str(args.signals): hashlib.sha256(raw).hexdigest()},
        "limitations": [
            "The direct X-Perp sample spans only July through mid-September 2026.",
            "Many simple atoms were inspected, so selection bias remains.",
            "September is a mechanical holdout here but had already been inspected in earlier research.",
            "The candidate must collect new paper-forward trades before live activation.",
            "Gross R excludes fees and client-selected position sizing at user request.",
        ],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    output.to_csv(args.out.with_suffix(".csv"), index=False)
    print(json.dumps({
        "status": report["status"],
        "accepted_entry_filters": accepted_rules,
        "entry_set": report["entry_set"],
        "base": report["base"],
        "candidate": report["candidate"],
    }, indent=2))


if __name__ == "__main__":
    main()
