"""Materialize the frozen robust-dynamic stock paper candidate.

The target rules in this file are already selected.  This command performs no
search: it applies the July-ranked, August-confirmed rules, removes the frozen
calibration-only symbol exclusions, and reveals September only in the report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from stock_strict_causal_modules import add_context, mask_for


MODULE_NAMES = (
    "drive_short_bear_open",
    "reclaim_long_bear_45_90",
    "gap_fade_short_bear_open",
    "breakout_long_bear_45_90",
    "breakout_long_bear_open",
    "drive_long_bear_open",
)
BASE_TARGETS = (1.0, .25, 1.0, .25, .5, .25)
EXCLUDED_SYMBOLS = frozenset({
    "GOOGLUSDT", "INTCUSDT", "SPYUSDT", "TSLAUSDT",
})
DYNAMIC_TARGET_RULES = {
    "breakout_long_bear_45_90": ("gap_dir_atr", "le", -1.0, 1.0),
    "breakout_long_bear_open": ("signal_range_atr", "ge", 2.0, 1.0),
    "drive_long_bear_open": ("prior_day_dir_atr", "le", 1.0, 1.0),
}
CALIBRATION_MONTHS = ("2026-07", "2026-08")
HOLDOUT_MONTH = "2026-09"


def wilson_lower(wins: int, count: int, z: float = 1.959963984540054) -> float:
    if not count:
        return 0.0
    p = wins / count
    denominator = 1 + z * z / count
    center = p + z * z / (2 * count)
    spread = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count))
    return 100 * (center - spread) / denominator


def metric(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {
            "n": 0, "wr": 0.0, "wr_lower_95": 0.0, "net_r": 0.0,
            "pf": None, "dd_r": 0.0, "max_loss_streak": 0,
        }
    ordered = frame.sort_values(["exit_time", "entry_time", "symbol"])
    values = ordered["net_r"].to_numpy(float)
    wins = int(np.sum(values > 0))
    gains, losses = values[values > 0].sum(), -values[values < 0].sum()
    curve = np.cumsum(values)
    drawdown = np.maximum.accumulate(np.r_[0.0, curve])[1:] - curve
    loss_streak = longest = 0
    for value in values:
        loss_streak = loss_streak + 1 if value < 0 else 0
        longest = max(longest, loss_streak)
    return {
        "n": int(len(values)), "wr": float(100 * wins / len(values)),
        "wr_lower_95": float(wilson_lower(wins, len(values))),
        "net_r": float(values.sum()),
        "pf": float(gains / losses) if losses else None,
        "dd_r": float(drawdown.max(initial=0.0)),
        "max_loss_streak": longest,
    }


def route_rows(frame: pd.DataFrame, candidate: dict, target: float) -> pd.DataFrame:
    route = {key: value for key, value in candidate["route"].items()
             if key != "target_r"}
    route_mask = np.ones(len(frame), dtype=bool)
    for field, value in route.items():
        route_mask &= frame[field].astype(str).to_numpy() == str(value)
    target_mask = np.isclose(frame["target_r"].to_numpy(float), target)
    conditions = tuple(tuple(item) for item in candidate["conditions"])
    return frame.loc[route_mask & target_mask & mask_for(frame, conditions)].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("module_report", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    dataset_raw = args.dataset.read_bytes()
    module_raw = args.module_report.read_bytes()
    frame = add_context(pd.read_csv(args.dataset))
    selected = json.loads(module_raw)["selected"]
    if len(selected) != len(MODULE_NAMES):
        raise ValueError(f"expected {len(MODULE_NAMES)} selected modules, got {len(selected)}")

    parts = []
    base_parts = []
    frozen_modules = []
    for priority, (candidate, name, base_target) in enumerate(
            zip(selected, MODULE_NAMES, BASE_TARGETS, strict=True)):
        base = route_rows(frame, candidate, base_target)
        static_base = base.copy()
        rule = DYNAMIC_TARGET_RULES.get(name)
        if rule:
            field, operator, threshold, upgraded_target = rule
            high = route_rows(frame, candidate, upgraded_target).set_index(
                ["symbol", "entry_time"], drop=False
            )
            values = pd.to_numeric(base[field], errors="coerce")
            upgrade_mask = values.notna() & (
                values.ge(threshold) if operator == "ge" else values.le(threshold)
            )
            upgraded = []
            for row_index in base.index[upgrade_mask]:
                item = base.loc[row_index]
                key = (item["symbol"], item["entry_time"])
                if key not in high.index:
                    raise KeyError(f"missing {upgraded_target}R replay row for {name} {key}")
                upgraded.append(high.loc[key])
            if upgraded:
                replacements = pd.DataFrame(upgraded)
                base = base.loc[~upgrade_mask].copy()
                base = pd.concat([base, replacements], ignore_index=True)
        base["module_priority"] = priority
        base["module"] = name
        base["portfolio_profile"] = "robust_dynamic"
        parts.append(base)
        static_base["module_priority"] = priority
        static_base["module"] = name
        static_base["portfolio_profile"] = "robust_frequency"
        base_parts.append(static_base)
        frozen_modules.append({
            "priority": priority,
            "name": name,
            "route": {key: value for key, value in candidate["route"].items()
                      if key != "target_r"},
            "conditions": candidate["conditions"],
            "base_target_r": base_target,
            "dynamic_target_rule": (
                {"field": rule[0], "operator": rule[1], "threshold": rule[2],
                 "target_r": rule[3]} if rule else None
            ),
        })

    def union(source_parts: list[pd.DataFrame]) -> pd.DataFrame:
        result = pd.concat(source_parts, ignore_index=True)
        result = result[~result["symbol"].isin(EXCLUDED_SYMBOLS)].copy()
        result = result.sort_values(
            ["module_priority", "family", "symbol", "entry_time"])
        result = result.drop_duplicates(["symbol", "entry_time"], keep="first")
        return result.sort_values(["entry_time", "symbol"]).reset_index(drop=True)

    output = union(parts)
    static_output = union(base_parts)
    keys = ["symbol", "entry_time"]
    if set(map(tuple, output[keys].to_numpy())) != set(
            map(tuple, static_output[keys].to_numpy())):
        raise AssertionError("dynamic targets changed the entry set")
    comparison = static_output.merge(
        output, on=keys, suffixes=("_base", "_dynamic"), validate="one_to_one")
    upgraded = comparison[
        comparison["target_r_dynamic"] > comparison["target_r_base"]
    ].copy()
    upgraded["delta_r"] = upgraded["net_r_dynamic"] - upgraded["net_r_base"]

    months = {
        month: metric(output[output["month"] == month])
        for month in (*CALIBRATION_MONTHS, HOLDOUT_MONTH)
    }
    for month, values in months.items():
        end = pd.Period(month).end_time.date()
        if month == HOLDOUT_MONTH:
            end = pd.to_datetime(output.loc[output["month"] == month, "entry_time"].max(),
                                 unit="s", utc=True).date()
        sessions = int(len(pd.bdate_range(pd.Period(month).start_time.date(), end)))
        values["sessions"] = sessions
        values["trades_per_session"] = values["n"] / sessions if sessions else 0.0

    local_dates = pd.to_datetime(output["entry_time"], unit="s", utc=True).dt.tz_convert(
        "America/New_York").dt.date
    by_day = output.assign(local_date=local_dates).groupby("local_date")["net_r"].sum()
    report = {
        "status": "PAPER_CANDIDATE",
        "profile": "robust_dynamic",
        "selection_policy": {
            "sequence": [
                "rank one-atom target upgrades on July",
                "require improvement/acceptance on August",
                "reveal September only after freezing",
            ],
            "calibration_months": list(CALIBRATION_MONTHS),
            "holdout_month": HOLDOUT_MONTH,
            "holdout_used_for_rule_ranking": False,
            "symbol_exclusions": (
                "frozen calibration-only gate from July and August; September excluded"
            ),
        },
        "excluded_symbols": sorted(EXCLUDED_SYMBOLS),
        "modules": frozen_modules,
        "target_upgrade_audit": {
            "same_entry_set_as_robust_frequency": True,
            "upgraded_trades": int(len(upgraded)),
            "downgraded_trades": int((
                comparison["target_r_dynamic"] < comparison["target_r_base"]
            ).sum()),
            "by_month": {
                str(month): {
                    "n": int(len(part)),
                    "base_net_r": float(part["net_r_base"].sum()),
                    "dynamic_net_r": float(part["net_r_dynamic"].sum()),
                    "incremental_net_r": float(part["delta_r"].sum()),
                    "base_wr": float(100 * (part["net_r_base"] > 0).mean()),
                    "dynamic_wr": float(100 * (part["net_r_dynamic"] > 0).mean()),
                }
                for month, part in upgraded.groupby("month_base")
            },
            "by_module": {
                str(module): {
                    "n": int(len(part)),
                    "base_net_r": float(part["net_r_base"].sum()),
                    "dynamic_net_r": float(part["net_r_dynamic"].sum()),
                    "incremental_net_r": float(part["delta_r"].sum()),
                    "base_wr": float(100 * (part["net_r_base"] > 0).mean()),
                    "dynamic_wr": float(100 * (part["net_r_dynamic"] > 0).mean()),
                }
                for module, part in upgraded.groupby("module_dynamic")
            },
        },
        "months": months,
        "all": metric(output),
        "leave_one_symbol_out": {
            symbol: metric(output[output["symbol"] != symbol])
            for symbol in sorted(output["symbol"].unique())
        },
        "days": {
            "active": int(len(by_day)),
            "positive": int((by_day > 0).sum()),
            "flat": int((by_day == 0).sum()),
            "negative": int((by_day < 0).sum()),
            "positive_pct": float(100 * (by_day > 0).mean()) if len(by_day) else 0.0,
            "worst_r": float(by_day.min()) if len(by_day) else 0.0,
        },
        "sources": {
            str(args.dataset): hashlib.sha256(dataset_raw).hexdigest(),
            str(args.module_report): hashlib.sha256(module_raw).hexdigest(),
        },
        "limitations": [
            "Direct stock X-Perp evidence is short and begins in June 2026.",
            "Target atoms and symbol subsets were explored, so reported results retain selection bias.",
            "September was excluded from programmatic rule ranking but was inspected during research.",
            "Older cross-venue history did not confirm simple higher targets; venue transfer is uncertain.",
            "A future paper-forward sample is required before any live activation.",
            "Gross R excludes costs and client-selected sizing at user request.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    output.to_csv(args.out.with_suffix(".csv"), index=False)
    print(json.dumps({
        "status": report["status"], "months": months,
        "all": report["all"], "days": report["days"],
    }, indent=2))


if __name__ == "__main__":
    main()
