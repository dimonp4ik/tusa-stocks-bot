"""Vectorized cross-era replay of the exact frozen production stock router.

The vectorized selector is first required to match ``frozen_rows`` exactly on
the direct X-Perp sample.  Only then is it applied to the 2017-2026 cash proxy.
The current >=0.5% first-target gate is applied before the portfolio gate.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pandas as pd

from src.stock_venue_router import (
    DYNAMIC_TARGET_RULES,
    ENTRY_FILTER_RULES,
    GAP_FADE_EXHAUSTION_MAX_QQQ_VOL_RATIO,
    GAP_FADE_EXHAUSTION_PRIOR_DAY_ATR,
    MAX_INDEX_INTRADAY_MOVE_ATR,
    MIN_FIRST_TARGET_MOVE_PCT,
    PROFILE_MODULES,
    ROBUST_FREQUENCY_EXCLUDED_SYMBOLS,
    _session_bucket,
)
from stock_forward_monitor import frozen_rows
from stock_module_cluster_audit import portfolio_gate


ROOT = Path("reports/audit_2026_09_08")
LONG = ROOT / "stock_session_family_market_context_2017_2026.csv"
DIRECT = ROOT / "stock_forward_sep30_families.csv"
OUT = ROOT / "stock_current_router_cross_era_oct11.json"
ROWS_OUT = ROOT / "stock_current_router_cross_era_oct11_rows.csv"
PROFILE = "robust_dynamic_entry_filtered"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def vector_frozen_rows(frame: pd.DataFrame) -> list[dict]:
    data = frame.copy()
    if "session_bucket" not in data:
        data["session_bucket"] = data["signal_minutes"].map(_session_bucket)
    data = data[~data.symbol.isin(ROBUST_FREQUENCY_EXCLUDED_SYMBOLS)].copy()
    sane = (
        data.qqq_intraday_move_atr.notna()
        & data.spy_intraday_move_atr.notna()
        & (data.qqq_intraday_move_atr.abs() <= MAX_INDEX_INTRADAY_MOVE_ATR)
        & (data.spy_intraday_move_atr.abs() <= MAX_INDEX_INTRADAY_MOVE_ATR)
    )
    selected = []
    for priority, module in enumerate(PROFILE_MODULES[PROFILE]):
        mask = (sane & (data.family == module.family)
                & (data.direction == module.direction)
                & (data.qqq_regime == module.qqq_regime)
                & (data.session_bucket == module.session_bucket))
        for field, operator, threshold in module.conditions:
            mask &= data[field].ge(threshold) if operator == "ge" else data[field].le(threshold)
        if module.name in ENTRY_FILTER_RULES:
            field, operator, threshold = ENTRY_FILTER_RULES[module.name]
            mask &= data[field].ge(threshold) if operator == "ge" else data[field].le(threshold)
            if module.name == "gap_fade_short_bear_open":
                mask &= ~(
                    data.qqq_prior_day_dir_atr.gt(GAP_FADE_EXHAUSTION_PRIOR_DAY_ATR)
                    & data.qqq_vol_ratio.le(GAP_FADE_EXHAUSTION_MAX_QQQ_VOL_RATIO)
                )
        target = pd.Series(float(module.target_r), index=data.index)
        if module.name in DYNAMIC_TARGET_RULES:
            field, operator, threshold, upgraded = DYNAMIC_TARGET_RULES[module.name]
            upgrade = data[field].ge(threshold) if operator == "ge" else data[field].le(threshold)
            target = target.mask(upgrade, float(upgraded))
        exact = mask & (data.target_r.sub(target).abs() < 1e-9)
        part = data.loc[exact].copy()
        part["module"] = module.name
        part["module_priority"] = priority
        part["profile"] = PROFILE
        selected.append(part)
    merged = pd.concat(selected, ignore_index=True)
    merged = (merged.sort_values(["entry_time", "symbol", "module_priority"])
              .drop_duplicates(["symbol", "entry_time"], keep="first"))
    return merged.sort_values(["entry_time", "symbol"]).to_dict("records")


def keys(rows: list[dict]) -> list[tuple]:
    return [(str(row["symbol"]), int(float(row["entry_time"])),
             str(row["module"]), float(row["target_r"])) for row in rows]


def metrics(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0, "wr": 0.0, "net_r": 0.0, "return_pct": 0.0,
                "dd_pct": 0.0, "negative_months": 0, "trades_per_market_day": 0.0}
    ordered = sorted(rows, key=lambda r: (float(r["exit_time"]), float(r["entry_time"]), r["symbol"]))
    equity = peak = 1.0
    dd = 0.0
    months: dict[str, float] = {}
    days = set()
    for row in ordered:
        signed = float(row["risk_pct"]) * float(row["net_r"])
        equity *= 1 + .03 * 10 * signed
        peak = max(peak, equity)
        dd = max(dd, (peak - equity) / peak)
        stamp = pd.Timestamp(float(row["entry_time"]), unit="s", tz="UTC")
        months[stamp.strftime("%Y-%m")] = months.get(stamp.strftime("%Y-%m"), 0.0) + signed
        days.add(stamp.strftime("%Y-%m-%d"))
    wins = sum(float(row["net_r"]) > 0 for row in rows)
    return {
        "n": len(rows), "wr": round(100 * wins / len(rows), 4),
        "net_r": round(sum(float(row["net_r"]) for row in rows), 6),
        "signed_price_return_pct": round(100 * sum(
            float(row["risk_pct"]) * float(row["net_r"]) for row in rows), 6),
        "return_pct": round(100 * (equity - 1), 6),
        "dd_pct": round(100 * dd, 6),
        "negative_months": sum(value < 0 for value in months.values()),
        "trades_per_market_day": round(len(rows) / len(days), 4),
    }


def annual(rows: list[dict]) -> dict[str, dict]:
    result = {}
    for year in range(2017, 2027):
        part = [row for row in rows if pd.Timestamp(
            float(row["entry_time"]), unit="s", tz="UTC").year == year]
        result[str(year)] = metrics(part)
    return result


def module_breakdown(rows: list[dict]) -> dict[str, dict]:
    result = {}
    for module in sorted({str(row["module"]) for row in rows}):
        part = [row for row in rows if str(row["module"]) == module]
        result[module] = {"all": metrics(part), "by_year": annual(part)}
    return result


def main() -> None:
    direct_frame = pd.read_csv(DIRECT)
    scalar = frozen_rows(direct_frame, PROFILE)
    vector = vector_frozen_rows(direct_frame)
    parity = {
        "scalar": len(scalar), "vector": len(vector),
        "exact_ordered_key_match": keys(scalar) == keys(vector),
    }
    if not parity["exact_ordered_key_match"]:
        scalar_keys, vector_keys = set(keys(scalar)), set(keys(vector))
        parity.update({
            "missing": sorted(scalar_keys - vector_keys)[:20],
            "unexpected": sorted(vector_keys - scalar_keys)[:20],
        })
        raise ValueError(f"vector selector parity failed: {parity}")

    long_frame = pd.read_csv(LONG)
    raw = vector_frozen_rows(long_frame)
    pre_material = portfolio_gate(raw, None)
    material_raw = [row for row in raw if 100 * float(row["risk_pct"])
                    * float(row["target_r"]) + 1e-12 >= MIN_FIRST_TARGET_MOVE_PCT]
    material = portfolio_gate(material_raw, None)
    compact = pd.DataFrame(raw)
    compact["first_target_move_pct"] = (
        100 * compact["risk_pct"] * compact["target_r"])
    compact["material_eligible"] = (
        compact["first_target_move_pct"] + 1e-12 >= MIN_FIRST_TARGET_MOVE_PCT)
    compact.to_csv(ROWS_OUT, index=False)
    report = {
        "status": "CURRENT_ROUTER_CROSS_ERA_REJECTED_FOR_LIVE",
        "protocol": {
            "router": PROFILE,
            "entry": "next 15m cash-proxy bar after closed causal signal",
            "history": "2017-2026 cash proxy",
            "portfolio": "one symbol open, 3h cooldown, cap2/scan, direction cap4, pause after 3 losses",
            "material_gate_pct": MIN_FIRST_TARGET_MOVE_PCT,
            "money": "3% entry-equity margin, fixed 10x gross",
        },
        "selector_parity_on_direct_xperp": parity,
        "pre_material_route": {
            "all": metrics(pre_material), "by_year": annual(pre_material),
            "by_module": module_breakdown(pre_material),
        },
        "current_material_route": {
            "raw_eligible": len(material_raw),
            "all": metrics(material), "by_year": annual(material),
            "by_module": module_breakdown(material),
        },
        "activation": {
            "allowed": False,
            "reason": "The current material route lacks adequate frequency and stable positive yearly evidence.",
        },
        "sources": {
            str(DIRECT): digest(DIRECT), str(LONG): digest(LONG),
            str(Path(__file__)): digest(Path(__file__)),
        },
        "limitations": [
            "Cash-proxy bars test cross-era structure but are not X-Perp fills.",
            "Gross results exclude fees and slippage at user request.",
            "Historical outcomes cannot guarantee future return or remove losing months.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "parity": parity,
        "pre_material": report["pre_material_route"],
        "material": report["current_material_route"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
