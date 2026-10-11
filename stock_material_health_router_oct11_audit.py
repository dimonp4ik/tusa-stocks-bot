"""Causal health gating for the current >=0.5% stock routes.

The gate observes only candidate outcomes whose exits occurred before the next
entry.  Parameters are ranked on 2018-2021 after a 2017 warm-up, must pass an
untouched 2022-2023 validation, and are then frozen before 2024 and 2025-2026
are revealed.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict, deque
from pathlib import Path

import pandas as pd

from stock_current_router_cross_era_oct11_audit import metrics
from stock_module_cluster_audit import portfolio_gate


ROOT = Path("reports/audit_2026_09_08")
SOURCE = ROOT / "stock_current_router_cross_era_oct11_rows.csv"
OUT = ROOT / "stock_material_health_router_oct11.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def years(rows: list[dict], first: int, last: int) -> list[dict]:
    return [row for row in rows if first <= pd.Timestamp(
        float(row["entry_time"]), unit="s", tz="UTC").year <= last]


def health_filter(rows: list[dict], *, lookback_days: int, min_trades: int,
                  min_mean_r: float, scope: str) -> list[dict]:
    history: dict[tuple, deque[dict]] = defaultdict(deque)
    pending: list[dict] = []
    output = []
    ordered = sorted(rows, key=lambda row: (float(row["entry_time"]), row["symbol"]))
    for row in ordered:
        now = float(row["entry_time"])
        completed = [item for item in pending if float(item["exit_time"]) < now]
        pending = [item for item in pending if float(item["exit_time"]) >= now]
        for done in completed:
            key = ((str(done["module"]), str(done["symbol"])) if scope == "module_symbol"
                   else (str(done["module"]),))
            history[key].append(done)
        key = ((str(row["module"]), str(row["symbol"])) if scope == "module_symbol"
               else (str(row["module"]),))
        cutoff = now - lookback_days * 86400
        while history[key] and float(history[key][0]["exit_time"]) < cutoff:
            history[key].popleft()
        prior = list(history[key])
        # Warm-up fails closed: an unmeasured module does not receive capital.
        if len(prior) >= min_trades:
            mean_r = sum(float(item["net_r"]) for item in prior) / len(prior)
            if mean_r >= min_mean_r:
                output.append(row)
        pending.append(row)
    return output


def evaluate(rows: list[dict]) -> dict:
    accepted = portfolio_gate(rows, None)
    return {
        "development_2018_2021": metrics(years(accepted, 2018, 2021)),
        "validation_2022_2023": metrics(years(accepted, 2022, 2023)),
        "test_2024": metrics(years(accepted, 2024, 2024)),
        "holdout_2025_2026": metrics(years(accepted, 2025, 2026)),
        "all_2018_2026": metrics(years(accepted, 2018, 2026)),
    }


def main() -> None:
    frame = pd.read_csv(SOURCE)
    raw = frame[frame.material_eligible.astype(bool)].to_dict("records")
    baseline = evaluate(raw)
    candidates = []
    for scope in ("module", "module_symbol"):
        for lookback in (60, 120, 252, 504):
            for minimum in (8, 16, 24):
                for threshold in (-.05, 0.0, .05):
                    filtered = health_filter(
                        raw, lookback_days=lookback, min_trades=minimum,
                        min_mean_r=threshold, scope=scope)
                    result = evaluate(filtered)
                    dev = result["development_2018_2021"]
                    val = result["validation_2022_2023"]
                    dev_base = baseline["development_2018_2021"]
                    val_base = baseline["validation_2022_2023"]
                    dev_pass = (
                        dev["n"] >= .35 * dev_base["n"]
                        and dev["net_r"] > dev_base["net_r"]
                        and dev["dd_pct"] <= dev_base["dd_pct"]
                    )
                    val_pass = (
                        val["n"] >= .35 * val_base["n"]
                        and val["net_r"] > val_base["net_r"]
                        and val["dd_pct"] <= val_base["dd_pct"]
                    )
                    candidates.append({
                        "params": {"scope": scope, "lookback_days": lookback,
                                   "min_trades": minimum, "min_mean_r": threshold},
                        "development": dev, "validation": val,
                        "development_pass": dev_pass, "validation_pass": val_pass,
                        "_result": result,
                    })
    qualified = [item for item in candidates
                 if item["development_pass"] and item["validation_pass"]]
    qualified.sort(key=lambda item: (
        item["validation"]["net_r"] - baseline["validation_2022_2023"]["net_r"],
        item["development"]["net_r"] - baseline["development_2018_2021"]["net_r"],
        -item["validation"]["dd_pct"]), reverse=True)
    selected = None
    if qualified:
        winner = qualified[0]
        selected = {
            "params": winner["params"],
            "development": winner["development"],
            "validation": winner["validation"],
            "test_2024_revealed_after_freeze": winner["_result"]["test_2024"],
            "test_2024_pass": (
                winner["_result"]["test_2024"]["net_r"] > 0
                and winner["_result"]["test_2024"]["net_r"]
                > baseline["test_2024"]["net_r"]
                and winner["_result"]["test_2024"]["dd_pct"]
                <= baseline["test_2024"]["dd_pct"]),
            "holdout_2025_2026_revealed_after_freeze": (
                winner["_result"]["holdout_2025_2026"]),
            "holdout_baseline": baseline["holdout_2025_2026"],
        }
        selected["holdout_pass"] = (
            selected["holdout_2025_2026_revealed_after_freeze"]["net_r"] > 0
            and selected["holdout_2025_2026_revealed_after_freeze"]["net_r"]
            >= selected["holdout_baseline"]["net_r"]
            and selected["holdout_2025_2026_revealed_after_freeze"]["dd_pct"]
            <= selected["holdout_baseline"]["dd_pct"]
        )
    report = {
        "status": "REJECTED" if selected is None or not (
            selected["test_2024_pass"] and selected["holdout_pass"])
                  else "RESEARCH_CANDIDATE_ONLY",
        "protocol": {
            "warmup": 2017,
            "development": [2018, 2019, 2020, 2021],
            "validation": [2022, 2023],
            "test_revealed_after_freeze": [2024],
            "holdout_revealed_after_freeze": [2025, 2026],
            "candidate_count": len(candidates),
            "gate": "only completed prior candidate outcomes; trailing module or module-symbol health",
            "selection": "strict net-R improvement and non-worse drawdown in development and validation",
        },
        "baseline": baseline,
        "qualified_before_test": len(qualified),
        "selected": selected,
        "top_pre_test": [{key: value for key, value in item.items() if key != "_result"}
                         for item in qualified[:10]],
        "runtime_change_allowed": False,
        "sources": {str(SOURCE): digest(SOURCE), str(Path(__file__)): digest(Path(__file__))},
        "limitations": [
            "Health uses paper outcomes of every frozen candidate, including candidates not funded by the portfolio.",
            "A bounded 72-policy grid still creates selection risk.",
            "The cash proxy is not a direct X-Perp fill history.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "baseline": baseline,
        "qualified_before_test": len(qualified), "selected": selected,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
