"""Causal walk-forward audit for a blended stock target router.

The router chooses one fixed target for each observable market route.  Every
annual fold uses completed training years plus the immediately preceding year
for validation.  Configuration ranking is limited to 2022-2023, 2024 is the
acceptance year, and 2025-2026 are revealed only after the configuration is
frozen.
"""
from __future__ import annotations

import hashlib
import heapq
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from stock_module_cluster_audit import portfolio_gate


ROOT = Path("reports/audit_2026_09_08")
LOW_SOURCE = ROOT / "stock_session_family_market_context_2017_2026.csv"
HIGH_SOURCE = ROOT / "stock_session_family_market_context_high_targets_2017_2026.csv"
OUTPUT = ROOT / "stock_blended_target_walkforward_oct22.json"
ROUTE = ("family", "direction", "qqq_regime", "session_bucket")
DEV_YEARS = (2022, 2023)
ACCEPTANCE_YEAR = 2024
HIDDEN_YEARS = (2025, 2026)
MARGIN_FRACTION = .03
LEVERAGE = 10.0
TRAINING_CACHE: dict[tuple[int, int], pd.DataFrame] = {}
VALIDATION_CACHE: dict[int, pd.DataFrame] = {}
APPLY_CACHE: dict[tuple, pd.DataFrame] = {}


def load() -> pd.DataFrame:
    columns = [
        "symbol", "family", "direction", "target_r", "entry_time",
        "exit_time", "outcome", "net_r", "risk_pct", "signal_minutes",
        "qqq_regime",
    ]
    parts = [pd.read_csv(path, usecols=columns) for path in (LOW_SOURCE, HIGH_SOURCE)]
    frame = pd.concat(parts, ignore_index=True)
    stamp = pd.to_datetime(frame["entry_time"], unit="s", utc=True)
    frame["year"] = stamp.dt.year.astype(np.int16)
    frame["month"] = stamp.dt.strftime("%Y-%m")
    frame["session_bucket"] = pd.cut(
        frame["signal_minutes"], (-1, 30, 90, 150, np.inf),
        labels=("00_30", "45_90", "105_150", "165_plus"),
    ).astype(str)
    frame["signed_return"] = (
        pd.to_numeric(frame["risk_pct"], errors="coerce")
        * pd.to_numeric(frame["net_r"], errors="coerce")
    )
    return frame.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["signed_return", "exit_time"])


def training_stats(train: pd.DataFrame) -> pd.DataFrame:
    keys = [*ROUTE, "target_r"]
    annual = train.groupby([*keys, "year"], observed=True).agg(
        n=("signed_return", "size"), wins=("signed_return", lambda x: int((x > 0).sum())),
        net=("signed_return", "sum"), mean=("signed_return", "mean"),
    ).reset_index()
    total = annual.groupby(keys, observed=True).agg(
        n=("n", "sum"), wins=("wins", "sum"), net=("net", "sum"),
        years=("year", "nunique"), positive_years=("net", lambda x: int((x > 0).sum())),
        min_year_n=("n", "min"), worst_year=("net", "min"),
        worst_mean=("mean", "min"), q25_mean=("mean", lambda x: float(x.quantile(.25))),
        mean=("mean", "mean"), annual_std=("net", "std"),
    ).reset_index()
    total["wr"] = 100 * total["wins"] / total["n"]
    total["positive_share"] = total["positive_years"] / total["years"]
    total["annual_std"] = total["annual_std"].fillna(0.0)
    total["train_score"] = (
        4 * total["worst_mean"] + 3 * total["q25_mean"] + total["mean"]
        + total["net"] / (100 + 100 * total["annual_std"])
    )
    return total


def validation_stats(validation: pd.DataFrame) -> pd.DataFrame:
    return validation.groupby([*ROUTE, "target_r"], observed=True).agg(
        val_n=("signed_return", "size"),
        val_wins=("signed_return", lambda x: int((x > 0).sum())),
        val_net=("signed_return", "sum"), val_mean=("signed_return", "mean"),
    ).reset_index()


def select(frame: pd.DataFrame, test_year: int, cfg: dict) -> pd.DataFrame:
    validation_year = test_year - 1
    available = sorted(int(year) for year in frame.year.unique() if year < validation_year)
    if cfg["window"]:
        available = available[-cfg["window"]:]
    cache_key = (test_year, cfg["window"])
    if cache_key not in TRAINING_CACHE:
        TRAINING_CACHE[cache_key] = training_stats(frame[frame.year.isin(available)])
    if validation_year not in VALIDATION_CACHE:
        VALIDATION_CACHE[validation_year] = validation_stats(
            frame[frame.year.eq(validation_year)])
    stats = TRAINING_CACHE[cache_key]
    eligible = stats[
        (stats.n >= cfg["min_train_n"])
        & (stats.min_year_n >= cfg["min_year_n"])
        & (stats.positive_share >= cfg["positive_share"])
        & (stats.wr >= cfg["min_wr"])
        & (stats.net > 0)
        & (stats.target_r <= cfg["max_target"])
    ].copy()
    # Target choice is made from training years before validation is inspected.
    eligible = eligible.sort_values(
        [*ROUTE, "train_score", "net"], ascending=[True] * len(ROUTE) + [False, False]
    ).drop_duplicates(list(ROUTE), keep="first")
    eligible = eligible.merge(
        VALIDATION_CACHE[validation_year], on=[*ROUTE, "target_r"], how="left")
    eligible = eligible[
        (eligible.val_n >= cfg["min_validation_n"])
        & (eligible.val_net > 0)
    ].copy()
    eligible["validation_wr"] = 100 * eligible.val_wins / eligible.val_n
    eligible["rank_score"] = (
        eligible.train_score + 3 * eligible.val_mean
        + eligible.val_net / (25 + eligible.val_n)
    )
    return eligible.sort_values(
        ["rank_score", "val_net", "train_score"], ascending=False
    ).head(cfg["top_k"])


def apply(frame: pd.DataFrame, test_year: int, selected: pd.DataFrame) -> pd.DataFrame:
    signature = tuple(
        tuple(str(row[key]) for key in (*ROUTE, "target_r"))
        for row in selected.to_dict("records")
    )
    cache_key = (test_year, signature)
    if cache_key in APPLY_CACHE:
        return APPLY_CACHE[cache_key]
    test = frame[frame.year.eq(test_year)]
    parts = []
    for priority, route in enumerate(selected.to_dict("records")):
        mask = np.ones(len(test), dtype=bool)
        for key in (*ROUTE, "target_r"):
            mask &= test[key].astype(str).to_numpy() == str(route[key])
        part = test.loc[mask].copy()
        if part.empty:
            continue
        part["priority"] = priority
        part["module"] = "|".join(str(route[key]) for key in ROUTE)
        part["selection_score"] = float(route["rank_score"])
        parts.append(part)
    if not parts:
        APPLY_CACHE[cache_key] = test.iloc[0:0].copy()
        return APPLY_CACHE[cache_key]
    candidates = pd.concat(parts, ignore_index=True).drop_duplicates(
        ["symbol", "entry_time"], keep="first")
    rows = portfolio_gate(
        candidates.to_dict("records"), None,
        rank_key=lambda row: (int(row["priority"]),),
    )
    APPLY_CACHE[cache_key] = pd.DataFrame(rows) if rows else candidates.iloc[0:0].copy()
    return APPLY_CACHE[cache_key]


def metrics(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {
            "n": 0, "wr": 0.0, "return_pct": 0.0, "dd_pct": 0.0,
            "negative_months": 0, "months": 0, "months_ge_30pct": 0,
            "max_loss_streak": 0, "trades_per_market_day": 0.0,
            "mean_move_pct": 0.0, "monthly_return_pct": {},
        }
    equity = peak = 1.0
    maximum_dd = 0.0
    pending = []
    completed = []
    serial = 0

    def realize(until: float | None) -> None:
        nonlocal equity, peak, maximum_dd
        while pending and (until is None or pending[0][0] < until):
            exit_time, _, pnl, row = heapq.heappop(pending)
            equity += pnl
            peak = max(peak, equity)
            maximum_dd = max(maximum_dd, (peak - equity) / peak)
            completed.append((exit_time, pnl, row))

    for row in frame.sort_values(["entry_time", "priority", "symbol"]).to_dict("records"):
        realize(float(row["entry_time"]))
        pnl = equity * MARGIN_FRACTION * LEVERAGE * float(row["signed_return"])
        serial += 1
        heapq.heappush(pending, (float(row["exit_time"]), serial, pnl, row))
    realize(None)
    ledger = 1.0
    month_start, month_end = {}, {}
    wins = streak = max_streak = 0
    for exit_time, pnl, row in sorted(completed, key=lambda x: (x[0], x[2]["entry_time"], x[2]["symbol"])):
        won = pnl > 0
        wins += int(won)
        streak = 0 if won else streak + 1
        max_streak = max(max_streak, streak)
        month = pd.Timestamp(exit_time, unit="s", tz="UTC").strftime("%Y-%m")
        month_start.setdefault(month, ledger)
        ledger += pnl
        month_end[month] = ledger
    monthly = {month: 100 * (month_end[month] / month_start[month] - 1)
               for month in sorted(month_start)}
    days = pd.to_datetime(frame.entry_time, unit="s", utc=True).dt.date.nunique()
    return {
        "n": int(len(frame)), "wr": round(100 * wins / len(frame), 2),
        "return_pct": round(100 * (equity - 1), 4),
        "dd_pct": round(100 * maximum_dd, 4),
        "negative_months": int(sum(value < -1e-12 for value in monthly.values())),
        "months": len(monthly),
        "months_ge_30pct": int(sum(value >= 30 for value in monthly.values())),
        "max_loss_streak": int(max_streak),
        "trades_per_market_day": round(len(frame) / max(1, days), 3),
        "mean_move_pct": round(100 * float(frame.signed_return.mean()), 5),
        "monthly_return_pct": {key: round(value, 4) for key, value in monthly.items()},
    }


def fold(frame: pd.DataFrame, test_year: int, cfg: dict, keep_routes: bool = False) -> dict:
    selected = select(frame, test_year, cfg)
    rows = apply(frame, test_year, selected)
    result = {
        "year": test_year,
        "training_years": sorted(int(year) for year in frame.year.unique()
                                 if year < test_year - 1)[-cfg["window"]:] if cfg["window"] else
                          sorted(int(year) for year in frame.year.unique() if year < test_year - 1),
        "validation_year": test_year - 1,
        "selected_count": int(len(selected)),
        "target_mix": {str(key): int(value) for key, value in rows.target_r.value_counts().sort_index().items()},
        "result": metrics(rows),
        "_rows": rows,
    }
    if keep_routes:
        fields = [*ROUTE, "target_r", "n", "wr", "net", "positive_share",
                  "val_n", "validation_wr", "val_net", "rank_score"]
        result["routes"] = selected[fields].to_dict("records")
    return result


def public_config(cfg: dict) -> dict:
    return {key: (float(value) if isinstance(value, np.floating) else int(value)
                  if isinstance(value, np.integer) else value)
            for key, value in cfg.items()}


def main() -> None:
    frame = load()
    configs = []
    for values in itertools.product(
            (3, 5, 0), (60, 100), (5, 10), (.67, 1.0),
            (40.0, 50.0), (1.0, 2.0, 4.0), (8,), (8, 12, 16, 24)):
        keys = ("window", "min_train_n", "min_year_n", "positive_share",
                "min_wr", "max_target", "min_validation_n", "top_k")
        configs.append(dict(zip(keys, values)))

    ranked = []
    for index, cfg in enumerate(configs, 1):
        folds = [fold(frame, year, cfg) for year in DEV_YEARS]
        values = [item["result"] for item in folds]
        if (all(value["n"] >= 40 and value["return_pct"] > 0 for value in values)
                and sum(value["n"] for value in values) >= 120):
            score = (
                sum(value["return_pct"] for value in values)
                + 2 * min(value["return_pct"] for value in values)
                - sum(value["dd_pct"] for value in values)
                - 2 * sum(value["negative_months"] for value in values)
            )
            ranked.append((score, cfg, values))
        if index % 96 == 0:
            print(json.dumps({"phase": "development", "tested": index,
                              "qualified": len(ranked)}), flush=True)
    ranked.sort(key=lambda item: item[0], reverse=True)
    chosen = ranked[0][1] if ranked else None
    development = [fold(frame, year, chosen, True) for year in DEV_YEARS] if chosen else []
    acceptance = fold(frame, ACCEPTANCE_YEAR, chosen, True) if chosen else None
    acceptance_pass = bool(acceptance and acceptance["result"]["n"] >= 40
                           and acceptance["result"]["return_pct"] > 0
                           and acceptance["result"]["dd_pct"] <= 10)
    # Hidden years are computed only after the chosen configuration is frozen.
    hidden = [fold(frame, year, chosen, True) for year in HIDDEN_YEARS] if chosen else []
    hidden_pass = bool(hidden and all(
        item["result"]["n"] >= 40 and item["result"]["return_pct"] > 0
        and item["result"]["dd_pct"] <= 10 for item in hidden))
    for item in [*development, *hidden, *([acceptance] if acceptance else [])]:
        item.pop("_rows", None)
    objective_checks = {
        "acceptance_pass": acceptance_pass,
        "hidden_years_pass": hidden_pass,
        "each_hidden_year_wr_75_to_95": bool(hidden and all(
            75 <= item["result"]["wr"] <= 95 for item in hidden)),
        "each_hidden_year_at_least_3_trades_per_day": bool(hidden and all(
            item["result"]["trades_per_market_day"] >= 3 for item in hidden)),
        "every_hidden_month_positive": bool(hidden and all(
            item["result"]["negative_months"] == 0 for item in hidden)),
        "every_hidden_month_ge_30pct": bool(hidden and all(
            item["result"]["months_ge_30pct"] == item["result"]["months"] for item in hidden)),
    }
    report = {
        "status": "SHADOW_CANDIDATE" if all(objective_checks.values()) else "REJECTED",
        "protocol": {
            "configuration_selection_years": list(DEV_YEARS),
            "acceptance_year": ACCEPTANCE_YEAR,
            "hidden_reveal_years": list(HIDDEN_YEARS),
            "annual_fold": "train through year-2; validate on year-1; trade test year",
            "target_choices_r": sorted(float(value) for value in frame.target_r.unique()),
            "route": list(ROUTE),
            "entry": "next 15-minute market open in source replay",
            "portfolio_gate": "cap2 per scan, 3h symbol-direction cooldown, max4 same direction, pause after 3 losses",
            "sizing": "client-selected 3% current-equity margin, 10x leverage",
        },
        "tested_configurations": len(configs),
        "development_qualified": len(ranked),
        "chosen": public_config(chosen) if chosen else None,
        "development": development,
        "acceptance_2024": acceptance,
        "hidden_2025_2026": hidden,
        "objective_checks": objective_checks,
        "top_development": [
            {"score": round(float(score), 6), "config": public_config(cfg),
             "years": values}
            for score, cfg, values in ranked[:20]
        ],
        "sources": {
            str(LOW_SOURCE): hashlib.sha256(LOW_SOURCE.read_bytes()).hexdigest(),
            str(HIGH_SOURCE): hashlib.sha256(HIGH_SOURCE.read_bytes()).hexdigest(),
        },
        "limitations": [
            "Cash-equity candles are structural transfer evidence for stock perpetuals.",
            "The target and route configuration is selected from a finite grid and requires direct X-Perp forward confirmation.",
            "Gross outcomes exclude fees at the user's request.",
            "Historical results cannot guarantee future profit or eliminate losing months.",
        ],
    }
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "qualified": len(ranked),
        "chosen": report["chosen"],
        "development": [item["result"] for item in development],
        "acceptance": acceptance["result"] if acceptance else None,
        "hidden": [item["result"] for item in hidden],
        "checks": objective_checks,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
