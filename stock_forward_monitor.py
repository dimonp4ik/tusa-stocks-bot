"""Replay the frozen stock venue router after its last audited entry.

The input family file is built from public, closed X-Perp candles.  Selection
uses the production module order, entry filters and dynamic targets, followed
by the same causal portfolio gate used by the research replay.  No API keys or
orders are used.
"""
from __future__ import annotations

import argparse
import collections
import heapq
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from filter_lab import metrics
from src.stock_venue_router import (
    PROFILE_MODULES,
    ROBUST_FREQUENCY_EXCLUDED_SYMBOLS,
    _entry_filter_matches,
    _matches,
    _session_bucket,
    _target_for,
)


def frozen_rows(frame: pd.DataFrame, profile: str) -> list[dict]:
    """Select exactly one production module for each symbol/entry timestamp."""
    if "session_bucket" not in frame:
        frame = frame.copy()
        frame["session_bucket"] = frame["signal_minutes"].map(_session_bucket)
    modules = PROFILE_MODULES[profile]
    selected: list[dict] = []
    allowed = frame[~frame["symbol"].isin(ROBUST_FREQUENCY_EXCLUDED_SYMBOLS)]
    for (_, _), group in allowed.groupby(["symbol", "entry_time"], sort=True):
        # Target variants carry identical causal entry features.  Use one row
        # per family/direction to decide the module, then fetch its target row.
        bases = group.sort_values(["family", "direction", "target_r"]).drop_duplicates(
            ["family", "direction"], keep="first"
        )
        chosen = None
        for priority, module in enumerate(modules):
            for _, base_series in bases.iterrows():
                base = base_series.to_dict()
                if not _matches(module, base) or not _entry_filter_matches(
                    module, base, profile
                ):
                    continue
                target = _target_for(module, base, profile)
                exact = group[
                    (group["family"] == base["family"])
                    & (group["direction"] == base["direction"])
                    & ((group["target_r"] - target).abs() < 1e-9)
                ]
                if exact.empty:
                    raise ValueError(
                        f"missing target {target} for {base['symbol']} "
                        f"at {int(base['entry_time'])}"
                    )
                chosen = exact.iloc[0].to_dict()
                chosen.update(
                    module=module.name,
                    module_priority=priority,
                    profile=profile,
                )
                break
            if chosen is not None:
                break
        if chosen is not None:
            selected.append(chosen)
    return sorted(selected, key=lambda row: (row["entry_time"], row["symbol"]))


def portfolio_gate(rows: list[dict], direction_cap: int, scan_cap: int) -> list[dict]:
    """Causal portfolio replay with an explicit per-scan candidate cap."""
    pending: list[tuple[float, int, dict]] = []
    opened: dict[str, dict] = {}
    last: dict[tuple[str, str], float] = {}
    per_scan: collections.Counter[int] = collections.Counter()
    output = []
    day = None
    streak = 0
    paused = False
    for row in sorted(rows, key=lambda item: (item["entry_time"], item["symbol"])):
        now = row["entry_time"]
        new_day = int(now // 86400)
        if new_day != day:
            day, streak, paused = new_day, 0, False
        while pending and pending[0][0] < now:
            end, _, done = heapq.heappop(pending)
            opened.pop(done["symbol"], None)
            if int(end // 86400) == day:
                streak = streak + 1 if done["outcome"] == "SL" else 0
                if streak >= 3:
                    paused = True
        if paused or row["symbol"] in opened:
            continue
        key = (row["symbol"], row["direction"])
        if now - last.get(key, -1e20) < 10800:
            continue
        scan = int(now // 900)
        if per_scan[scan] >= scan_cap:
            continue
        if sum(item["direction"] == row["direction"]
               for item in opened.values()) >= direction_cap:
            continue
        opened[row["symbol"]] = row
        last[key] = now
        per_scan[scan] += 1
        output.append(row)
        heapq.heappush(pending, (row["exit_time"], len(output), row))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-signals", type=Path, required=True)
    parser.add_argument("--families", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--profile", default="robust_dynamic_entry_filtered")
    parser.add_argument("--direction-cap", type=int, default=4)
    parser.add_argument("--scan-cap", type=int, default=2)
    args = parser.parse_args()

    reference_raw = args.reference_signals.read_bytes()
    family_raw = args.families.read_bytes()
    reference = pd.read_csv(args.reference_signals)
    families = pd.read_csv(args.families)
    previous_max = int(reference["entry_time"].max())
    raw = frozen_rows(families, args.profile)
    fresh_raw = [row for row in raw if int(row["entry_time"]) > previous_max]
    accepted = portfolio_gate(fresh_raw, args.direction_cap, args.scan_cap)

    report = {
        "status": "PASS",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "profile": args.profile,
        "scan_cap": args.scan_cap,
        "previous_max_entry_time": previous_max,
        "previous_max_entry_iso": datetime.fromtimestamp(
            previous_max, timezone.utc
        ).isoformat(),
        "family_range": {
            "first": int(families["entry_time"].min()),
            "last": int(families["entry_time"].max()),
        },
        "raw_frozen_signals": len(raw),
        "fresh_raw_signals": len(fresh_raw),
        "fresh_raw": fresh_raw,
        "fresh_portfolio_signals": len(accepted),
        "fresh_metrics": metrics(accepted),
        "fresh": accepted,
        "sources": {
            str(args.reference_signals): hashlib.sha256(reference_raw).hexdigest(),
            str(args.families): hashlib.sha256(family_raw).hexdigest(),
        },
        "invariant": (
            "production module order, entry filters and dynamic targets; only "
            "entries later than the audited reference; causal one-symbol, cooldown, "
            "three-per-scan, loss-pause and direction-cap portfolio gate"
        ),
        "limitations": [
            "OHLC stop-first replay is a market-entry proxy, not account fills.",
            "The fresh slice is too short to establish a stable win rate.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(accepted).to_csv(args.out.with_suffix(".csv"), index=False)
    print(json.dumps({key: report[key] for key in (
        "status", "previous_max_entry_iso", "family_range", "raw_frozen_signals",
        "fresh_raw_signals", "fresh_portfolio_signals", "fresh_metrics",
    )}, indent=2))


if __name__ == "__main__":
    main()
