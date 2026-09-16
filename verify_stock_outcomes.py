"""Recompute every selected stock X-Perp outcome from hashed venue bars."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import pandas as pd

from stock_session_family_lab import replay


def key(row: dict) -> tuple:
    return (
        str(row["symbol"]), str(row["family"]), str(row["direction"]),
        round(float(row["target_r"]), 8), int(row["entry_time"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signals", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    signal_raw = args.signals.read_bytes()
    selected = pd.read_csv(args.signals)
    hashes = {str(args.signals): hashlib.sha256(signal_raw).hexdigest()}
    recomputed = {}
    duplicates = []
    for symbol in sorted(selected["symbol"].astype(str).unique()):
        path = args.cache_dir / f"{symbol}_15min_current.pkl"
        raw = path.read_bytes()
        hashes[str(path)] = hashlib.sha256(raw).hexdigest()
        candles = pickle.loads(raw)
        for row in replay(candles, symbol):
            item_key = key(row)
            if item_key in recomputed:
                duplicates.append(item_key)
            recomputed[item_key] = row

    failures = []
    for stored in selected.to_dict("records"):
        item_key = key(stored)
        actual = recomputed.get(item_key)
        if actual is None:
            failures.append({"key": item_key, "reason": "replay row missing"})
            continue
        matched = (
            actual["outcome"] == stored["outcome"]
            and int(actual["exit_time"]) == int(stored["exit_time"])
            and abs(float(actual["net_r"]) - float(stored["net_r"])) < 1e-9
            and abs(float(actual["risk_pct"]) - float(stored["risk_pct"])) < 1e-9
        )
        if not matched:
            failures.append({
                "key": item_key,
                "stored": {name: stored[name] for name in
                           ("outcome", "exit_time", "net_r", "risk_pct")},
                "recomputed": {name: actual[name] for name in
                               ("outcome", "exit_time", "net_r", "risk_pct")},
            })

    if duplicates:
        failures.append({"reason": "duplicate replay keys", "count": len(duplicates)})
    report = {
        "status": "PASS" if not failures else "FAIL",
        "rows": len(selected), "matched": len(selected) - len(failures),
        "failures": failures, "sources": hashes,
        "invariant": (
            "selected outcome, exit time, gross R and risk distance exactly match "
            "a fresh causal session-family replay on direct X-Perp OHLC"
        ),
    }
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
