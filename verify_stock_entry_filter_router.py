"""Verify accepted and rejected rows for the stock entry-filter profile."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import pandas as pd

from src.stock_venue_router import stock_venue_setups


PROFILE = "robust_dynamic_entry_filtered"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-signals", type=Path, required=True)
    parser.add_argument("--candidate-signals", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    base_raw = args.base_signals.read_bytes()
    candidate_raw = args.candidate_signals.read_bytes()
    base = pd.read_csv(args.base_signals)
    candidate = pd.read_csv(args.candidate_signals)
    accepted_keys = set(zip(candidate["symbol"], candidate["entry_time"].astype(int)))
    base_keys = set(zip(base["symbol"], base["entry_time"].astype(int)))
    if not accepted_keys.issubset(base_keys):
        raise AssertionError("candidate is not a subset of the base profile")

    caches = {}
    lookups = {}
    hashes = {
        str(args.base_signals): hashlib.sha256(base_raw).hexdigest(),
        str(args.candidate_signals): hashlib.sha256(candidate_raw).hexdigest(),
    }

    def candles(symbol: str) -> dict:
        if symbol not in caches:
            path = args.cache_dir / f"{symbol}_15min_current.pkl"
            raw = path.read_bytes()
            hashes[str(path)] = hashlib.sha256(raw).hexdigest()
            caches[symbol] = pickle.loads(raw)
            lookups[symbol] = {
                int(timestamp): index
                for index, timestamp in enumerate(caches[symbol]["time"])
            }
        return caches[symbol]

    qqq, spy = candles("QQQUSDT"), candles("SPYUSDT")
    failures = []
    accepted = rejected = 0
    for row in base.to_dict("records"):
        symbol = str(row["symbol"])
        timestamp = int(row["entry_time"])
        series = candles(symbol)
        index = lookups[symbol][timestamp]
        qqq_index = lookups["QQQUSDT"][timestamp]
        spy_index = lookups["SPYUSDT"][timestamp]
        setups = stock_venue_setups(
            {key: value[:index] for key, value in series.items()},
            {key: value[:qqq_index] for key, value in qqq.items()},
            {key: value[:spy_index] for key, value in spy.items()},
            market_price=float(series["open"][index]), profile=PROFILE,
            symbol=symbol,
        )
        expected = (symbol, timestamp) in accepted_keys
        matched = (len(setups) == 1 and expected
                   and setups[0]["family"] == row["family"]
                   and setups[0]["direction"] == row["direction"]
                   and abs(setups[0]["target_r"] - float(row["target_r"])) < 1e-12)
        if not expected:
            matched = len(setups) == 0
        if matched:
            accepted += int(expected)
            rejected += int(not expected)
        else:
            failures.append({
                "symbol": symbol, "entry_time": timestamp,
                "expected": "accepted" if expected else "rejected",
                "returned_setups": len(setups),
            })

    report = {
        "status": "PASS" if not failures else "FAIL",
        "profile": PROFILE, "base_rows": int(len(base)),
        "expected_accepted": int(len(candidate)),
        "matched_accepted": accepted,
        "expected_rejected": int(len(base) - len(candidate)),
        "matched_rejected": rejected,
        "failures": failures, "sources": hashes,
        "invariant": (
            "every frozen base row is accepted iff it belongs to the filtered "
            "candidate; accepted family, direction and target match from closed bars"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
