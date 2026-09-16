"""Verify production stock venue router against every frozen audit signal."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import pandas as pd

from src.stock_venue_router import stock_venue_setups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signals", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--cache-suffix", default="15min_current.pkl")
    parser.add_argument("--profile", choices=(
        "profit", "precision", "balanced", "frequency", "low_drawdown",
        "robust_frequency", "robust_dynamic", "robust_dynamic_entry_filtered"),
                        default="profit")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    signal_bytes = args.signals.read_bytes()
    frame = pd.read_csv(args.signals)
    caches: dict[str, dict] = {}
    lookups: dict[str, dict[int, int]] = {}
    hashes = {str(args.signals): hashlib.sha256(signal_bytes).hexdigest()}

    def candles(symbol: str) -> dict:
        if symbol not in caches:
            path = args.cache_dir / f"{symbol}_{args.cache_suffix}"
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
    for row in frame.to_dict("records"):
        series = candles(row["symbol"])
        timestamp = int(row["entry_time"])
        index = lookups[row["symbol"]][timestamp]
        qqq_index = lookups["QQQUSDT"][timestamp]
        spy_index = lookups["SPYUSDT"][timestamp]
        prefix = {key: value[:index] for key, value in series.items()}
        qqq_prefix = {key: value[:qqq_index] for key, value in qqq.items()}
        spy_prefix = {key: value[:spy_index] for key, value in spy.items()}
        setups = stock_venue_setups(
            prefix, qqq_prefix, spy_prefix,
            market_price=float(series["open"][index]), profile=args.profile,
            symbol=row["symbol"],
        )
        matched = (
            len(setups) == 1
            and setups[0]["family"] == row["family"]
            and setups[0]["direction"] == row["direction"]
            and abs(setups[0]["target_r"] - float(row["target_r"])) < 1e-12
        )
        if not matched:
            failures.append({
                "symbol": row["symbol"], "entry_time": timestamp,
                "family": row["family"], "direction": row["direction"],
                "target_r": row["target_r"],
            })
    report = {
        "status": "PASS" if not failures else "FAIL",
        "rows": len(frame), "matched": len(frame) - len(failures),
        "failures": failures, "sources": hashes,
        "profile": args.profile,
        "invariant": "exactly one setup; family, direction and target match from closed-bar prefixes only",
    }
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
