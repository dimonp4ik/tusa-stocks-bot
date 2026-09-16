"""Read-only snapshot of the production stock router on closed venue bars."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

from src.stock_venue_router import (
    AUDITED_SYMBOLS, ROBUST_FREQUENCY_EXCLUDED_SYMBOLS, stock_venue_setups,
)


def load_candles(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    candles = pickle.loads(raw)
    lengths = {key: len(candles.get(key, []))
               for key in ("time", "open", "high", "low", "close", "volume")}
    if len(set(lengths.values())) != 1 or not lengths["time"]:
        raise ValueError(f"{path}: inconsistent candle lengths {lengths}")
    times = list(map(int, candles["time"]))
    if times != sorted(times) or len(times) != len(set(times)):
        raise ValueError(f"{path}: timestamps are not sorted and unique")
    return candles, hashlib.sha256(raw).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--profile", choices=("robust_frequency", "robust_dynamic",
                              "robust_dynamic_entry_filtered"),
        default="robust_dynamic_entry_filtered",
    )
    args = parser.parse_args()

    allowed = sorted(AUDITED_SYMBOLS - ROBUST_FREQUENCY_EXCLUDED_SYMBOLS)
    required = sorted(set(allowed) | {"QQQUSDT", "SPYUSDT"})
    loaded = {}
    hashes = {}
    for symbol in required:
        path = args.cache_dir / f"{symbol}_15min_current.pkl"
        candles, digest = load_candles(path)
        loaded[symbol] = candles
        hashes[str(path)] = digest

    qqq, spy = loaded["QQQUSDT"], loaded["SPYUSDT"]
    reference_time = int(qqq["time"][-1])
    if int(spy["time"][-1]) != reference_time:
        raise ValueError("QQQ and SPY last closed bars are not aligned")
    rows = []
    for symbol in allowed:
        candles = loaded[symbol]
        last_closed = int(candles["time"][-1])
        if last_closed != reference_time:
            raise ValueError(
                f"{symbol}: last closed bar {last_closed} does not match QQQ {reference_time}"
            )
        setups = stock_venue_setups(
            candles, qqq, spy, market_price=float(candles["close"][-1]),
            profile=args.profile, symbol=symbol,
        )
        if len(setups) > 1:
            raise AssertionError(f"{symbol}: router returned {len(setups)} setups")
        rows.append({
            "symbol": symbol, "bars": len(candles["time"]),
            "last_closed_bar": last_closed,
            "setup": setups[0] if setups else None,
        })

    report = {
        "status": "PASS",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "invariant": (
            "sorted unique closed bars; symbol, QQQ and SPY aligned; production "
            f"{args.profile} router returned at most one next-bar market setup"
        ),
        "profile": args.profile,
        "signals": sum(row["setup"] is not None for row in rows),
        "rows": rows, "sources": hashes,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
