"""Download and validate extended Dukascopy histories for stock research."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.dukascopy_client import CACHE_DIR, fetch_history_duka


DEFAULT_SYMBOLS = (
    "AAPLUSDT", "AMZNUSDT", "GOOGLUSDT", "METAUSDT", "MSFTUSDT",
    "NVDAUSDT", "TSLAUSDT", "INTCUSDT", "MRVLUSDT", "MUUSDT",
    "QQQUSDT", "SPYUSDT",
)
SHORT_HISTORY_MINIMUMS = {"MRVLUSDT": 25_000}


def validate(symbol: str, candles: dict, minimum: int) -> None:
    fields = ("time", "open", "high", "low", "close", "volume")
    lengths = {len(candles.get(field, [])) for field in fields}
    if len(lengths) != 1 or next(iter(lengths), 0) < minimum:
        raise ValueError(f"{symbol}: invalid lengths {sorted(lengths)}")
    times = list(map(int, candles["time"]))
    if any(right <= left for left, right in zip(times, times[1:])):
        raise ValueError(f"{symbol}: timestamps are not strictly increasing")
    for index in range(len(times)):
        opening = float(candles["open"][index])
        high = float(candles["high"][index])
        low = float(candles["low"][index])
        close = float(candles["close"][index])
        if (not all(math.isfinite(value) and value > 0
                    for value in (opening, high, low, close))
                or high < max(opening, close) or low > min(opening, close)):
            raise ValueError(f"{symbol}: invalid OHLC at row {index}")


def fetch(symbol: str, count: int, minimum: int) -> tuple[str, dict, Path]:
    candles = fetch_history_duka(symbol, "15min", 900, count)
    required = SHORT_HISTORY_MINIMUMS.get(symbol, minimum)
    validate(symbol, candles, required)
    path = CACHE_DIR / f"{symbol}_15min_{count}.pkl"
    # Re-read the serialized artifact to catch truncated writes immediately.
    stored = pickle.loads(path.read_bytes())
    validate(symbol, stored, required)
    return symbol, stored, path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--count", type=int, default=65_000)
    parser.add_argument("--minimum", type=int, default=40_000)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    symbols = tuple(dict.fromkeys(value.strip().upper()
                                  for value in args.symbols.split(",") if value.strip()))
    if args.count < args.minimum or args.minimum < 1000 or args.workers < 1:
        parser.error("count >= minimum >= 1000 and workers >= 1 are required")
    report = {"status": "VALIDATED", "requested_count": args.count,
              "minimum_count": args.minimum, "symbols": {}}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(fetch, symbol, args.count, args.minimum): symbol
                   for symbol in symbols}
        for future in as_completed(futures):
            symbol, candles, path = future.result()
            report["symbols"][symbol] = {
                "path": str(path), "rows": len(candles["time"]),
                "start": int(candles["time"][0]), "end": int(candles["time"][-1]),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            print(json.dumps({"symbol": symbol,
                              "rows": report["symbols"][symbol]["rows"],
                              "start": report["symbols"][symbol]["start"],
                              "end": report["symbols"][symbol]["end"]}), flush=True)
    report["symbols"] = {symbol: report["symbols"][symbol] for symbol in symbols}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
