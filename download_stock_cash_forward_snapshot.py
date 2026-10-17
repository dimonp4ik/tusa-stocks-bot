"""Download an isolated Dukascopy 15-minute cash snapshot for forward tests."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import dukascopy_python as dp

from src.dukascopy_client import INSTRUMENT_MAP


def parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def fetch(symbol: str, start: datetime, end: datetime, out: Path) -> tuple[str, dict]:
    instrument, _ = INSTRUMENT_MAP[symbol]
    frame = dp.fetch(instrument, dp.INTERVAL_MIN_15, dp.OFFER_SIDE_BID, start, end)
    if frame is None or frame.empty:
        raise ValueError(f"No Dukascopy data for {symbol}")
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    candles = {
        "time": [int(stamp.timestamp()) for stamp in frame.index],
        "open": [float(value) for value in frame["open"]],
        "high": [float(value) for value in frame["high"]],
        "low": [float(value) for value in frame["low"]],
        "close": [float(value) for value in frame["close"]],
        "volume": [float(value) for value in frame["volume"]],
    }
    if len(candles["time"]) < 100 or any(
            right <= left for left, right in zip(candles["time"], candles["time"][1:])):
        raise ValueError(f"Invalid Dukascopy snapshot for {symbol}")
    raw = pickle.dumps(candles, protocol=pickle.HIGHEST_PROTOCOL)
    path = out / f"{symbol}_15min_forward.pkl"
    path.write_bytes(raw)
    return symbol, {
        "path": str(path), "rows": len(candles["time"]),
        "start": candles["time"][0], "end": candles["time"][-1],
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", required=True)
    parser.add_argument("--start", default="2026-08-15T00:00:00Z")
    parser.add_argument("--end", default=None)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    symbols = tuple(dict.fromkeys(part.strip().upper() for part in args.symbols.split(",") if part.strip()))
    unknown = [symbol for symbol in symbols if symbol not in INSTRUMENT_MAP]
    if unknown or args.workers < 1:
        parser.error(f"invalid symbols/workers: {unknown}")
    start = parse_utc(args.start)
    end = parse_utc(args.end) if args.end else datetime.now(timezone.utc)
    if end <= start:
        parser.error("end must be after start")
    args.out.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "ISOLATED_PUBLIC_FORWARD_SNAPSHOT",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "requested_start": start.isoformat(), "requested_end": end.isoformat(),
        "symbols": {},
    }
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(fetch, symbol, start, end, args.out): symbol for symbol in symbols}
        for future in as_completed(futures):
            symbol, item = future.result()
            report["symbols"][symbol] = item
            print(json.dumps({"symbol": symbol, **item}), flush=True)
    report["symbols"] = {symbol: report["symbols"][symbol] for symbol in symbols}
    manifest = args.out / "manifest.json"
    manifest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest), "symbols": len(symbols)}), flush=True)


if __name__ == "__main__":
    main()
