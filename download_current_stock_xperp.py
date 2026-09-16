"""Download current public stock X-Perp 15-minute candles; no keys or orders."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path

import requests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", required=True)
    parser.add_argument("--bars", type=int, default=12000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    instruments = requests.get(
        "https://www.okx.com/api/v5/public/instruments",
        params={"instType": "FUTURES"}, timeout=25).json().get("data", [])
    mapping = {item["instId"].split("-")[0] + "USDT": item["instId"]
               for item in instruments
               if "_UM_XPERP-" in item.get("instId", "") and item.get("state") == "live"}
    args.out.mkdir(parents=True, exist_ok=True)
    report = {"status": "PUBLIC_CURRENT_STOCK_XPERP", "created_at": datetime.now(timezone.utc).isoformat(),
              "bars_requested": args.bars, "symbols": {}}
    for symbol in (value.strip() for value in args.symbols.split(",") if value.strip()):
        instrument = mapping.get(symbol)
        if not instrument:
            report["symbols"][symbol] = {"status": "NO_INSTRUMENT"}
            continue
        after = int(time.time() * 1000) + 60_000
        received = {}
        for _ in range((args.bars + 299) // 300):
            response = requests.get(
                "https://www.okx.com/api/v5/market/history-candles",
                params={"instId": instrument, "bar": "15m", "after": after, "limit": 300},
                timeout=25)
            response.raise_for_status()
            body = response.json()
            if body.get("code") != "0":
                raise RuntimeError(str(body))
            batch = body.get("data", [])
            if not batch:
                break
            for row in batch:
                if row[-1] == "1":
                    received[int(row[0]) // 1000] = row
            oldest = min(int(row[0]) for row in batch)
            if oldest >= after:
                raise RuntimeError(f"{symbol}: pagination stalled")
            after = oldest
            if len(received) >= args.bars:
                break
            time.sleep(.16)
        timestamps = sorted(received)[-args.bars:]
        candles = {"time": timestamps, **{
            key: [float(received[timestamp][index]) for timestamp in timestamps]
            for index, key in enumerate(("open", "high", "low", "close", "volume"), 1)}}
        raw = pickle.dumps(candles)
        path = args.out / f"{symbol}_15min_current.pkl"
        path.write_bytes(raw)
        item = {"status": "OK" if len(timestamps) >= 1500 else "SHORT_HISTORY",
                "instrument": instrument, "bars": len(timestamps),
                "start": timestamps[0] if timestamps else None,
                "end": timestamps[-1] if timestamps else None,
                "sha256": hashlib.sha256(raw).hexdigest(), "path": str(path)}
        report["symbols"][symbol] = item
        print(json.dumps({"symbol": symbol, **item}), flush=True)
    (args.out / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
