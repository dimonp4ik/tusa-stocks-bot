"""Download current public stock X-Perp 15-minute candles; no keys or orders."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests


ENDPOINT = "https://www.okx.com/api/v5/market/history-candles"


def get_json(session: requests.Session, url: str, params: dict) -> dict:
    last_error = None
    for attempt in range(4):
        try:
            response = session.get(url, params=params, timeout=25)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(1.0 + attempt)
    raise RuntimeError(f"request failed after retries: {last_error}")


def fetch(symbol: str, instrument: str, bars: int, request_anchor_ms: int,
          out: Path) -> tuple[str, dict]:
    session = requests.Session()
    after = request_anchor_ms
    received = {}
    for _ in range((bars + 299) // 300 + 1):
        body = get_json(session, ENDPOINT, {
            "instId": instrument, "bar": "15m", "after": after,
            "limit": 300,
        })
        if body.get("code") != "0":
            raise RuntimeError(f"{symbol}: {body}")
        batch = body.get("data", [])
        if not batch:
            break
        for row in batch:
            if row[-1] == "1":
                received[int(row[0]) // 1000] = row
        oldest = min(int(row[0]) for row in batch)
        if oldest >= after:
            raise RuntimeError(f"{symbol}: pagination stalled at {oldest}")
        after = oldest - 1
        if len(received) >= bars:
            break
        time.sleep(.12)
    timestamps = sorted(received)[-bars:]
    if not timestamps:
        raise ValueError(f"No closed candles for {symbol}")
    candles = {"time": timestamps, **{
        key: [float(received[timestamp][index]) for timestamp in timestamps]
        for index, key in enumerate(("open", "high", "low", "close", "volume"), 1)}}
    raw = pickle.dumps(candles)
    path = out / f"{symbol}_15min_current.pkl"
    path.write_bytes(raw)
    return symbol, {
        "status": "OK" if len(timestamps) >= 1500 else "SHORT_HISTORY",
        "instrument": instrument, "bars": len(timestamps),
        "start": timestamps[0], "end": timestamps[-1],
        "sha256": hashlib.sha256(raw).hexdigest(), "path": str(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", required=True)
    parser.add_argument("--bars", type=int, default=12000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    instruments = get_json(requests.Session(),
        "https://www.okx.com/api/v5/public/instruments",
        {"instType": "FUTURES"}).get("data", [])
    mapping = {item["instId"].split("-")[0] + "USDT": item["instId"]
               for item in instruments
               if "_UM_XPERP-" in item.get("instId", "") and item.get("state") == "live"}
    args.out.mkdir(parents=True, exist_ok=True)
    # All symbols must describe the same point-in-time snapshot.  A per-symbol
    # clock could cross a 15-minute close during this sequential download and
    # give later symbols one additional completed bar.
    request_anchor_ms = int(time.time() * 1000) + 60_000
    report = {"status": "PUBLIC_CURRENT_STOCK_XPERP", "created_at": datetime.now(timezone.utc).isoformat(),
              "bars_requested": args.bars, "request_anchor_ms": request_anchor_ms,
              "symbols": {}}
    symbols = tuple(dict.fromkeys(
        value.strip().upper() for value in args.symbols.split(",")
        if value.strip()))
    available = []
    for symbol in symbols:
        instrument = mapping.get(symbol)
        if instrument:
            available.append((symbol, instrument))
        else:
            report["symbols"][symbol] = {"status": "NO_INSTRUMENT"}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(fetch, symbol, instrument, args.bars,
                            request_anchor_ms, args.out): symbol
            for symbol, instrument in available
        }
        for future in as_completed(futures):
            requested_symbol = futures[future]
            try:
                symbol, item = future.result()
            except Exception as exc:
                symbol = requested_symbol
                item = {"status": "ERROR", "error": str(exc)}
            report["symbols"][symbol] = item
            print(json.dumps({"symbol": symbol, **item}), flush=True)
    report["symbols"] = {
        symbol: report["symbols"][symbol] for symbol in symbols
    }
    (args.out / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
