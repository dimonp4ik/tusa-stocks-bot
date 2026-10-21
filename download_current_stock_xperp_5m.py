"""Download closed public stock X-Perp five-minute candles; no keys or orders."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import time
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", required=True)
    parser.add_argument("--bars", type=int, default=18000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    session = requests.Session()
    instruments = get_json(session,
        "https://www.okx.com/api/v5/public/instruments",
        {"instType": "FUTURES"}).get("data", [])
    mapping = {item["instId"].split("-")[0] + "USDT": item["instId"]
               for item in instruments
               if "_UM_XPERP-" in item.get("instId", "")
               and item.get("state") == "live"}
    args.out.mkdir(parents=True, exist_ok=True)
    anchor_ms = int(time.time() * 1000) + 60_000
    report = {
        "status": "PUBLIC_CURRENT_STOCK_XPERP_5M",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "bars_requested": args.bars,
        "request_anchor_ms": anchor_ms,
        "symbols": {},
    }
    for symbol in (value.strip() for value in args.symbols.split(",") if value.strip()):
        instrument = mapping.get(symbol)
        if not instrument:
            raise ValueError(f"No live X-Perp instrument for {symbol}")
        after = anchor_ms
        received = {}
        for _ in range((args.bars + 299) // 300 + 1):
            body = get_json(session,
                ENDPOINT,
                {"instId": instrument, "bar": "5m",
                 "after": after, "limit": 300})
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
            if len(received) >= args.bars:
                break
            time.sleep(.12)
        timestamps = sorted(received)[-args.bars:]
        if not timestamps:
            raise ValueError(f"No closed candles for {symbol}")
        candles = {"time": timestamps, **{
            key: [float(received[timestamp][index]) for timestamp in timestamps]
            for index, key in enumerate(("open", "high", "low", "close", "volume"), 1)}}
        raw = pickle.dumps(candles)
        path = args.out / f"{symbol}_5min_current.pkl"
        path.write_bytes(raw)
        item = {
            "status": "OK" if len(timestamps) >= args.bars else "SHORT_HISTORY",
            "instrument": instrument,
            "bars": len(timestamps),
            "start": timestamps[0],
            "end": timestamps[-1],
            "sha256": hashlib.sha256(raw).hexdigest(),
            "path": str(path),
        }
        report["symbols"][symbol] = item
        print(json.dumps({"symbol": symbol, **item}), flush=True)
    (args.out / "manifest.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
