"""Download timestamp-matched public OKX stock-index candles for research."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import time
from pathlib import Path

import pandas as pd
import requests


def merge_execution(paths: list[Path], symbol: str) -> dict:
    rows = {}
    for root in paths:
        for path in root.glob(f"{symbol}_15min*.pkl"):
            candles = pickle.loads(path.read_bytes())
            for values in zip(*(candles[key] for key in
                                ("time", "open", "high", "low", "close", "volume"))):
                rows[int(values[0])] = tuple(float(value) for value in values[1:])
    times = sorted(rows)
    return {"time": times, **{
        key: [rows[timestamp][position] for timestamp in times]
        for position, key in enumerate(("open", "high", "low", "close", "volume"))
    }}


def download(symbol: str, execution: dict, start_ts: int) -> dict:
    wanted = {int(timestamp): index for index, timestamp in enumerate(execution["time"])
              if int(timestamp) >= start_ts}
    if not wanted:
        raise ValueError(f"{symbol}: no execution timestamps after start")
    after = max(wanted) * 1000 + 1
    minimum = min(wanted)
    rows = {}
    session = requests.Session()
    for _ in range(120):
        response = session.get(
            "https://www.okx.com/api/v5/market/history-index-candles",
            params={"instId": symbol.removesuffix("USDT") + "-USD",
                    "bar": "15m", "after": after, "limit": 100},
            timeout=25,
        )
        response.raise_for_status()
        body = response.json()
        if body.get("code") != "0":
            raise RuntimeError(f"{symbol}: {body}")
        batch = body.get("data", [])
        if not batch:
            break
        for row in batch:
            timestamp = int(row[0]) // 1000
            if timestamp in wanted and row[-1] == "1":
                rows[timestamp] = row
        oldest = min(int(row[0]) for row in batch)
        if oldest <= minimum * 1000:
            break
        if oldest >= after:
            raise RuntimeError(f"{symbol}: pagination stalled")
        after = oldest
        time.sleep(.12)
    times = sorted(rows)
    if not times:
        raise ValueError(f"{symbol}: no matched index candles")
    return {
        "time": times,
        "open": [float(rows[timestamp][1]) for timestamp in times],
        "high": [float(rows[timestamp][2]) for timestamp in times],
        "low": [float(rows[timestamp][3]) for timestamp in times],
        "close": [float(rows[timestamp][4]) for timestamp in times],
        "volume": [float(execution["volume"][wanted[timestamp]]) for timestamp in times],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-dir", action="append", type=Path, required=True)
    parser.add_argument("--symbols", required=True)
    parser.add_argument("--start", default="2026-07-01")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    start_ts = int(pd.Timestamp(args.start, tz="UTC").timestamp())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"status": "PUBLIC_OKX_INDEX", "start": args.start, "symbols": {}}
    for symbol in (value.strip() for value in args.symbols.split(",") if value.strip()):
        execution = merge_execution(args.execution_dir, symbol)
        candles = download(symbol, execution, start_ts)
        raw = pickle.dumps(candles)
        path = args.out_dir / f"{symbol}_15min_current.pkl"
        path.write_bytes(raw)
        item = {
            "bars": len(candles["time"]),
            "range": [candles["time"][0], candles["time"][-1]],
            "matched_fraction": len(candles["time"]) / sum(
                int(timestamp) >= start_ts for timestamp in execution["time"]),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        manifest["symbols"][symbol] = item
        print(json.dumps({"symbol": symbol, **item}), flush=True)
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
