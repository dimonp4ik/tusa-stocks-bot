"""Download public OKX underlying-index candles for offline signal research.

OHLC comes from the X-Perp underlying index. Volume is copied from the
timestamp-matched X-Perp candle because index candles contain no volume.
No authentication or trading endpoints are used.
"""
import argparse
import hashlib
import json
import pickle
import time
from pathlib import Path

import requests


BARS = {900: "15m", 3600: "1H", 14400: "4H", 86400: "1Dutc"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", required=True)
    parser.add_argument("--only-1hour", action="store_true")
    args = parser.parse_args()
    symbols = args.symbols.split(",")
    folder = Path("reports/audit_2026_09_08")
    xdir = folder / "xperp_cache"
    out = folder / "index_signal_cache"
    out.mkdir(exist_ok=True)
    source = json.loads((folder / "filter_universe.json").read_text(encoding="utf-8"))
    results = []

    for name, meta in source["cache_manifest"].items():
        symbol = name.split("_")[0]
        if symbol not in symbols:
            continue
        if args.only_1hour and meta["interval_sec"] != 3600:
            continue
        xraw = (xdir / name).read_bytes()
        xperp = pickle.loads(xraw)
        xlookup = {t: i for i, t in enumerate(xperp["time"])}
        if not xlookup:
            raise ValueError(f"Empty X-Perp cache: {name}")
        index_id = symbol.removesuffix("USDT") + "-USD"
        after = max(xlookup) * 1000 + 1
        wanted_start = min(xlookup)
        rows = {}
        for _ in range(100):
            response = requests.get(
                "https://www.okx.com/api/v5/market/history-index-candles",
                params={"instId": index_id, "bar": BARS[meta["interval_sec"]],
                        "after": after, "limit": 100},
                timeout=25,
            )
            response.raise_for_status()
            body = response.json()
            if body.get("code") != "0":
                raise RuntimeError(str(body))
            batch = body.get("data", [])
            if not batch:
                break
            for row in batch:
                ts = int(row[0]) // 1000
                if ts in xlookup and row[-1] == "1":
                    rows[ts] = row
            oldest = min(int(row[0]) for row in batch)
            if oldest <= wanted_start * 1000:
                break
            if oldest >= after:
                raise RuntimeError(f"Pagination stalled for {name}")
            after = oldest
            time.sleep(0.12)
        times = sorted(rows)
        if not times:
            raise ValueError(f"No matched index candles: {name}")
        data = {
            "time": times,
            "open": [float(rows[t][1]) for t in times],
            "high": [float(rows[t][2]) for t in times],
            "low": [float(rows[t][3]) for t in times],
            "close": [float(rows[t][4]) for t in times],
            "volume": [float(xperp["volume"][xlookup[t]]) for t in times],
        }
        raw = pickle.dumps(data)
        (out / name).write_bytes(raw)
        item = {
            "file": name,
            "index": index_id,
            "bars": len(times),
            "xperp_bars": len(xperp["time"]),
            "matched_fraction": len(times) / len(xperp["time"]),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "xperp_sha256": hashlib.sha256(xraw).hexdigest(),
        }
        results.append(item)
        print(json.dumps(item), flush=True)
    expected = len(symbols) * (1 if args.only_1hour else 4)
    if len(results) != expected:
        raise ValueError(f"Expected {expected} outputs, got {len(results)}")
    manifest = {
        "status": "PUBLIC_INDEX_DATA",
        "symbols": symbols,
        "volume_source": "timestamp-matched X-Perp candles",
        "results": results,
    }
    (out / ("manifest_" + "_".join(symbols) + ".json")).write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
