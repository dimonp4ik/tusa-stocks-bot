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
BAR_SECONDS = 15 * 60


def equity_mapping(instruments: list[dict], listed_before_ms: int | None = None) -> dict[str, str]:
    """Return live equity X-Perp futures from OKX's public schema."""
    output = {}
    for item in instruments:
        if (item.get("state") != "live"
                or item.get("instType") != "FUTURES"
                or item.get("instCategory") != "3"
                or item.get("ruleType") != "xperp"
                or "_UM_XPERP-" not in str(item.get("instId", ""))):
            continue
        try:
            listed = int(item.get("listTime") or 0)
        except (TypeError, ValueError):
            continue
        if listed_before_ms is not None and listed > listed_before_ms:
            continue
        base = str(item.get("instId") or "").split("-")[0].strip().upper()
        if base:
            output[f"{base}USDT"] = str(item["instId"])
    return output


def snapshot_anchor_ms(now_seconds: float | None = None) -> int:
    """Return the open time of the currently forming 15-minute candle.

    OKX ``after`` returns candles older than the anchor.  Anchoring at the
    current interval open therefore includes the same last fully closed bar
    for every worker, even when the download crosses a candle boundary.
    """
    now = time.time() if now_seconds is None else float(now_seconds)
    return int(now // BAR_SECONDS) * BAR_SECONDS * 1000


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
    parser.add_argument("--append", action="store_true",
                        help="Merge requested symbols into an existing anchored snapshot")
    parser.add_argument(
        "--listed-before",
        help="For --symbols ALL, include instruments listed on or before YYYY-MM-DD UTC",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    listed_before_ms = None
    if args.listed_before:
        try:
            listed_before_ms = int(
                datetime.fromisoformat(args.listed_before)
                .replace(tzinfo=timezone.utc).timestamp() * 1000)
        except ValueError as exc:
            parser.error(f"invalid --listed-before date: {exc}")
    instruments = get_json(
        requests.Session(),
        "https://www.okx.com/api/v5/public/instruments",
        {"instType": "FUTURES", "ruleType": "xperp"},
    ).get("data", [])
    mapping = equity_mapping(instruments, listed_before_ms)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.json"
    # All symbols must describe the same point-in-time snapshot.  A per-symbol
    # clock could cross a 15-minute close during this sequential download and
    # give later symbols one additional completed bar.
    if args.append and manifest_path.exists():
        report = json.loads(manifest_path.read_text(encoding="utf-8"))
        request_anchor_ms = int(report["request_anchor_ms"])
        if int(report["bars_requested"]) != args.bars:
            parser.error("--append requires the original --bars value")
    else:
        request_anchor_ms = snapshot_anchor_ms()
        report = {"status": "PUBLIC_CURRENT_STOCK_XPERP",
                  "created_at": datetime.now(timezone.utc).isoformat(),
                  "bars_requested": args.bars,
                  "request_anchor_ms": request_anchor_ms, "symbols": {}}
    symbols = (tuple(sorted(mapping)) if args.symbols.strip().upper() == "ALL"
               else tuple(dict.fromkeys(
                   value.strip().upper() for value in args.symbols.split(",")
                   if value.strip())))
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
        symbol: report["symbols"][symbol] for symbol in sorted(report["symbols"])
    }
    manifest_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
