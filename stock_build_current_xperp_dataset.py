"""Build causal stock-session families directly from current X-Perp candles."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import pandas as pd

from stock_session_family_lab import (
    attach_intraday_market_context,
    attach_market_context,
    market_context_by_day,
    market_intraday_context_by_time,
    replay,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    caches = {}
    sources = {}
    for path in sorted(args.cache_dir.glob("*_15min_current.pkl")):
        symbol = path.name.split("_15min_")[0]
        raw = path.read_bytes()
        caches[symbol] = pickle.loads(raw)
        sources[str(path)] = hashlib.sha256(raw).hexdigest()
    rows = []
    for symbol, candles in caches.items():
        rows.extend(replay(candles, symbol, execution=candles))
    if "QQQUSDT" in caches:
        attach_market_context(rows, market_context_by_day(caches["QQQUSDT"]))
        attach_intraday_market_context(
            rows, market_intraday_context_by_time(caches["QQQUSDT"], "qqq"), "qqq")
    if "SPYUSDT" in caches:
        attach_intraday_market_context(
            rows, market_intraday_context_by_time(caches["SPYUSDT"], "spy"), "spy")
    frame = pd.DataFrame(rows).sort_values(["entry_time", "symbol", "family", "target_r"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    month = pd.to_datetime(frame["entry_time"], unit="s", utc=True).dt.strftime("%Y-%m")
    report = {
        "status": "DIRECT_CURRENT_XPERP_FAMILIES",
        "rows": len(frame), "symbols": sorted(caches),
        "range": [pd.to_datetime(frame["entry_time"].min(), unit="s", utc=True).isoformat(),
                  pd.to_datetime(frame["entry_time"].max(), unit="s", utc=True).isoformat()],
        "months": {key: {"n": len(part), "net_r": float(part["net_r"].sum()),
                         "wr": float((part["net_r"] > 0).mean() * 100)}
                   for key, part in frame.groupby(month)},
        "csv": {"path": str(args.out), "sha256": hashlib.sha256(args.out.read_bytes()).hexdigest()},
        "sources": sources,
        "limitations": ["Direct X-Perp entries use the next 15-minute market open proxy.",
                        "Gross outcomes exclude fees and position sizing at user request."],
    }
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("rows", "symbols", "range", "months")},
                     indent=2))


if __name__ == "__main__":
    main()
