"""Build a streamed 2017-2026 stock session-family research dataset."""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

from stock_session_family_lab import (
    attach_intraday_market_context, attach_market_context,
    market_context_by_day, market_intraday_context_by_time, replay,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest_raw = args.manifest.read_bytes()
    manifest = json.loads(manifest_raw)
    source_meta = manifest["symbols"]

    qqq_meta = source_meta["QQQUSDT"]
    qqq_path = Path(qqq_meta["path"])
    if hashlib.sha256(qqq_path.read_bytes()).hexdigest() != qqq_meta["sha256"]:
        raise ValueError("QQQ cache hash changed after manifest creation")
    qqq_context = market_context_by_day(pickle.loads(qqq_path.read_bytes()))
    qqq_intraday = market_intraday_context_by_time(
        pickle.loads(qqq_path.read_bytes()), "qqq")
    spy_meta = source_meta["SPYUSDT"]
    spy_path = Path(spy_meta["path"])
    if hashlib.sha256(spy_path.read_bytes()).hexdigest() != spy_meta["sha256"]:
        raise ValueError("SPY cache hash changed after manifest creation")
    spy_intraday = market_intraday_context_by_time(
        pickle.loads(spy_path.read_bytes()), "spy")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    total = 0
    counts = collections.Counter()
    symbol_counts = {}
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        for symbol, meta in source_meta.items():
            path = Path(meta["path"])
            raw = path.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            if digest != meta["sha256"]:
                raise ValueError(f"{symbol}: cache hash changed after manifest creation")
            rows = replay(pickle.loads(raw), symbol)
            attach_market_context(rows, qqq_context)
            attach_intraday_market_context(rows, qqq_intraday, "qqq")
            attach_intraday_market_context(rows, spy_intraday, "spy")
            rows = [row for row in rows
                    if row.get("qqq_regime")
                    and row.get("qqq_intraday_move_atr") is not None
                    and row.get("spy_intraday_move_atr") is not None]
            if rows and writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
            if writer:
                writer.writerows(rows)
            symbol_counts[symbol] = len(rows)
            total += len(rows)
            for row in rows:
                year = datetime.fromtimestamp(row["entry_time"], timezone.utc).year
                counts[(year, float(row["target_r"]))] += 1
            print(json.dumps({"symbol": symbol, "rows": len(rows),
                              "total": total}), flush=True)

    report = {
        "status": "RESEARCH_INPUT_VALIDATED",
        "rows": total,
        "csv": str(args.out),
        "csv_sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
        "manifest": str(args.manifest),
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "symbol_rows": symbol_counts,
        "year_target_rows": {
            str(year): {str(target): counts[(year, target)]
                        for target in (.25, .5, .75, 1.0)}
            for year in sorted({key[0] for key in counts})
        },
        "method": ("next 15-minute bar market-entry proxy; causal prior-session QQQ "
                   "and exact closed-bar intraday QQQ/SPY context"),
        "limitations": [
            "Dukascopy validates long-run session structure; exact entries require X-Perp transfer checks.",
            "MRVL history begins in 2022 and MU history begins in late 2017.",
            "Gross outcomes exclude fees and client-selected position sizing at user request.",
        ],
    }
    report_path = args.out.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"rows": total, "years": report["year_target_rows"],
                      "sha256": report["csv_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
