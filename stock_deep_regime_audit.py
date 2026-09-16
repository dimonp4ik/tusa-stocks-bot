"""Replay the frozen stock opening-breakout rule on multi-year cash-session data."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

from filter_lab import gate, metrics
from index_session_hypotheses import atr_at, candidate_records
from src.backtest_integrity import simulate_exit


def replay(candles: dict, symbol: str) -> list[dict]:
    rows = []
    for item in candidate_records(candles, "opening_breakout"):
        prior = item["prior_day_directional_atr"]
        if item["direction"] != -1 or prior is None or prior > 1:
            continue
        first = item["entry_bar"]
        last = item["session_end"]
        atr = atr_at(candles, first)
        if not atr or first > last:
            continue
        entry = float(candles["open"][first])
        stop = entry + atr
        target = entry - .5 * atr
        if target <= 0:
            continue
        result = simulate_exit(
            candles, range(first, last + 1), direction="SHORT", entry=entry,
            sl=stop, tp1=target, tp2=target, atr=atr,
            tp1_fraction=1, trail=False, trail_mult=0, stop_on_close=False,
            backstop_r=1, choose_trail=lambda *_: 0,
        )
        rows.append({
            "symbol": symbol, "direction": "SHORT",
            "entry_time": candles["time"][first],
            "exit_time": candles["time"][result.bar] + 900,
            "outcome": result.outcome, "net_r": result.gross_r,
            "risk_pct": atr / entry, "size_mult": 1,
            **{key: value for key, value in item.items()
               if key not in {"direction", "signal_bar", "entry_bar", "session_end"}},
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--symbols", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    sources = {}
    for symbol in args.symbols.split(","):
        path = args.cache_dir / f"{symbol}_15min_26000.pkl"
        raw = path.read_bytes()
        sources[symbol] = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}
        rows.extend(replay(pickle.loads(raw), symbol))
    accepted = gate(sorted(rows, key=lambda row: (row["entry_time"], row["symbol"])), 4)
    detail = args.out.with_suffix(".csv")
    if rows:
        with detail.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    years = sorted({datetime.fromtimestamp(row["entry_time"], timezone.utc).year
                    for row in accepted})
    report = {
        "status": "DEEP_FEED_CONFIRMATION",
        "rule": [["direction", "eq", "SHORT"],
                 ["prior_day_directional_atr", "le", 1]],
        "target_r": .5, "stop_atr": 1,
        "raw_detail": str(detail),
        "all": metrics(accepted),
        "years": {str(year): metrics([row for row in accepted if
            datetime.fromtimestamp(row["entry_time"], timezone.utc).year == year])
            for year in years},
        "symbols": {symbol: metrics([row for row in accepted if row["symbol"] == symbol])
                    for symbol in args.symbols.split(",")},
        "sources": sources,
        "limitations": [
            "Dukascopy bid OHLC confirms the underlying pattern, not exact OKX X-Perp fills.",
            "The separate recent X-Perp replay is required for execution-venue confirmation.",
            "Gross outcomes only; fees and position sizing are excluded.",
        ],
    }
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"all": report["all"], "years": report["years"],
                      "symbols": report["symbols"]}, indent=2))


if __name__ == "__main__":
    main()
