"""
Simple walk-forward validator for exported backtest trades.

For each month after the first, train on the previous month and select groups
whose train net R/trade and win rate pass thresholds. Then report the next
month's out-of-sample result for those selected groups.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone


# A "win" is a trade that CLOSED IN PROFIT after costs, not one whose outcome
# label looks favourable. The old set counted every TRAIL exit, but the trail is
# floored at breakeven in PRICE, so the round-trip fee pushes a flat trail exit
# negative. Measured 2026-08-28 on the current book: 48 trail exits and 25
# expiries closed below zero, which is why the backtest headline reads ~4pp
# higher than the share of trades that actually made money. The live reporting
# in src/db.py has always used r > 0; this now agrees with it.
WIN_OUTCOMES = {"TP1", "TP2", "TRAIL"}   # kept for reference, no longer used


@dataclass
class Bucket:
    trades: int = 0
    wins: int = 0
    net_r: float = 0.0

    def add(self, row: dict[str, str]) -> None:
        self.trades += 1
        try:
            _nr = float(row.get("net_r") or 0.0)
        except (TypeError, ValueError):
            _nr = 0.0
        self.wins += int(_nr > 0.0)
        self.net_r += float(row.get("net_r") or 0.0)

    @property
    def wr(self) -> float:
        return self.wins / self.trades * 100.0 if self.trades else 0.0

    @property
    def rpt(self) -> float:
        return self.net_r / self.trades if self.trades else 0.0


def month_key(row: dict[str, str]) -> str:
    ts = int(float(row.get("entry_time") or 0))
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m")


def group_key(row: dict[str, str], fields: list[str]) -> tuple[str, ...]:
    return tuple((row.get(f) or "-") for f in fields)


def load_rows(path: str) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: int(float(r.get("entry_time") or 0)))
    return rows


def summarize(rows: list[dict[str, str]]) -> Bucket:
    bucket = Bucket()
    for row in rows:
        bucket.add(row)
    return bucket


def main() -> int:
    parser = argparse.ArgumentParser(description="Walk-forward validation for exported backtest trades")
    parser.add_argument("csv_path")
    parser.add_argument("--group-fields", default="adaptive_pack,entry_source")
    parser.add_argument("--min-train-trades", type=int, default=30)
    parser.add_argument("--min-train-wr", type=float, default=50.0)
    parser.add_argument("--min-train-rpt", type=float, default=0.05)
    args = parser.parse_args()

    fields = [f.strip() for f in args.group_fields.split(",") if f.strip()]
    rows = load_rows(args.csv_path)
    by_month: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_month[month_key(row)].append(row)
    months = sorted(by_month)

    total_test: list[dict[str, str]] = []
    print(f"months={','.join(months)} group={'+'.join(fields)}")
    for prev, cur in zip(months, months[1:]):
        train_groups: dict[tuple[str, ...], Bucket] = defaultdict(Bucket)
        for row in by_month[prev]:
            train_groups[group_key(row, fields)].add(row)

        selected = {
            key for key, bucket in train_groups.items()
            if bucket.trades >= args.min_train_trades
            and bucket.wr >= args.min_train_wr
            and bucket.rpt >= args.min_train_rpt
        }
        test_rows = [row for row in by_month[cur] if group_key(row, fields) in selected]
        total_test.extend(test_rows)
        bucket = summarize(test_rows)
        print(
            f"train={prev} test={cur} selected={len(selected)} "
            f"test_tr={bucket.trades} wr={bucket.wr:.1f}% "
            f"net={bucket.net_r:+.2f}R rpt={bucket.rpt:+.3f}"
        )

    total = summarize(total_test)
    print(
        f"TOTAL_OOS tr={total.trades} wr={total.wr:.1f}% "
        f"net={total.net_r:+.2f}R rpt={total.rpt:+.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
