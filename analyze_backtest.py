"""
Analyze exported backtest trades.

Usage:
    python analyze_backtest.py trades.csv
    python analyze_backtest.py trades.csv --groups session,entry_source,trend_pair
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from statistics import mean, pstdev


WIN_OUTCOMES = {"TP1", "TP2", "TRAIL"}
LOSS_OUTCOMES = {"SL"}


@dataclass
class Bucket:
    trades: int = 0
    wins: int = 0
    sl: int = 0
    expired: int = 0
    net_r: float = 0.0
    gross_r: float = 0.0

    def add(self, row: dict[str, str]) -> None:
        outcome = row["outcome"]
        self.trades += 1
        self.wins += int(outcome in WIN_OUTCOMES)
        self.sl += int(outcome in LOSS_OUTCOMES)
        self.expired += int(outcome == "EXPIRED")
        self.net_r += float(row.get("net_r") or 0.0)
        self.gross_r += float(row.get("gross_r") or 0.0)

    @property
    def win_rate(self) -> float:
        return self.wins / self.trades * 100 if self.trades else 0.0

    @property
    def resolved_wr(self) -> float:
        resolved = self.wins + self.sl
        return self.wins / resolved * 100 if resolved else 0.0

    @property
    def expired_pct(self) -> float:
        return self.expired / self.trades * 100 if self.trades else 0.0

    @property
    def sl_pct(self) -> float:
        return self.sl / self.trades * 100 if self.trades else 0.0

    @property
    def net_rpt(self) -> float:
        return self.net_r / self.trades if self.trades else 0.0


def month_key(ts: str) -> str:
    dt = datetime.fromtimestamp(int(float(ts)), tz=timezone.utc)
    return dt.strftime("%Y-%m")


def trend_pair(row: dict[str, str]) -> str:
    return f"{row.get('trend_1h', '')}/{row.get('trend_4h', '')}"


def numeric_band(row: dict[str, str], key: str, cuts: list[float]) -> str:
    value = float(row.get(key) or 0.0)
    prev = "-inf"
    for cut in cuts:
        if value < cut:
            return f"{key}:{prev}-{cut:g}"
        prev = f"{cut:g}"
    return f"{key}:{prev}+"


def group_value(row: dict[str, str], group: str) -> str:
    if group == "trend_pair":
        return trend_pair(row)
    if group == "volume_band":
        return numeric_band(row, "volume_ratio", [1.6, 1.8, 2.0, 2.5, 3.0])
    if group == "eff_band":
        return numeric_band(row, "eff_ratio", [0.15, 0.20, 0.25, 0.30, 0.40])
    if group == "volreg_band":
        return numeric_band(row, "vol_ratio_regime", [0.55, 0.8, 1.2, 1.8, 3.0])
    if group == "rsi_band":
        return numeric_band(row, "rsi", [30, 40, 50, 60, 70])
    return row.get(group, "") or "-"


def print_bucket(label: str, bucket: Bucket) -> None:
    print(
        f"{label:<18} tr={bucket.trades:<5} wr={bucket.win_rate:>5.1f}% "
        f"resWR={bucket.resolved_wr:>5.1f}% exp={bucket.expired_pct:>5.1f}% "
        f"sl={bucket.sl_pct:>5.1f}% net={bucket.net_r:>9.2f}R "
        f"rpt={bucket.net_rpt:>7.3f}"
    )


def load_rows(path: str) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: (int(float(r.get("entry_time") or 0)), r.get("symbol", "")))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("--groups", default="adaptive_pack,session,entry_source,trend_pair,volume_band,eff_band,volreg_band")
    parser.add_argument("--min-trades", type=int, default=50)
    args = parser.parse_args()

    rows = load_rows(args.csv_path)
    total = Bucket()
    by_month: dict[str, Bucket] = defaultdict(Bucket)
    for row in rows:
        total.add(row)
        by_month[month_key(row["entry_time"])].add(row)

    print("TOTAL")
    print_bucket("all", total)

    print("\nMONTHLY")
    month_wrs = []
    month_net = []
    for key in sorted(by_month):
        bucket = by_month[key]
        print_bucket(key, bucket)
        if bucket.trades >= args.min_trades:
            month_wrs.append(bucket.win_rate)
            month_net.append(bucket.net_r)

    if month_wrs:
        print("\nSTABILITY")
        print(f"months={len(month_wrs)} avg_wr={mean(month_wrs):.2f}% std_wr={pstdev(month_wrs):.2f}%")
        print(f"min_wr={min(month_wrs):.2f}% max_wr={max(month_wrs):.2f}%")
        print(f"positive_net_months={sum(1 for x in month_net if x > 0)}/{len(month_net)}")

    for group in [g.strip() for g in args.groups.split(",") if g.strip()]:
        buckets: dict[str, Bucket] = defaultdict(Bucket)
        for row in rows:
            buckets[group_value(row, group)].add(row)
        print(f"\nGROUP {group}")
        filtered = [(k, b) for k, b in buckets.items() if b.trades >= args.min_trades]
        for key, bucket in sorted(filtered, key=lambda kv: kv[1].net_r, reverse=True):
            print_bucket(key, bucket)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
