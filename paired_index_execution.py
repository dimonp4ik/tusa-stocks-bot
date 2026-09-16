"""Replay index-derived callbacks on the timestamp-matched X-Perp OHLC."""
import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path

from config import STOP_CLOSE_CONFIRM, STOP_EXCHANGE_BACKSTOP_R
from filter_lab import gate, metrics


ROUNDTRIP_SIDE_COST = 0.0006


def replay(row, candles):
    lookup = {t: i for i, t in enumerate(candles["time"])}
    entry_time = int(float(row["entry_time"]))
    callback_bar_time = int(float(row["exit_time"])) - 900
    if entry_time not in lookup or callback_bar_time not in lookup:
        raise ValueError("missing execution timestamp")
    first, last = lookup[entry_time], lookup[callback_bar_time]
    if last < first:
        raise ValueError("exit before entry")
    expected = list(range(entry_time, callback_bar_time + 900, 900))
    if candles["time"][first:last + 1] != expected:
        raise ValueError("execution gap")
    planned = float(row["entry"])
    entry = float(candles["open"][first])
    if planned <= 0 or entry <= 0:
        raise ValueError("invalid entry")
    basis = entry / planned
    direction = row["direction"]
    sign = 1 if direction == "LONG" else -1
    sl = float(row["sl"]) * basis
    tp1 = float(row["tp1"]) * basis
    tp2 = float(row["tp2"]) * basis
    risk = abs(entry - sl)
    if risk <= 0 or not (sl < entry < tp1 if sign == 1 else tp1 < entry < sl):
        return None
    stop = entry - sign * risk * max(1.0, STOP_EXCHANGE_BACKSTOP_R) \
        if STOP_CLOSE_CONFIRM else sl
    outcome = row["outcome"]
    exit_price = float(candles["close"][last])
    exit_bar = last
    for j in range(first, last + 1):
        opening, high, low = (float(candles[k][j]) for k in ("open", "high", "low"))
        if low <= stop if sign == 1 else high >= stop:
            exit_price = min(opening, stop) if sign == 1 else max(opening, stop)
            outcome, exit_bar = "SL", j
            break
        if high >= tp2 if sign == 1 else low <= tp2:
            exit_price = tp2
            outcome, exit_bar = "TP2", j
            break
    weight = float(row["size_mult"])
    net_r = (sign * (exit_price - entry)
             - ROUNDTRIP_SIDE_COST * (entry + exit_price)) / risk * weight
    return {
        "symbol": row["symbol"], "direction": direction,
        "entry_time": entry_time, "exit_time": candles["time"][exit_bar] + 900,
        "net_r": net_r, "outcome": outcome, "risk_pct": risk / entry,
        "size_mult": weight, "basis_at_entry": basis,
        "source_outcome": row["outcome"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-label", required=True)
    parser.add_argument("--output-label", required=True)
    parser.add_argument("--cap", type=int, required=True)
    args = parser.parse_args()
    folder = Path("reports/audit_2026_09_08")
    source = folder / f"{args.source_label}_raw.csv"
    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    market = [row for row in rows if row["entry_is_intrabar"] == "False"]
    candles, hashes = {}, {}
    for symbol in sorted({row["symbol"] for row in market}):
        path = folder / "xperp_cache" / f"{symbol}_15min_18000.pkl"
        raw = path.read_bytes()
        candles[symbol] = pickle.loads(raw)
        hashes[str(path)] = hashlib.sha256(raw).hexdigest()
    actual, errors, rejected = [], [], 0
    for row in market:
        try:
            item = replay(row, candles[row["symbol"]])
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if item is None:
            rejected += 1
        else:
            actual.append(item)
    accepted = gate(sorted(actual, key=lambda r: (r["entry_time"], r["symbol"])), args.cap)
    report = {
        "status": "EXECUTION_DIAGNOSTIC",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_hashes": hashes,
        "market_entries": len(market),
        "excluded_intrabar_entries": len(rows) - len(market),
        "bracket_rejections": rejected,
        "errors": errors,
        "raw": metrics(actual), "gated": metrics(accepted),
        "cost_per_side": ROUNDTRIP_SIDE_COST,
        "limitations": [
            "Index signal levels are rescaled by entry-time X-Perp/index basis.",
            "15-minute OHLC approximates fills; stop wins ambiguous bars.",
            "Source callback timing is modeled, not a production polling log.",
            "Intrabar watched entries excluded. Funding and historical book depth omitted.",
            "Previously inspected history; no independent live proof.",
        ],
    }
    (folder / f"{args.output_label}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report))


if __name__ == "__main__":
    main()
