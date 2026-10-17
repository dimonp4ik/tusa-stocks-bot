"""Immutable public-data forward monitor for the frozen native growth profile.

The monitor has no exchange-authentication, Telegram, or order code.  It reads
one point-in-time public candle manifest, keeps only entries at or after the
frozen audit cutoff, and refuses to overwrite an existing report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import stock_cash_xperp_leadlag_oct6_audit as lead
from stock_module_cluster_audit import portfolio_gate
import stock_xperp_native_cross_section_oct18_audit as parent
import stock_xperp_native_economics_oct18_audit as economics
import stock_xperp_native_growth_profile_oct18_audit as growth


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest(path: Path) -> tuple[dict, dict]:
    frozen = json.loads(parent.MANIFEST.read_text(encoding="utf-8"))
    current = json.loads(path.read_text(encoding="utf-8"))
    if set(current["symbols"]) != set(frozen["symbols"]):
        raise ValueError("current universe differs from the frozen universe")
    candles, sources = {}, {str(path): digest(path)}
    for symbol, item in current["symbols"].items():
        if symbol in parent.EXCLUDED:
            continue
        if item["status"] not in ("OK", "SHORT_HISTORY"):
            raise ValueError(f"missing frozen symbol: {symbol}")
        source = Path(item["path"])
        raw = source.read_bytes()
        if hashlib.sha256(raw).hexdigest() != item["sha256"]:
            raise ValueError(f"source hash mismatch: {symbol}")
        candles[symbol] = pickle.loads(raw)
        sources[str(source)] = hashlib.sha256(raw).hexdigest()
    return candles, sources


def resolve_row(row: dict, candle: dict, target_pct: float, stop_pct: float,
                hold_bars: int) -> dict:
    output = dict(row)
    open_, high, low, close = [np.asarray(candle[key], dtype=float)
                               for key in ("open", "high", "low", "close")]
    times = np.asarray(candle["time"], dtype=np.int64)
    entry_index = int(row["entry_index"])
    direction = int(row["direction_n"])
    entry = float(open_[entry_index])
    target_fraction, stop_fraction = target_pct / 100, stop_pct / 100
    target = entry * (1 + direction * target_fraction)
    stop = entry * (1 - direction * stop_fraction)
    required_last = entry_index + hold_bars - 1
    available_last = min(required_last, len(times) - 1)
    signed_return = None
    outcome_code = None
    exit_index = available_last
    for index in range(entry_index, available_last + 1):
        if index > entry_index and times[index] - times[index - 1] != 900:
            exit_index = index - 1
            signed_return = direction * (close[exit_index] / entry - 1)
            outcome_code = 0
            break
        opening = float(open_[index])
        if direction > 0:
            if opening <= stop:
                exit_index, signed_return, outcome_code = index, opening / entry - 1, -1
                break
            if opening >= target:
                exit_index, signed_return, outcome_code = index, opening / entry - 1, 1
                break
            if low[index] <= stop:  # stop first on a same-bar ambiguity
                exit_index, signed_return, outcome_code = index, -stop_fraction, -1
                break
            if high[index] >= target:
                exit_index, signed_return, outcome_code = index, target_fraction, 1
                break
        else:
            if opening >= stop:
                exit_index, signed_return, outcome_code = index, 1 - opening / entry, -1
                break
            if opening <= target:
                exit_index, signed_return, outcome_code = index, 1 - opening / entry, 1
                break
            if high[index] >= stop:
                exit_index, signed_return, outcome_code = index, -stop_fraction, -1
                break
            if low[index] <= target:
                exit_index, signed_return, outcome_code = index, target_fraction, 1
                break
    if signed_return is None and available_last >= required_last:
        signed_return = direction * (close[required_last] / entry - 1)
        outcome_code = 0
        exit_index = required_last
    pending = signed_return is None
    output.update({
        "entry": entry, "target_pct": target_pct, "stop_pct": stop_pct,
        "hold_bars": hold_bars,
        "direction": "LONG" if direction > 0 else "SHORT",
        "signed_return": None if pending else float(signed_return),
        "outcome_code": outcome_code,
        "outcome": ("PENDING" if pending else
                    "TP" if outcome_code == 1 else
                    "SL" if outcome_code == -1 else "TIME"),
        "exit_time": (int(times[entry_index]) + hold_bars * 900
                      if pending else int(times[exit_index]) + 900),
        "risk_pct": stop_fraction,
        "net_r": None if pending else float(signed_return / stop_fraction),
        "module": "native_xperp_growth_profile",
        "module_priority": 0,
    })
    return output


def resolve_entries(entries: pd.DataFrame, candles: dict,
                    parameters: dict) -> list[dict]:
    output = []
    for row in entries.to_dict("records"):
        output.append(resolve_row(
            row, candles[row["symbol"]],
            float(parameters["target_pct"]),
            float(parameters["stop_pct"]),
            int(parameters["hold_bars"])))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=growth.ROWS_OUT)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    csv_out = args.out.with_suffix(".csv")
    if args.out.exists() or csv_out.exists():
        raise FileExistsError("forward output is immutable; choose a new path")
    profile = json.loads(growth.OUT.read_text(encoding="utf-8"))
    parameters = profile["selection"]["selected"]["parameters"]
    candles, sources = load_manifest(args.manifest)
    features = parent.feature_rows(candles, int(parameters["lookback_bars"]))
    entries = parent.select_entries(
        features, float(parameters["z_threshold"]),
        float(parameters["minimum_move_pct"]) / 100)
    cutoff = pd.Timestamp(parent.FRESH[1], tz="UTC").timestamp()
    fresh_entries = entries[entries.entry_time.ge(cutoff)]
    fresh_raw = resolve_entries(fresh_entries, candles, parameters)
    reference = pd.read_csv(args.reference).to_dict("records")
    gated = portfolio_gate(reference + fresh_raw, None)
    fresh = [row for row in gated if float(row["entry_time"]) >= cutoff]
    resolved = [row for row in fresh if row["outcome"] != "PENDING"]
    pending = [row for row in fresh if row["outcome"] == "PENDING"]
    result = {
        "status": "SHADOW_ONLY_FORWARD_SNAPSHOT",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "ready_for_live_orders": False,
        "orders_possible": False,
        "audit_cutoff": parent.FRESH[1],
        "parameters": parameters,
        "fresh_raw_signals": len(fresh_raw),
        "fresh_portfolio_signals": len(fresh),
        "resolved_signals": len(resolved),
        "pending_signals": len(pending),
        "resolved_metrics_3pct": economics.simulate(resolved, .03),
        "resolved_metrics_5pct": economics.simulate(resolved, .05),
        "fresh": fresh,
        "sources": {
            **sources, str(args.reference): digest(args.reference),
            str(growth.OUT): digest(growth.OUT),
            str(Path(__file__)): digest(Path(__file__)),
        },
        "invariant": (
            "frozen universe and parameters; closed 15-minute feature; "
            "next-bar market entry; stop-first OHLC; same causal portfolio gate"),
        "limitations": [
            "This is an immutable public-data shadow snapshot, not account fills.",
            "Pending rows remain unresolved until a later immutable snapshot.",
            "A tiny forward sample cannot establish monthly profitability.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    pd.DataFrame(fresh).to_csv(csv_out, index=False)
    print(json.dumps({key: result[key] for key in (
        "status", "audit_cutoff", "fresh_raw_signals",
        "fresh_portfolio_signals", "resolved_signals", "pending_signals",
        "resolved_metrics_3pct", "resolved_metrics_5pct")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
