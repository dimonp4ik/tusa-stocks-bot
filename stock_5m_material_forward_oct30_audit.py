"""Oct-30 forward audit of the exact frozen material-move stock 5m shadow router.

No route or threshold is fitted here.  The script calls the production
``candidate_context`` rules on closed public X-Perp bars, enters at the next
five-minute market-open proxy, enforces the current >=0.5% first target, and
replays the frozen 70/30 near-target/runner exits through the cash-session
close.  The output separates the prior-data append from observations after the
material-target code freeze.
"""
from __future__ import annotations

import hashlib
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

import stock_5m_barbell_runner_oct3_audit as barbell
import stock_5m_family_oct3_audit as family
from stock_5m_high_payoff_oct3_audit import accepted, money
from src import stock_5m_shadow_router as runtime


ROOT = Path("reports/audit_2026_09_08")
OLD_CACHE = ROOT / "stock_xperp_5m_oct3"
NEW_CACHE = ROOT / "stock_xperp_5m_oct30"
NEW_MANIFEST = NEW_CACHE / "manifest.json"
OLD_SELECTED = ROOT / "stock_5m_regime_filter_oct3_rows.csv"
OUT = ROOT / "stock_5m_material_forward_oct30.json"
ROWS_OUT = ROOT / "stock_5m_material_forward_oct30.csv"
FIELDS = ("time", "open", "high", "low", "close", "volume")
RUNNERS = {
    runtime.RANGE_MODULE: 1.0,
    runtime.PULLBACK_MODULE: 4.0,
}
MODULE_PRIORITY = {
    runtime.RANGE_MODULE: 100,
    runtime.PULLBACK_MODULE: 101,
}
LOCK_FRACTION = 0.70
# daf3b0d: stocks: enforce material shadow targets
MATERIAL_FREEZE = int(pd.Timestamp("2026-10-06T05:10:51Z").timestamp())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_caches() -> tuple[dict, dict, dict]:
    manifest = json.loads(NEW_MANIFEST.read_text(encoding="utf-8"))
    sources = {str(NEW_MANIFEST): digest(NEW_MANIFEST)}
    integrity = {}
    caches = {}
    for symbol in sorted(runtime.AUDITED_SYMBOLS):
        item = manifest["symbols"][symbol]
        path = Path(item["path"])
        raw = path.read_bytes()
        found = hashlib.sha256(raw).hexdigest()
        if found != item["sha256"]:
            raise ValueError(f"hash mismatch: {path}")
        candle = pickle.loads(raw)
        lengths = {field: len(candle.get(field, [])) for field in FIELDS}
        if len(set(lengths.values())) != 1 or not lengths["time"]:
            raise ValueError(f"bad candle lengths: {symbol} {lengths}")
        times = [int(value) for value in candle["time"]]
        if any(right - left != 300 for left, right in zip(times, times[1:])):
            raise ValueError(f"non-contiguous 5m history: {symbol}")
        caches[symbol] = candle
        sources[str(path)] = found

        old_path = OLD_CACHE / f"{symbol}_5min_current.pkl"
        old = pickle.loads(old_path.read_bytes())
        sources[str(old_path)] = digest(old_path)
        old_index = {int(value): index for index, value in enumerate(old["time"])}
        new_index = {int(value): index for index, value in enumerate(candle["time"])}
        common = sorted(set(old_index) & set(new_index))
        changes = {field: 0 for field in FIELDS[1:]}
        for timestamp in common:
            for field in changes:
                if old[field][old_index[timestamp]] != candle[field][new_index[timestamp]]:
                    changes[field] += 1
        integrity[symbol] = {
            "overlap_bars": len(common),
            "overlap_start": common[0] if common else None,
            "overlap_end": common[-1] if common else None,
            "revisions": changes,
        }
        if any(changes.values()):
            raise ValueError(f"confirmed candle revision: {symbol} {changes}")
    return caches, sources, integrity


def candle_slice(candle: dict, end: int, length: int = 110) -> dict:
    start = max(0, end - length + 1)
    return {field: candle[field][start:end + 1] for field in FIELDS}


def context(candle: dict) -> dict:
    lookup = {int(value): index for index, value in enumerate(candle["time"])}
    session_last = {}
    for bars in family.regular_groups(candle).values():
        for index in bars:
            session_last[index] = bars[-1]
    return {"candles": candle, "index": lookup, "session_last": session_last}


def replay(caches: dict) -> tuple[list[dict], list[dict]]:
    qqq = caches["QQQUSDT"]
    qqq_index = {int(value): index for index, value in enumerate(qqq["time"])}
    contexts = {symbol: context(candle) for symbol, candle in caches.items()}
    raw, material = [], []
    for symbol in sorted(runtime.AUDITED_SYMBOLS):
        candle = caches[symbol]
        times = [int(value) for value in candle["time"]]
        for signal_index in range(97, len(times) - 1):
            signal_time = times[signal_index]
            next_time = times[signal_index + 1]
            q_index = qqq_index.get(signal_time)
            if q_index is None or q_index < 97 or next_time != signal_time + 300:
                continue
            candidate = runtime.candidate_context(
                candle_slice(candle, signal_index),
                candle_slice(qqq, q_index), symbol)
            if candidate is None:
                continue
            entry = float(candle["open"][signal_index + 1])
            risk = (float(candidate["source_atr"])
                    * float(candidate["stop_atr_mult"]))
            move_pct = 100.0 * risk * float(candidate["target_r"]) / entry
            row = {
                **candidate,
                "symbol": symbol,
                "entry_time": next_time,
                "entry_price": entry,
                "risk_pct": risk / entry,
                "first_target_move_pct": move_pct,
                "module_priority": MODULE_PRIORITY[candidate["module"]],
            }
            raw.append(row)
            if move_pct + 1e-12 < runtime.MIN_FIRST_TARGET_MOVE_PCT:
                continue
            resolved = barbell.simulate(
                row, contexts[symbol], RUNNERS[candidate["module"]],
                LOCK_FRACTION)
            outcome = str(resolved["outcome"])
            resolved["literal_target_hit"] = outcome.startswith("TP_")
            resolved["signed_price_return"] = (
                float(resolved["risk_pct"]) * float(resolved["net_r"]))
            material.append(resolved)
    return raw, material


def event_key(row: dict) -> tuple[str, int]:
    return str(row["symbol"]), int(float(row["entry_time"]))


def metrics(rows: list[dict]) -> dict:
    if not rows:
        return {
            "n": 0, "wr": 0.0, "return_pct": 0.0, "dd_pct": 0.0,
            "literal_target_hits": 0, "literal_target_hit_rate": 0.0,
            "first_entry_utc": None, "last_entry_utc": None,
            "module_count": {}, "outcomes": {},
        }
    value = money(rows)
    selected = accepted(rows)
    literal_hits = sum(bool(row["literal_target_hit"]) for row in selected)
    moves = [float(row["first_target_move_pct"]) for row in selected]
    value.update({
        "literal_target_hits": literal_hits,
        "literal_target_hit_rate": round(
            100 * literal_hits / len(selected), 4) if selected else 0.0,
        "module_count": dict(sorted(Counter(
            str(row["module"]) for row in selected).items())),
        "outcomes": dict(sorted(Counter(
            str(row["outcome"]) for row in selected).items())),
        "first_target_move_pct": {
            "min": round(min(moves), 6), "median": round(float(pd.Series(moves).median()), 6),
            "max": round(max(moves), 6),
            "above_2pct": sum(value > 2.0 for value in moves),
        } if moves else {},
        "first_entry_utc": pd.Timestamp(
            min(float(row["entry_time"]) for row in selected),
            unit="s", tz="UTC").isoformat() if selected else None,
        "last_entry_utc": pd.Timestamp(
            max(float(row["entry_time"]) for row in selected),
            unit="s", tz="UTC").isoformat() if selected else None,
    })
    return value


def breakdown(rows: list[dict], fmt: str) -> dict:
    groups = defaultdict(list)
    for row in rows:
        key = pd.Timestamp(float(row["entry_time"]), unit="s", tz="UTC").strftime(fmt)
        groups[key].append(row)
    return {key: metrics(part) for key, part in sorted(groups.items())}


def main() -> None:
    caches, sources, integrity = load_caches()
    raw, material = replay(caches)
    accepted_all = accepted(material)
    old = pd.read_csv(OLD_SELECTED)
    old_max = int(float(old["entry_time"].max()))
    overlap_start = max(int(float(old["entry_time"].min())),
                        min(int(row["entry_time"]) for row in raw))
    old_overlap = old[
        (old.entry_time >= overlap_start) & (old.entry_time <= old_max)]
    raw_overlap = [row for row in raw
                   if overlap_start <= int(row["entry_time"]) <= old_max]
    old_keys = {event_key(row) for row in old_overlap.to_dict("records")}
    replay_keys = {event_key(row) for row in raw_overlap}

    post_prior_raw = [row for row in raw if int(row["entry_time"]) > old_max]
    post_prior_material = [row for row in material
                           if int(row["entry_time"]) > old_max]
    post_prior_accepted = [row for row in accepted_all
                           if int(row["entry_time"]) > old_max]
    post_freeze_raw = [row for row in raw
                       if int(row["entry_time"]) > MATERIAL_FREEZE]
    post_freeze_material = [row for row in material
                            if int(row["entry_time"]) > MATERIAL_FREEZE]
    post_freeze_accepted = [row for row in accepted_all
                            if int(row["entry_time"]) > MATERIAL_FREEZE]

    forward_result = metrics(post_freeze_accepted)
    forward_gate_met = bool(
        len(post_freeze_accepted) >= 30
        and forward_result["return_pct"] > 0
        and forward_result["negative_months"] == 0)
    report = {
        "status": ("FORWARD_GATE_PASSED_SHADOW_ONLY" if forward_gate_met
                   else "FORWARD_GATE_FAILED"),
        "ready_for_live_orders": False,
        "orders_possible": False,
        "protocol": {
            "selection": "exact current src.stock_5m_shadow_router; no fitting",
            "signal": "fresh closed 5m X-Perp event in US regular session",
            "entry": "next 5m market open proxy",
            "minimum_first_target_move_pct": runtime.MIN_FIRST_TARGET_MOVE_PCT,
            "exit": "70% near target, 30% runner, breakeven from next bar, stop-first",
            "runners_r": RUNNERS,
            "expiry": "same US cash-session close",
            "portfolio": "one symbol open, 3h cooldown, cap2/15m, direction cap4, daily pause after 3 losses",
            "money": "3% current-equity margin, fixed 10x gross",
            "orders": "market proxy only; shadow runtime cannot place real orders",
        },
        "integrity": {
            "candle_overlap": integrity,
            "legacy_research_event_parity": {
                "period_start": overlap_start,
                "period_end": old_max,
                "legacy_events": len(old_overlap),
                "current_runtime_events": len(raw_overlap),
                "matched": len(old_keys & replay_keys),
                "legacy_missing_from_runtime": len(old_keys - replay_keys),
                "runtime_unexpected": len(replay_keys - old_keys),
                "interpretation": (
                    "Exact parity proves the current fresh-transition runtime reproduces "
                    "the legacy selected events on their shared period. Material-target "
                    "eligibility is measured separately below."),
            },
        },
        "material_filter": {
            "all_current_runtime_events": len(raw),
            "all_events_passing_0_5pct": len(material),
            "rejected_below_0_5pct": len(raw) - len(material),
            "pass_rate_pct": round(100 * len(material) / len(raw), 4) if raw else 0.0,
        },
        "post_prior_data": {
            "cutoff": old_max,
            "cutoff_utc": pd.Timestamp(old_max, unit="s", tz="UTC").isoformat(),
            "raw_events": len(post_prior_raw),
            "material_events": len(post_prior_material),
            "accepted_events": len(post_prior_accepted),
            "result": metrics(post_prior_accepted),
            "by_day": breakdown(post_prior_accepted, "%Y-%m-%d"),
            "by_month": breakdown(post_prior_accepted, "%Y-%m"),
        },
        "genuinely_post_material_freeze": {
            "cutoff": MATERIAL_FREEZE,
            "cutoff_utc": pd.Timestamp(MATERIAL_FREEZE, unit="s", tz="UTC").isoformat(),
            "raw_events": len(post_freeze_raw),
            "material_events": len(post_freeze_material),
            "accepted_events": len(post_freeze_accepted),
            "result": forward_result,
            "by_day": breakdown(post_freeze_accepted, "%Y-%m-%d"),
        },
        "full_current_runtime_window": metrics(accepted_all),
        "readiness": {
            "required_genuinely_new_accepted": 30,
            "observed_genuinely_new_accepted": len(post_freeze_accepted),
            "statistically_ready": len(post_freeze_accepted) >= 30,
            "positive_return": forward_result["return_pct"] > 0,
            "no_negative_months": forward_result["negative_months"] == 0,
            "met": forward_gate_met,
            "live_activation_allowed": False,
        },
        "sources": {
            **sources,
            str(OLD_SELECTED): digest(OLD_SELECTED),
            str(Path(runtime.__file__)): digest(Path(runtime.__file__)),
            str(Path(__file__)): digest(Path(__file__)),
        },
        "limitations": [
            "The material-target rule produced only one accepted post-freeze trade through Oct 29.",
            "Public OHLC cannot reproduce live spread, latency or order-book slippage.",
            "The 30-200% monthly objective and absence of losing months remain unproven.",
        ],
    }
    pd.DataFrame(material).to_csv(ROWS_OUT, index=False)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "legacy_research_event_parity": report["integrity"]["legacy_research_event_parity"],
        "material_filter": report["material_filter"],
        "post_prior_data": report["post_prior_data"],
        "genuinely_post_material_freeze": report["genuinely_post_material_freeze"],
        "full_current_runtime_window": report["full_current_runtime_window"],
        "readiness": report["readiness"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()

