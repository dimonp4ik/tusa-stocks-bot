"""Oct-13 append-only check of the frozen stock 5m material router."""
from __future__ import annotations

import hashlib
import json
import pickle
from collections import Counter
from pathlib import Path

import pandas as pd

import stock_5m_material_forward_oct8_audit as audit


ROOT = Path("reports/audit_2026_09_08")
PREVIOUS = ROOT / "stock_xperp_5m_oct9_forward" / "manifest.json"
CURRENT = ROOT / "stock_xperp_5m_oct13_forward" / "manifest.json"
OUT = ROOT / "stock_5m_material_forward_oct13.json"
ROWS = ROOT / "stock_5m_material_forward_oct13.csv"
FIELDS = ("open", "high", "low", "close", "volume")
REQUIRED_PRICE_RETURN_PCT = 1.327709


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path.cwd() / path


def verified_manifest(path: Path) -> tuple[dict, dict]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    candles = {}
    for symbol in sorted(audit.runtime.AUDITED_SYMBOLS):
        item = manifest["symbols"][symbol]
        source = resolve_path(item["path"])
        raw = source.read_bytes()
        if hashlib.sha256(raw).hexdigest() != item["sha256"]:
            raise ValueError(f"source hash mismatch: {symbol}")
        candles[symbol] = pickle.loads(raw)
    return manifest, candles


def append_parity(previous: dict, current: dict) -> dict:
    result = {}
    for symbol in sorted(audit.runtime.AUDITED_SYMBOLS):
        old = previous[symbol]
        new = current[symbol]
        old_index = {int(value): index for index, value in enumerate(old["time"])}
        new_index = {int(value): index for index, value in enumerate(new["time"])}
        common = sorted(set(old_index) & set(new_index))
        mismatches = {field: sum(
            float(old[field][old_index[timestamp]])
            != float(new[field][new_index[timestamp]])
            for timestamp in common) for field in FIELDS}
        if any(mismatches.values()):
            raise ValueError(f"confirmed source revision: {symbol} {mismatches}")
        result[symbol] = {
            "overlap_bars_verified": len(common),
            "overlap_start": common[0] if common else None,
            "overlap_end": common[-1] if common else None,
            "field_mismatches": mismatches,
            "new_bars_after_previous": sum(
                int(value) > max(old_index) for value in new["time"]),
        }
    return result


def economics(rows: list[dict], value: dict) -> dict:
    n = len(rows)
    mean_price = (100 * sum(float(row["signed_price_return"])
                            for row in rows) / n if n else 0.0)
    daily = float(value.get("trades_per_active_day", 0))
    per_trade_equity = .03 * 10 * mean_price / 100
    projected = (100 * ((1 + per_trade_equity) ** (22 * daily) - 1)
                 if n and 1 + per_trade_equity > 0 else 0.0)
    clusters = Counter(int(float(row["entry_time"])) for row in rows)
    return {
        "mean_signed_price_return_pct": round(mean_price, 6),
        "required_mean_price_return_pct_at_3_per_market_day":
            REQUIRED_PRICE_RETURN_PCT,
        "projected_22_session_return_at_observed_frequency_pct":
            round(projected, 4),
        "economic_30pct_gate": bool(
            daily >= 3 and mean_price >= REQUIRED_PRICE_RETURN_PCT),
        "max_entries_same_5m_open": max(clusters.values(), default=0),
        "clustered_5m_opens": sum(count > 1 for count in clusters.values()),
    }


def raw_signal_diagnostics(raw: list[dict]) -> dict:
    fresh = [row for row in raw
             if int(float(row["entry_time"])) > audit.MATERIAL_FREEZE]
    moves = pd.Series([float(row["first_target_move_pct"])
                       for row in fresh], dtype=float)
    risks = pd.Series([100 * float(row["risk_pct"])
                       for row in fresh], dtype=float)
    qqq = pd.Series([float(row["qqq_trend_dir_atr"])
                     for row in fresh], dtype=float)
    return {
        "n": len(fresh),
        "direction_count": dict(sorted(Counter(
            str(row["direction"]) for row in fresh).items())),
        "module_count": dict(sorted(Counter(
            str(row["module"]) for row in fresh).items())),
        "first_target_move_pct": {
            "min": round(float(moves.min()), 6) if len(moves) else None,
            "median": round(float(moves.median()), 6) if len(moves) else None,
            "max": round(float(moves.max()), 6) if len(moves) else None,
            "at_least_0_5pct": int((moves >= .5).sum()),
        },
        "current_stop_distance_pct": {
            "min": round(float(risks.min()), 6) if len(risks) else None,
            "median": round(float(risks.median()), 6) if len(risks) else None,
            "max": round(float(risks.max()), 6) if len(risks) else None,
        },
        "qqq_direction_context_atr": {
            "min": round(float(qqq.min()), 6) if len(qqq) else None,
            "median": round(float(qqq.median()), 6) if len(qqq) else None,
            "max": round(float(qqq.max()), 6) if len(qqq) else None,
            "shorts_against_qqq_uptrend": sum(
                str(row["direction"]) == "SHORT"
                and float(row["qqq_trend_dir_atr"]) < 0
                for row in fresh),
            "shorts_with_qqq_downtrend": sum(
                str(row["direction"]) == "SHORT"
                and float(row["qqq_trend_dir_atr"]) > 0
                for row in fresh),
        },
        "zero_reported_volume_signals": sum(
            float(row["volume_ratio"]) == 0 for row in fresh),
        "interpretation": (
            "These are causally generated runtime proposals before the 0.5% "
            "minimum-move gate. They diagnose why trading is absent; they are "
            "not counterfactual trades and are not used to tune the runtime."),
    }


def _literal_path(row: dict, candle: dict, stop_pct: float) -> dict:
    lookup = {int(value): index for index, value in enumerate(candle["time"])}
    entry_index = lookup[int(float(row["entry_time"]))]
    session_end = audit.context(candle)["session_last"][entry_index]
    entry = float(row["entry_price"])
    target = entry * .995
    stop = entry * (1 + stop_pct / 100)
    outcome = "TIMEOUT"
    for index in range(entry_index, session_end + 1):
        # Deliberately pessimistic when target and stop share one OHLC bar.
        if float(candle["high"][index]) >= stop:
            outcome = "SL"
            break
        if float(candle["low"][index]) <= target:
            outcome = "TP"
            break
    lows = [float(value) for value in
            candle["low"][entry_index:session_end + 1]]
    highs = [float(value) for value in
             candle["high"][entry_index:session_end + 1]]
    return {
        "outcome": outcome,
        "full_session_mfe_pct": 100 * (entry - min(lows)) / entry,
        "full_session_mae_pct": 100 * (max(highs) - entry) / entry,
    }


def literal_move_diagnostics(raw: list[dict], candles: dict) -> dict:
    fresh = [row for row in raw
             if int(float(row["entry_time"])) > audit.MATERIAL_FREEZE]
    result = {}
    for stop_pct in (.5, 1.0):
        paths = [_literal_path(row, candles[str(row["symbol"])], stop_pct)
                 for row in fresh]
        outcomes = Counter(path["outcome"] for path in paths)
        by_qqq_context: dict[str, Counter] = {}
        for row, path in zip(fresh, paths):
            label = ("short_with_qqq_downtrend"
                     if float(row["qqq_trend_dir_atr"]) > 0
                     else "short_against_qqq_uptrend")
            by_qqq_context.setdefault(label, Counter())[path["outcome"]] += 1
        result[f"tp_0_5pct_sl_{str(stop_pct).replace('.', '_')}pct"] = {
            "n": len(paths),
            "outcomes": dict(sorted(outcomes.items())),
            "literal_tp_rate_pct": round(
                100 * outcomes["TP"] / len(paths), 4) if paths else 0.0,
            "mean_full_session_mfe_pct": round(sum(
                path["full_session_mfe_pct"] for path in paths)
                / len(paths), 6) if paths else 0.0,
            "mean_full_session_mae_pct": round(sum(
                path["full_session_mae_pct"] for path in paths)
                / len(paths), 6) if paths else 0.0,
            "outcomes_by_qqq_context": {
                label: dict(sorted(counts.items()))
                for label, counts in sorted(by_qqq_context.items())
            },
        }
    result["interpretation"] = (
        "Counterfactual price-path diagnostic only: fixed 0.5% short target, "
        "market entry on the next 5m open, expiry at the cash-session close, "
        "and stop-first OHLC ambiguity. It does not alter or fit the runtime.")
    return result


def main() -> None:
    _, previous = verified_manifest(PREVIOUS)
    _, current = verified_manifest(CURRENT)
    parity = append_parity(previous, current)

    audit.NEW_CACHE = ROOT / "stock_xperp_5m_oct13_forward"
    audit.NEW_MANIFEST = CURRENT
    audit.OUT = OUT
    audit.ROWS_OUT = ROWS
    audit.main()

    report = json.loads(OUT.read_text(encoding="utf-8"))
    raw, _ = audit.replay(current)
    material = pd.read_csv(ROWS).to_dict("records") if ROWS.stat().st_size else []
    accepted = audit.accepted(material)
    fresh = [row for row in accepted
             if int(float(row["entry_time"])) > audit.MATERIAL_FREEZE]
    fresh_value = audit.money(fresh)
    report["integrity"]["previous_snapshot_append_parity"] = parity
    report["genuinely_post_material_freeze"]["economic_projection"] = \
        economics(fresh, fresh_value)
    report["genuinely_post_material_freeze"]["raw_signal_diagnostics"] = \
        raw_signal_diagnostics(raw)
    report["genuinely_post_material_freeze"]["literal_move_diagnostics"] = \
        literal_move_diagnostics(raw, current)
    report["sources"][str(PREVIOUS)] = digest(PREVIOUS)
    report["sources"][str(Path(__file__))] = digest(Path(__file__))
    report["limitations"].append(
        "The +30% economic gate requires 1.327709% mean signed price return at three trades per market day.")
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "append_parity": parity,
        "genuinely_post_material_freeze":
            report["genuinely_post_material_freeze"],
        "readiness": report["readiness"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
