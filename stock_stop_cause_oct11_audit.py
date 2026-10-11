"""Causal structure and post-stop audit for the frozen stock venue router.

Entry filters use only bars strictly before the market-entry timestamp.  The
post-stop section is diagnostic and is never exposed to candidate selection.
July is development, August is validation and September is revealed only
after a rule has passed both earlier periods.
"""
from __future__ import annotations

import hashlib
import json
import math
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from stock_forward_monitor import frozen_rows
from stock_module_cluster_audit import portfolio_gate


ROOT = Path("reports/audit_2026_09_08")
SOURCE = ROOT / "stock_forward_sep30_families.csv"
CACHE = ROOT / "stock_xperp_forward_sep30"
MANIFEST = CACHE / "manifest.json"
OUT = ROOT / "stock_stop_cause_oct11.json"
ROWS_OUT = ROOT / "stock_stop_cause_oct11_rows.csv"
PROFILE = "robust_dynamic_entry_filtered"
NY = ZoneInfo("America/New_York")
FIELDS = ("time", "open", "high", "low", "close", "volume")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_caches() -> tuple[dict[str, dict], dict[str, str]]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    sources = {str(MANIFEST): digest(MANIFEST)}
    caches: dict[str, dict] = {}
    for symbol, item in sorted(manifest["symbols"].items()):
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
        if any(right - left != 900 for left, right in zip(times, times[1:])):
            raise ValueError(f"non-contiguous 15m history: {symbol}")
        caches[symbol] = candle
        sources[str(path)] = found
    return caches, sources


def finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def swing_levels(candle: dict, index: int, lookback: int = 96) -> tuple[list[float], list[float]]:
    """Confirmed local levels using only candles closed before entry."""
    start = max(2, index - lookback)
    end = index - 1
    highs: list[float] = []
    lows: list[float] = []
    for j in range(start, end - 1):
        local = range(j - 2, j + 3)
        high = float(candle["high"][j])
        low = float(candle["low"][j])
        if high >= max(float(candle["high"][k]) for k in local):
            highs.append(high)
        if low <= min(float(candle["low"][k]) for k in local):
            lows.append(low)
    return highs, lows


def nearest(values: list[float], entry: float, above: bool) -> float | None:
    eligible = [value for value in values if value > entry] if above else [
        value for value in values if value < entry]
    if not eligible:
        return None
    return min(eligible) if above else max(eligible)


def opposing_order_block(candle: dict, index: int, entry: float,
                         target: float, direction: int) -> bool:
    """Simple causal body-zone proxy for resistance/support in the target path."""
    for j in range(index - 1, max(-1, index - 49), -1):
        opening = float(candle["open"][j])
        close = float(candle["close"][j])
        low, high = sorted((opening, close))
        if direction > 0 and close < opening and high > entry and low < target:
            return True
        if direction < 0 and close > opening and low < entry and high > target:
            return True
    return False


def causal_features(row: dict, candle: dict, raw_cluster: Counter,
                    accepted_cluster: Counter) -> dict:
    lookup = {int(value): i for i, value in enumerate(candle["time"])}
    entry_time = int(float(row["entry_time"]))
    index = lookup.get(entry_time)
    if index is None or index < 100:
        raise ValueError(f"entry candle missing/warmup: {row['symbol']} {entry_time}")
    entry = float(candle["open"][index])
    risk = entry * float(row["risk_pct"])
    direction = 1 if row["direction"] == "LONG" else -1
    stop = entry - direction * risk
    target = entry + direction * risk * float(row["target_r"])
    highs, lows = swing_levels(candle, index)
    target_level = nearest(highs if direction > 0 else lows, entry, direction > 0)
    stop_level = nearest(lows if direction > 0 else highs, entry, direction < 0)
    target_level_r = (direction * (target_level - entry) / risk
                      if target_level is not None else None)
    stop_level_r = (-direction * (stop_level - entry) / risk
                    if stop_level is not None else None)

    signal = index - 1
    prior_start = max(0, signal - 20)
    prior_high = max(float(v) for v in candle["high"][prior_start:signal])
    prior_low = min(float(v) for v in candle["low"][prior_start:signal])
    signal_high = float(candle["high"][signal])
    signal_low = float(candle["low"][signal])
    signal_close = float(candle["close"][signal])
    adverse_failed_breakout = (
        signal_high > prior_high and signal_close < prior_high
        if direction > 0 else signal_low < prior_low and signal_close > prior_low)
    supportive_sweep = (
        signal_low < prior_low and signal_close > prior_low
        if direction > 0 else signal_high > prior_high and signal_close < prior_high)

    volumes = [float(value) for value in candle["volume"][max(0, signal - 20):signal]]
    median_volume = float(pd.Series(volumes).median()) if volumes else 0.0
    volume_ratio_20 = (float(candle["volume"][signal]) / median_volume
                       if median_volume > 0 else None)
    q_move = row.get("qqq_intraday_move_dir_atr")
    s_move = row.get("spy_intraday_move_dir_atr")
    both_indices_against = (finite(q_move) and finite(s_move)
                            and float(q_move) < 0 and float(s_move) < 0)
    index_divergence = (finite(q_move) and finite(s_move)
                        and float(q_move) * float(s_move) < 0)
    target_r = float(row["target_r"])
    return {
        **row,
        "entry_price_rebuilt": entry,
        "stop_price_rebuilt": stop,
        "target_price_rebuilt": target,
        "first_target_move_pct": 100 * risk * target_r / entry,
        "target_side_swing_r": target_level_r,
        "stop_side_swing_r": stop_level_r,
        "target_path_has_prior_swing": bool(
            target_level_r is not None and target_level_r <= target_r),
        "stop_zone_has_prior_swing": bool(
            stop_level_r is not None and stop_level_r <= 1.0),
        "liquidity_closer_to_stop": bool(
            stop_level_r is not None and target_level_r is not None
            and stop_level_r < target_level_r),
        "opposing_order_block_proxy": opposing_order_block(
            candle, index, entry, target, direction),
        "adverse_failed_breakout": bool(adverse_failed_breakout),
        "supportive_sweep": bool(supportive_sweep),
        "both_indices_against": bool(both_indices_against),
        "index_divergence": bool(index_divergence),
        "volume_ratio_20": volume_ratio_20,
        "raw_same_scan": int(raw_cluster[(entry_time, str(row["direction"]))]),
        "accepted_same_scan": int(accepted_cluster[(entry_time, str(row["direction"]))]),
    }


def post_stop_path(row: dict, candle: dict) -> dict:
    if row["outcome"] != "SL":
        return {}
    lookup = {int(value): i for i, value in enumerate(candle["time"])}
    entry_i = lookup[int(float(row["entry_time"]))]
    stop_i = lookup[int(float(row["exit_time"]))]
    entry = float(row["entry_price_rebuilt"])
    stop = float(row["stop_price_rebuilt"])
    target = float(row["target_price_rebuilt"])
    direction = 1 if row["direction"] == "LONG" else -1
    stop_close = float(candle["close"][stop_i])
    stop_candle_closed_back_inside = stop_close > stop if direction > 0 else stop_close < stop
    stop_high = float(candle["high"][stop_i])
    stop_low = float(candle["low"][stop_i])
    same_bar_target = stop_high >= target if direction > 0 else stop_low <= target

    entry_day = pd.Timestamp(int(float(row["entry_time"])), unit="s", tz="UTC").tz_convert(NY).date()
    session_end = stop_i
    for j in range(stop_i + 1, len(candle["time"])):
        local = pd.Timestamp(int(candle["time"][j]), unit="s", tz="UTC").tz_convert(NY)
        if local.date() != entry_day or local.hour >= 16:
            break
        session_end = j
    session_slice = range(stop_i + 1, session_end + 1)
    double_end = min(len(candle["time"]) - 1, entry_i + 52)
    double_slice = range(stop_i + 1, double_end + 1)

    def target_hit(indices: range) -> bool:
        return any((float(candle["high"][j]) >= target if direction > 0
                    else float(candle["low"][j]) <= target) for j in indices)

    def entry_hit(indices: range) -> bool:
        return any((float(candle["high"][j]) >= entry if direction > 0
                    else float(candle["low"][j]) <= entry) for j in indices)

    hard_reversal = (not entry_hit(session_slice)
                     and not stop_candle_closed_back_inside)
    return {
        "stop_candle_closed_back_inside": bool(stop_candle_closed_back_inside),
        "stop_and_target_same_bar": bool(same_bar_target),
        "target_after_stop_same_session": target_hit(session_slice),
        "entry_after_stop_same_session": entry_hit(session_slice),
        "target_after_stop_13h": target_hit(double_slice),
        "hard_reversal_proxy": bool(hard_reversal),
    }


def metrics(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0, "wr": 0.0, "stops": 0, "net_r": 0.0,
                "signed_price_return_pct": 0.0, "return_3pct_margin_pct": 0.0,
                "max_loss_streak": 0, "dd_pct": 0.0}
    ordered = sorted(rows, key=lambda r: (float(r["exit_time"]), float(r["entry_time"]), r["symbol"]))
    equity = peak = 1.0
    dd = 0.0
    wins = streak = max_streak = 0
    for row in ordered:
        won = float(row["net_r"]) > 0
        wins += int(won)
        streak = 0 if won else streak + 1
        max_streak = max(max_streak, streak)
        equity *= 1 + .03 * 10 * float(row["risk_pct"]) * float(row["net_r"])
        peak = max(peak, equity)
        dd = max(dd, (peak - equity) / peak)
    return {
        "n": len(rows),
        "wr": round(100 * wins / len(rows), 4),
        "stops": sum(str(row["outcome"]) == "SL" for row in rows),
        "net_r": round(sum(float(row["net_r"]) for row in rows), 6),
        "signed_price_return_pct": round(100 * sum(
            float(row["risk_pct"]) * float(row["net_r"]) for row in rows), 6),
        "return_3pct_margin_pct": round(100 * (equity - 1), 6),
        "max_loss_streak": max_streak,
        "dd_pct": round(100 * dd, 6),
    }


def by_month(rows: list[dict]) -> dict[str, dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        month = pd.Timestamp(float(row["entry_time"]), unit="s", tz="UTC").strftime("%Y-%m")
        groups[month].append(row)
    return {month: metrics(part) for month, part in sorted(groups.items())}


def factor_stats(rows: list[dict], fields: list[str]) -> dict:
    base_rate = sum(row["outcome"] == "SL" for row in rows) / len(rows)
    result = {}
    for field in fields:
        exposed = [row for row in rows if bool(row[field])]
        clear = [row for row in rows if not bool(row[field])]
        rate = (sum(row["outcome"] == "SL" for row in exposed) / len(exposed)
                if exposed else 0.0)
        result[field] = {
            "exposed": metrics(exposed),
            "clear": metrics(clear),
            "stop_rate_lift": round(rate / base_rate, 4) if base_rate else None,
            "exposed_by_month": by_month(exposed),
        }
    return result


def apply_rule(rows: list[dict], name: str) -> list[dict]:
    def keep(row: dict) -> bool:
        if name == "reject_adverse_failed_breakout":
            return not row["adverse_failed_breakout"]
        if name == "reject_index_opposition":
            return not row["both_indices_against"]
        if name == "reject_opposing_ob":
            return not row["opposing_order_block_proxy"]
        if name == "reject_stop_zone_swing":
            return not row["stop_zone_has_prior_swing"]
        if name == "require_target_liquidity_closer":
            return not row["liquidity_closer_to_stop"]
        if name == "reject_structure_conflict":
            return not (row["adverse_failed_breakout"] or (
                row["stop_zone_has_prior_swing"]
                and row["liquidity_closer_to_stop"]))
        if name == "single_directional_signal_per_scan":
            return row["raw_same_scan"] <= 1
        if name == "material_target_0_5pct":
            return row["first_target_move_pct"] >= .5
        raise KeyError(name)
    return [row for row in rows if keep(row)]


def period(rows: list[dict], month: str) -> list[dict]:
    return [row for row in rows if pd.Timestamp(
        float(row["entry_time"]), unit="s", tz="UTC").strftime("%Y-%m") == month]


def candidate_audit(raw: list[dict]) -> dict:
    names = [
        "reject_adverse_failed_breakout",
        "reject_index_opposition",
        "reject_opposing_ob",
        "reject_stop_zone_swing",
        "require_target_liquidity_closer",
        "reject_structure_conflict",
        "single_directional_signal_per_scan",
        "material_target_0_5pct",
    ]
    base = {month: portfolio_gate(period(raw, month), None)
            for month in ("2026-07", "2026-08", "2026-09")}
    reports = []
    qualified = []
    for name in names:
        parts = {month: portfolio_gate(apply_rule(period(raw, month), name), None)
                 for month in base}
        dev, dev_base = metrics(parts["2026-07"]), metrics(base["2026-07"])
        val, val_base = metrics(parts["2026-08"]), metrics(base["2026-08"])
        dev_pass = (dev["n"] >= .60 * dev_base["n"] and dev["wr"] >= dev_base["wr"]
                    and dev["net_r"] >= dev_base["net_r"]
                    and dev["dd_pct"] <= dev_base["dd_pct"] + 1e-9)
        val_pass = (val["n"] >= .60 * val_base["n"] and val["wr"] >= val_base["wr"]
                    and val["net_r"] >= val_base["net_r"]
                    and val["dd_pct"] <= val_base["dd_pct"] + 1e-9)
        item = {
            "rule": name,
            "development": dev,
            "development_baseline": dev_base,
            "development_pass": dev_pass,
            "validation": val,
            "validation_baseline": val_base,
            "validation_pass": val_pass,
            "hidden_revealed": metrics(parts["2026-09"]) if dev_pass and val_pass else None,
            "hidden_baseline": metrics(base["2026-09"]) if dev_pass and val_pass else None,
        }
        reports.append(item)
        if dev_pass and val_pass:
            qualified.append(item)
    qualified.sort(key=lambda item: (
        item["validation"]["net_r"] - item["validation_baseline"]["net_r"],
        item["validation"]["wr"] - item["validation_baseline"]["wr"],
        item["development"]["net_r"] - item["development_baseline"]["net_r"]),
        reverse=True)
    return {
        "protocol": {
            "development": "2026-07", "validation": "2026-08",
            "hidden": "2026-09; revealed only for rules passing both earlier gates",
            "gate": "retain >=60%; non-worse WR/net R/drawdown in development and validation",
            "thresholds": "fixed structural definitions; no outcome-fitted numeric search",
        },
        "baseline": {month: metrics(rows) for month, rows in base.items()},
        "rules": reports,
        "selected": qualified[0] if qualified else None,
        "runtime_change_allowed": False,
    }


def main() -> None:
    caches, sources = load_caches()
    frame = pd.read_csv(SOURCE)
    raw = frozen_rows(frame, PROFILE)
    accepted = portfolio_gate(raw, None)
    raw_cluster = Counter((int(float(row["entry_time"])), str(row["direction"])) for row in raw)
    accepted_cluster = Counter((int(float(row["entry_time"])), str(row["direction"]))
                               for row in accepted)
    enriched = [causal_features(row, caches[str(row["symbol"])], raw_cluster,
                                accepted_cluster) for row in raw]
    enriched_by_key = {(str(row["symbol"]), int(float(row["entry_time"]))): row
                       for row in enriched}
    accepted_enriched = [enriched_by_key[(str(row["symbol"]), int(float(row["entry_time"])))]
                         for row in accepted]
    for row in accepted_enriched:
        row.update(post_stop_path(row, caches[str(row["symbol"])]))

    stop_rows = [row for row in accepted_enriched if row["outcome"] == "SL"]
    path_fields = [
        "stop_candle_closed_back_inside", "stop_and_target_same_bar",
        "target_after_stop_same_session", "entry_after_stop_same_session",
        "target_after_stop_13h", "hard_reversal_proxy",
    ]
    path = {field: {
        "count": sum(bool(row.get(field)) for row in stop_rows),
        "pct": round(100 * sum(bool(row.get(field)) for row in stop_rows) / len(stop_rows), 4)
               if stop_rows else 0.0,
    } for field in path_fields}
    boolean_fields = [
        "target_path_has_prior_swing", "stop_zone_has_prior_swing",
        "liquidity_closer_to_stop", "opposing_order_block_proxy",
        "adverse_failed_breakout", "supportive_sweep", "both_indices_against",
        "index_divergence",
    ]
    report = {
        "status": "DIAGNOSTIC_COMPLETE_NO_RUNTIME_CHANGE",
        "scope": {
            "router": PROFILE,
            "raw_frozen_entries": len(raw),
            "portfolio_accepted_entries": len(accepted_enriched),
            "period": [int(frame.entry_time.min()), int(frame.entry_time.max())],
            "causal_rule": "all filter features use bars closed before entry",
            "post_stop_is_diagnostic_only": True,
        },
        "baseline": {"all": metrics(accepted_enriched), "by_month": by_month(accepted_enriched)},
        "stop_factors": factor_stats(accepted_enriched, boolean_fields),
        "stop_path": {"stops": len(stop_rows), **path},
        "cluster": {
            "accepted_same_scan_distribution": dict(sorted(Counter(
                int(row["accepted_same_scan"]) for row in accepted_enriched).items())),
            "same_scan_stops": {
                str(size): metrics([row for row in accepted_enriched
                                    if int(row["accepted_same_scan"]) == size])
                for size in sorted({int(row["accepted_same_scan"]) for row in accepted_enriched})
            },
        },
        "candidate_filter_audit": candidate_audit(enriched),
        "material_move_subset": metrics([row for row in accepted_enriched
                                         if row["first_target_move_pct"] >= .5]),
        "sources": {
            str(SOURCE): digest(SOURCE),
            str(Path(__file__)): digest(Path(__file__)),
            **sources,
        },
        "limitations": [
            "Direct X-Perp evidence covers July-September 2026, so it cannot establish cross-era robustness.",
            "Swing and order-block fields are explicit OHLC proxies, not exchange order-book liquidity.",
            "Only the 0.5% material subset matches the current minimum-move runtime; it is too small for activation.",
            "Post-stop recovery describes stop geometry and cannot be used as an entry feature.",
        ],
    }
    pd.DataFrame(accepted_enriched).to_csv(ROWS_OUT, index=False)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "baseline": report["baseline"],
        "stop_path": report["stop_path"], "cluster": report["cluster"],
        "selected": report["candidate_filter_audit"]["selected"],
        "material_move_subset": report["material_move_subset"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
