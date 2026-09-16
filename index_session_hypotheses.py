"""Prespecified US cash-session index signals executed on X-Perp OHLC."""
import hashlib
import json
import math
import pickle
import csv
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from filter_lab import gate, metrics, timestamp
from src.backtest_integrity import simulate_exit


NY = ZoneInfo("America/New_York")
FAMILIES = ("opening_breakout", "opening_reclaim", "gap_fade")
TARGETS = (0.25, 0.5, 1.0)
SYMBOLS = ("MSFTUSDT", "AMZNUSDT", "TSLAUSDT", "QQQUSDT")
# Filter research is ranked by gross outcomes. Client position size and
# transaction-cost assumptions do not decide whether a setup exists.
ROUNDTRIP_COST = 0.0


def atr_at(candles, index, period=14):
    if index < period + 1:
        return None
    closes = candles["close"]
    values = []
    for j in range(index - period, index):
        values.append(max(
            candles["high"][j] - candles["low"][j],
            abs(candles["high"][j] - closes[j - 1]),
            abs(candles["low"][j] - closes[j - 1]),
        ))
    value = sum(values) / period
    return value if math.isfinite(value) and value > 0 else None


def core_groups(candles):
    groups = {}
    for i, ts in enumerate(candles["time"]):
        local = datetime.fromtimestamp(ts, NY)
        minute = local.hour * 60 + local.minute
        if local.weekday() < 5 and 570 <= minute < 960:
            groups.setdefault(local.date().isoformat(), []).append(i)
    return groups


def candidate_records(index_candles, family):
    groups = core_groups(index_candles)
    days = sorted(groups)
    signals = []
    for position, day in enumerate(days):
        bars = groups[day]
        if len(bars) < 4:
            continue
        first = bars[0]
        if datetime.fromtimestamp(index_candles["time"][first], NY).strftime("%H:%M") != "09:30":
            continue
        direction = signal_bar = None
        high = low = previous_close = None
        if family in ("opening_breakout", "opening_reclaim"):
            opening = bars[:2]
            high = max(index_candles["high"][j] for j in opening)
            low = min(index_candles["low"][j] for j in opening)
            for j in bars[2:-1]:
                local = datetime.fromtimestamp(index_candles["time"][j], NY)
                if local.hour >= 14:
                    break
                close = index_candles["close"][j]
                previous = index_candles["close"][j - 1]
                if family == "opening_breakout":
                    if close > high and previous <= high:
                        direction, signal_bar = 1, j
                    elif close < low and previous >= low:
                        direction, signal_bar = -1, j
                else:
                    if index_candles["low"][j] < low and close > low:
                        direction, signal_bar = 1, j
                    elif index_candles["high"][j] > high and close < high:
                        direction, signal_bar = -1, j
                if direction:
                    break
        else:
            if position == 0:
                continue
            previous_bars = groups[days[position - 1]]
            previous_close = index_candles["close"][previous_bars[-1]]
            opening = index_candles["open"][first]
            atr = atr_at(index_candles, first)
            first_close = index_candles["close"][first]
            if atr and opening < previous_close - 0.5 * atr and first_close > opening:
                direction, signal_bar = 1, first
            elif atr and opening > previous_close + 0.5 * atr and first_close < opening:
                direction, signal_bar = -1, first
        if direction and signal_bar + 1 in bars:
            signal_atr = atr_at(index_candles, signal_bar)
            prior_bars = groups[days[position - 1]] if position else []
            prior_close = index_candles["close"][prior_bars[-1]] if prior_bars else None
            prior_open = index_candles["open"][prior_bars[0]] if prior_bars else None
            signal_close = index_candles["close"][signal_bar]
            signal_open = index_candles["open"][signal_bar]
            signal_high = index_candles["high"][signal_bar]
            signal_low = index_candles["low"][signal_bar]
            local = datetime.fromtimestamp(index_candles["time"][signal_bar], NY)
            boundary = high if direction == 1 else low
            opening_width = (high - low) if high is not None and low is not None else None
            signals.append({
                "direction": direction, "signal_bar": signal_bar,
                "entry_bar": signal_bar + 1, "session_end": bars[-1],
                "signal_hour": local.hour + local.minute / 60,
                "minutes_after_open": (index_candles["time"][signal_bar]
                                       - index_candles["time"][first]) / 60,
                "opening_range_atr": opening_width / signal_atr
                if opening_width is not None and signal_atr else None,
                "breakout_extension_atr": direction * (signal_close - boundary) / signal_atr
                if boundary is not None and signal_atr else None,
                "signal_body_atr": direction * (signal_close - signal_open) / signal_atr
                if signal_atr else None,
                "signal_range_atr": (signal_high - signal_low) / signal_atr if signal_atr else None,
                "pre_signal_move_atr": direction * (index_candles["close"][signal_bar - 1]
                                                       - index_candles["open"][first]) / signal_atr
                if signal_atr else None,
                "gap_directional_atr": direction * (index_candles["open"][first] - prior_close) / signal_atr
                if prior_close is not None and signal_atr else None,
                "prior_day_directional_atr": direction * (prior_close - prior_open) / signal_atr
                if prior_close is not None and prior_open is not None and signal_atr else None,
            })
    return signals


def candidates(index_candles, family):
    """Backward-compatible compact candidate tuples used by existing callers."""
    return [(item["direction"], item["signal_bar"], item["entry_bar"], item["session_end"])
            for item in candidate_records(index_candles, family)]


def replay(index_candles, xperp, symbol, family, target):
    xlookup = {t: i for i, t in enumerate(xperp["time"])}
    rows = []
    for candidate in candidate_records(index_candles, family):
        direction = candidate["direction"]
        signal_bar = candidate["signal_bar"]
        entry_bar = candidate["entry_bar"]
        session_end = candidate["session_end"]
        entry_time = index_candles["time"][entry_bar]
        end_time = index_candles["time"][session_end]
        if entry_time not in xlookup or end_time not in xlookup:
            continue
        first, last = xlookup[entry_time], xlookup[end_time]
        if xperp["time"][first:last + 1] != list(range(entry_time, end_time + 900, 900)):
            continue
        atr = atr_at(index_candles, entry_bar)
        index_entry = index_candles["open"][entry_bar]
        actual_entry = xperp["open"][first]
        if not atr or index_entry <= 0 or actual_entry <= 0:
            continue
        basis = actual_entry / index_entry
        risk = atr * basis
        if risk <= 0 or ROUNDTRIP_COST * actual_entry > 0.25 * risk * target:
            continue
        side = "LONG" if direction == 1 else "SHORT"
        stop = actual_entry - direction * risk
        take = actual_entry + direction * risk * target
        if min(stop, take) <= 0:
            continue
        result = simulate_exit(
            xperp, range(first, last + 1), direction=side, entry=actual_entry,
            sl=stop, tp1=take, tp2=take, atr=atr * basis,
            tp1_fraction=0, trail=False, trail_mult=0, stop_on_close=False,
            backstop_r=1, choose_trail=lambda *args: 0,
        )
        rows.append({
            "symbol": symbol, "direction": side, "entry_time": entry_time,
            "exit_time": xperp["time"][result.bar] + 900,
            "outcome": result.outcome,
            "net_r": result.gross_r - ROUNDTRIP_COST * actual_entry / risk,
            "risk_pct": risk / actual_entry, "size_mult": 1,
            **{key: value for key, value in candidate.items()
               if key not in {"direction", "signal_bar", "entry_bar", "session_end"}},
        })
    return rows


def main():
    folder = Path("reports/audit_2026_09_08")
    inputs, data = {}, {}
    for symbol in SYMBOLS:
        paths = {
            "index": folder / "index_signal_cache" / f"{symbol}_15min_18000.pkl",
            "xperp": folder / "xperp_cache" / f"{symbol}_15min_18000.pkl",
        }
        data[symbol] = {}
        for label, path in paths.items():
            raw = path.read_bytes()
            inputs[str(path)] = hashlib.sha256(raw).hexdigest()
            data[symbol][label] = pickle.loads(raw)
    report = {
        "status": "RESEARCH_ONLY", "families": FAMILIES, "targets": TARGETS,
        "roundtrip_cost": ROUNDTRIP_COST, "inputs": inputs, "results": [],
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "limitations": [
            "Prespecified small hypothesis set on already inspected June-August 2026 history.",
            "Index OHLC signals use timestamp-matched X-Perp volume only indirectly through no rule.",
            "X-Perp 15-minute OHLC approximates execution; historical depth and funding excluded.",
            "One signal per symbol/family/day; flat by cash-session close; equal initial risk.",
        ],
    }
    for family in FAMILIES:
        for target in TARGETS:
            raw = []
            for symbol in SYMBOLS:
                raw.extend(replay(data[symbol]["index"], data[symbol]["xperp"], symbol, family, target))
            accepted = gate(sorted(raw, key=lambda row: (row["entry_time"], row["symbol"])), 4)
            raw = sorted(raw, key=lambda row: (row["entry_time"], row["symbol"]))
            july, august = timestamp("2026-07-01"), timestamp("2026-08-01")
            item = {
                "family": family, "target": target, "all": metrics(accepted),
                "june": metrics([r for r in accepted if r["entry_time"] < july]),
                "july": metrics([r for r in accepted if july <= r["entry_time"] < august]),
                "august": metrics([r for r in accepted if r["entry_time"] >= august]),
            }
            report["results"].append(item)
            detail = folder / f"index_session_{family}_{target}.csv"
            raw_detail = folder / f"index_session_{family}_{target}_raw.csv"
            if accepted:
                with detail.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(accepted[0]))
                    writer.writeheader()
                    writer.writerows(accepted)
                item["detail"] = str(detail)
            if raw:
                with raw_detail.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(raw[0]))
                    writer.writeheader()
                    writer.writerows(raw)
                item["raw_detail"] = str(raw_detail)
            print(json.dumps(item), flush=True)
    (folder / "index_session_hypotheses.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
