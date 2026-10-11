"""Cross-era audit of causal liquidity-sweep and level-break stock entries."""
from __future__ import annotations

import hashlib
import heapq
import json
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from stock_module_cluster_audit import portfolio_gate


ROOT = Path("reports/audit_2026_09_08")
HISTORY_MANIFEST = ROOT / "stock_extended_history_manifest.json"
DIRECT_MANIFEST = ROOT / "stock_xperp_forward_sep30" / "manifest.json"
OUT = ROOT / "stock_liquidity_structure_oct11.json"
ROWS_OUT = ROOT / "stock_liquidity_structure_oct11_rows.csv"
HISTORY_FEATURE_CACHE = ROOT / "stock_liquidity_structure_oct11_history.pkl"
DIRECT_FEATURE_CACHE = ROOT / "stock_liquidity_structure_oct11_direct.pkl"
TARGET_PCT = .5
STOP_PCT = 1.0
STOCKS = ("AAPLUSDT", "AMZNUSDT", "GOOGLUSDT", "INTCUSDT", "METAUSDT",
          "MRVLUSDT", "MSFTUSDT", "MUUSDT", "NVDAUSDT", "TSLAUSDT")
NUMERIC = (
    "trend_dir_atr", "ret20_dir", "volume_ratio", "body_dir_atr",
    "range_atr", "wick_fraction", "level_excursion_atr", "reclaim_atr",
    "qqq_trend_dir_atr", "qqq_body_dir_atr", "signal_minutes",
)
CATEGORICAL = ("route", "symbol")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest(path: Path) -> tuple[dict[str, dict], dict[str, str]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    caches, sources = {}, {str(path): digest(path)}
    for symbol, item in manifest["symbols"].items():
        source = Path(item["path"])
        raw = source.read_bytes()
        found = hashlib.sha256(raw).hexdigest()
        if found != item["sha256"]:
            raise ValueError(f"hash mismatch {source}")
        caches[symbol] = pickle.loads(raw)
        sources[str(source)] = found
    return caches, sources


def frame(candle: dict) -> pd.DataFrame:
    value = pd.DataFrame({name: candle[name] for name in
                          ("time", "open", "high", "low", "close", "volume")})
    stamp = pd.to_datetime(value.time.astype(int), unit="s", utc=True).dt.tz_convert(
        "America/New_York")
    value["date"] = stamp.dt.strftime("%Y-%m-%d")
    value["minute"] = stamp.dt.hour * 60 + stamp.dt.minute
    value["regular"] = value.minute.between(570, 945)
    previous = value.close.shift(1)
    tr = pd.concat([(value.high-value.low), (value.high-previous).abs(),
                    (value.low-previous).abs()], axis=1).max(axis=1)
    value["atr"] = tr.rolling(14, min_periods=14).mean()
    value["prior_high"] = value.high.shift(1).rolling(20, min_periods=20).max()
    value["prior_low"] = value.low.shift(1).rolling(20, min_periods=20).min()
    value["sma20"] = value.close.rolling(20, min_periods=20).mean()
    value["ret20"] = value.close/value.close.shift(20)-1
    value["volume_ratio"] = value.volume/value.volume.shift(1).rolling(
        20, min_periods=10).median().replace(0, np.nan)
    value["body"] = value.close-value.open
    value["range"] = value.high-value.low
    return value


def qqq_context(candle: dict) -> dict[int, dict]:
    q = frame(candle)
    return {int(row.time): {
        "qqq_trend_atr": ((float(row.close)-float(row.sma20))/float(row.atr)
                          if row.atr and math.isfinite(row.atr) else np.nan),
        "qqq_body_atr": (float(row.body)/float(row.atr)
                         if row.atr and math.isfinite(row.atr) else np.nan),
    } for row in q.itertuples()}


def outcome(data: pd.DataFrame, entry_i: int, end_i: int,
            direction: int) -> tuple[float, int, bool, int]:
    entry = float(data.open.iat[entry_i])
    target = entry*(1+direction*TARGET_PCT/100)
    stop = entry*(1-direction*STOP_PCT/100)
    signed = 0.0
    reason = 2
    hit = False
    exit_i = entry_i
    for j in range(entry_i, end_i+1):
        exit_i = j
        stopped = float(data.low.iat[j]) <= stop if direction > 0 else float(data.high.iat[j]) >= stop
        reached = float(data.high.iat[j]) >= target if direction > 0 else float(data.low.iat[j]) <= target
        if stopped:
            opening = float(data.open.iat[j])
            fill = min(opening, stop) if direction > 0 else max(opening, stop)
            signed = direction*(fill/entry-1)
            reason = 0
            break
        if reached:
            signed = TARGET_PCT/100
            reason = 1
            hit = True
            break
    else:
        signed = direction*(float(data.close.iat[end_i])/entry-1)
        signed = max(-STOP_PCT/100, min(TARGET_PCT/100, signed))
    return signed, reason, hit, exit_i


def events(candle: dict, symbol: str, qqq: dict[int, dict]) -> list[dict]:
    data = frame(candle)
    output = []
    groups = data[data.regular].groupby("date", sort=True).indices
    for date, raw_indices in groups.items():
        indices = sorted(int(v) for v in raw_indices)
        if len(indices) < 3:
            continue
        session_end = indices[-1]
        for position, i in enumerate(indices[:-1]):
            entry_i = indices[position+1]
            if entry_i != i+1 or i < 21:
                continue
            atr = float(data.atr.iat[i])
            price_range = float(data.range.iat[i])
            if not atr or not math.isfinite(atr) or price_range <= 0:
                continue
            opening, high, low, close = (float(data[name].iat[i]) for name in
                                         ("open", "high", "low", "close"))
            prior_high, prior_low = float(data.prior_high.iat[i]), float(data.prior_low.iat[i])
            definitions = []
            if low < prior_low < close and close > opening:
                definitions.append(("liquidity_sweep_reclaim_long", 1,
                                    (min(opening, close)-low)/price_range,
                                    (prior_low-low)/atr, (close-prior_low)/atr))
            if high > prior_high > close and close < opening:
                definitions.append(("liquidity_sweep_reclaim_short", -1,
                                    (high-max(opening, close))/price_range,
                                    (high-prior_high)/atr, (prior_high-close)/atr))
            if close > prior_high and close > opening and abs(close-opening)/price_range >= .5:
                definitions.append(("confirmed_level_break_long", 1,
                                    (close-opening)/price_range,
                                    (close-prior_high)/atr, 0.0))
            if close < prior_low and close < opening and abs(close-opening)/price_range >= .5:
                definitions.append(("confirmed_level_break_short", -1,
                                    (opening-close)/price_range,
                                    (prior_low-close)/atr, 0.0))
            context = qqq.get(int(data.time.iat[i]))
            if not definitions or context is None:
                continue
            for route, direction, wick, excursion, reclaim in definitions:
                signed, reason, hit, exit_i = outcome(data, entry_i, session_end, direction)
                output.append({
                    "symbol": symbol, "route": route,
                    "module": route, "direction": "LONG" if direction > 0 else "SHORT",
                    "entry_time": int(data.time.iat[entry_i]),
                    "exit_time": int(data.time.iat[exit_i])+900,
                    "outcome": "SL" if signed < 0 else "TP",
                    "net_r": signed/(STOP_PCT/100), "risk_pct": STOP_PCT/100,
                    "signed_return": signed, "target_hit": hit, "exit_reason": reason,
                    "year": int(pd.Timestamp(date).year),
                    "month": str(date)[:7],
                    "signal_minutes": int(data.minute.iat[i]-570),
                    "trend_dir_atr": direction*(close-float(data.sma20.iat[i]))/atr,
                    "ret20_dir": direction*float(data.ret20.iat[i]),
                    "volume_ratio": float(data.volume_ratio.iat[i]),
                    "body_dir_atr": direction*(close-opening)/atr,
                    "range_atr": price_range/atr,
                    "wick_fraction": wick,
                    "level_excursion_atr": excursion,
                    "reclaim_atr": reclaim,
                    "qqq_trend_dir_atr": direction*float(context["qqq_trend_atr"]),
                    "qqq_body_dir_atr": direction*float(context["qqq_body_atr"]),
                })
    return output


def build(caches: dict[str, dict]) -> pd.DataFrame:
    qqq = qqq_context(caches["QQQUSDT"])
    rows = []
    for symbol in STOCKS:
        if symbol in caches:
            rows.extend(events(caches[symbol], symbol, qqq))
    value = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
    value["won"] = value.target_hit.astype(int)
    return value


def account(rows: list[dict]) -> dict:
    rows = portfolio_gate(rows, None, rank_key=lambda row: (-float(row.get("quality", 0)),))
    if not rows:
        return {"n": 0, "wr": 0.0, "return_pct": 0.0, "dd_pct": 0.0,
                "negative_months": 0, "trades_per_market_day": 0.0,
                "max_loss_streak": 0}
    equity = peak = 1.0
    drawdown = 0.0
    pending = []
    ledger = []
    for serial, row in enumerate(sorted(rows, key=lambda r: (r["entry_time"], r["symbol"]))):
        pnl = equity*.03*10*float(row["signed_return"])
        heapq.heappush(pending, (float(row["exit_time"]), serial, pnl, row))
        while pending and (serial == len(rows)-1 or pending[0][0] < float(rows[min(serial+1, len(rows)-1)]["entry_time"])):
            end, _, value, done = heapq.heappop(pending)
            equity += value
            peak = max(peak, equity)
            drawdown = max(drawdown, (peak-equity)/peak)
            ledger.append((end, done))
    while pending:
        end, _, value, done = heapq.heappop(pending)
        equity += value
        peak = max(peak, equity)
        drawdown = max(drawdown, (peak-equity)/peak)
        ledger.append((end, done))
    monthly = {}
    streak = maximum = 0
    for end, row in sorted(ledger, key=lambda item: (
            item[0], float(item[1]["entry_time"]), str(item[1]["symbol"]))):
        month = pd.Timestamp(end, unit="s", tz="UTC").strftime("%Y-%m")
        monthly[month] = monthly.get(month, 0.0)+float(row["signed_return"])
        streak = 0 if row["target_hit"] else streak+1
        maximum = max(maximum, streak)
    days = {pd.Timestamp(float(row["entry_time"]), unit="s", tz="UTC").date() for row in rows}
    hits = sum(bool(row["target_hit"]) for row in rows)
    return {"n": len(rows), "wr": round(100*hits/len(rows), 4),
            "return_pct": round(100*(equity-1), 4), "dd_pct": round(100*drawdown, 4),
            "negative_months": sum(v < 0 for v in monthly.values()),
            "months": len(monthly), "trades_per_market_day": round(len(rows)/len(days), 3),
            "max_loss_streak": maximum}


def model(config: dict) -> object:
    prep = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), list(NUMERIC)),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), list(CATEGORICAL)),
    ])
    tree = DecisionTreeClassifier(max_depth=config["depth"],
                                  min_samples_leaf=config["leaf"],
                                  random_state=17)
    return make_pipeline(prep, tree)


def score(frame: pd.DataFrame, probability: np.ndarray, years: tuple[int, ...],
          threshold: float) -> dict:
    selected = frame[frame.year.isin(years) & (probability >= threshold)].copy()
    selected["quality"] = probability[selected.index]
    return account(selected.sort_values(["entry_time", "quality", "symbol"],
                                        ascending=[True, False, True]).to_dict("records"))


def main() -> None:
    history, history_sources = load_manifest(HISTORY_MANIFEST)
    direct, direct_sources = load_manifest(DIRECT_MANIFEST)
    if HISTORY_FEATURE_CACHE.exists():
        data = pd.read_pickle(HISTORY_FEATURE_CACHE).reset_index(drop=True)
    else:
        data = build(history).reset_index(drop=True)
        data.to_pickle(HISTORY_FEATURE_CACHE)
    print(json.dumps({"phase": "history_features", "rows": len(data)}), flush=True)
    if DIRECT_FEATURE_CACHE.exists():
        direct_data = pd.read_pickle(DIRECT_FEATURE_CACHE).reset_index(drop=True)
    else:
        direct_data = build(direct).reset_index(drop=True)
        direct_data.to_pickle(DIRECT_FEATURE_CACHE)
    print(json.dumps({"phase": "direct_features", "rows": len(direct_data)}), flush=True)
    fit = data.year.between(2017, 2020)
    configs = [{"depth": depth, "leaf": leaf}
               for depth in (2, 3, 4, 5) for leaf in (100, 200, 400)]
    candidates = []
    frontier = []
    for config in configs:
        estimator = model(config)
        estimator.fit(data.loc[fit, [*NUMERIC, *CATEGORICAL]], data.loc[fit, "won"])
        probability = estimator.predict_proba(data[[*NUMERIC, *CATEGORICAL]])[:, 1]
        selection_values = probability[data.year.isin((2021, 2022))]
        thresholds = sorted(set([float(np.quantile(selection_values, q))
                                 for q in (.5, .6, .7, .75, .8, .85, .9, .925, .95, .975)]
                                + list(np.arange(.65, .901, .025))))
        for threshold in thresholds:
            selection = score(data, probability, (2021, 2022), threshold)
            frontier.append({"config": config, "threshold": threshold,
                             "selection_2021_2022": selection})
            if not (selection["n"] >= 350 and selection["wr"] >= 72
                    and selection["return_pct"] > 0 and selection["dd_pct"] <= 8
                    and selection["trades_per_market_day"] >= 1.5):
                continue
            validation = score(data, probability, (2023, 2024), threshold)
            if not (validation["n"] >= 250 and validation["wr"] >= 70
                    and validation["return_pct"] > 0 and validation["dd_pct"] <= 8
                    and validation["trades_per_market_day"] >= 1.3):
                continue
            candidates.append({"config": config, "threshold": threshold,
                               "selection_2021_2022": selection,
                               "validation_2023_2024": validation,
                               "_model": estimator})
    candidates.sort(key=lambda item: (
        item["validation_2023_2024"]["return_pct"]
        - 2*item["validation_2023_2024"]["dd_pct"],
        item["validation_2023_2024"]["wr"]), reverse=True)
    frontier.sort(key=lambda item: (
        item["selection_2021_2022"]["wr"],
        item["selection_2021_2022"]["return_pct"],
        item["selection_2021_2022"]["trades_per_market_day"]), reverse=True)
    frozen = candidates[0] if candidates else None
    result = None
    status = "REJECTED_BEFORE_TEST"
    if frozen:
        estimator = frozen["_model"]
        probability = estimator.predict_proba(data[[*NUMERIC, *CATEGORICAL]])[:, 1]
        direct_probability = estimator.predict_proba(
            direct_data[[*NUMERIC, *CATEGORICAL]])[:, 1]
        threshold = frozen["threshold"]
        result = {
            "config": frozen["config"], "threshold": threshold,
            "selection_2021_2022": frozen["selection_2021_2022"],
            "validation_2023_2024": frozen["validation_2023_2024"],
            "test_2025": score(data, probability, (2025,), threshold),
            "observed_2026_cash": score(data, probability, (2026,), threshold),
            "direct_xperp_2026": score(
                direct_data, direct_probability, (2026,), threshold),
        }
        test = result["test_2025"]
        observed = result["observed_2026_cash"]
        status = ("RESEARCH_CANDIDATE_AWAITING_FORWARD"
                  if test["return_pct"] > 0 and test["wr"] >= 70
                  and observed["return_pct"] > 0 and observed["wr"] >= 70
                  else "REJECTED_TEST_OR_OBSERVED")
        chosen = data[probability >= threshold].copy()
        chosen["quality"] = probability[probability >= threshold]
        chosen.to_csv(ROWS_OUT, index=False)
    report = {
        "status": status,
        "protocol": {"fit": [2017, 2018, 2019, 2020],
                     "selection": [2021, 2022], "validation": [2023, 2024],
                     "test": 2025, "observed": "2026 cash and direct X-Perp",
                     "entry": "next 15m market open", "target_pct": TARGET_PCT,
                     "stop_pct": STOP_PCT, "orders": "market only",
                     "routes": ["liquidity sweep reclaim", "confirmed level break"],
                     "models": configs, "candidate_count": len(candidates)},
        "rows": {"history": len(data), "direct": len(direct_data)},
        "base": {str(year): account(data[data.year.eq(year)].to_dict("records"))
                 for year in range(2017, 2027)},
        "frozen_before_test": ({key: value for key, value in frozen.items()
                                if key != "_model"} if frozen else None),
        "top_selection_frontier": frontier[:20],
        "result": result,
        "sources": {**history_sources, **direct_sources,
                    str(Path(__file__)): digest(Path(__file__))},
        "limitations": [
            "Cash CFD candles test structure across eras but are not X-Perp fills.",
            "The bounded tree/threshold grid creates selection risk.",
            "Only genuinely new paper-forward outcomes can authorize runtime use.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": status, "rows": report["rows"],
                      "qualified_before_test": len(candidates),
                      "frozen": report["frozen_before_test"],
                      "result": result}, indent=2), flush=True)


if __name__ == "__main__":
    main()
