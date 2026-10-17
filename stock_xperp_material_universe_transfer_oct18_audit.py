"""Transfer the frozen material X-Perp module to eight unseen stocks."""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

import stock_cash_xperp_leadlag_oct6_audit as lead
import stock_cash_xperp_tree_filter_oct6_audit as tree
import stock_xperp_material_barbell_oct18_audit as parent


ROOT = parent.ROOT
CASH_MANIFEST = ROOT / "stock_cash_expansion_oct17" / "manifest.json"
XPERP_MANIFEST = ROOT / "stock_xperp_expansion_oct17" / "manifest.json"
PARENT_REPORT = parent.OUT
PARENT_ROWS = parent.ROWS_OUT
OUT = ROOT / "stock_xperp_material_universe_transfer_oct18.json"
ROWS_OUT = ROOT / "stock_xperp_material_universe_transfer_oct18.csv"
EXPANSION_SYMBOLS = (
    "AMDUSDT", "QCOMUSDT", "PLTRUSDT", "DELLUSDT",
    "AVGOUSDT", "OKTAUSDT", "WDCUSDT", "MRNAUSDT",
)
PRE_FRESH = ("2026-08-01", "2026-10-07")
FULL_TRANSFER = ("2026-08-01", "2026-10-17")
AMBIGUOUS_TICKER_COLLISIONS = {
    "SUIUSDT", "OUSDT", "REUSDT", "MONUSDT", "BIOUSDT",
    "HUSDT", "ZENUSDT", "ARUSDT", "WUSDT",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verified_manifest(path: Path) -> tuple[dict, dict]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    candles, sources = {}, {str(path): digest(path)}
    if tuple(manifest["symbols"]) != EXPANSION_SYMBOLS:
        raise ValueError(f"unexpected expansion universe in {path}")
    for symbol, item in manifest["symbols"].items():
        source = Path(item["path"])
        raw = source.read_bytes()
        found = hashlib.sha256(raw).hexdigest()
        if found != item["sha256"]:
            raise ValueError(f"source hash mismatch: {symbol}")
        candles[symbol] = pickle.loads(raw)
        sources[str(source)] = found
    return candles, sources


def rebuild_frozen_model(cash: dict, xperp: dict, parameters: dict):
    raw = lead.raw_features(cash, xperp)
    entry = lead.apply_spec(raw, parameters["entry"])
    entry = entry[entry.lookback.eq(parameters["lookback_bars"])]
    resolved = lead.add_outcomes(
        entry, xperp, parameters["target_pct"], 1.0,
        parameters["hold_bars"])
    train = tree.inside(resolved, parent.DEVELOPMENT, require_closed=True)
    train_x, columns = tree.matrix(train)
    model = GradientBoostingClassifier(
        random_state=261018, n_estimators=60, learning_rate=.05,
        max_depth=2, min_samples_leaf=30, subsample=.8,
    )
    model.fit(train_x, train.signed_return.gt(0).astype(int))
    return model, columns


def transfer_rows(cash: dict, xperp: dict, model, columns: list[str],
                  parameters: dict, threshold: float,
                  exit_parameters: dict) -> pd.DataFrame:
    old_symbols = lead.SYMBOLS
    try:
        lead.SYMBOLS = EXPANSION_SYMBOLS
        raw = lead.raw_features(cash, xperp)
    finally:
        lead.SYMBOLS = old_symbols
    entries = lead.apply_spec(raw, parameters["entry"])
    entries = entries[entries.lookback.eq(parameters["lookback_bars"])]
    base = lead.add_outcomes(entries, xperp, parameters["target_pct"],
                             1.0, parameters["hold_bars"])
    matrix, _ = tree.matrix(base, columns)
    base["probability"] = model.predict_proba(matrix)[:, 1]
    accepted_keys = set(zip(
        base.loc[base.probability.ge(threshold), "symbol"],
        base.loc[base.probability.ge(threshold), "entry_time"],
    ))
    resolved = parent.resolve_barbell(
        entries, xperp, exit_parameters["runner_target_pct"],
        exit_parameters["hold_bars"], exit_parameters["near_fraction"])
    return resolved[[key in accepted_keys for key in zip(
        resolved.symbol, resolved.entry_time)]].copy()


def main() -> None:
    report = json.loads(PARENT_REPORT.read_text(encoding="utf-8"))
    if report["status"] != "REJECTED_MONTHLY_OBJECTIVE":
        raise ValueError("unexpected parent audit state")
    parameters = report["frozen_base"]
    threshold = float(report["frozen_threshold"]["threshold"])
    exit_parameters = report["frozen_before_oct7"]

    original_cash, original_xperp, original_sources = parent.load_sources()
    model, columns = rebuild_frozen_model(
        original_cash, original_xperp, parameters)
    cash, cash_sources = verified_manifest(CASH_MANIFEST)
    xperp, venue_sources = verified_manifest(XPERP_MANIFEST)
    cash["QQQUSDT"] = original_cash["QQQUSDT"]
    xperp["QQQUSDT"] = original_xperp["QQQUSDT"]
    expansion = transfer_rows(cash, xperp, model, columns, parameters,
                              threshold, exit_parameters)
    expansion.to_csv(ROWS_OUT, index=False)
    original = pd.read_csv(PARENT_ROWS)
    combined = pd.concat([original, expansion], ignore_index=True)

    values = {
        "expansion_pre_fresh": lead.evaluate(expansion, PRE_FRESH),
        "expansion_fresh": lead.evaluate(expansion, parent.FRESH),
        "expansion_full": lead.evaluate(expansion, FULL_TRANSFER),
        "original_full": lead.evaluate(original, FULL_TRANSFER),
        "combined_pre_fresh": lead.evaluate(combined, PRE_FRESH),
        "combined_fresh": lead.evaluate(combined, parent.FRESH),
        "combined_full": lead.evaluate(combined, FULL_TRANSFER),
    }
    checks = {
        "unseen_universe_fresh_positive":
            values["expansion_fresh"]["return_pct"] > 0,
        "combined_full_improves_original":
            values["combined_full"]["return_pct"]
            > values["original_full"]["return_pct"],
        "combined_full_no_negative_months":
            values["combined_full"]["negative_months"] == 0,
        "combined_fresh_at_least_3_trades_per_day":
            values["combined_fresh"]["trades_per_calendar_day"] >= 3,
        "combined_fresh_return_at_least_30pct":
            values["combined_fresh"]["return_pct"] >= 30,
    }
    result = {
        "status": ("UNIVERSE_TRANSFER_CANDIDATE"
                   if all(checks.values()) else "REJECTED_MONTHLY_OBJECTIVE"),
        "ready_for_live_orders": False,
        "protocol": {
            "frozen_parent": str(PARENT_REPORT),
            "expansion_symbols": list(EXPANSION_SYMBOLS),
            "symbol_selection": "all unambiguous matched stocks; no outcome selection",
            "ambiguous_ticker_collisions_excluded":
                sorted(AMBIGUOUS_TICKER_COLLISIONS),
            "fresh": parent.FRESH,
            "entry": "next 15-minute X-Perp market open",
            "orders": "market only",
            "money": "3% current-equity margin, fixed 10x gross",
        },
        "expansion_rows": len(expansion),
        "values": values, "checks": checks,
        "sources": {**original_sources, **cash_sources, **venue_sources,
                    str(PARENT_REPORT): digest(PARENT_REPORT),
                    str(PARENT_ROWS): digest(PARENT_ROWS),
                    str(Path(__file__)): digest(Path(__file__))},
        "limitations": [
            "New X-Perps have unequal listing dates and short histories.",
            "The expansion aggregate was negative in September before recovering in October.",
            "The expanded portfolio still falls far below the 30% monthly objective.",
            "Gross returns exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": result["status"], "values": values,
                      "checks": checks}, indent=2), flush=True)


if __name__ == "__main__":
    main()
