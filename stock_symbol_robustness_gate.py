"""Freeze a calibration-only stability gate for a stock portfolio."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


CALIBRATION_MONTHS = ("2026-07", "2026-08")
HOLDOUT_MONTH = "2026-09"


def metric(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {"n": 0, "wr": 0.0, "net_r": 0.0, "pf": 0.0, "dd_r": 0.0}
    values = frame.sort_values(["exit_time", "entry_time"])["net_r"].to_numpy(float)
    curve = np.cumsum(values)
    drawdown = np.maximum.accumulate(np.r_[0.0, curve])[1:] - curve
    wins, losses = values[values > 0].sum(), -values[values < 0].sum()
    return {
        "n": int(len(values)), "wr": float(100 * np.mean(values > 0)),
        "net_r": float(values.sum()),
        "pf": float(wins / losses) if losses else None,
        "dd_r": float(drawdown.max(initial=0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("signals", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    raw = args.signals.read_bytes()
    frame = pd.read_csv(args.signals)
    calibration = frame[frame["month"].isin(CALIBRATION_MONTHS)]
    calibration_by_symbol = calibration.groupby("symbol")["net_r"].sum()
    calibration_wr_by_symbol = calibration.groupby("symbol")["net_r"].apply(
        lambda values: float(100 * (values > 0).mean())
    )
    calibration_month_r = calibration.pivot_table(
        index="symbol", columns="month", values="net_r", aggfunc="sum", fill_value=0.0
    )
    excluded = sorted(
        symbol for symbol in calibration_by_symbol.index
        if calibration_by_symbol[symbol] <= 0
        or (
            calibration_wr_by_symbol[symbol] < 75.0
            and any(float(calibration_month_r.loc[symbol, month]) < 0
                    for month in CALIBRATION_MONTHS)
        )
    )
    output = frame[~frame["symbol"].isin(excluded)].copy()
    report = {
        "status": "PAPER_CANDIDATE",
        "selection_policy": {
            "calibration_months": list(CALIBRATION_MONTHS),
            "rule": (
                "exclude when total calibration net_r <= 0, or when calibration "
                "WR < 75% and at least one calibration month is negative"
            ),
            "holdout_month": HOLDOUT_MONTH,
            "holdout_used_for_selection": False,
        },
        "excluded_symbols": excluded,
        "calibration_net_r_by_symbol": {
            key: float(value) for key, value in calibration_by_symbol.items()
        },
        "calibration_wr_by_symbol": {
            key: float(value) for key, value in calibration_wr_by_symbol.items()
        },
        "calibration_month_net_r_by_symbol": {
            symbol: {
                month: float(calibration_month_r.loc[symbol, month])
                for month in CALIBRATION_MONTHS
            }
            for symbol in calibration_month_r.index
        },
        "july_calibration": metric(output[output["month"] == "2026-07"]),
        "august_calibration": metric(output[output["month"] == "2026-08"]),
        "september_holdout": metric(output[output["month"] == HOLDOUT_MONTH]),
        "all": metric(output),
        "source": {"path": str(args.signals), "sha256": hashlib.sha256(raw).hexdigest()},
        "limitations": [
            "The symbol gate has only two direct X-Perp calibration months.",
            "Symbol-subset alternatives were explored, increasing selection bias.",
            "A future paper-forward sample is required before live activation.",
            "Gross R excludes costs and client-selected sizing at user request.",
        ],
    }
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    output.to_csv(args.out.with_suffix(".csv"), index=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
