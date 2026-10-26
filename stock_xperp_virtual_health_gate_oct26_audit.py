"""Chronological virtual-outcome health gate for the native X-Perp route."""
from __future__ import annotations

import hashlib
import heapq
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import stock_xperp_native_economics_oct18_audit as economics


ROOT = Path("reports/audit_2026_09_08")
REFERENCE = ROOT / "stock_xperp_backing_level_guard_oct20.csv"
FORWARD = ROOT / "stock_xperp_backing_hour_cap_forward_oct26.csv"
OUT = ROOT / "stock_xperp_virtual_health_gate_oct26.json"
ROWS_OUT = ROOT / "stock_xperp_virtual_health_gate_oct26.csv"
WINDOWS = (10, 20, 30, 50, 75)
THRESHOLDS = (-.001, 0.0, .001)
PERIODS = {
    "development_july": ("2026-07-01", "2026-08-01"),
    "validation_august": ("2026-08-01", "2026-09-01"),
    "confirmation_september": ("2026-09-01", "2026-10-01"),
    "observed_oct1_oct16": ("2026-10-01", "2026-10-17"),
    "observed_oct17_oct26": ("2026-10-17", "2026-10-27"),
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def apply_gate(frame: pd.DataFrame, window: int, threshold: float) -> pd.DataFrame:
    """Use only virtual outcomes whose exit was known before each entry."""
    pending: list[tuple[float, int, dict]] = []
    history: list[float] = []
    accepted = []
    for sequence, row in enumerate(frame.sort_values(
            ["entry_time", "symbol"], kind="mergesort").to_dict("records")):
        now = float(row["entry_time"])
        while pending and pending[0][0] < now:
            _, _, completed = heapq.heappop(pending)
            history.append(float(completed["signed_return"]))
        minimum = min(20, window)
        healthy = (len(history) < minimum
                   or float(np.mean(history[-window:])) > threshold)
        if healthy:
            accepted.append(row)
        # Every setup remains virtually scored, including blocked setups.
        heapq.heappush(pending, (float(row["exit_time"]), sequence, row))
    return pd.DataFrame(accepted, columns=frame.columns)


def period(frame: pd.DataFrame, bounds: tuple[str, str]) -> pd.DataFrame:
    lo = pd.Timestamp(bounds[0], tz="UTC").timestamp()
    hi = pd.Timestamp(bounds[1], tz="UTC").timestamp()
    return frame[frame.entry_time.ge(lo) & frame.entry_time.lt(hi)]


def metrics(frame: pd.DataFrame) -> dict:
    value = economics.simulate(frame.to_dict("records"), .03)
    days = max(1, pd.to_datetime(
        frame.entry_time, unit="s", utc=True).dt.date.nunique())
    value["trades_per_active_day"] = round(len(frame) / days, 4)
    value["mean_signed_move_pct"] = round(
        100 * float(frame.signed_return.mean()), 6) if len(frame) else 0.0
    return value


def main() -> None:
    full = pd.concat([pd.read_csv(REFERENCE), pd.read_csv(FORWARD)],
                     ignore_index=True)
    baseline = {name: metrics(period(full, bounds))
                for name, bounds in PERIODS.items()}
    candidates = []
    for window, threshold in itertools.product(WINDOWS, THRESHOLDS):
        accepted = apply_gate(full, window, threshold)
        values = {name: metrics(period(accepted, bounds))
                  for name, bounds in PERIODS.items()}
        development = values["development_july"]
        candidates.append({
            "window": window, "minimum_mean_signed_move_pct": 100 * threshold,
            "development_score": development["return_pct"] - development["dd_pct"],
            "values": values, "_rows": accepted,
        })
    eligible = [item for item in candidates
                if item["values"]["development_july"]["return_pct"]
                >= baseline["development_july"]["return_pct"]
                and item["values"]["development_july"]["trades_per_active_day"] >= 3]
    eligible.sort(key=lambda item: item["development_score"], reverse=True)
    selected = eligible[0] if eligible else None
    validation_pass = bool(selected and all(
        selected["values"][name]["return_pct"] > 0
        and selected["values"][name]["trades_per_active_day"] >= 3
        for name in ("validation_august", "confirmation_september")))
    fresh_pass = bool(validation_pass and
        selected["values"]["observed_oct17_oct26"]["return_pct"] > 0)
    rows = selected["_rows"] if selected else full.iloc[:0]
    rows.to_csv(ROWS_OUT, index=False)
    public = ({k: v for k, v in selected.items() if not k.startswith("_")}
              if selected else None)
    report = {
        "status": ("SHADOW_CANDIDATE_REQUIRES_NEW_FORWARD" if fresh_pass
                   else "REJECTED_FRESH" if validation_pass
                   else "REJECTED_VALIDATION"),
        "ready_for_live_orders": False, "orders_possible": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "mechanism": "allow entries only while the mean of recent completed virtual outcomes exceeds a frozen threshold",
            "blocked_rows_continue_virtual_scoring": True,
            "selection": "July only", "validation": "August",
            "confirmation": "September", "observed_reveal": "October 1-26",
            "orders": "market only", "money": "3% current-equity margin; fixed 10x",
        },
        "grid": {"windows": list(WINDOWS),
                 "threshold_signed_move_pct": [100 * x for x in THRESHOLDS],
                 "candidates": len(candidates), "development_eligible": len(eligible)},
        "baseline": baseline, "selected": public,
        "validation_pass": validation_pass, "fresh_pass": fresh_pass,
        "sources": {str(REFERENCE): digest(REFERENCE), str(FORWARD): digest(FORWARD),
                    str(Path(__file__).resolve()): digest(Path(__file__).resolve())},
        "limitations": [
            "October was already observed before this audit and is diagnostic.",
            "The gate post-filters admitted rows and does not backfill portfolio slots.",
            "Native history begins in July 2026.",
            "Gross returns exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: report[k] for k in (
        "status", "grid", "selected", "validation_pass", "fresh_pass")},
        indent=2), flush=True)


if __name__ == "__main__":
    main()
