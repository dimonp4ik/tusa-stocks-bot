"""Audit a frozen US cash-session exclusion for the native X-Perp route.

Stock X-Perps trade continuously, but the underlying equities enter their
strongest price-discovery regime from 09:30 through 16:00 New York time.  The
native route fades relative extremes, so this audit removes that interval and
keeps the existing market entries everywhere else.  The gate is causal and
DST-aware.  October 17-26 had already been observed before this hypothesis and
is diagnostic only; genuinely new shadow-forward evidence is still required.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, time, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

import stock_xperp_native_economics_oct18_audit as economics


ROOT = Path("reports/audit_2026_09_08")
REFERENCE = ROOT / "stock_xperp_backing_level_guard_oct20.csv"
FORWARD = ROOT / "stock_xperp_backing_hour_cap_forward_oct26.csv"
OUT = ROOT / "stock_xperp_cash_session_exclusion_oct26.json"
ROWS_OUT = ROOT / "stock_xperp_cash_session_exclusion_oct26.csv"
ET = ZoneInfo("America/New_York")
PERIODS = {
    "development_2026_07": ("2026-07-01", "2026-08-01"),
    "validation_2026_08": ("2026-08-01", "2026-09-01"),
    "confirmation_2026_09": ("2026-09-01", "2026-10-01"),
    "observed_oct1_oct16": ("2026-10-01", "2026-10-17"),
    "observed_oct17_oct26": ("2026-10-17", "2026-10-27"),
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_us_cash_session(timestamp: float) -> bool:
    local = datetime.fromtimestamp(float(timestamp), timezone.utc).astimezone(ET)
    return local.weekday() < 5 and time(9, 30) <= local.time() < time(16, 0)


def outside_cash(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[~frame.entry_time.map(is_us_cash_session)].copy()


def window(frame: pd.DataFrame, bounds: tuple[str, str]) -> pd.DataFrame:
    lower = pd.Timestamp(bounds[0], tz="UTC").timestamp()
    upper = pd.Timestamp(bounds[1], tz="UTC").timestamp()
    return frame[frame.entry_time.ge(lower) & frame.entry_time.lt(upper)]


def metrics(frame: pd.DataFrame, margin: float) -> dict:
    value = economics.simulate(frame.to_dict("records"), margin)
    dates = pd.to_datetime(frame.entry_time, unit="s", utc=True).dt.date
    days = max(1, dates.nunique())
    value.update({
        "trades_per_active_day": round(len(frame) / days, 4),
        "profitable_exit_rate_pct": round(
            100 * float(frame.signed_return.gt(0).mean()), 4) if len(frame) else 0.0,
        "mean_signed_move_pct": round(
            100 * float(frame.signed_return.mean()), 6) if len(frame) else 0.0,
    })
    return value


def main() -> None:
    reference = pd.read_csv(REFERENCE)
    forward = pd.read_csv(FORWARD)
    full = pd.concat([reference, forward], ignore_index=True).sort_values(
        ["entry_time", "symbol"], kind="mergesort")
    filtered = outside_cash(full)
    periods = {}
    for name, bounds in PERIODS.items():
        before = window(full, bounds)
        after = window(filtered, bounds)
        periods[name] = {
            "removed_cash_session_rows": int(len(before) - len(after)),
            "baseline_3pct": metrics(before, .03),
            "filtered_3pct": metrics(after, .03),
            "baseline_5pct": metrics(before, .05),
            "filtered_5pct": metrics(after, .05),
        }
    filtered.to_csv(ROWS_OUT, index=False)
    old = periods["development_2026_07"]
    aug = periods["validation_2026_08"]
    sep = periods["confirmation_2026_09"]
    preserved = all(stage["filtered_3pct"]["return_pct"] > 0
                    for stage in (old, aug, sep))
    enough_validation_occurrences = (
        aug["removed_cash_session_rows"] >= 20
        and sep["removed_cash_session_rows"] >= 20)
    report = {
        "status": ("SHADOW_CANDIDATE_REQUIRES_GENUINELY_NEW_FORWARD"
                   if preserved and enough_validation_occurrences
                   else "REJECTED_INSUFFICIENT_OCCURRENCES"),
        "ready_for_live_orders": False,
        "orders_possible": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "rule": "reject weekday entries from 09:30 inclusive to 16:00 exclusive America/New_York",
            "entry": "unchanged next-15m market entry outside the excluded session",
            "money": "client-selected 3% or 5% current-equity margin; fixed 10x",
            "orders": "market only",
            "chronology": "July development, August validation, September confirmation; October already observed and diagnostic",
            "application": "conservative post-filter of already portfolio-admitted rows",
        },
        "periods": periods,
        "checks": {
            "positive_july_august_september_at_3pct": preserved,
            "improves_july": old["filtered_3pct"]["return_pct"] > old["baseline_3pct"]["return_pct"],
            "improves_august": aug["filtered_3pct"]["return_pct"] > aug["baseline_3pct"]["return_pct"],
            "september_remains_positive": sep["filtered_3pct"]["return_pct"] > 0,
            "enough_validation_occurrences": enough_validation_occurrences,
            "new_forward_available": False,
        },
        "sources": {
            str(REFERENCE): digest(REFERENCE), str(FORWARD): digest(FORWARD),
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "The October slices were already observed before this audit and are diagnostic only.",
            "The exact cash session occurred too rarely in admitted rows to validate the hypothesis.",
            "Post-filtering is conservative: it does not backfill later trades that a removed loss might have unpaused.",
            "Native X-Perp history begins in July 2026.",
            "Gross returns exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": report["status"], "checks": report["checks"],
                      "periods": periods}, indent=2), flush=True)


if __name__ == "__main__":
    main()
