"""Route the frozen material stock module by causal US-session bucket.

Buckets are selected only on the original ten-symbol universe before October
7.  The frozen buckets are then applied unchanged to the original plus eight
unseen stock X-Perps, with October 7-16 retained as the final fresh reveal.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

import stock_cash_xperp_leadlag_oct6_audit as lead
import stock_xperp_material_barbell_oct18_audit as parent
import stock_xperp_material_universe_transfer_oct18_audit as transfer


ROOT = parent.ROOT
OUT = ROOT / "stock_xperp_material_session_router_oct18.json"
ROWS_OUT = ROOT / "stock_xperp_material_session_router_oct18.csv"
BUCKETS = ("open", "early", "mid", "close")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def attach_bucket(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    stamp = pd.to_datetime(output.entry_time, unit="s", utc=True)
    hour = stamp.dt.hour + stamp.dt.minute / 60
    output["session_bucket"] = pd.cut(
        hour, [-1, 14.5, 16.5, 18.5, 25], labels=BUCKETS,
        include_lowest=True, right=False).astype(str)
    return output


def bucket_development(frame: pd.DataFrame, bucket: str) -> dict:
    part = frame[frame.session_bucket.eq(bucket)]
    return {
        "development": lead.evaluate(part, parent.DEVELOPMENT),
        "validation": lead.evaluate(part, parent.VALIDATION),
        "confirmation": lead.evaluate(part, parent.CONFIRMATION),
    }


def bucket_ok(values: dict) -> bool:
    return all(value["n"] >= 5 and value["return_pct"] > 0
               and value["dd_pct"] <= 5 for value in values.values())


def main() -> None:
    original = attach_bucket(pd.read_csv(parent.ROWS_OUT))
    expansion = attach_bucket(pd.read_csv(transfer.ROWS_OUT))
    development = {bucket: bucket_development(original, bucket)
                   for bucket in BUCKETS}
    selected = tuple(bucket for bucket in BUCKETS
                     if bucket_ok(development[bucket]))
    if not selected:
        raise ValueError("no session bucket survived pre-fresh development")

    original_selected = original[original.session_bucket.isin(selected)].copy()
    combined = pd.concat([original, expansion], ignore_index=True)
    combined_selected = combined[combined.session_bucket.isin(selected)].copy()
    combined_selected.to_csv(ROWS_OUT, index=False)

    values = {
        "original_selected_development":
            lead.evaluate(original_selected, parent.DEVELOPMENT),
        "original_selected_validation":
            lead.evaluate(original_selected, parent.VALIDATION),
        "original_selected_confirmation":
            lead.evaluate(original_selected, parent.CONFIRMATION),
        "original_selected_fresh":
            lead.evaluate(original_selected, parent.FRESH),
        "combined_pre_fresh":
            lead.evaluate(combined_selected, transfer.PRE_FRESH),
        "combined_fresh":
            lead.evaluate(combined_selected, parent.FRESH),
        "combined_full":
            lead.evaluate(combined_selected, transfer.FULL_TRANSFER),
    }
    fresh = values["combined_fresh"]
    projected = (100 * ((1 + fresh["return_pct"] / 100)
                        ** (22 / max(fresh["active_days"], 1)) - 1)
                 if fresh["return_pct"] > -100 else -100)
    checks = {
        "buckets_selected_before_fresh": bool(selected),
        "fresh_positive": fresh["return_pct"] > 0,
        "fresh_no_negative_month": fresh["negative_months"] == 0,
        "fresh_at_least_3_trades_per_day":
            fresh["trades_per_calendar_day"] >= 3,
        "fresh_drawdown_at_most_5pct": fresh["dd_pct"] <= 5,
        "fresh_return_at_least_30pct": fresh["return_pct"] >= 30,
    }
    report = {
        "status": ("STOCK_SESSION_ROUTER_CANDIDATE"
                   if all(checks.values()) else "REJECTED_MONTHLY_OBJECTIVE"),
        "ready_for_live_orders": False,
        "protocol": {
            "bucket_selection_universe": "original ten stock X-Perps only",
            "bucket_selection_periods": {
                "development": parent.DEVELOPMENT,
                "validation": parent.VALIDATION,
                "confirmation": parent.CONFIRMATION,
            },
            "untouched_fresh_reveal": parent.FRESH,
            "bucket_boundaries_utc": {
                "open": "before 14:30", "early": "14:30-16:29",
                "mid": "16:30-18:29", "close": "18:30 onward",
            },
            "selected_buckets": list(selected),
            "expansion_universe_selected_by_outcome": False,
            "entry": "next 15-minute X-Perp market open",
            "orders": "market only",
            "exit": "50% at +1%; 50% runner to +1.5%; 1% initial stop",
            "money": "3% current-equity margin, fixed 10x gross",
        },
        "bucket_development": development,
        "values": values,
        "fresh_22_session_extrapolation_pct": round(projected, 4),
        "objective_checks": checks,
        "sources": {
            str(parent.ROWS_OUT): digest(parent.ROWS_OUT),
            str(parent.OUT): digest(parent.OUT),
            str(transfer.ROWS_OUT): digest(transfer.ROWS_OUT),
            str(transfer.OUT): digest(transfer.OUT),
            str(Path(__file__)): digest(Path(__file__)),
        },
        "limitations": [
            "The fresh result contains only eight active sessions.",
            "The 22-session extrapolation is arithmetic context, not evidence of a full month.",
            "Fixed UTC boundaries require a daylight-saving-time mapping in runtime.",
            "The combined pre-fresh expansion period contains a partial negative October segment.",
            "Gross returns exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": report["status"], "selected": selected,
                      "values": values, "projected_22_sessions": projected,
                      "checks": checks}, indent=2), flush=True)


if __name__ == "__main__":
    main()
