"""Capture hashed daily Cboe volatility-surface index histories."""
from __future__ import annotations

import hashlib
import io
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests


ROOT = Path("reports/audit_2026_09_08/stock_cboe_vol_surface_oct9")
INDICES = ("VIX", "VIX9D", "VIX3M", "VIX6M", "VIX1Y", "SKEW", "VVIX")
URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{name}_History.csv"


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    pieces, sources = [], {}
    for name in INDICES:
        response = requests.get(
            URL.format(name=name), timeout=60,
            headers={"User-Agent": "material-move-research/1.0"},
        )
        response.raise_for_status()
        raw = response.content
        raw_path = ROOT / f"{name}_History.csv"
        raw_path.write_bytes(raw)
        frame = pd.read_csv(io.BytesIO(raw))
        if "DATE" not in frame or frame.DATE.duplicated().any():
            raise ValueError(f"invalid {name} date column")
        value_column = "CLOSE" if "CLOSE" in frame else name
        if value_column not in frame:
            raise ValueError(f"invalid {name} value schema: {list(frame.columns)}")
        frame["date"] = pd.to_datetime(frame.DATE, format="%m/%d/%Y", errors="coerce")
        frame[name.lower()] = pd.to_numeric(frame[value_column], errors="coerce")
        frame = frame.dropna(subset=["date", name.lower()])[["date", name.lower()]]
        frame = frame[frame.date.between("2016-01-01", "2026-12-31")]
        pieces.append(frame)
        sources[name] = {
            "url": response.url, "path": str(raw_path),
            "sha256": digest(raw), "bytes": len(raw), "rows": len(frame),
            "first_date": str(frame.date.min().date()),
            "last_date": str(frame.date.max().date()),
        }
    combined = pieces[0]
    for frame in pieces[1:]:
        combined = combined.merge(frame, on="date", how="inner", validate="one_to_one")
    combined = combined.sort_values("date").reset_index(drop=True)
    if combined.empty or combined.isna().any().any():
        raise ValueError("empty or incomplete Cboe joined history")
    combined["date_epoch"] = combined.date.map(
        lambda value: int(value.tz_localize("UTC").timestamp()))
    payload = {column: combined[column].to_numpy() for column in (
        "date_epoch", *[name.lower() for name in INDICES])}
    data_path = ROOT / "cboe_vol_surface_daily.pkl"
    data_path.write_bytes(pickle.dumps(payload, protocol=5))
    business_dates = pd.bdate_range(combined.date.min(), combined.date.max())
    missing_weekdays = business_dates.difference(combined.date)
    manifest = {
        "status": "IMMUTABLE_PUBLIC_RESEARCH_INPUT",
        "provider": "Cboe Global Markets",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "indices": list(INDICES), "sources": sources,
        "causal_availability": (
            "Daily closing indices are joined only to a later stock session; "
            "same-session values are forbidden."),
        "data_path": str(data_path),
        "data_sha256": digest(data_path.read_bytes()),
        "quality": {
            "joined_rows": len(combined),
            "first_joined_date": str(combined.date.min().date()),
            "last_joined_date": str(combined.date.max().date()),
            "missing_weekdays_including_us_holidays": [
                str(value.date()) for value in missing_weekdays],
            "monotonic_unique_dates": bool(
                combined.date.is_monotonic_increasing and combined.date.is_unique),
        },
    }
    manifest_path = ROOT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True),
                             encoding="utf-8")
    print(json.dumps({
        "data_sha256": manifest["data_sha256"],
        "quality": manifest["quality"], "sources": sources,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
