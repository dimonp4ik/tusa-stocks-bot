"""Download and normalize official FRED daily US rate series."""
from __future__ import annotations

import hashlib
import io
import json
import pickle
from pathlib import Path

import pandas as pd
import requests


ROOT = Path("reports/audit_2026_09_08/stock_fred_rates_oct10")
START = "2015-01-01"
END = "2025-12-31"
SERIES = {
    "dgs2": "DGS2",
    "dgs10": "DGS10",
    "dfii10": "DFII10",
    "t10yie": "T10YIE",
}
URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}&cosd={start}&coed={end}"


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    sources: dict[str, dict] = {}
    frames = []
    for label, series in SERIES.items():
        url = URL.format(series=series, start=START, end=END)
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        raw = response.content
        path = ROOT / f"{series}.csv"
        path.write_bytes(raw)
        value = pd.read_csv(io.BytesIO(raw), na_values=["."])
        expected = ["observation_date", series]
        if list(value.columns) != expected:
            raise ValueError(f"unexpected FRED schema for {series}: {list(value.columns)}")
        value["date"] = pd.to_datetime(value.observation_date, utc=True)
        value[label] = pd.to_numeric(value[series], errors="coerce")
        value = value.dropna(subset=[label]).sort_values("date")
        if value.date.duplicated().any() or not value.date.is_monotonic_increasing:
            raise ValueError(f"invalid FRED chronology for {series}")
        frames.append(value[["date", label]])
        sources[label] = {
            "series": series,
            "url": url,
            "path": str(path),
            "sha256": digest(raw),
            "rows": len(value),
            "first_date": str(value.date.min().date()),
            "last_date": str(value.date.max().date()),
        }
    frame = frames[0]
    for value in frames[1:]:
        frame = frame.merge(value, on="date", how="inner", validate="one_to_one")
    frame = frame.sort_values("date")
    frame["date_epoch"] = frame.date.map(lambda value: int(value.timestamp()))
    if frame.date_epoch.duplicated().any() or len(frame) < 2500:
        raise ValueError("insufficient or duplicate normalized FRED history")
    payload = {name: frame[name].to_numpy() for name in (
        "date_epoch", "dgs2", "dgs10", "dfii10", "t10yie")}
    data_path = ROOT / "fred_rates_daily.pkl"
    data_path.write_bytes(pickle.dumps(payload, protocol=5))
    manifest = {
        "status": "IMMUTABLE_PUBLIC_RESEARCH_INPUT",
        "provider": "Federal Reserve Bank of St. Louis FRED",
        "requested_start": START,
        "requested_end": END,
        "sources": sources,
        "data_path": str(data_path),
        "data_sha256": digest(data_path.read_bytes()),
        "causal_availability": (
            "Each observation is delayed by two later US stock sessions before use; "
            "this is conservative versus the next-business-day H.15 update."),
        "quality": {
            "normalized_complete_rows": len(frame),
            "first_date": str(frame.date.min().date()),
            "last_date": str(frame.date.max().date()),
            "missing_calendar_days_are_not_forward_filled": True,
        },
    }
    manifest_path = ROOT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True),
                             encoding="utf-8")
    print(json.dumps({
        "data_sha256": manifest["data_sha256"],
        "quality": manifest["quality"],
        "sources": sources,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
