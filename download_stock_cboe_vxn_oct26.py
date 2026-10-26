"""Download and normalize the official Cboe VXN daily history."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import io
import json
import pickle
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd


ROOT = Path("reports/audit_2026_09_08/stock_cboe_vxn_oct26")
URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VXN_History.csv"


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    request = Request(URL, headers={"User-Agent": "Mozilla/5.0 research-audit"})
    with urlopen(request, timeout=60) as response:
        raw = response.read()
    raw_path = ROOT / "VXN_History.csv"
    raw_path.write_bytes(raw)
    frame = pd.read_csv(io.BytesIO(raw))
    frame.columns = [str(name).strip().lower() for name in frame.columns]
    required = {"date", "open", "high", "low", "close"}
    if not required.issubset(frame.columns):
        raise ValueError(f"unexpected VXN schema: {list(frame.columns)}")
    frame["date"] = pd.to_datetime(frame.date, format="%m/%d/%Y", utc=True)
    for name in ("open", "high", "low", "close"):
        frame[name] = pd.to_numeric(frame[name], errors="raise")
    source_rows = len(frame)
    frame = frame.sort_values("date").drop_duplicates("date", keep="last")
    if (frame[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("non-positive VXN value")
    if (frame.high < frame[["open", "close"]].max(axis=1)).any():
        raise ValueError("invalid VXN high")
    if (frame.low > frame[["open", "close"]].min(axis=1)).any():
        raise ValueError("invalid VXN low")
    normalized = pd.DataFrame({
        "date_epoch": frame.date.map(lambda value: int(value.timestamp())),
        "vxn_open": frame.open.astype(float),
        "vxn_high": frame.high.astype(float),
        "vxn_low": frame.low.astype(float),
        "vxn_close": frame.close.astype(float),
    })
    data = pickle.dumps(normalized.to_dict("list"), protocol=5)
    data_path = ROOT / "vxn_daily.pkl"
    data_path.write_bytes(data)
    manifest = {
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "provider": "Cboe Global Markets",
            "url": URL,
            "path": str(raw_path),
            "sha256": sha256(raw),
        },
        "data_path": str(data_path),
        "data_sha256": sha256(data),
        "rows": len(normalized),
        "start": frame.date.iloc[0].isoformat(),
        "end": frame.date.iloc[-1].isoformat(),
        "causal_availability": (
            "A VXN close may gate only a strictly later stock session; "
            "same-session attachment is forbidden."),
        "quality": {
            "duplicates_removed": int(source_rows-len(frame)),
            "monotonic_dates": bool(normalized.date_epoch.is_monotonic_increasing),
            "positive_ohlc": True,
            "valid_ohlc_envelope": True,
        },
    }
    manifest_path = ROOT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True),
                             encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
