"""Download official Cboe implied-correlation tenor histories."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd


ROOT = Path("reports/audit_2026_09_08/stock_cboe_correlation_oct26")
SYMBOLS = ("COR1M", "COR3M", "COR6M")
URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{symbol}_History.csv"


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    frames, sources = [], {}
    for symbol in SYMBOLS:
        url = URL.format(symbol=symbol)
        request = Request(url, headers={"User-Agent": "Mozilla/5.0 research-audit"})
        with urlopen(request, timeout=60) as response:
            raw = response.read()
        raw_path = ROOT / f"{symbol}_History.csv"
        raw_path.write_bytes(raw)
        frame = pd.read_csv(io.BytesIO(raw))
        frame.columns = [str(name).strip().lower() for name in frame.columns]
        if not {"date", "open", "high", "low", "close"}.issubset(frame.columns):
            raise ValueError(f"unexpected {symbol} schema")
        frame["date"] = pd.to_datetime(frame.date, format="%m/%d/%Y", utc=True)
        source_rows = len(frame)
        frame = frame.sort_values("date").drop_duplicates("date", keep="last")
        for name in ("open", "high", "low", "close"):
            frame[name] = pd.to_numeric(frame[name], errors="raise")
        if (frame.high < frame[["open", "close"]].max(axis=1)).any():
            raise ValueError(f"invalid {symbol} high")
        if (frame.low > frame[["open", "close"]].min(axis=1)).any():
            raise ValueError(f"invalid {symbol} low")
        prefix = symbol.lower()
        frames.append(frame[["date", "open", "high", "low", "close"]].rename(
            columns={name: f"{prefix}_{name}" for name in ("open", "high", "low", "close")}))
        sources[symbol] = {"url": url, "path": str(raw_path),
                           "sha256": sha256(raw), "rows": len(frame),
                           "duplicates_removed": source_rows-len(frame)}
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="date", how="inner", validate="one_to_one")
    merged.insert(0, "date_epoch", merged.date.map(lambda value: int(value.timestamp())))
    merged = merged.drop(columns="date")
    data_path = ROOT / "correlation_daily.csv"
    merged.to_csv(data_path, index=False, lineterminator="\n")
    payload = data_path.read_bytes()
    manifest = {
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "provider": "Cboe Global Markets", "sources": sources,
        "data_path": str(data_path), "data_sha256": sha256(payload),
        "rows": len(merged),
        "start": pd.to_datetime(merged.date_epoch.iloc[0], unit="s", utc=True).isoformat(),
        "end": pd.to_datetime(merged.date_epoch.iloc[-1], unit="s", utc=True).isoformat(),
        "causal_availability": (
            "A completed Cboe correlation close may gate only a strictly later "
            "stock session; same-session use is forbidden."),
        "quality": {"monotonic_dates": bool(merged.date_epoch.is_monotonic_increasing),
                    "duplicates": int(merged.date_epoch.duplicated().sum()),
                    "missing_values": int(merged.isna().sum().sum())},
    }
    (ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
