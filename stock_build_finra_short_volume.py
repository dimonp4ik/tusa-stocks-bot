"""Build a hashed daily stock short-volume archive from FINRA Reg SHO files."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import io
import json
import pickle
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests


ROOT = Path("reports/audit_2026_09_08/stock_finra_short_volume_oct9")
CACHE = ROOT / "daily_cache"
BASE = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{day}.txt"
SYMBOLS = ("AAPL", "AMZN", "FB", "META", "MRVL", "MSFT", "MU", "NVDA", "QQQ")


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def download(day: pd.Timestamp, retries: int = 4) -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    label = day.strftime("%Y%m%d")
    target = CACHE / f"CNMSshvol{label}.txt"
    url = BASE.format(day=label)
    if target.exists() and target.stat().st_size > 100:
        raw = target.read_bytes()
        return {"day": str(day.date()), "path": str(target), "url": url,
                "sha256": digest(raw), "bytes": len(raw), "cached": True}
    error = None
    for attempt in range(retries):
        try:
            response = requests.get(
                url, timeout=45,
                headers={"User-Agent": "material-move-research/1.0"},
            )
            if response.status_code in (403, 404):
                return {"day": str(day.date()), "url": url, "missing": True,
                        "http_status": response.status_code}
            response.raise_for_status()
            raw = response.content
            if not raw.startswith(b"Date|Symbol|ShortVolume|"):
                raise ValueError("unexpected FINRA schema")
            target.write_bytes(raw)
            return {"day": str(day.date()), "path": str(target), "url": url,
                    "sha256": digest(raw), "bytes": len(raw), "cached": False}
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            time.sleep(1.5*(attempt+1))
    return {"day": str(day.date()), "url": url, "error": error}


def parse(item: dict) -> tuple[pd.DataFrame, dict]:
    if item.get("missing") or item.get("error"):
        return pd.DataFrame(), item
    raw = Path(item["path"]).read_bytes()
    lines = raw.decode("utf-8-sig").splitlines()
    if len(lines) < 2:
        raise ValueError(f"truncated FINRA file for {item['day']}")
    trailer = lines[-1].strip()
    if trailer.isdigit():
        if int(trailer) != len(lines)-2:
            raise ValueError(f"FINRA record count mismatch for {item['day']}")
    elif trailer.startswith("Record Count:"):
        if int(trailer.split(":", 1)[1].strip()) != len(lines)-2:
            raise ValueError(f"FINRA record count mismatch for {item['day']}")
    else:
        raise ValueError(f"missing FINRA trailer for {item['day']}")
    frame = pd.read_csv(io.StringIO("\n".join(lines[:-1])), sep="|")
    required = {"Date", "Symbol", "ShortVolume", "ShortExemptVolume", "TotalVolume"}
    if not required.issubset(frame.columns):
        raise ValueError(f"FINRA schema mismatch for {item['day']}")
    frame = frame[frame.Symbol.isin(SYMBOLS)].copy()
    for name in ("ShortVolume", "ShortExemptVolume", "TotalVolume"):
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    frame = frame.dropna(subset=["Date", "Symbol", "ShortVolume",
                                 "ShortExemptVolume", "TotalVolume"])
    frame = frame[frame.TotalVolume.gt(0)]
    diagnostic = {**item, "target_rows": len(frame),
                  "symbols": sorted(frame.Symbol.unique().tolist())}
    return frame, diagnostic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2018-08-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    requested = list(pd.bdate_range(args.start, args.end))
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        downloaded = list(pool.map(download, requested))
    pieces, diagnostics = [], []
    for index, item in enumerate(downloaded, 1):
        frame, diagnostic = parse(item)
        diagnostics.append(diagnostic)
        if not frame.empty:
            pieces.append(frame)
        if index % 250 == 0:
            print(json.dumps({"parsed": index, "requested": len(requested)}),
                  flush=True)
    if not pieces:
        raise ValueError("no FINRA target rows")
    frame = pd.concat(pieces, ignore_index=True)
    dates_numeric = pd.to_numeric(frame.Date, errors="coerce")
    meta_company = ((frame.Symbol.eq("FB") & dates_numeric.le(20220608))
                    | (frame.Symbol.eq("META") & dates_numeric.ge(20220609)))
    frame = frame[~frame.Symbol.isin(("FB", "META")) | meta_company].copy()
    frame["date"] = pd.to_datetime(frame.Date.astype(str), format="%Y%m%d", utc=True)
    frame["date_epoch"] = frame.date.map(lambda value: int(value.timestamp()))
    frame["asset"] = frame.Symbol
    frame.loc[frame.Symbol.isin(("FB", "META")), "asset"] = "META"
    frame = frame.sort_values(["date_epoch", "asset"])
    if frame.duplicated(["date_epoch", "asset"]).any():
        raise ValueError("duplicate FINRA asset/day after FB-META mapping")
    frame["short_ratio"] = frame.ShortVolume/frame.TotalVolume
    frame["short_exempt_ratio"] = frame.ShortExemptVolume/frame.TotalVolume
    frame = frame.rename(columns={
        "ShortVolume": "short_volume", "ShortExemptVolume": "short_exempt_volume",
        "TotalVolume": "total_volume",
    })
    payload = {column: frame[column].to_numpy() for column in (
        "date_epoch", "asset", "short_volume", "short_exempt_volume",
        "total_volume", "short_ratio", "short_exempt_ratio",
    )}
    ROOT.mkdir(parents=True, exist_ok=True)
    data_path = ROOT / "finra_short_volume_daily.pkl"
    data_path.write_bytes(pickle.dumps(payload, protocol=5))
    coverage = {}
    for asset, value in frame.groupby("asset"):
        coverage[asset] = {
            "rows": len(value), "first_date": str(value.date.min().date()),
            "last_date": str(value.date.max().date()),
            "short_ratio_min": round(float(value.short_ratio.min()), 8),
            "short_ratio_median": round(float(value.short_ratio.median()), 8),
            "short_ratio_max": round(float(value.short_ratio.max()), 8),
        }
    quality = {
        "requested_weekdays": len(requested),
        "available_files": sum(not x.get("missing") and not x.get("error")
                               for x in diagnostics),
        "missing_files": [x["day"] for x in diagnostics if x.get("missing")],
        "error_files": [{"day": x["day"], "error": x.get("error")}
                        for x in diagnostics if x.get("error")],
        "target_rows": len(frame), "coverage": coverage,
    }
    manifest = {
        "status": "IMMUTABLE_PUBLIC_RESEARCH_INPUT",
        "provider": "FINRA Reg SHO Daily Short Sale Volume",
        "source_template": BASE, "start": args.start, "end": args.end,
        "requested_symbols": list(SYMBOLS),
        "meta_symbol_history": (
            "Meta Platforms uses FINRA FB through 2022-06-08 and META from "
            "2022-06-09; unrelated META-before and FB-after rows are excluded."),
        "causal_availability": (
            "Trade-date aggregate is published after the regular session and is "
            "joined only to a later stock session."),
        "semantic_warning": (
            "FINRA short-sale volume is not short interest and excludes exchange data."),
        "data_path": str(data_path),
        "data_sha256": digest(data_path.read_bytes()), "quality": quality,
    }
    manifest_path = ROOT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True),
                             encoding="utf-8")
    print(json.dumps({
        "data_sha256": manifest["data_sha256"],
        "requested_weekdays": quality["requested_weekdays"],
        "available_files": quality["available_files"],
        "missing_files": len(quality["missing_files"]),
        "error_files": len(quality["error_files"]),
        "target_rows": quality["target_rows"], "coverage": coverage,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
