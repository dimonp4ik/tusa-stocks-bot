"""Capture official Cboe daily put/call ratios with raw-page hashes."""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import pickle
import time
from datetime import date, timedelta
from pathlib import Path

import requests


OUT = Path("reports/audit_2026_09_08/stock_cboe_put_call_oct16")
START = date(2019, 10, 7)
END = date(2026, 9, 30)
URL = "https://www.cboe.com/us/options/market_statistics/daily/?dt={day}"


def weekdays(start: date, end: date):
    current = start
    while current <= end:
        if current.weekday() < 5:
            yield current
        current += timedelta(days=1)


def parse_page(text: str, requested: str) -> dict | None:
    position = text.find("optionsData")
    if position < 0:
        raise ValueError("optionsData marker missing")
    decoded = text[position:].replace(chr(92)+chr(34), chr(34))
    if decoded.startswith("optionsData"+chr(34)+":null"):
        return None
    start = decoded.find("{")
    marker = ","+chr(34)+"selectedDate"+chr(34)
    end = decoded.find(marker)
    if start < 0 or end < 0:
        raise ValueError("optionsData payload boundary missing")
    payload = json.loads(decoded[start:end])
    suffix = decoded[end:]
    selected = suffix.split(chr(34))[3]
    if selected != requested:
        raise ValueError(f"selected date mismatch: {selected} != {requested}")
    ratios = {item["name"]: float(item["value"])
              for item in payload["ratios"]}
    names = {
        "total_pc": "TOTAL PUT/CALL RATIO",
        "index_pc": "INDEX PUT/CALL RATIO",
        "etp_pc": "EXCHANGE TRADED PRODUCTS PUT/CALL RATIO",
        "equity_pc": "EQUITY PUT/CALL RATIO",
        "vix_pc": "CBOE VOLATILITY INDEX (VIX) PUT/CALL RATIO",
        "spx_pc": "SPX + SPXW PUT/CALL RATIO",
    }
    row = {"date": requested}
    for target, source in names.items():
        if source not in ratios:
            raise ValueError(f"ratio missing: {source}")
        row[target] = ratios[source]
    categories = {
        "total": "SUM OF ALL PRODUCTS", "index": "INDEX OPTIONS",
        "etp": "EXCHANGE TRADED PRODUCTS", "equity": "EQUITY OPTIONS",
        "vix": "CBOE VOLATILITY INDEX (VIX)", "spx": "SPX + SPXW",
    }
    for prefix, source in categories.items():
        records = {item["name"]: item for item in payload[source]}
        for metric, key in (("volume", "VOLUME"), ("open_interest", "OPEN INTEREST")):
            item = records[key]
            for side in ("call", "put", "total"):
                row[f"{prefix}_{metric}_{side}"] = int(item[side])
    return row


def fetch(day: date) -> tuple[dict, dict | None]:
    label = day.isoformat()
    url = URL.format(day=label)
    error = None
    for attempt in range(4):
        try:
            response = requests.get(url, headers={"RSC": "1"}, timeout=45)
            response.raise_for_status()
            raw = response.content
            row = parse_page(response.text, label)
            return {
                "date": label, "url": url,
                "status": "ok" if row else "closed",
                "raw_sha256": hashlib.sha256(raw).hexdigest(),
                "raw_bytes": len(raw),
            }, row
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            time.sleep(1+attempt)
    return {"date": label, "url": url, "status": "error", "error": error}, None


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    checkpoint = OUT / "checkpoint.pkl"
    if checkpoint.exists():
        saved = pickle.loads(checkpoint.read_bytes())
        records, rows = saved["records"], saved["rows"]
    else:
        records, rows = [], []
    completed = {item["date"] for item in records}
    all_days = list(weekdays(START, END))
    schedule = [day for day in all_days if day.isoformat() not in completed]
    with concurrent.futures.ThreadPoolExecutor(max_workers=24) as pool:
        futures = {pool.submit(fetch, day): day for day in schedule}
        done = 0
        for future in concurrent.futures.as_completed(futures):
            meta, row = future.result()
            records.append(meta)
            if row:
                rows.append(row)
            done += 1
            if done % 100 == 0:
                checkpoint.write_bytes(pickle.dumps(
                    {"records": records, "rows": rows},
                    protocol=pickle.HIGHEST_PROTOCOL))
                print("processed", len(records), "of", len(all_days), flush=True)
    records.sort(key=lambda item: item["date"])
    rows.sort(key=lambda item: item["date"])
    if any(item["status"] == "error" for item in records):
        errors = [item for item in records if item["status"] == "error"]
        raise RuntimeError(f"Cboe download errors: {errors[:5]}")
    data_path = OUT / "cboe_put_call_2019_2026.pkl"
    data_path.write_bytes(pickle.dumps(rows, protocol=pickle.HIGHEST_PROTOCOL))
    manifest = {
        "status": "VALIDATED_CAPTURE",
        "source": "official Cboe U.S. Options Daily Market Statistics",
        "start": START.isoformat(), "end": END.isoformat(),
        "causal_availability": (
            "end-of-day ratios are eligible only for a later stock session"),
        "data_path": str(data_path), "rows": len(rows),
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "requests": records,
        "quality": {
            "weekdays_requested": len(all_days),
            "trading_days": len(rows),
            "closed_weekdays": sum(item["status"] == "closed" for item in records),
            "errors": 0,
            "unique_dates": len({row["date"] for row in rows}),
        },
    }
    manifest_path = OUT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"data_path": str(data_path), **manifest["quality"]}, indent=2))


if __name__ == "__main__":
    main()
