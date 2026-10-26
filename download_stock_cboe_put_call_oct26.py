"""Download the official Cboe daily put/call ratios used by the stock audit."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import threading
import time

import pandas as pd
import requests

import stock_15m_multisession_regime_oct6_audit as base


ROOT = Path("reports/audit_2026_09_08/stock_cboe_put_call_oct26")
DATA = ROOT / "put_call_daily.csv"
MANIFEST = ROOT / "manifest.json"
URL = "https://www.cboe.com/markets/us/options/market-statistics/daily?dt={date}"
FIRST_DATE = "2019-10-07"
NAMES = {
    "TOTAL PUT/CALL RATIO": "total_put_call",
    "INDEX PUT/CALL RATIO": "index_put_call",
    "EXCHANGE TRADED PRODUCTS PUT/CALL RATIO": "etp_put_call",
    "EQUITY PUT/CALL RATIO": "equity_put_call",
    "CBOE VOLATILITY INDEX (VIX) PUT/CALL RATIO": "vix_put_call",
    "SPX + SPXW PUT/CALL RATIO": "spx_put_call",
}
LOCAL = threading.local()


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def session() -> requests.Session:
    if not hasattr(LOCAL, "session"):
        value = requests.Session()
        value.headers.update({"User-Agent": "Mozilla/5.0 research-audit"})
        LOCAL.session = value
    return LOCAL.session


def parse(date: str, payload: str) -> dict | None:
    selected = re.search(r'\\"selectedDate\\":\\"([^\\"]+)', payload)
    if selected is None:
        selected = re.search(r'"selectedDate":"([^"]+)', payload)
    if selected is None or selected.group(1) != date:
        return None
    row = {"date": date}
    for label, field in NAMES.items():
        escaped = re.escape(label).replace(r"\ ", " ")
        match = re.search(
            rf'\\"name\\":\\"{escaped}\\",\\"value\\":\\"([0-9.]+)',
            payload)
        if match is None:
            match = re.search(
                rf'"name":"{escaped}","value":"([0-9.]+)', payload)
        if match is None:
            raise ValueError(f"missing {label} on {date}")
        row[field] = float(match.group(1))
    return row


def fetch(date: str) -> dict | None:
    last_error = None
    for attempt in range(4):
        try:
            response = session().get(URL.format(date=date), timeout=45)
            response.raise_for_status()
            return parse(date, response.text)
        except Exception as error:  # network retry is part of acquisition
            last_error = error
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Cboe download failed for {date}: {last_error}")


def write(rows: list[dict]) -> None:
    frame = pd.DataFrame(rows).sort_values("date").drop_duplicates("date")
    frame.to_csv(DATA, index=False, lineterminator="\n")


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    candles, _ = base.load_history()
    features, _ = base.build_features(candles)
    wanted = sorted(date for date in features.session_date.unique()
                    if FIRST_DATE <= date <= "2026-10-23")
    existing = []
    if DATA.exists():
        existing = pd.read_csv(DATA).to_dict("records")
    have = {str(row["date"]) for row in existing}
    missing = [date for date in wanted if date not in have]
    rows = list(existing)
    print(json.dumps({"wanted": len(wanted), "cached": len(have),
                      "missing": len(missing)}), flush=True)
    with ThreadPoolExecutor(max_workers=18) as pool:
        futures = {pool.submit(fetch, date): date for date in missing}
        for index, future in enumerate(as_completed(futures), 1):
            row = future.result()
            if row is not None:
                rows.append(row)
            if index % 50 == 0:
                write(rows)
                print(json.dumps({"completed": index, "total": len(missing),
                                  "valid_rows": len(rows)}), flush=True)
    write(rows)
    payload = DATA.read_bytes()
    frame = pd.read_csv(DATA)
    numeric = list(NAMES.values())
    quality = {
        "duplicates": int(frame.date.duplicated().sum()),
        "monotonic_dates": bool(frame.date.is_monotonic_increasing),
        "missing_values": int(frame[numeric].isna().sum().sum()),
        "nonnegative_ratios": bool((frame[numeric] >= 0).all().all()),
    }
    if quality != {"duplicates": 0, "monotonic_dates": True,
                    "missing_values": 0, "nonnegative_ratios": True}:
        raise ValueError(f"put/call quality failure: {quality}")
    manifest = {
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "provider": "Cboe Global Markets",
        "source_url_template": URL,
        "causal_availability": (
            "A completed Cboe ratio may gate only a strictly later stock "
            "session; same-session use is forbidden."),
        "data_path": str(DATA), "data_sha256": sha256(payload),
        "rows": len(frame), "start": str(frame.date.iloc[0]),
        "end": str(frame.date.iloc[-1]), "quality": quality,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True),
                        encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
