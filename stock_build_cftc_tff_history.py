"""Capture official CFTC financial-futures positioning for causal research."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
from pathlib import Path
import time
import zipfile

import requests


URL = "https://www.cftc.gov/files/dea/history/fut_fin_txt_{year}.zip"
CONTRACTS = {
    "bitcoin_cme": "133741",
    "nasdaq_100_consolidated": "20974+",
}
NUMERIC = {
    "open_interest": "Open_Interest_All",
    "dealer_long": "Dealer_Positions_Long_All",
    "dealer_short": "Dealer_Positions_Short_All",
    "asset_manager_long": "Asset_Mgr_Positions_Long_All",
    "asset_manager_short": "Asset_Mgr_Positions_Short_All",
    "leveraged_money_long": "Lev_Money_Positions_Long_All",
    "leveraged_money_short": "Lev_Money_Positions_Short_All",
    "other_reportable_long": "Other_Rept_Positions_Long_All",
    "other_reportable_short": "Other_Rept_Positions_Short_All",
}


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def integer(value: str) -> int:
    text = str(value).strip().replace(",", "")
    return int(float(text)) if text else 0


def fetch(session: requests.Session, url: str) -> bytes:
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            response = session.get(url, timeout=60)
            if response.status_code == 429:
                time.sleep(1.5*(attempt+1))
                continue
            response.raise_for_status()
            return response.content
        except requests.RequestException as exc:
            last_error = exc
            if attempt < 4:
                time.sleep(1.5*(attempt+1))
    raise RuntimeError(f"CFTC request failed: {url}: {last_error}")


def parse(payload: bytes, year: int) -> list[dict]:
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = [name for name in archive.namelist()
                 if name.lower().endswith(".txt")]
        if len(names) != 1:
            raise ValueError(f"unexpected CFTC archive members for {year}: {names}")
        raw = archive.read(names[0])
    rows = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
    inverse = {code: name for name, code in CONTRACTS.items()}
    output = []
    for row in rows:
        code = str(row["CFTC_Contract_Market_Code"]).strip()
        if code not in inverse:
            continue
        report_date = datetime.strptime(
            row["Report_Date_as_YYYY-MM-DD"], "%Y-%m-%d").replace(
                tzinfo=timezone.utc)
        # COT values are dated Tuesday and are normally published Friday.
        # A full seven-day delay is deliberately more conservative and avoids
        # holiday/publication-time ambiguity in historical files.
        available = report_date + timedelta(days=7)
        item = {
            "contract": inverse[code],
            "contract_code": code,
            "market_name": row["Market_and_Exchange_Names"].strip(),
            "report_date": report_date.date().isoformat(),
            "available_at_utc": available.isoformat(),
            "source_year": year,
        }
        item.update({name: integer(row[column])
                     for name, column in NUMERIC.items()})
        if item["open_interest"] <= 0:
            raise ValueError(f"non-positive open interest: {item}")
        output.append(item)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=2017)
    parser.add_argument("--end-year", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 CFTC research audit"})
    sources, records = {}, []
    for year in range(args.start_year, args.end_year+1):
        url = URL.format(year=year)
        payload = fetch(session, url)
        sources[url] = {"sha256": digest(payload), "bytes": len(payload)}
        records.extend(parse(payload, year))
        time.sleep(.1)
    identities = {(row["contract"], row["report_date"]) for row in records}
    if len(identities) != len(records):
        raise ValueError("duplicate CFTC contract/report date")
    records.sort(key=lambda row: (row["available_at_utc"], row["contract"]))
    coverage = {}
    for contract in CONTRACTS:
        selected = [row for row in records if row["contract"] == contract]
        coverage[contract] = {
            "records": len(selected),
            "first_report_date": selected[0]["report_date"] if selected else None,
            "last_report_date": selected[-1]["report_date"] if selected else None,
        }
    report = {
        "status": "IMMUTABLE_OFFICIAL_CFTC_RESEARCH_INPUT",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "provider": "U.S. Commodity Futures Trading Commission",
        "report_family": "Traders in Financial Futures, Futures Only",
        "requested_years": [args.start_year, args.end_year],
        "causal_policy": (
            "report_date Tuesday is usable only at 00:00 UTC seven days later; "
            "this is later than the normal Friday publication"),
        "contract_codes": CONTRACTS,
        "coverage": coverage,
        "sources": sources,
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True),
                        encoding="utf-8")
    print(json.dumps({
        "out": str(args.out), "coverage": coverage,
        "source_count": len(sources),
        "sha256": digest(args.out.read_bytes()),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
