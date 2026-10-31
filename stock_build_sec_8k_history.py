"""Capture official SEC 8-K metadata for causal stock-event research.

The SEC submissions endpoint exposes an acceptance timestamp and the reported
8-K item numbers.  The normalized output deliberately excludes earnings item
2.02 because that mechanism already has a separate audit.  A filing is made
usable only from the next complete regular stock session by downstream code.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import requests


SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik}.json"
ARCHIVE = "https://data.sec.gov/submissions/{name}"
COMPANIES = {
    "AAPL": "0000320193",
    "AMZN": "0001018724",
    "META": "0001326801",
    "MSFT": "0000789019",
    "MU": "0000723125",
    "NVDA": "0001045810",
}
DISTRESS_ITEMS = {"2.03", "2.04", "2.05", "2.06", "3.01", "4.02"}
TRANSACTION_ITEMS = {"1.01", "1.02", "2.01", "3.02", "5.01"}
GOVERNANCE_ITEMS = {"4.01", "5.02", "5.03", "5.07"}


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def fetch(session: requests.Session, url: str) -> bytes:
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            response = session.get(url, timeout=45)
            if response.status_code == 429:
                time.sleep(1.0 + attempt)
                continue
            response.raise_for_status()
            return response.content
        except requests.RequestException as exc:
            last_error = exc
            if attempt < 4:
                time.sleep(1.0 + attempt)
    raise RuntimeError(f"SEC request failed: {url}: {last_error}")


def category(items: set[str]) -> str:
    if items & DISTRESS_ITEMS:
        return "distress_or_accounting"
    if items & TRANSACTION_ITEMS:
        return "material_transaction"
    if items & GOVERNANCE_ITEMS:
        return "governance"
    if items & {"7.01", "8.01"}:
        return "reg_fd_or_other"
    return "other_8k"


def normalize(symbol: str, cik: str, recent: dict,
              start: str, end: str) -> list[dict]:
    output = []
    for index, form in enumerate(recent["form"]):
        if form not in {"8-K", "8-K/A"}:
            continue
        accepted = str(recent["acceptanceDateTime"][index])
        if not accepted or not (start <= accepted[:10] < end):
            continue
        items = {value.strip() for value in
                 str(recent["items"][index]).split(",") if value.strip()}
        # Earnings and amendments that contain only the earnings item are a
        # previously tested source, not a new event mechanism.
        non_earnings = items - {"2.02", "9.01"}
        if not non_earnings:
            continue
        output.append({
            "symbol": symbol,
            "cik": cik,
            "accession": str(recent["accessionNumber"][index]),
            "form": form,
            "filing_date": str(recent["filingDate"][index]),
            "report_date": str(recent["reportDate"][index]),
            "acceptance_datetime_utc": accepted,
            "items": sorted(items),
            "non_earnings_items": sorted(non_earnings),
            "category": category(non_earnings),
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2017-01-01")
    parser.add_argument("--end", required=True, help="exclusive YYYY-MM-DD")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    session = requests.Session()
    session.headers.update({
        "User-Agent": "OpenAI-Codex research@openai.com",
        "Accept-Encoding": "gzip, deflate",
        "Accept": "application/json",
    })
    sources, events = {}, []
    for symbol, cik in COMPANIES.items():
        url = SUBMISSIONS.format(cik=cik)
        payload = fetch(session, url)
        document = json.loads(payload)
        sources[url] = {
            "sha256": digest(payload), "bytes": len(payload),
            "company_name": document.get("name"),
        }
        events.extend(normalize(
            symbol, cik, document["filings"]["recent"], args.start, args.end))
        # High-volume issuers roll older submission metadata into numbered
        # official JSON files.  Omitting these files creates symbol-specific
        # start dates and invalidates a cross-era audit.
        for item in document["filings"].get("files", []):
            if item.get("filingTo", "") < args.start:
                continue
            archive_url = ARCHIVE.format(name=item["name"])
            archive_payload = fetch(session, archive_url)
            archive_document = json.loads(archive_payload)
            sources[archive_url] = {
                "sha256": digest(archive_payload),
                "bytes": len(archive_payload),
                "company_name": document.get("name"),
                "filing_from": item.get("filingFrom"),
                "filing_to": item.get("filingTo"),
            }
            events.extend(normalize(
                symbol, cik, archive_document, args.start, args.end))
            time.sleep(.12)
        time.sleep(.12)
    # The recent/archive boundary may overlap by a few filings.  Accession is
    # the SEC's immutable filing identity.
    events = list({(row["symbol"], row["accession"]): row
                   for row in events}.values())
    events.sort(key=lambda row: (
        row["acceptance_datetime_utc"], row["symbol"], row["accession"]))
    counts = {symbol: sum(row["symbol"] == symbol for row in events)
              for symbol in COMPANIES}
    categories = {}
    for row in events:
        categories[row["category"]] = categories.get(row["category"], 0) + 1
    report = {
        "status": "IMMUTABLE_OFFICIAL_SEC_RESEARCH_INPUT",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "requested_start": args.start,
        "requested_end_exclusive": args.end,
        "provider": "U.S. Securities and Exchange Commission",
        "source_policy": "official data.sec.gov submissions JSON",
        "causal_policy": (
            "acceptance_datetime_utc is authoritative; downstream strategies "
            "may first use an event in the next complete regular session"),
        "exclusions": [
            "8-K filings whose only substantive item is earnings Item 2.02",
            "Item 9.01 because it is normally an attachment index",
        ],
        "symbols": list(COMPANIES),
        "records": len(events),
        "records_by_symbol": counts,
        "records_by_category": dict(sorted(categories.items())),
        "sources": sources,
        "events": events,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True),
                        encoding="utf-8")
    print(json.dumps({
        "out": str(args.out), "records": len(events),
        "records_by_symbol": counts,
        "records_by_category": report["records_by_category"],
        "sha256": digest(args.out.read_bytes()),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
