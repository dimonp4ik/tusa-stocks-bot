"""Download hashed historical earnings surprises through Yahoo visualization."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import yfinance as yf


ENDPOINT = "https://query1.finance.yahoo.com/v1/finance/visualization"
DEFAULT_SYMBOLS = "AAPL,AMZN,META,MRVL,MSFT,MU,NVDA"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    parser.add_argument("--start", default="2017-01-01")
    parser.add_argument("--end", required=True, help="exclusive YYYY-MM-DD")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    symbols = [value.strip().upper() for value in args.symbols.split(",")
               if value.strip()]
    events, payload_hashes, response_rows = [], {}, {}
    ny = ZoneInfo("America/New_York")
    for symbol in symbols:
        ticker = yf.Ticker(symbol)
        params = {"lang": "en-US", "region": "US"}
        body = {
            "size": 100,
            "query": {"operator": "eq", "operands": ["ticker", symbol]},
            "sortField": "startdatetime", "sortType": "DESC",
            "entityIdType": "earnings",
            "includeFields": ["startdatetime", "timeZoneShortName",
                              "epsestimate", "epsactual", "epssurprisepct",
                              "eventtype"],
        }
        response = ticker._data.post(ENDPOINT, params=params, body=body)
        response.raise_for_status()
        payload = response.content
        document = response.json()["finance"]["result"][0]["documents"][0]
        columns = [column["id"] for column in document["columns"]]
        rows = [dict(zip(columns, row)) for row in document["rows"]]
        payload_hashes[symbol] = hashlib.sha256(payload).hexdigest()
        response_rows[symbol] = len(rows)
        for row in rows:
            if str(row.get("eventtype")) != "2":
                continue
            instant = datetime.fromisoformat(
                str(row["startdatetime"]).replace("Z", "+00:00"))
            local = instant.astimezone(ny)
            day = local.date().isoformat()
            if args.start <= day < args.end:
                events.append({
                    "symbol": symbol, "announcement_date": day,
                    "event_time_utc": instant.isoformat(),
                    "event_time_et": local.isoformat(),
                    "eps_estimate": row.get("epsestimate"),
                    "eps_actual": row.get("epsactual"),
                    "surprise_pct": row.get("epssurprisepct"),
                })
    events.sort(key=lambda row: (row["announcement_date"], row["symbol"]))
    counts = {symbol: sum(row["symbol"] == symbol for row in events)
              for symbol in symbols}
    report = {
        "status": "IMMUTABLE_PUBLIC_RESEARCH_INPUT",
        "provider": "Yahoo Finance visualization earnings entity",
        "source": ENDPOINT,
        "known_provider_limit": "endpoint stopped updating during 2025",
        "requested_start": args.start, "requested_end_exclusive": args.end,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "symbols": symbols, "payload_sha256": payload_hashes,
        "response_rows": response_rows, "records": len(events),
        "records_by_symbol": counts, "earnings": events,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "records": len(events),
                      "records_by_symbol": counts,
                      "payload_sha256": payload_hashes,
                      "sha256": hashlib.sha256(args.out.read_bytes()).hexdigest()}),
          flush=True)


if __name__ == "__main__":
    main()
