"""Download and filter the CC0 SEC earnings-announcement archive."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


SOURCE = "https://quant500.com/api/descarga/anuncios.csv"
DEFAULT_SYMBOLS = "AAPL,AMZN,META,MRVL,MSFT,MU,NVDA"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    parser.add_argument("--start", default="2017-01-01")
    parser.add_argument("--end", required=True, help="exclusive YYYY-MM-DD")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    symbols = {value.strip().upper() for value in args.symbols.split(",")
               if value.strip()}
    request = urllib.request.Request(
        SOURCE, headers={"User-Agent": "Mozilla/5.0", "Accept": "text/csv"})
    with urllib.request.urlopen(request, timeout=90) as response:
        payload = response.read()
    text = payload.decode("utf-8-sig")
    data_lines = [line for line in text.splitlines() if not line.startswith("#")]
    reader = csv.DictReader(io.StringIO("\n".join(data_lines)))
    selected = []
    for row in reader:
        symbol = str(row.get("ticker", "")).upper()
        day = str(row.get("announcement_date", ""))
        if symbol in symbols and args.start <= day < args.end:
            selected.append(row)
    selected.sort(key=lambda row: (row["announcement_date"], row["ticker"],
                                   row.get("accession", "")))
    counts = {symbol: sum(row["ticker"].upper() == symbol for row in selected)
              for symbol in sorted(symbols)}
    report = {
        "status": "IMMUTABLE_PUBLIC_RESEARCH_INPUT",
        "provider": "Quant500 SEC 8-K Item 2.02 compilation",
        "source": SOURCE, "license": "CC0 1.0",
        "source_sha256": hashlib.sha256(payload).hexdigest(),
        "source_bytes": len(payload), "requested_start": args.start,
        "requested_end_exclusive": args.end,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "symbols": sorted(symbols), "records": len(selected),
        "records_by_symbol": counts, "earnings": selected,
        "causal_policy": (
            "announcement_date D is usable only from the next regular session; "
            "provisional SEC acceptance time/session fields are ignored"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "source_bytes": len(payload),
                      "source_sha256": report["source_sha256"],
                      "records": len(selected), "records_by_symbol": counts,
                      "sha256": hashlib.sha256(args.out.read_bytes()).hexdigest()}),
          flush=True)


if __name__ == "__main__":
    main()
