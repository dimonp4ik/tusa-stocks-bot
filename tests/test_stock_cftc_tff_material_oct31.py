from __future__ import annotations

import io
import zipfile

import pandas as pd

import stock_build_cftc_tff_history as build
import stock_cftc_tff_material_oct31_audit as audit


def _archive(text: str) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("FinFut24.txt", text)
    return output.getvalue()


def test_parse_keeps_named_contracts_and_delays_report_seven_days():
    header = [
        "CFTC_Contract_Market_Code", "Market_and_Exchange_Names",
        "Report_Date_as_YYYY-MM-DD", *build.NUMERIC.values(),
    ]
    values = ["1,000", "600", "500", "700", "400", "300", "350",
              "100", "90"]
    rows = [
        ["20974+", "NASDAQ-100 Consolidated", "2024-01-02", *values],
        ["133741", "BITCOIN CME", "2024-01-02", *values],
        ["999999", "UNUSED", "2024-01-02", *values],
    ]
    import csv
    text = io.StringIO()
    writer = csv.writer(text, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)

    parsed = build.parse(_archive(text.getvalue()), 2024)

    assert {row["contract"] for row in parsed} == {
        "nasdaq_100_consolidated", "bitcoin_cme"}
    assert {row["available_at_utc"] for row in parsed} == {
        "2024-01-09T00:00:00+00:00"}
    assert all(row["open_interest"] == 1000 for row in parsed)


def test_attach_never_uses_a_future_cftc_release():
    features = pd.DataFrame({
        "entry_time": [99, 100, 199, 200],
        "session_date": ["2024-01-01"]*4,
    })
    cftc = pd.DataFrame({
        "available_time": [100, 200],
        "report_date": ["2023-12-26", "2024-01-02"],
        "am_z": [1.0, 2.0], "lev_z": [3.0, 4.0],
        "dealer_z": [5.0, 6.0], "spread_z": [-2.0, -2.0],
    })

    attached = audit.attach(features, cftc)

    assert attached.entry_time.tolist() == [100, 199, 200]
    assert attached.am_z.tolist() == [1.0, 1.0, 2.0]


def test_directional_rule_treats_long_and_short_symmetrically():
    entries = pd.DataFrame({
        "direction_n": [1, -1, 1, -1],
        "am_z": [.2, -.2, -.2, .2],
        "lev_z": [0.0]*4, "dealer_z": [0.0]*4,
        "spread_z": [0.0]*4, "score": [1.0]*4,
    })

    result = audit.apply_rule(entries, {
        "name": "asset_manager_confirms", "am_min": 0.0})

    assert result.index.tolist() == [0, 1]
