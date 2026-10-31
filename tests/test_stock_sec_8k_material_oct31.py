import hashlib
import json
from pathlib import Path


ROOT = Path("reports/audit_2026_09_08")


def load(name: str) -> dict:
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def test_official_sec_archive_is_complete_and_excludes_earnings_only_events():
    value = load("stock_sec_8k_history_oct31.json")
    assert value["status"] == "IMMUTABLE_OFFICIAL_SEC_RESEARCH_INPUT"
    assert value["records"] == len(value["events"]) == 406
    assert set(value["records_by_symbol"]) == {
        "AAPL", "AMZN", "META", "MSFT", "MU", "NVDA"}
    assert all(count > 0 for count in value["records_by_symbol"].values())
    assert len(value["sources"]) > len(value["records_by_symbol"])
    identities = {(row["symbol"], row["accession"])
                  for row in value["events"]}
    assert len(identities) == len(value["events"])
    for row in value["events"]:
        assert row["non_earnings_items"]
        assert set(row["non_earnings_items"]).isdisjoint({"2.02", "9.01"})
        assert "2017-01-01" <= row["acceptance_datetime_utc"][:10] < "2026-11-01"
    for source in value["sources"].values():
        assert len(source["sha256"]) == 64


def test_sec_event_route_fails_hidden_test_and_cannot_change_runtime():
    archive_path = ROOT / "stock_sec_8k_history_oct31.json"
    value = load("stock_sec_8k_material_oct31.json")
    assert value["status"] == "REJECTED"
    assert value["ready_for_live_orders"] is False
    assert value["orders_possible"] is False
    assert value["runtime_change_allowed"] is False
    assert value["qualified_supplements_before_2024"] == 2
    assert value["test_2024"]["return_pct"] < 0
    assert value["test_2024"]["economic_gate"]["passes"] is False
    assert value["confirmation_2025"] is None
    assert value["untouched_2026"] is None
    assert value["sources"][str(archive_path)] == hashlib.sha256(
        archive_path.read_bytes()).hexdigest()
    assert all(item["first_acceptance"][:4] == "2017"
               for item in value["coverage"].values())
