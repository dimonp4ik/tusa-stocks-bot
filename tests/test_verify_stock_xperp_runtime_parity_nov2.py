import json
from pathlib import Path


REPORT = (Path(__file__).parents[1] / "reports" / "audit_2026_09_08"
          / "verify_stock_xperp_runtime_parity_nov2.json")


def test_runtime_matches_independent_november_replay():
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    checks = report["checks"]
    assert report["status"] == "RUNTIME_REPLAY_PARITY_PASSED"
    assert checks["passes"] is True
    assert checks["timestamps_checked"] >= 150
    assert checks["expected_candidates"] >= 100
    assert checks["runtime_candidates"] == checks["expected_candidates"]
    assert checks["matched_feature_rows"] == checks["expected_candidates"]
    assert checks["missing_candidates"] == []
    assert checks["unexpected_candidates"] == []
    assert checks["rank_mismatches"] == []
    assert checks["feature_mismatches"] == []
    assert checks["target_mismatches"] == []
