"""Oct-9 append-only wrapper for the frozen stock 5m material audit."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import stock_5m_material_forward_oct8_audit as audit


ROOT = Path("reports/audit_2026_09_08")

def main() -> None:
    audit.NEW_CACHE = ROOT / "stock_xperp_5m_oct9_forward"
    audit.NEW_MANIFEST = audit.NEW_CACHE / "manifest.json"
    audit.OUT = ROOT / "stock_5m_material_forward_oct9.json"
    audit.ROWS_OUT = ROOT / "stock_5m_material_forward_oct9.csv"
    audit.main()
    report = json.loads(audit.OUT.read_text(encoding="utf-8"))
    report.setdefault("sources", {})[str(Path(__file__))] = hashlib.sha256(
        Path(__file__).read_bytes()).hexdigest()
    audit.OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
