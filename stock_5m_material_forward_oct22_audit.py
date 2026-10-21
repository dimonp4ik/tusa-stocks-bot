"""Oct-22 append-only check of the frozen stock 5m material router."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import stock_5m_material_forward_oct13_audit as previous


ROOT = Path("reports/audit_2026_09_08")
PREVIOUS = ROOT / "stock_xperp_5m_oct15_forward" / "manifest.json"
CURRENT = ROOT / "stock_xperp_5m_oct21_forward" / "manifest.json"
OUT = ROOT / "stock_5m_material_forward_oct22.json"
ROWS = ROOT / "stock_5m_material_forward_oct22.csv"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    previous.PREVIOUS = PREVIOUS
    previous.CURRENT = CURRENT
    previous.OUT = OUT
    previous.ROWS = ROWS
    previous.main()
    report = json.loads(OUT.read_text(encoding="utf-8"))
    report["protocol"]["append_snapshot"] = "2026-10-22 closed public data"
    report["sources"][str(Path(__file__).resolve())] = digest(
        Path(__file__).resolve())
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "append_integrity":
            report["integrity"]["previous_snapshot_append_parity"],
        "genuinely_post_material_freeze":
            report["genuinely_post_material_freeze"],
        "readiness": report["readiness"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
