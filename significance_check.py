#!/usr/bin/env python3
"""Bootstrap significance check for baseline vs candidate trade CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path


def _float(value: object, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        out = float(value)
        return out if math.isfinite(out) else default
    except (TypeError, ValueError):
        return default


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=row_key)
    return rows


def row_key(row: dict[str, str]) -> tuple:
    return (
        row.get("symbol", ""),
        row.get("direction", ""),
        row.get("entry_time", ""),
        row.get("entry_bar", ""),
        row.get("exit_time", ""),
        row.get("outcome", ""),
    )


def entry_key(row: dict[str, str]) -> tuple:
    return (
        row.get("symbol", ""),
        row.get("direction", ""),
        row.get("entry_time", ""),
        row.get("entry_bar", ""),
    )


def effective_r(row: dict[str, str], *, use_risk_mult: bool) -> float:
    risk_mult = _float(row.get("risk_mult"), 1.0) if use_risk_mult else 1.0
    return _float(row.get("net_r")) * risk_mult


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * pct
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return values[lo]
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def summarize(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "net_r": round(sum(values), 6),
        "rpt": round(sum(values) / len(values), 8) if values else 0.0,
    }


def paired_deltas(
    base_rows: list[dict[str, str]],
    cand_rows: list[dict[str, str]],
    *,
    use_risk_mult: bool,
    pair_key: str = "full",
) -> list[float]:
    key_fn = entry_key if pair_key == "entry" else row_key
    base_by_key: dict[tuple, list[dict[str, str]]] = {}
    cand_by_key: dict[tuple, list[dict[str, str]]] = {}
    for row in base_rows:
        base_by_key.setdefault(key_fn(row), []).append(row)
    for row in cand_rows:
        cand_by_key.setdefault(key_fn(row), []).append(row)

    deltas: list[float] = []
    for key in sorted(set(base_by_key) & set(cand_by_key)):
        base_group = base_by_key[key]
        cand_group = cand_by_key[key]
        for base_row, cand_row in zip(base_group, cand_group):
            deltas.append(
                effective_r(cand_row, use_risk_mult=use_risk_mult)
                - effective_r(base_row, use_risk_mult=use_risk_mult)
            )
    return deltas


def resolve_pair_key(value: str) -> str:
    value = (value or "auto").strip().lower()
    if value not in {"auto", "full", "entry"}:
        raise ValueError("--pair-key must be auto, full, or entry")
    return value


def bootstrap_paired(deltas: list[float], *, runs: int, seed: int) -> dict[str, float | int]:
    rng = random.Random(seed)
    n = len(deltas)
    samples: list[float] = []
    for _ in range(runs):
        samples.append(sum(deltas[rng.randrange(n)] for _ in range(n)))
    return {
        "runs": runs,
        "p_gt_zero": round(sum(1 for value in samples if value > 0) / runs, 6),
        "p05_delta_net_r": round(percentile(samples, 0.05), 6),
        "p50_delta_net_r": round(percentile(samples, 0.50), 6),
        "p95_delta_net_r": round(percentile(samples, 0.95), 6),
    }


def bootstrap_unpaired(
    base_values: list[float],
    cand_values: list[float],
    *,
    runs: int,
    seed: int,
) -> dict[str, float | int]:
    rng = random.Random(seed)
    base_n = len(base_values)
    cand_n = len(cand_values)
    net_samples: list[float] = []
    rpt_samples: list[float] = []
    for _ in range(runs):
        base_sample = [base_values[rng.randrange(base_n)] for _ in range(base_n)]
        cand_sample = [cand_values[rng.randrange(cand_n)] for _ in range(cand_n)]
        net_samples.append(sum(cand_sample) - sum(base_sample))
        rpt_samples.append((sum(cand_sample) / cand_n) - (sum(base_sample) / base_n))
    return {
        "runs": runs,
        "p_gt_zero_net": round(sum(1 for value in net_samples if value > 0) / runs, 6),
        "p_gt_zero_rpt": round(sum(1 for value in rpt_samples if value > 0) / runs, 6),
        "p05_delta_net_r": round(percentile(net_samples, 0.05), 6),
        "p50_delta_net_r": round(percentile(net_samples, 0.50), 6),
        "p95_delta_net_r": round(percentile(net_samples, 0.95), 6),
        "p05_delta_rpt": round(percentile(rpt_samples, 0.05), 8),
        "p50_delta_rpt": round(percentile(rpt_samples, 0.50), 8),
        "p95_delta_rpt": round(percentile(rpt_samples, 0.95), 8),
    }


def evaluate(
    baseline_csv: Path,
    candidate_csv: Path,
    *,
    use_risk_mult: bool,
    runs: int,
    seed: int,
    pair_key: str = "auto",
) -> dict[str, object]:
    pair_key = resolve_pair_key(pair_key)
    base_rows = load_rows(baseline_csv)
    cand_rows = load_rows(candidate_csv)
    base_values = [effective_r(row, use_risk_mult=use_risk_mult) for row in base_rows]
    cand_values = [effective_r(row, use_risk_mult=use_risk_mult) for row in cand_rows]
    full_deltas = paired_deltas(base_rows, cand_rows, use_risk_mult=use_risk_mult, pair_key="full")
    entry_deltas = paired_deltas(base_rows, cand_rows, use_risk_mult=use_risk_mult, pair_key="entry")
    full_identity = len(full_deltas) == len(base_rows) == len(cand_rows)
    entry_identity = len(entry_deltas) == len(base_rows) == len(cand_rows)
    if pair_key == "entry":
        deltas = entry_deltas
        mode = "paired_entry" if entry_identity else "unpaired"
    elif pair_key == "full":
        deltas = full_deltas
        mode = "paired" if full_identity else "unpaired"
    elif full_identity:
        deltas = full_deltas
        mode = "paired"
    elif entry_identity:
        deltas = entry_deltas
        mode = "paired_entry"
    else:
        deltas = full_deltas
        mode = "unpaired"
    result: dict[str, object] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "baseline_csv": str(baseline_csv),
        "candidate_csv": str(candidate_csv),
        "use_risk_mult": use_risk_mult,
        "pair_key": pair_key,
        "mode": mode,
        "paired_count": len(deltas),
        "full_paired_count": len(full_deltas),
        "entry_paired_count": len(entry_deltas),
        "baseline": summarize(base_values),
        "candidate": summarize(cand_values),
        "observed_delta_net_r": round(sum(cand_values) - sum(base_values), 6),
        "observed_delta_rpt": round(
            (sum(cand_values) / len(cand_values) if cand_values else 0.0)
            - (sum(base_values) / len(base_values) if base_values else 0.0),
            8,
        ),
    }
    if mode in {"paired", "paired_entry"} and deltas:
        result["bootstrap"] = bootstrap_paired(deltas, runs=runs, seed=seed)
    elif base_values and cand_values:
        result["bootstrap"] = bootstrap_unpaired(base_values, cand_values, runs=runs, seed=seed)
    else:
        result["bootstrap"] = {"runs": 0, "error": "missing_values"}
    return result


def write_markdown(path: Path, result: dict[str, object]) -> None:
    boot = result.get("bootstrap", {}) if isinstance(result.get("bootstrap"), dict) else {}
    lines = [
        "# Significance Check",
        "",
        f"Generated: {result.get('generated_utc', '')}",
        f"Mode: `{result.get('mode', '')}`",
        f"Pair key: `{result.get('pair_key', '')}`",
        f"Risk mult: `{'on' if result.get('use_risk_mult') else 'off'}`",
        f"Paired rows: `{result.get('paired_count', 0)}`",
        f"Full paired rows: `{result.get('full_paired_count', 0)}`",
        f"Entry paired rows: `{result.get('entry_paired_count', 0)}`",
        "",
        "## Observed",
        "",
        f"- baseline net: `{(result.get('baseline') or {}).get('net_r', 0)}`",
        f"- candidate net: `{(result.get('candidate') or {}).get('net_r', 0)}`",
        f"- delta net: `{result.get('observed_delta_net_r', 0)}`",
        f"- delta R/tr: `{result.get('observed_delta_rpt', 0)}`",
        "",
        "## Bootstrap",
        "",
    ]
    for key, value in boot.items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    lines.extend(
        [
            "## Rule",
            "",
            "Treat weak improvements as suspicious when bootstrap lower-tail delta is near",
            "or below zero. For risk-only overlays, full paired mode is expected.",
            "For exit-policy experiments, entry-paired mode is expected because",
            "the same entries can intentionally produce different exits/outcomes.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Bootstrap significance check for two trade CSVs.")
    parser.add_argument("baseline_csv", type=Path)
    parser.add_argument("candidate_csv", type=Path)
    parser.add_argument("--use-risk-mult", action="store_true")
    parser.add_argument("--runs", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pair-key", choices=["auto", "full", "entry"], default="auto")
    parser.add_argument("--out", type=Path, default=Path("reports/significance_check.md"))
    parser.add_argument("--json-out", type=Path, default=Path("reports/significance_check.json"))
    args = parser.parse_args()

    result = evaluate(
        args.baseline_csv,
        args.candidate_csv,
        use_risk_mult=args.use_risk_mult,
        runs=args.runs,
        seed=args.seed,
        pair_key=args.pair_key,
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    write_markdown(args.out, result)
    print(args.out.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
