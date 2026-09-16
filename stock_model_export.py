"""Export the frozen sklearn stock model to dependency-free JSON."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def export_bundle(bundle: dict, *, source_hashes: dict) -> dict:
    pipeline = bundle["model"]
    transform = pipeline.named_steps["transform"]
    classifier = pipeline.named_steps["classifier"]
    numeric = list(bundle["numeric"])
    categorical = list(bundle["categorical"])
    numeric_pipe = transform.named_transformers_["numeric"]
    categorical_pipe = transform.named_transformers_["categorical"]
    imputer = numeric_pipe.named_steps["impute"]
    scaler = numeric_pipe.named_steps["scale"]
    categorical_imputer = categorical_pipe.named_steps["impute"]
    encoder = categorical_pipe.named_steps["encode"]
    trees = []
    for iteration in classifier._predictors:
        if len(iteration) != 1:
            raise ValueError("Only binary HistGradientBoosting models are supported")
        nodes = iteration[0].nodes
        if any(bool(node["is_categorical"]) for node in nodes):
            raise ValueError("Categorical tree nodes are not supported after one-hot encoding")
        trees.append([{
            "value": float(node["value"]),
            "feature": int(node["feature_idx"]),
            "threshold": float(node["num_threshold"]),
            "missing_left": bool(node["missing_go_to_left"]),
            "left": int(node["left"]),
            "right": int(node["right"]),
            "leaf": bool(node["is_leaf"]),
        } for node in nodes])
    return {
        "format": "stock-session-hgb-v1",
        "status": "PAPER_ONLY",
        "thresholds": {"broad": .82, "premium": float(bundle["threshold"])},
        "numeric": numeric,
        "numeric_medians": [float(value) for value in imputer.statistics_],
        "numeric_means": [float(value) for value in scaler.mean_],
        "numeric_scales": [float(value) for value in scaler.scale_],
        "categorical": categorical,
        "categorical_fill": [str(value) for value in categorical_imputer.statistics_],
        "categorical_values": [[str(value) for value in values]
                               for values in encoder.categories_],
        "baseline": float(classifier._baseline_prediction.ravel()[0]),
        "trees": trees,
        "source_hashes": source_hashes,
    }


def predict_one(model: dict, row: dict) -> float:
    values = []
    for index, field in enumerate(model["numeric"]):
        try:
            value = float(row.get(field))
        except (TypeError, ValueError):
            value = model["numeric_medians"][index]
        if not math.isfinite(value):
            value = model["numeric_medians"][index]
        values.append((value - model["numeric_means"][index])
                      / model["numeric_scales"][index])
    for index, field in enumerate(model["categorical"]):
        raw = row.get(field)
        value = model["categorical_fill"][index] if raw is None else str(raw)
        values.extend(1.0 if value == category else 0.0
                      for category in model["categorical_values"][index])
    score = model["baseline"]
    for tree in model["trees"]:
        position = 0
        while not tree[position]["leaf"]:
            node = tree[position]
            value = values[node["feature"]]
            go_left = node["missing_left"] if not math.isfinite(value) \
                else value <= node["threshold"]
            position = node["left"] if go_left else node["right"]
        score += tree[position]["value"]
    return 1 / (1 + math.exp(-score))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("joblib", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--sample-csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    joblib_raw = args.joblib.read_bytes()
    report_raw = args.report.read_bytes()
    sample_raw_hash = hashlib.sha256(args.sample_csv.read_bytes()).hexdigest()
    bundle = joblib.load(args.joblib)
    model = export_bundle(bundle, source_hashes={
        "joblib": hashlib.sha256(joblib_raw).hexdigest(),
        "walk_forward_report": hashlib.sha256(report_raw).hexdigest(),
        "training_dataset": sample_raw_hash,
    })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(model, separators=(",", ":")), encoding="utf-8")

    columns = list(bundle["numeric"] + bundle["categorical"])
    sample = pd.read_csv(args.sample_csv, usecols=columns).sample(
        n=2000, random_state=20260915)
    expected = bundle["model"].predict_proba(sample[columns])[:, 1]
    actual = np.asarray([predict_one(model, row) for row in sample.to_dict("records")])
    difference = np.abs(expected - actual)
    threshold = float(bundle["threshold"])
    mismatches = int(np.sum((expected >= threshold) != (actual >= threshold)))
    if float(difference.max()) > 1e-12 or mismatches:
        raise ValueError(f"export mismatch max={difference.max()} decisions={mismatches}")
    verification = {
        "status": "EXACT",
        "samples": len(sample),
        "maximum_probability_difference": float(difference.max()),
        "premium_decision_mismatches": mismatches,
        "json": str(args.out),
        "json_sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
        "json_bytes": args.out.stat().st_size,
    }
    verify_path = args.out.with_suffix(".verification.json")
    verify_path.write_text(json.dumps(verification, indent=2), encoding="utf-8")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
