from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biodynformer.feature_bank import FEATURE_SCHEMA, load_feature_bank_index


def audit_feature_bank(feature_bank: str | Path) -> dict:
    root = Path(feature_bank)
    index = load_feature_bank_index(root)
    centers = Counter(row["center"] for row in index)
    subjects = {row["subject_id"] for row in index}
    valid_windows = Counter()
    errors = []
    for row in index:
        arr = np.load(row["tensor_path"], allow_pickle=True)
        for key in [
            "node_features",
            "hfo_features",
            "quality_features",
            "causal_edge",
            "sync_edge",
            "structural_edge",
            "coverage_features",
            "window_mask",
            "channel_mask",
            "labels_ez",
        ]:
            if key not in arr.files:
                errors.append(f"{row['tensor_path']} missing {key}")
        if "window_mask" in arr.files:
            valid_windows[int(np.asarray(arr["window_mask"], dtype=bool).sum())] += 1
    summary = {
        "feature_bank": str(root),
        "num_runs": len(index),
        "num_subjects": len(subjects),
        "centers": dict(centers),
        "valid_window_count_distribution": dict(valid_windows),
        "feature_schema": FEATURE_SCHEMA,
        "errors": errors,
    }
    (root / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a BioDynFormer preictal feature bank.")
    parser.add_argument("--feature-bank", "--cache", dest="feature_bank", type=Path, required=True)
    args = parser.parse_args()
    summary = audit_feature_bank(args.feature_bank)
    print(json.dumps(summary, indent=2), flush=True)
    if summary["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
