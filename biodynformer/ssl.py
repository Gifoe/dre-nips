from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def run_preictal_ssl(index: Sequence[dict[str, Any]], *, output_dir: str | Path, mask_fraction: float = 0.15) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    means = []
    losses = []
    for row in index:
        arr = np.load(row["tensor_path"], allow_pickle=True)
        x = np.asarray(arr["node_features"], dtype=np.float32)
        mask = np.asarray(arr["window_mask"], dtype=bool)
        if not mask.any():
            continue
        valid = x[mask]
        mean = valid.mean(axis=(0, 1))
        means.append(mean)
        rng = np.random.default_rng(abs(hash(row["subject_id"] + row["run_id"])) % (2**32))
        sample_mask = rng.random(valid.shape) < float(mask_fraction)
        if sample_mask.any():
            recon = np.broadcast_to(mean, valid.shape)
            losses.append(float(np.mean((valid[sample_mask] - recon[sample_mask]) ** 2)))
    global_mean = np.mean(np.stack(means), axis=0).tolist() if means else []
    result = {
        "ssl_objective": "masked_preictal_feature_reconstruction",
        "num_runs": len(means),
        "mask_fraction": float(mask_fraction),
        "mean_reconstruction_loss": float(np.mean(losses)) if losses else float("nan"),
        "node_feature_mean": global_mean,
    }
    (output / "ssl_checkpoint.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
