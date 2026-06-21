from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_physics_window_cache import build_cache_payload


def _toy_cache() -> dict[str, Any]:
    rng = np.random.default_rng(123)
    run_records: list[dict[str, Any]] = []
    patient_index: dict[str, dict[str, Any]] = {}
    outcome_index: dict[str, dict[str, Any]] = {}
    centers = ["lzu", "hup", "multicenter", "pediatric"]
    for idx in range(4):
        sid = f"toy:{idx}"
        channels = [f"A{ch}" for ch in range(4)]
        labels_nez = np.asarray([0.0, 1.0, 1.0, 0.0], dtype=np.float32)
        labels_ez = 1.0 - labels_nez
        patient_index[sid] = {
            "canonical_channels": channels,
            "labels": labels_nez,
            "labels_nez": labels_nez,
            "labels_ez": labels_ez,
            "label_mask": np.ones((4,), dtype=bool),
            "center": centers[idx],
        }
        outcome_index[sid] = {
            "Engel": "I" if idx % 2 == 0 else "II",
            "success_failure": 1 if idx % 2 == 0 else 0,
            "center": centers[idx],
        }
        adjacency = rng.random((5, 4, 4), dtype=np.float32)
        for t in range(adjacency.shape[0]):
            np.fill_diagonal(adjacency[t], 0.0)
        run_records.append(
            {
                "subject_id": sid,
                "run_id": f"run-{idx}",
                "seizure_id": f"sz-{idx}",
                "center": centers[idx],
                "channel_names_norm": channels,
                "sample": {
                    "window_features": rng.normal(size=(5, 4, 9)).astype(np.float32),
                    "physics_node_features": rng.normal(size=(5, 4, 6)).astype(np.float32),
                    "tfccm_adjacency": adjacency,
                    "tfccm_delay": rng.random((5, 4, 4), dtype=np.float32) * adjacency,
                    "causal_node_features": rng.normal(size=(5, 4, 7)).astype(np.float32),
                    "topology_graph_features": rng.normal(size=(5, 8)).astype(np.float32),
                    "window_relative_centers_sec": np.arange(5, dtype=np.float32) - 2.0,
                    "window_mask": np.ones((5,), dtype=bool),
                    "channel_names": channels,
                },
            }
        )
    return {
        "run_records": run_records,
        "patient_index": patient_index,
        "outcome_index": outcome_index,
        "cache_meta": {
            "cache_name": "toy_all_window_cache_physics_v1.pkl",
            "label_semantics": "internal_positive_class=NEZ; labels_nez: 1=NEZ, 0=EZ",
            "causal_graph_algorithm": "tfccm_lite_nearest_neighbor_cross_mapping",
            "physics_feature_level": "physics_proxy_v1",
            "feature_names_b0": [f"b0_{idx}" for idx in range(9)],
            "feature_names_physics": [f"phys_{idx}" for idx in range(6)],
            "feature_names_causal_node": [f"causal_{idx}" for idx in range(7)],
            "feature_names_topology": [f"topo_{idx}" for idx in range(8)],
        },
    }


def _patient_records_for_builder() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    centers = ["lzu", "hup", "multicenter", "pediatric"]
    for idx, center in enumerate(centers):
        samples = 512
        t = np.linspace(0, 4.0, samples, dtype=np.float32)
        signal = np.vstack(
            [
                np.sin(2 * np.pi * (6 + idx) * t),
                np.cos(2 * np.pi * (10 + idx) * t),
                np.sin(2 * np.pi * (20 + idx) * t) + 0.1 * np.sin(2 * np.pi * 100 * t),
                np.cos(2 * np.pi * (35 + idx) * t),
            ]
        ).astype(np.float32)
        records.append(
            {
                "center": center,
                "subject_id": f"strict-{idx}",
                "outcome_success": idx % 2 == 0,
                "Engel": "I" if idx % 2 == 0 else "II",
                "canonical_channels": ["A0", "A1", "A2", "A3"],
                "labels": np.asarray([0.0, 1.0, 1.0, 0.0], dtype=np.float32),
                "seizures": [
                    {
                        "run_id": f"run-{idx}",
                        "seizure_id": f"sz-{idx}",
                        "signal": signal,
                        "sfreq": 128.0,
                        "seizure_onset_sec": 2.0,
                        "channel_names": ["A0", "A1", "A2", "A3"],
                        "labels": np.asarray([0.0, 1.0, 1.0, 0.0], dtype=np.float32),
                    }
                ],
            }
        )
    return records


def _run(cmd: list[str], *, cwd: Path, expect_success: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if expect_success and result.returncode != 0:
        raise RuntimeError(f"Command failed with exit {result.returncode}:\n{' '.join(cmd)}\n{result.stdout}")
    if not expect_success and result.returncode == 0:
        raise RuntimeError(f"Command unexpectedly succeeded:\n{' '.join(cmd)}\n{result.stdout}")
    return result


def _assert_file(path: Path) -> None:
    if not path.exists():
        raise AssertionError(f"Missing expected file: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small PGC-SEEG v1.1 smoke test without pytest.")
    parser.add_argument("--work-dir", type=Path, default=None)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="pgc-smoke-", dir=str(args.work_dir) if args.work_dir else None) as tmp:
        root = Path(tmp)
        cache_path = root / "toy_cache.pkl"
        with open(cache_path, "wb") as fout:
            pickle.dump(_toy_cache(), fout, protocol=pickle.HIGHEST_PROTOCOL)
        strict_cache = build_cache_payload(
            _patient_records_for_builder(),
            source_patient_records_pkl="synthetic-strict.pkl",
            window_length_sec=0.5,
            window_step_sec=0.5,
            pre_onset_sec=1.0,
            post_onset_sec=1.0,
            physics_mode="strict",
        )
        strict_cache_path = root / "toy_cache_strict.pkl"
        with open(strict_cache_path, "wb") as fout:
            pickle.dump(strict_cache, fout, protocol=pickle.HIGHEST_PROTOCOL)

        task1_base = root / "task1_base"
        task1_full = root / "task1_full"
        task2_base = root / "task2_base"
        task2_full = root / "task2_full"
        task1_strict = root / "task1_strict"
        task2_strict = root / "task2_strict"

        for experiment, output in [("T1_B0_BASELINE", task1_base), ("T1_FULL_PGC", task1_full)]:
            _run(
                [
                    sys.executable,
                    "run_task1_pgc_ez.py",
                    "--window_cache_path",
                    str(cache_path),
                    "--experiment_name",
                    experiment,
                    "--output_dir",
                    str(output),
                    "--split_strategy",
                    "5fold",
                    "--n_splits",
                    "2",
                    "--epochs",
                    "1",
                    "--batch_size",
                    "2",
                    "--model_dim",
                    "8",
                    "--patience",
                    "1",
                ],
                cwd=repo,
            )
            _assert_file(output / "fold_0" / "best_task1_backbone.pt")
            _assert_file(output / "fold_0" / "split_subjects.json")
            _assert_file(output / "splits.json")

        _run(
            [
                sys.executable,
                "run_task2_pgc_outcome.py",
                "--window_cache_path",
                str(cache_path),
                "--task1_checkpoint",
                str(task1_full / "best_task1_backbone.pt"),
                "--experiment_name",
                "T2_B0_GLOBAL",
                "--output_dir",
                str(root / "task2_bad_single"),
                "--split_strategy",
                "5fold",
                "--n_splits",
                "2",
                "--epochs",
                "1",
                "--batch_size",
                "2",
                "--model_dim",
                "8",
            ],
            cwd=repo,
            expect_success=False,
        )

        for experiment, task1_dir, output in [
            ("T2_B0_GLOBAL", task1_base, task2_base),
            ("T2_FULL_ATTENTION_TOPOLOGY", task1_full, task2_full),
        ]:
            _run(
                [
                    sys.executable,
                    "run_task2_pgc_outcome.py",
                    "--window_cache_path",
                    str(cache_path),
                    "--task1_checkpoint_dir",
                    str(task1_dir),
                    "--experiment_name",
                    experiment,
                    "--output_dir",
                    str(output),
                    "--split_strategy",
                    "5fold",
                    "--n_splits",
                    "2",
                    "--epochs",
                    "1",
                    "--batch_size",
                    "2",
                    "--model_dim",
                    "8",
                    "--patience",
                    "1",
                    "--freeze_backbone",
                    "true",
                ],
                cwd=repo,
            )
            _assert_file(output / "fold_0" / "split_subjects.json")
            _assert_file(output / "summary_metrics.json")
            _assert_file(output / "best_checkpoint.pt")

        _run(
            [
                sys.executable,
                "run_task1_pgc_ez.py",
                "--window_cache_path",
                str(strict_cache_path),
                "--experiment_name",
                "T1_FULL_PGC",
                "--output_dir",
                str(task1_strict),
                "--split_strategy",
                "5fold",
                "--n_splits",
                "2",
                "--epochs",
                "1",
                "--batch_size",
                "2",
                "--model_dim",
                "8",
                "--patience",
                "1",
            ],
            cwd=repo,
        )
        _run(
            [
                sys.executable,
                "run_task2_pgc_outcome.py",
                "--window_cache_path",
                str(strict_cache_path),
                "--task1_checkpoint_dir",
                str(task1_strict),
                "--experiment_name",
                "T2_FULL_ATTENTION_TOPOLOGY",
                "--output_dir",
                str(task2_strict),
                "--split_strategy",
                "5fold",
                "--n_splits",
                "2",
                "--epochs",
                "1",
                "--batch_size",
                "2",
                "--model_dim",
                "8",
                "--patience",
                "1",
                "--freeze_backbone",
                "true",
            ],
            cwd=repo,
        )
        _assert_file(task2_strict / "summary_metrics.json")

        print(json.dumps({"status": "ok", "work_dir": str(root)}, indent=2))


if __name__ == "__main__":
    main()
