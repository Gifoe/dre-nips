from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from .feature_bank import load_feature_bank_index
from .splits import Split, build_five_fold_splits, build_leave_one_center_out_splits
from .ssl import run_preictal_ssl
from .train_task1 import train_and_evaluate_task1
from .train_task2 import train_and_evaluate_task2


def should_skip_run(run_dir: str | Path, *, resume: bool) -> bool:
    return bool(resume) and (Path(run_dir) / "done.json").exists()


def write_run_done(run_dir: str | Path, payload: dict[str, Any]) -> None:
    path = Path(run_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / "done.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _patients_from_index(index: Sequence[dict[str, Any]], task: str) -> list[dict[str, Any]]:
    seen = {}
    for row in index:
        if task == "task1" and row.get("outcome_success") is not True:
            continue
        seen[str(row["subject_id"])] = {"subject_id": str(row["subject_id"]), "center": str(row["center"]), "outcome_success": row.get("outcome_success")}
    return list(seen.values())


def build_requested_splits(index: Sequence[dict[str, Any]], *, task: str, run_5fold: bool, run_loco: bool, n_splits: int, seed: int) -> list[Split]:
    patients = _patients_from_index(index, task)
    splits: list[Split] = []
    if run_5fold:
        splits.extend(build_five_fold_splits(patients, n_splits=n_splits, seed=seed))
    if run_loco:
        splits.extend(build_leave_one_center_out_splits(patients))
    return splits


def _run_one(
    index: Sequence[dict[str, Any]],
    *,
    version: str,
    task: str,
    split: Split,
    output_dir: Path,
    resume: bool,
    learning_rate: float,
    epochs: int,
    mode: str = "full",
) -> dict[str, Any] | None:
    run_dir = output_dir / version / task / split.kind / split.name
    if should_skip_run(run_dir, resume=resume):
        return None
    train_subjects = set(split.train_subjects)
    test_subjects = set(split.test_subjects)
    if task == "task1":
        result = train_and_evaluate_task1(index, version=version, train_subjects=train_subjects, test_subjects=test_subjects, learning_rate=learning_rate, epochs=epochs)
    else:
        result = train_and_evaluate_task2(index, version=version, train_subjects=train_subjects, test_subjects=test_subjects, mode=mode, learning_rate=learning_rate, epochs=epochs)
    payload = {
        "version": version,
        "task": task,
        "split": split.__dict__,
        **result,
    }
    write_run_done(run_dir, payload)
    return payload


def run_all_versions(
    *,
    feature_bank: str | Path,
    output_dir: str | Path,
    versions: Sequence[str],
    tasks: Sequence[str],
    run_5fold: bool = True,
    run_loco: bool = True,
    resume: bool = True,
    n_splits: int = 5,
    seed: int = 42,
    learning_rate: float = 0.05,
    epochs: int = 200,
) -> list[dict[str, Any]]:
    index = load_feature_bank_index(feature_bank)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    if "final" in {str(v).lower() for v in versions}:
        ssl_dir = output / "final" / "ssl"
        if not should_skip_run(ssl_dir, resume=resume):
            ssl_result = run_preictal_ssl(index, output_dir=ssl_dir)
            write_run_done(ssl_dir, ssl_result)
    for version in versions:
        for task in tasks:
            splits = build_requested_splits(index, task=task, run_5fold=run_5fold, run_loco=run_loco, n_splits=n_splits, seed=seed)
            for split in splits:
                result = _run_one(
                    index,
                    version=str(version),
                    task=str(task),
                    split=split,
                    output_dir=output,
                    resume=resume,
                    learning_rate=learning_rate,
                    epochs=epochs,
                )
                if result is not None:
                    results.append(result)
        if str(version).lower() == "final":
            for mode in ["label_only", "biomarker_only", "metadata_only"]:
                task = "task2"
                splits = build_requested_splits(index, task=task, run_5fold=run_5fold, run_loco=run_loco, n_splits=n_splits, seed=seed)
                for split in splits:
                    result = _run_one(
                        index,
                        version="final",
                        task=f"task2_{mode}",
                        split=split,
                        output_dir=output,
                        resume=resume,
                        learning_rate=learning_rate,
                        epochs=epochs,
                        mode=mode,
                    )
                    if result is not None:
                        results.append(result)
    aggregate_results(output)
    return results


def _flatten_metric_rows(output_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for done in output_dir.rglob("done.json"):
        payload = json.loads(done.read_text(encoding="utf-8"))
        if "metrics" not in payload:
            continue
        split = payload.get("split", {})
        row = {
            "version": payload.get("version"),
            "task": payload.get("task"),
            "split_kind": split.get("kind"),
            "split_name": split.get("name"),
            "held_out_center": split.get("held_out_center"),
            "num_train_examples": payload.get("num_train_examples"),
            "num_test_examples": payload.get("num_test_examples"),
        }
        row.update(payload["metrics"])
        rows.append(row)
    return rows


def aggregate_results(output_dir: str | Path) -> Path:
    output = Path(output_dir)
    rows = _flatten_metric_rows(output)
    path = output / "aggregate_results.csv"
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8-sig", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path
