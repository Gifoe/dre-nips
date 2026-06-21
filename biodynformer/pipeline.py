from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .feature_bank import build_feature_bank_from_records, load_feature_bank_index
from .orchestrator import run_all_versions
from .source_adapters import load_four_center_records
from .source_metadata import audit_source_metadata, write_audit_outputs


REQUIRED_MANIFEST_COLUMNS = (
    "subject_id",
    "run_id",
    "signal_path",
    "sfreq",
    "seizure_onset_sec",
    "channel_names",
    "labels_ez",
    "outcome",
    "quality_rating",
)


class ManifestInputError(RuntimeError):
    """Raised when per-run manifests required for tensor extraction are missing."""


class FeatureBankInputError(RuntimeError):
    """Raised when a feature bank directory is missing its index or tensors."""


def resolve_required_manifests(
    centers: Sequence[str],
    *,
    manifest_paths: Mapping[str, str | Path | None] | None,
    roots: Mapping[str, str | Path | None] | None,
) -> dict[str, Path]:
    manifests = dict(manifest_paths or {})
    root_map = dict(roots or {})
    resolved: dict[str, Path] = {}
    missing: list[str] = []

    for center in centers:
        center_key = str(center).strip().lower()
        explicit = manifests.get(center_key)
        if explicit:
            path = Path(explicit)
        else:
            root = root_map.get(center_key)
            if root:
                path = Path(root) / "manifest.csv"
            else:
                missing.append(f"{center_key}: no --{center_key}-manifest and no --{center_key}-root")
                continue
        if not path.exists():
            missing.append(f"{center_key}: {path}")
            continue
        resolved[center_key] = path

    if missing:
        required = ", ".join(REQUIRED_MANIFEST_COLUMNS)
        detail = "\n".join(f"  - {item}" for item in missing)
        raise ManifestInputError(
            "Missing per-run manifest input. The source metadata audit is only a center-level check; "
            "it does not contain the per-run signal_path, seizure_onset_sec, channel_names, or labels_ez "
            "needed to build tensors.\n"
            f"Required manifest columns: {required}\n"
            f"Missing manifests:\n{detail}"
        )
    return resolved


def validate_feature_bank(feature_bank: str | Path) -> dict[str, Any]:
    root = Path(feature_bank)
    manifest = root / "run_manifest.csv"
    if not manifest.exists():
        raise FeatureBankInputError(
            f"Feature bank is not built: {manifest} is missing. "
            "Run scripts/run_full_pipeline.py or scripts/build_feature_bank.py before scripts/run_all_versions.py."
        )
    index = load_feature_bank_index(root)
    if not index:
        raise FeatureBankInputError(f"Feature bank has an empty run_manifest.csv: {manifest}")
    missing_tensors = [row["tensor_path"] for row in index if not Path(row["tensor_path"]).exists()]
    if missing_tensors:
        sample = "\n".join(f"  - {path}" for path in missing_tensors[:10])
        raise FeatureBankInputError(f"Feature bank index points to missing tensor files:\n{sample}")
    return {
        "feature_bank": str(root),
        "num_runs": len(index),
        "num_subjects": len({row["subject_id"] for row in index}),
        "centers": sorted({row["center"] for row in index}),
    }


def run_full_pipeline(
    *,
    metadata_dir: str | Path,
    source_audit_output_dir: str | Path,
    feature_bank_output_dir: str | Path,
    runs_output_dir: str | Path,
    centers: Sequence[str],
    roots: Mapping[str, str | Path | None],
    manifest_paths: Mapping[str, str | Path | None],
    hup_participants_path: str | Path | None = None,
    rebuild_feature_bank: bool = False,
    quality_filter: bool = True,
    quality_keep_ratings: str = "GOOD,REVIEW",
    quality_drop_ratings: str = "POOR",
    quality_missing_policy: str = "drop",
    versions: Sequence[str] = ("v1", "v2", "final"),
    tasks: Sequence[str] = ("task1", "task2"),
    run_5fold: bool = True,
    run_loco: bool = True,
    resume: bool = True,
    n_splits: int = 5,
    seed: int = 42,
    learning_rate: float = 0.05,
    epochs: int = 200,
) -> dict[str, Any]:
    audit = audit_source_metadata(
        metadata_dir=metadata_dir,
        centers=centers,
        roots=dict(roots),
        hup_participants_path=hup_participants_path,
    )
    write_audit_outputs(audit, source_audit_output_dir)
    if not audit.get("can_build_feature_bank", False):
        blockers = {
            center: payload.get("blocker", "")
            for center, payload in audit.get("centers", {}).items()
            if payload.get("blocker")
        }
        raise RuntimeError(f"Source metadata audit blocks feature-bank build: {json.dumps(blockers, ensure_ascii=False)}")

    feature_bank_dir = Path(feature_bank_output_dir)
    bank_manifest = feature_bank_dir / "run_manifest.csv"
    if rebuild_feature_bank or not bank_manifest.exists():
        resolved_manifests = resolve_required_manifests(centers, manifest_paths=manifest_paths, roots=roots)
        records = load_four_center_records(centers=centers, manifest_paths=resolved_manifests, roots={})
        build_summary = build_feature_bank_from_records(
            records,
            output_dir=feature_bank_dir,
            quality_filter=quality_filter,
            keep_ratings=quality_keep_ratings,
            drop_ratings=quality_drop_ratings,
            missing_policy=quality_missing_policy,
        )
    else:
        build_summary = {"skipped": True, "reason": f"{bank_manifest} already exists"}

    bank_summary = validate_feature_bank(feature_bank_dir)
    results = run_all_versions(
        feature_bank=feature_bank_dir,
        output_dir=runs_output_dir,
        versions=versions,
        tasks=tasks,
        run_5fold=run_5fold,
        run_loco=run_loco,
        resume=resume,
        n_splits=n_splits,
        seed=seed,
        learning_rate=learning_rate,
        epochs=epochs,
    )
    return {
        "source_audit_output_dir": str(source_audit_output_dir),
        "feature_bank_output_dir": str(feature_bank_dir),
        "runs_output_dir": str(runs_output_dir),
        "feature_bank_build": build_summary,
        "feature_bank_audit": bank_summary,
        "new_runs": len(results),
    }
