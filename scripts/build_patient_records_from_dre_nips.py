from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
import sys
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biodynformer.source_adapters.common import PatientRecord, SeizureRecord, parse_bool_outcome


FORBIDDEN_RECORD_KEYS = {
    "interictal_sources",
    "interictal_source_paths",
    "interictal_source_type",
    "remote_interictal_buffer",
    "has_interictal",
    "interictal_missing_mask",
    "n_interictal_runs_subject",
}
LABEL_SEMANTICS = "0=EZ, 1=NEZ, inherited from dre-nips reader"

DEFAULT_METADATA_FILENAMES = {
    "lzu_ez": "label-seeg_with_engel.xlsx",
    "lzu_times": "SEEG\u6570\u636e\u5206\u6790\u65f6\u95f4\u6807\u7b7e.xlsx",
    "lzu_quality": "slope_quality_report_en1\u4fee\u6539.xlsx",
    "lzu_quality_failure": "slope_quality_report_en2-4.xlsx",
    "hup_quality": "slope_quality_report_HUP_S.xlsx",
    "hup_quality_failure": "slope_quality_report_HUP_F.xlsx",
    "multicenter_quality": "slope_quality_report_ds3029_S.xlsx",
    "multicenter_quality_failure": "slope_quality_report_ds003929_F.xlsx",
    "multicenter_participants": "participants-muticenter.tsv",
    "pediatric_metadata": "pediatric_ez_channels_final.xlsx",
    "pediatric_quality": "\u513f\u79d1\u6570\u636e\u5206\u7c7b.xlsx",
}


def parse_centers(value: str | Sequence[str]) -> list[str]:
    if isinstance(value, str):
        parts = re.split(r"[,;\s]+", value)
    else:
        parts = [str(item) for item in value]
    centers = [part.strip().lower() for part in parts if part and part.strip()]
    if "all" in centers:
        return ["lzu", "hup", "multicenter", "pediatric"]
    valid = {"lzu", "hup", "multicenter", "pediatric"}
    unknown = sorted(set(centers) - valid)
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown centers: {unknown}; expected {sorted(valid)}")
    return centers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build preictal-only BioDynFormer patient_records.pkl from local migrated dre-nips ictal readers."
    )
    parser.add_argument("--output-pkl", type=Path, required=True)
    parser.add_argument("--output-summary-json", type=Path, required=True)
    parser.add_argument("--quality-audit-csv", type=Path, required=True)
    parser.add_argument("--read-audit-dir", type=Path, required=True)
    parser.add_argument("--patient-record-shard-dir", type=Path, default=None)
    parser.add_argument("--centers", type=parse_centers, default=["lzu", "hup", "multicenter", "pediatric"])
    parser.add_argument("--force-rebuild-shards", action="store_true")

    parser.add_argument("--quality-report-root", type=Path, default=None)
    parser.add_argument("--lzu-root", type=Path, default=None)
    parser.add_argument("--lzu-ez-annotations-path", type=Path, default=None)
    parser.add_argument("--lzu-seizure-times-path", type=Path, default=None)
    parser.add_argument("--hup-root", type=Path, default=None)
    parser.add_argument("--hup-participants-path", type=Path, default=None)
    parser.add_argument("--multicenter-root", type=Path, default=None)
    parser.add_argument("--multicenter-sidecar-root", type=Path, default=None)
    parser.add_argument("--multicenter-participants-path", type=Path, default=None)
    parser.add_argument("--pediatric-root", type=Path, default=None)
    parser.add_argument("--pediatric-metadata-xlsx", type=Path, default=None)

    parser.add_argument("--lzu-quality-report", type=Path, default=None)
    parser.add_argument("--lzu-quality-report-failure", type=Path, default=None)
    parser.add_argument("--hup-quality-report", type=Path, default=None)
    parser.add_argument("--hup-quality-report-failure", type=Path, default=None)
    parser.add_argument("--multicenter-quality-report", type=Path, default=None)
    parser.add_argument("--multicenter-quality-report-failure", type=Path, default=None)
    parser.add_argument("--pediatric-quality-report", type=Path, default=None)

    parser.add_argument("--success-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--reader-target-sfreq", type=float, default=512.0)
    parser.add_argument("--reader-bandpass-low", type=float, default=1.0)
    parser.add_argument("--reader-bandpass-high", type=float, default=150.0)
    parser.add_argument("--reader-line-freq", type=float, default=None)
    parser.add_argument("--reader-ez-definition", type=str, default="soz_only")
    parser.add_argument("--subject-filter", type=str, default=None)
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--debug-limit", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        shard_paths, quality_audit_rows, removed_count = build_or_reuse_center_shards(args)
        records = load_records_from_shards(shard_paths)
        write_pickle(args.output_pkl, records)
        write_csv(args.quality_audit_csv, quality_audit_rows)
        read_audits = build_read_audits(records)
        write_read_audits(args.read_audit_dir, read_audits)
        summary = summarize_patient_records(records, interictal_fields_removed_count=removed_count)
        args.output_summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        if args.strict:
            fail_on_strict_audit_errors(read_audits)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    except MemoryError as exc:
        shard_dir = patient_record_shard_dir(args)
        print(
            "MemoryError while building patient_records. Completed center shards remain in "
            f"{shard_dir}. Re-run the same command after freeing memory; valid shards will be reused "
            "unless --force-rebuild-shards is set.",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(1) from exc


def patient_record_shard_dir(args: argparse.Namespace) -> Path:
    return args.patient_record_shard_dir or (args.output_pkl.parent / "patient_record_shards")


def build_or_reuse_center_shards(args: argparse.Namespace) -> tuple[list[Path], list[dict[str, Any]], int]:
    quality_rows = read_quality_reports(args)
    quality_index = build_quality_index(quality_rows)
    shard_dir = patient_record_shard_dir(args)
    shard_paths: list[Path] = []
    quality_audit_rows: list[dict[str, Any]] = []
    removed_count = 0

    for center in args.centers:
        shard_path = shard_dir / f"{center}.pkl"
        center_records = None if args.force_rebuild_shards else load_valid_shard(shard_path)
        if center_records is None:
            print(f"{center}: building patient record shard {shard_path}", flush=True)
            patients = load_dre_center_patient_records(args, center)
            center_records, center_quality_rows, center_removed_count = convert_dre_patients(
                {center: patients},
                quality_index=quality_index,
            )
            write_pickle(shard_path, center_records)
            write_csv(shard_path.with_suffix(".quality_audit.csv"), center_quality_rows)
            write_shard_summary(shard_path, center_records, removed_count=center_removed_count)
        else:
            print(f"{center}: reusing existing patient record shard {shard_path}", flush=True)
            center_quality_rows = read_csv_rows(shard_path.with_suffix(".quality_audit.csv"))
            if not center_quality_rows:
                center_quality_rows = build_quality_audit_rows_from_records(center_records)
            center_removed_count = read_shard_removed_count(shard_path)

        shard_paths.append(shard_path)
        quality_audit_rows.extend(center_quality_rows)
        removed_count += center_removed_count

    return shard_paths, quality_audit_rows, removed_count


def load_records_from_shards(shard_paths: Sequence[Path]) -> list[PatientRecord]:
    records: list[PatientRecord] = []
    for shard_path in shard_paths:
        shard_records = load_valid_shard(shard_path)
        if shard_records is None:
            raise RuntimeError(f"Center shard is missing or invalid: {shard_path}")
        records.extend(shard_records)
    return records


def load_dre_patient_records(args: argparse.Namespace) -> dict[str, list[Any]]:
    return {center: load_dre_center_patient_records(args, center) for center in args.centers}


def load_dre_center_patient_records(args: argparse.Namespace, center: str) -> list[Any]:
    from scripts.dre_nips_readers.audit import ReadAudit
    from scripts.dre_nips_readers.hup import load_hup_patient_records
    from scripts.dre_nips_readers.lzu import load_lzu_patient_records
    from scripts.dre_nips_readers.multicenter import load_multicenter_patient_records
    from scripts.dre_nips_readers.schemas import DataInterfaceConfig

    center = str(center).strip().lower()
    if center not in {"lzu", "hup", "multicenter", "pediatric"}:
        raise ValueError(f"Unknown center: {center}")
    requested_centers = [center]
    cfg = DataInterfaceConfig(
        datasets=tuple(name for name in requested_centers if name != "pediatric"),
        lzu_root=required_path(args.lzu_root, "lzu-root", "lzu" in requested_centers),
        lzu_ez_annotations_path=required_default_path(
            args.lzu_ez_annotations_path, args.quality_report_root, "lzu_ez", "lzu-ez-annotations-path", "lzu" in requested_centers
        ),
        lzu_seizure_times_path=required_default_path(
            args.lzu_seizure_times_path, args.quality_report_root, "lzu_times", "lzu-seizure-times-path", "lzu" in requested_centers
        ),
        hup_root=required_path(args.hup_root, "hup-root", "hup" in requested_centers),
        hup_participants_path=args.hup_participants_path,
        multicenter_root=required_path(args.multicenter_root, "multicenter-root", "multicenter" in requested_centers),
        multicenter_sidecar_root=args.multicenter_sidecar_root,
        multicenter_participants_path=resolve_default_path(
            args.multicenter_participants_path, args.quality_report_root, "multicenter_participants"
        ),
        read_audit_dir=args.read_audit_dir,
        success_only=bool(args.success_only),
        subject_filter=args.subject_filter,
        target_sfreq=args.reader_target_sfreq,
        bandpass_low=args.reader_bandpass_low,
        bandpass_high=args.reader_bandpass_high,
        line_freq=args.reader_line_freq,
        ez_definition=args.reader_ez_definition,
        strict=bool(args.strict),
        debug_limit=args.debug_limit,
        feature_num_workers=1,
        write_read_audit=True,
    )

    with preictal_only_dre_patches():
        if center == "lzu":
            return load_lzu_patient_records(cfg, audit=ReadAudit())
        if center == "hup":
            return load_hup_patient_records(cfg, audit=ReadAudit())
        if center == "multicenter":
            return load_multicenter_patient_records(cfg, audit=ReadAudit())
        return load_pediatric_records_from_dre(args)


def required_path(value: Path | None, name: str, required: bool) -> Path:
    if value is not None:
        return value
    if required:
        raise ValueError(f"--{name} is required when that center is requested.")
    return Path(".")


def resolve_default_path(value: Path | None, root: Path | None, key: str) -> Path | None:
    if value is not None:
        return value
    if root is None:
        return None
    filename = DEFAULT_METADATA_FILENAMES[key]
    return root / filename


def required_default_path(
    value: Path | None,
    root: Path | None,
    key: str,
    arg_name: str,
    required: bool,
) -> Path:
    resolved = resolve_default_path(value, root, key)
    if resolved is None:
        if required:
            raise ValueError(f"--{arg_name} or --quality-report-root is required when that center is requested.")
        return Path(".")
    return resolved


@contextmanager
def preictal_only_dre_patches() -> Iterable[None]:
    import scripts.dre_nips_readers.bids_loader as bids_loader
    import scripts.dre_nips_readers.lzu as lzu_reader

    original_lzu_sources = lzu_reader._lzu_standalone_interictal_sources_by_subject
    original_discover = bids_loader.discover_bids_edfs_for_participants

    def no_lzu_standalone_sources(edf_index: Mapping[Any, Path]) -> dict[str, list[dict[str, Any]]]:
        return {}

    def ictal_only_discover(*args: Any, **kwargs: Any) -> tuple[list[Path], Any]:
        edf_files, matched_dirs = original_discover(*args, **kwargs)
        filtered = [Path(path) for path in edf_files if not is_forbidden_source_path(Path(path))]
        return filtered, matched_dirs

    lzu_reader._lzu_standalone_interictal_sources_by_subject = no_lzu_standalone_sources
    bids_loader.discover_bids_edfs_for_participants = ictal_only_discover
    try:
        yield
    finally:
        lzu_reader._lzu_standalone_interictal_sources_by_subject = original_lzu_sources
        bids_loader.discover_bids_edfs_for_participants = original_discover


def is_forbidden_source_path(path: Path) -> bool:
    text = str(path).lower()
    return any(token in text for token in ("task-interictal", "interictal"))


def load_pediatric_records_from_dre(args: argparse.Namespace, dre_root: Path | None = None) -> list[Any]:
    from scripts.dre_nips_readers.pediatric import load_pediatric_patient_records

    return load_pediatric_patient_records(
        root_dir=required_path(args.pediatric_root, "pediatric-root", True),
        metadata_xlsx=args.pediatric_metadata_xlsx
        or resolve_default_path(None, args.quality_report_root, "pediatric_metadata"),
        success_only=bool(args.success_only),
        subject_filter=args.subject_filter,
        target_sfreq=args.reader_target_sfreq,
        bandpass_low=args.reader_bandpass_low,
        bandpass_high=args.reader_bandpass_high,
        line_freq=args.reader_line_freq,
        debug_limit=args.debug_limit,
    )


def read_quality_reports(args: argparse.Namespace) -> list[dict[str, Any]]:
    paths_by_center = {
        "lzu": [
            args.lzu_quality_report or resolve_default_path(None, args.quality_report_root, "lzu_quality"),
            args.lzu_quality_report_failure
            or resolve_default_path(None, args.quality_report_root, "lzu_quality_failure"),
        ],
        "hup": [
            args.hup_quality_report or resolve_default_path(None, args.quality_report_root, "hup_quality"),
            args.hup_quality_report_failure
            or resolve_default_path(None, args.quality_report_root, "hup_quality_failure"),
        ],
        "multicenter": [
            args.multicenter_quality_report
            or resolve_default_path(None, args.quality_report_root, "multicenter_quality"),
            args.multicenter_quality_report_failure
            or resolve_default_path(None, args.quality_report_root, "multicenter_quality_failure"),
        ],
        "pediatric": [
            args.pediatric_quality_report or resolve_default_path(None, args.quality_report_root, "pediatric_quality")
        ],
    }
    rows: list[dict[str, Any]] = []
    for center in args.centers:
        for path in paths_by_center.get(center, []):
            if path is None or not Path(path).exists():
                continue
            rows.extend(read_quality_workbook(Path(path), center=center))
    return rows


def read_quality_workbook(path: Path, *, center: str) -> list[dict[str, Any]]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Reading quality workbooks requires pandas and openpyxl.") from exc
    workbook = pd.read_excel(path, sheet_name=None)
    rows: list[dict[str, Any]] = []
    for sheet_name, df in workbook.items():
        subject_col = first_matching_column(df.columns, ("patient", "subject", "participant", "\u60a3\u8005"))
        seizure_col = first_matching_column(df.columns, ("seizure", "run", "\u53d1\u4f5c"))
        rating_col = first_matching_column(df.columns, ("quality", "rating", "\u8d28\u91cf"))
        path_col = first_matching_column(df.columns, ("path", "\u8def\u5f84"))
        if subject_col is None or rating_col is None:
            continue
        for row_index, row in df.iterrows():
            subject_id = clean_text(row.get(subject_col))
            rating = clean_text(row.get(rating_col)).upper()
            if not subject_id or not rating:
                continue
            run_id = clean_text(row.get(seizure_col)) if seizure_col is not None else ""
            rows.append(
                {
                    "center": center,
                    "subject_id": subject_id,
                    "run_id": run_id,
                    "seizure_id": run_id,
                    "quality_rating": rating,
                    "quality_report_path": str(path),
                    "quality_report_sheet": str(sheet_name),
                    "quality_report_row": int(row_index) + 2,
                    "signal_path": clean_text(row.get(path_col)) if path_col is not None else "",
                }
            )
    return rows


def first_matching_column(columns: Iterable[Any], tokens: Sequence[str]) -> Any | None:
    for col in columns:
        text = str(col).strip().lower()
        if any(token.lower() in text for token in tokens):
            return col
    return None


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        import pandas as pd

        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def build_quality_index(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    index: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        center = normalize_text(row.get("center", "")).lower()
        subject_id = row.get("subject_id", row.get("patient_id", ""))
        run_id = row.get("run_id", "")
        seizure_id = row.get("seizure_id", "")
        for key in quality_keys(center, subject_id, run_id, seizure_id):
            index.setdefault(key, row)
    return index


def quality_keys(center: str, subject_id: Any, run_id: Any, seizure_id: Any) -> list[str]:
    subject = normalize_key(subject_id)
    run = normalize_key(run_id)
    seizure = normalize_key(seizure_id)
    keys: list[str] = []
    if run and seizure:
        keys.append("|".join([center, subject, run, seizure]))
    if run:
        keys.append("|".join([center, subject, run]))
    if seizure:
        keys.append("|".join([center, subject, seizure]))
    keys.append("|".join([center, subject]))
    return keys


def normalize_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def normalize_key(value: Any) -> str:
    text = normalize_text(value).lower()
    text = re.sub(r"^sub[-_]*", "", text)
    return re.sub(r"[^a-z0-9]+", "", text)


def convert_dre_patients(
    patients_by_center: Mapping[str, Sequence[Any]],
    *,
    quality_index: Mapping[str, Mapping[str, Any]],
) -> tuple[list[PatientRecord], list[dict[str, Any]], int]:
    records: list[PatientRecord] = []
    quality_rows: list[dict[str, Any]] = []
    removed_total = 0
    for center, patients in patients_by_center.items():
        center_key = str(center).strip().lower()
        for patient in patients:
            converted_seizures: list[SeizureRecord] = []
            subject_id = str(get_value(patient, "subject_id", "")).strip()
            outcome = extract_outcome(patient)
            for seizure in list(get_value(patient, "seizures", []) or []):
                cleaned_seizure, removed = strip_forbidden_fields(seizure)
                removed_total += removed
                run_id = str(get_value(cleaned_seizure, "run_id", get_value(cleaned_seizure, "seizure_id", "run"))).strip()
                seizure_id = str(get_value(cleaned_seizure, "seizure_id", run_id)).strip()
                signal = np.asarray(get_value(cleaned_seizure, "signal"), dtype=np.float32)
                labels = np.asarray(get_value(cleaned_seizure, "labels_ez", get_value(cleaned_seizure, "labels")), dtype=np.float32)
                channel_names = [str(name) for name in get_value(cleaned_seizure, "channel_names")]
                channel_meta, removed = clean_meta_list(get_value(cleaned_seizure, "channel_meta", []))
                removed_total += removed
                quality_row = match_quality(center_key, subject_id, run_id, seizure_id, quality_index)
                quality_rating = str(
                    quality_row.get("quality_rating")
                    or get_value(cleaned_seizure, "quality_rating", "")
                    or "UNRATED"
                ).upper()
                attach_quality_metadata(channel_meta, quality_row)
                converted = SeizureRecord(
                    subject_id=subject_id,
                    run_id=run_id,
                    seizure_id=seizure_id,
                    signal=signal,
                    sfreq=float(get_value(cleaned_seizure, "sfreq")),
                    seizure_onset_sec=float(get_value(cleaned_seizure, "seizure_onset_sec")),
                    channel_names=channel_names,
                    labels_ez=labels,
                    quality_rating=quality_rating,
                    channel_meta=channel_meta,
                )
                setattr(converted, "file_quality_report_path", quality_row.get("quality_report_path", ""))
                setattr(converted, "file_quality_report_sheet", quality_row.get("quality_report_sheet", ""))
                setattr(converted, "file_quality_report_row", quality_row.get("quality_report_row", ""))
                setattr(converted, "file_quality_match_key", quality_row.get("match_key", ""))
                setattr(converted, "file_quality_patient_id", quality_row.get("subject_id", ""))
                setattr(
                    converted,
                    "file_quality_seizure_name",
                    quality_row.get("run_id", quality_row.get("seizure_id", "")),
                )
                converted_seizures.append(converted)
                quality_rows.append(
                    {
                        "center": center_key,
                        "subject_id": subject_id,
                        "seizure_id": seizure_id,
                        "run_id": run_id,
                        "outcome_success": "" if outcome is None else int(outcome),
                        "quality_rating": quality_rating,
                        "match_status": quality_row.get("match_status", "missing"),
                        "match_key": quality_row.get("match_key", ""),
                        "reason": quality_row.get("reason", ""),
                        "quality_report_path": quality_row.get("quality_report_path", ""),
                        "quality_report_row": quality_row.get("quality_report_row", ""),
                        "signal_path": first_meta_value(channel_meta, "source_path"),
                    }
                )
            patient_meta, removed = clean_meta_list(get_value(patient, "channel_meta", []))
            removed_total += removed
            patient_labels = np.asarray(get_value(patient, "labels_ez", get_value(patient, "labels", [])), dtype=np.float32)
            canonical_channels = [str(name) for name in get_value(patient, "canonical_channels", [])]
            if converted_seizures and (not canonical_channels or patient_labels.size == 0):
                canonical_channels = list(converted_seizures[0].channel_names)
                patient_labels = converted_seizures[0].labels_ez.copy()
                patient_meta = [dict(meta) for meta in converted_seizures[0].channel_meta]
            records.append(
                PatientRecord(
                    center=center_key,
                    subject_id=subject_id,
                    outcome_success=outcome,
                    seizures=converted_seizures,
                    canonical_channels=canonical_channels,
                    labels_ez=patient_labels,
                    channel_meta=patient_meta,
                )
            )
    return records, quality_rows, removed_total


def get_value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def strip_forbidden_fields(obj: Any) -> tuple[Any, int]:
    removed = 0
    if isinstance(obj, Mapping):
        cloned: dict[str, Any] = {}
        for key, value in obj.items():
            if is_forbidden_key(key):
                removed += 1
                continue
            cloned[key] = value
        return cloned, removed
    for key in list(vars(obj)):
        if is_forbidden_key(key):
            delattr(obj, key)
            removed += 1
    return obj, removed


def clean_meta_list(values: Any) -> tuple[list[dict[str, Any]], int]:
    metas: list[dict[str, Any]] = []
    removed = 0
    for item in list(values or []):
        if not isinstance(item, Mapping):
            continue
        cleaned: dict[str, Any] = {}
        for key, value in item.items():
            if is_forbidden_key(key):
                removed += 1
                continue
            cleaned[str(key)] = json_safe_meta_value(value)
        metas.append(cleaned)
    return metas, removed


def is_forbidden_key(key: Any) -> bool:
    text = str(key)
    return text in FORBIDDEN_RECORD_KEYS


def json_safe_meta_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple, set)):
        return [json_safe_meta_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def extract_outcome(patient: Any) -> bool | None:
    for key in ("outcome_success", "success_used", "surgery_success", "outcome"):
        parsed = parse_bool_outcome(get_value(patient, key, None))
        if parsed is not None:
            return parsed
    for meta in list(get_value(patient, "channel_meta", []) or []):
        if not isinstance(meta, Mapping):
            continue
        for key in ("success_used", "surgery_success", "outcome_success", "outcome", "engel_score", "engel"):
            parsed = parse_bool_outcome(meta.get(key))
            if parsed is not None:
                return parsed
    for seizure in list(get_value(patient, "seizures", []) or []):
        for meta in list(get_value(seizure, "channel_meta", []) or []):
            if not isinstance(meta, Mapping):
                continue
            for key in ("success_used", "surgery_success", "outcome_success", "outcome", "engel_score", "engel"):
                parsed = parse_bool_outcome(meta.get(key))
                if parsed is not None:
                    return parsed
    return None


def match_quality(
    center: str,
    subject_id: str,
    run_id: str,
    seizure_id: str,
    quality_index: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    for key in quality_keys(center, subject_id, run_id, seizure_id):
        row = quality_index.get(key)
        if row is not None:
            return {**dict(row), "match_status": "matched", "match_key": key, "reason": "quality_report_match"}
    return {"match_status": "missing", "match_key": "", "reason": "missing_quality_match"}


def attach_quality_metadata(channel_meta: list[dict[str, Any]], quality_row: Mapping[str, Any]) -> None:
    for meta in channel_meta:
        meta["file_quality_report_path"] = quality_row.get("quality_report_path", "")
        meta["file_quality_report_sheet"] = quality_row.get("quality_report_sheet", "")
        meta["file_quality_report_row"] = quality_row.get("quality_report_row", "")
        meta["file_quality_match_key"] = quality_row.get("match_key", "")
        meta["file_quality_patient_id"] = quality_row.get("subject_id", "")
        meta["file_quality_seizure_name"] = quality_row.get("run_id", quality_row.get("seizure_id", ""))


def first_meta_value(channel_meta: Sequence[Mapping[str, Any]], key: str) -> str:
    for meta in channel_meta:
        value = meta.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def build_read_audits(records: Sequence[PatientRecord]) -> dict[str, list[dict[str, Any]]]:
    edf_rows: list[dict[str, Any]] = []
    channel_rows: list[dict[str, Any]] = []
    onset_rows: list[dict[str, Any]] = []
    for patient in records:
        for seizure in patient.seizures:
            signal = np.asarray(seizure.signal)
            duration = float(signal.shape[1]) / float(seizure.sfreq) if signal.ndim == 2 and seizure.sfreq > 0 else 0.0
            row_base = {
                "center": patient.center,
                "subject_id": patient.subject_id,
                "seizure_id": seizure.seizure_id,
                "run_id": seizure.run_id,
                "signal_path": first_meta_value(seizure.channel_meta, "source_path"),
            }
            finite = np.isfinite(signal)
            std = np.std(signal, axis=1) if signal.ndim == 2 and signal.shape[0] else np.asarray([])
            abs_max = np.max(np.abs(signal), axis=1) if signal.ndim == 2 and signal.shape[0] else np.asarray([])
            edf_rows.append(
                {
                    **row_base,
                    "read_status": "ok",
                    "signal_ndim": int(signal.ndim),
                    "n_channels": int(signal.shape[0]) if signal.ndim == 2 else "",
                    "n_samples": int(signal.shape[1]) if signal.ndim == 2 else "",
                    "sfreq": float(seizure.sfreq),
                    "has_nan": bool(np.isnan(signal).any()),
                    "has_inf": bool(np.isinf(signal).any()),
                    "all_zero_channel_count": int(np.sum(np.all(signal == 0, axis=1))) if signal.ndim == 2 else "",
                    "near_constant_channel_count": int(np.sum(std < 1e-8)) if std.size else 0,
                    "extreme_amplitude_channel_count": int(np.sum(abs_max > 1e6)) if abs_max.size else 0,
                    "finite_fraction": float(finite.mean()) if finite.size else 0.0,
                }
            )
            channel_rows.append(
                {
                    **row_base,
                    "channel_names_count": len(seizure.channel_names),
                    "labels_ez_count": int(seizure.labels_ez.shape[0]),
                    "signal_channels": int(signal.shape[0]) if signal.ndim == 2 else "",
                    "channel_label_mismatch": bool(
                        signal.ndim != 2
                        or len(seizure.channel_names) != signal.shape[0]
                        or int(seizure.labels_ez.shape[0]) != signal.shape[0]
                    ),
                    "bad_channel_removed": any(bool(meta.get("is_bad")) for meta in seizure.channel_meta),
                    "unmatched_label_ids": ";".join(
                        sorted(
                            {
                                str(item)
                                for meta in seizure.channel_meta
                                for item in list_value(meta.get("unmatched_label_ids"))
                            }
                        )
                    ),
                }
            )
            onset_rows.append(
                {
                    **row_base,
                    "seizure_onset_sec": float(seizure.seizure_onset_sec),
                    "signal_duration_sec": duration,
                    "onset_in_range": bool(0.0 <= float(seizure.seizure_onset_sec) <= duration),
                    "enough_preictal_120_sec": bool(float(seizure.seizure_onset_sec) >= 120.0),
                }
            )
    return {"edf_read_audit": edf_rows, "channel_label_audit": channel_rows, "onset_audit": onset_rows}


def list_value(value: Any) -> list[Any]:
    if value in (None, ""):
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def summarize_patient_records(
    records: Sequence[PatientRecord],
    *,
    interictal_fields_removed_count: int,
) -> dict[str, Any]:
    total_seizures = sum(len(patient.seizures) for patient in records)
    centers = sorted({patient.center for patient in records})
    patients_by_center = Counter(patient.center for patient in records)
    seizures_by_center = Counter()
    success_by_center = Counter()
    failure_by_center = Counter()
    unknown_by_center = Counter()
    quality_by_center: dict[str, Counter[str]] = defaultdict(Counter)
    quality_missing = Counter()
    ez_count = 0
    nez_count = 0
    patients_without_ez = 0
    onset_out_of_range = 0
    not_enough_preictal = 0
    channel_label_mismatch = 0
    edf_read_error = 0
    for patient in records:
        seizures_by_center[patient.center] += len(patient.seizures)
        if patient.outcome_success is True:
            success_by_center[patient.center] += 1
        elif patient.outcome_success is False:
            failure_by_center[patient.center] += 1
        else:
            unknown_by_center[patient.center] += 1
        labels = np.asarray(patient.labels_ez)
        ez_count += int(np.sum(labels == 0.0))
        nez_count += int(np.sum(labels == 1.0))
        if not np.any(labels == 0.0):
            patients_without_ez += 1
        for seizure in patient.seizures:
            rating = str(seizure.quality_rating or "UNRATED").upper()
            quality_by_center[patient.center][rating] += 1
            if rating == "UNRATED" or not first_meta_value(seizure.channel_meta, "file_quality_match_key"):
                quality_missing[patient.center] += 1
            signal = np.asarray(seizure.signal)
            if signal.ndim != 2 or signal.shape[0] != len(seizure.channel_names):
                channel_label_mismatch += 1
            if signal.ndim != 2 or seizure.sfreq <= 0 or not np.isfinite(signal).all():
                edf_read_error += 1
                continue
            duration = float(signal.shape[1]) / float(seizure.sfreq)
            if not (0.0 <= float(seizure.seizure_onset_sec) <= duration):
                onset_out_of_range += 1
            if float(seizure.seizure_onset_sec) < 120.0:
                not_enough_preictal += 1
            if int(seizure.labels_ez.shape[0]) != signal.shape[0]:
                channel_label_mismatch += 1
    return {
        "total_patients": len(records),
        "total_seizures": total_seizures,
        "centers": centers,
        "patients_by_center": dict(sorted(patients_by_center.items())),
        "seizures_by_center": dict(sorted(seizures_by_center.items())),
        "success_patients_by_center": dict(sorted(success_by_center.items())),
        "failure_patients_by_center": dict(sorted(failure_by_center.items())),
        "unknown_outcome_patients_by_center": dict(sorted(unknown_by_center.items())),
        "quality_rating_counts_by_center": {
            center: dict(sorted(counter.items())) for center, counter in sorted(quality_by_center.items())
        },
        "quality_missing_count_by_center": dict(sorted(quality_missing.items())),
        "label_semantics": LABEL_SEMANTICS,
        "ez_channel_count": ez_count,
        "nez_channel_count": nez_count,
        "patients_without_ez": patients_without_ez,
        "onset_out_of_range_count": onset_out_of_range,
        "not_enough_preictal_window_count": not_enough_preictal,
        "channel_label_mismatch_count": channel_label_mismatch,
        "edf_read_error_count": edf_read_error,
        "interictal_fields_removed_count": interictal_fields_removed_count,
    }


def load_valid_shard(path: Path) -> list[PatientRecord] | None:
    if not path.exists():
        return None
    try:
        with open(path, "rb") as fin:
            payload = pickle.load(fin)
    except (EOFError, OSError, pickle.UnpicklingError, AttributeError, ImportError):
        return None
    if isinstance(payload, Mapping):
        payload = payload.get("records")
    if payload is None:
        return None
    if not isinstance(payload, list):
        try:
            payload = list(payload)
        except TypeError:
            return None
    if not all(hasattr(record, "seizures") for record in payload):
        return None
    return payload


def write_pickle(path: Path, records: Sequence[PatientRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = records if isinstance(records, list) else list(records)
    tmp_path = path.with_name(f".{path.name}.tmp")
    try:
        with open(tmp_path, "wb") as fout:
            pickle.dump(payload, fout, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_path.replace(path)
    except Exception:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise


def write_shard_summary(path: Path, records: Sequence[PatientRecord], *, removed_count: int) -> None:
    summary_path = path.with_suffix(".summary.json")
    summary = summarize_patient_records(records, interictal_fields_removed_count=removed_count)
    summary["shard_path"] = str(path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def read_shard_removed_count(path: Path) -> int:
    summary_path = path.with_suffix(".summary.json")
    if not summary_path.exists():
        return 0
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    return int(payload.get("interictal_fields_removed_count") or 0)


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as fin:
            return [dict(row) for row in csv.DictReader(fin)]
    except OSError:
        return []


def build_quality_audit_rows_from_records(records: Sequence[PatientRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for patient in records:
        outcome = "" if patient.outcome_success is None else int(bool(patient.outcome_success))
        for seizure in patient.seizures:
            rows.append(
                {
                    "center": patient.center,
                    "subject_id": patient.subject_id,
                    "seizure_id": seizure.seizure_id,
                    "run_id": seizure.run_id,
                    "outcome_success": outcome,
                    "quality_rating": str(seizure.quality_rating or "UNRATED").upper(),
                    "match_status": "reconstructed_from_shard",
                    "match_key": first_meta_value(seizure.channel_meta, "file_quality_match_key"),
                    "reason": "reused_center_shard",
                    "quality_report_path": first_meta_value(seizure.channel_meta, "file_quality_report_path"),
                    "quality_report_row": first_meta_value(seizure.channel_meta, "file_quality_report_row"),
                    "signal_path": first_meta_value(seizure.channel_meta, "source_path"),
                }
            )
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    if not fieldnames:
        fieldnames = ["status"]
        rows = [{"status": "empty"}]
    with open(path, "w", encoding="utf-8-sig", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_read_audits(output_dir: Path, audits: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in audits.items():
        write_csv(output_dir / f"{name}.csv", rows)


def fail_on_strict_audit_errors(audits: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    edf_errors = [
        row
        for row in audits.get("edf_read_audit", [])
        if row.get("signal_ndim") != 2 or float(row.get("sfreq") or 0) <= 0 or row.get("has_nan") or row.get("has_inf")
    ]
    channel_errors = [row for row in audits.get("channel_label_audit", []) if row.get("channel_label_mismatch")]
    onset_errors = [
        row
        for row in audits.get("onset_audit", [])
        if not row.get("onset_in_range") or not row.get("enough_preictal_120_sec")
    ]
    if edf_errors or channel_errors or onset_errors:
        raise SystemExit(
            "Strict reader audit failed: "
            f"edf_errors={len(edf_errors)}, channel_errors={len(channel_errors)}, onset_errors={len(onset_errors)}"
        )


if __name__ == "__main__":
    main()
