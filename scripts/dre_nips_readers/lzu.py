from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, time
from pathlib import Path
import math
import re
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .audit import ReadAudit
from .edf import close_raw, crop_raw_to_preictal_context, finalize_raw_data, read_raw_edf
from .schemas import (
    DataInterfaceConfig,
    PatientRecord,
    SeizureRecord,
    build_patient_records,
    clean_text,
    is_successful_surgery_value,
    make_unique,
    natural_channel_sort_key,
    normalize_channel_name,
    parse_contact_topology,
    resolve_cpu_workers,
    safe_float,
    subject_filter_set,
    subject_key,
)

try:
    from ..logging_utils import log
except Exception:
    def log(message: str) -> None:
        print(message)


LZU_NAME_COLUMNS = (
    "\u59d3\u540d",
    "\u60a3\u8005",
    "\u75c5\u4eba",
    "name",
    "subject",
    "subject_id",
    "patient",
    "patient_id",
)
LZU_EZ_COLUMNS = (
    "number",
    "numbers",
    "\u901a\u9053\u7f16\u53f7",
    "ez\u7f16\u53f7",
    "ez\u901a\u9053\u7f16\u53f7",
    "ez channel id",
    "ez channel ids",
)
LZU_BAD_COLUMNS = (
    "\u9700\u8981\u5220\u9664\u7684\u901a\u9053",
    "\u5220\u9664\u901a\u9053",
    "\u574f\u901a\u9053",
    "deleted",
    "bad",
    "bad channel",
    "bad channels",
)
LZU_OUTCOME_COLUMNS = (
    "Engel\u5206\u7ea7(S/F)",
    "Engel\u5206\u7ea7",
    "engel",
    "engel_score",
    "outcome",
    "\u624b\u672f\u7ed3\u679c",
    "\u9884\u540e",
)
LZU_TIME_SUBHEADERS = (
    "\u53d1\u4f5c\u95f4\u671f\uff08\u8d77-\u6b62\uff09",
    "\u53d1\u4f5c\u524d\u671f",
    "\u53d1\u4f5c\u671f\u5f00\u59cb",
    "\u53d1\u4f5c\u7ed3\u675f",
)
LZU_NUMBER_COLUMN = "\u53d1\u4f5c\u7f16\u53f7"
LZU_ANCHOR_COLUMN = "\u8111\u7535\u56fe\u8bb0\u5f55\u5f00\u59cb\u65f6\u95f4"
LZU_FALLBACK_ONSET_COLUMN = "\u53d1\u4f5c\u524d\u671f"
LZU_ONSET_COLUMN = "\u53d1\u4f5c\u671f\u5f00\u59cb"
LZU_OFFSET_COLUMN = "\u53d1\u4f5c\u7ed3\u675f"


@dataclass
class LzuTimeInfo:
    anchor_clock_sec: float | None
    onset_value: Any
    offset_value: Any
    onset_clock_sec: float | None
    offset_clock_sec: float | None
    source_columns: list[str] = field(default_factory=list)


def load_lzu_patient_records(cfg: DataInterfaceConfig, audit: ReadAudit | None = None) -> list[PatientRecord]:
    audit = audit or ReadAudit()
    subject_filter = subject_filter_set(cfg.subject_filter, add_sub_prefix=False)
    annotations = _load_lzu_annotations(cfg.lzu_ez_annotations_path)
    if cfg.success_only and all(item.get("is_successful") is None for item in annotations.values()):
        raise ValueError(
            "LZU success_only=True but no usable Engel/outcome column was found in "
            f"{cfg.lzu_ez_annotations_path}. Use label-seeg_with_engel.xlsx or disable --success-only."
        )

    seizure_rows = _load_lzu_seizure_times(cfg.lzu_seizure_times_path)
    edf_index = _build_lzu_edf_index(cfg.lzu_root)
    log(
        f"LZU: annotations={len(annotations)}, seizure_time_rows={len(seizure_rows)}, "
        f"indexed_edfs={len(edf_index)}"
    )
    if cfg.success_only:
        n_lzu_success = sum(1 for item in annotations.values() if item.get("is_successful") is True)
        n_lzu_failed = sum(1 for item in annotations.values() if item.get("is_successful") is False)
        n_lzu_unknown = sum(1 for item in annotations.values() if item.get("is_successful") is None)
        log(
            f"LZU: success_filter_applies=True, successful_subjects={n_lzu_success}, "
            f"failed_subjects={n_lzu_failed}, unknown_outcome_subjects={n_lzu_unknown}"
        )
    else:
        log("LZU: success_filter_applies=False")
    if seizure_rows.empty:
        log("LZU: seizure time table is empty after parsing.")
    else:
        time_keys = sorted({subject_key(clean_text(value)) for value in seizure_rows["name"] if clean_text(value)})
        ann_keys = sorted(annotations.keys())
        overlap = sorted(set(time_keys) & set(ann_keys))
        log(
            f"LZU: annotation_subjects={len(ann_keys)}, time_subjects={len(time_keys)}, "
            f"subject_overlap={len(overlap)}"
        )
        log(f"LZU: annotation_keys_sample={ann_keys[:10]}")
        log(f"LZU: time_keys_sample={time_keys[:10]}")

    seizures: list[SeizureRecord] = []
    missing_annotation = 0
    missing_edf = 0
    read_errors = 0
    missing_edf_details: list[dict[str, Any]] = []
    read_error_details: list[dict[str, Any]] = []
    read_jobs: list[dict[str, Any]] = []

    for row_idx, row in seizure_rows.iterrows():
        subject_name = clean_text(row.get("name"))
        seizure_number = clean_text(row.get("number"))
        if not subject_name or not seizure_number:
            continue
        if subject_filter is not None and subject_key(subject_name) not in subject_filter and subject_name not in subject_filter:
            continue

        annotation = annotations.get(subject_key(subject_name))
        if annotation is None:
            missing_annotation += 1
            audit.add_skipped_seizure("lzu", subject_name, seizure_number, "missing_lzu_annotation", time_row_index=int(row_idx))
            continue

        if cfg.success_only:
            success_value = annotation.get("is_successful")
            if success_value is not True:
                audit.add_skipped_seizure(
                    "lzu",
                    subject_name,
                    seizure_number,
                    "non_successful_surgery",
                    time_row_index=int(row_idx),
                    lzu_outcome_raw=annotation.get("outcome_raw"),
                )
                continue

        edf_stem_candidates = _lzu_edf_stem_candidates(seizure_number)
        edf_path = _resolve_lzu_edf_path(cfg.lzu_root, subject_name, seizure_number, edf_index)
        if edf_path is None:
            missing_edf += 1
            detail = {
                "time_row_index": int(row_idx),
                "subject_id": subject_name,
                "seizure_id": seizure_number,
                "lzu_raw_seizure_number": seizure_number,
                "lzu_edf_stem_candidates": ";".join(edf_stem_candidates),
                "candidate_paths": ";".join(_lzu_candidate_edf_paths(cfg.lzu_root, subject_name, edf_stem_candidates, seizure_number)),
            }
            missing_edf_details.append(detail)
            audit.add_skipped_seizure("lzu", subject_name, seizure_number, "missing_edf", **detail)
            continue

        read_jobs.append(
            {
                "row_idx": int(row_idx),
                "row": row,
                "subject_name": subject_name,
                "seizure_number": seizure_number,
                "annotation": annotation,
                "edf_path": edf_path,
            }
        )
        if cfg.debug_limit is not None and len(read_jobs) >= int(cfg.debug_limit):
            break

    workers = resolve_cpu_workers(cfg, len(read_jobs))
    log(f"LZU: EDF read jobs={len(read_jobs)}, workers={workers}")
    if workers <= 1:
        for completed, job in enumerate(read_jobs, start=1):
            seizure, detail = _read_lzu_job(job, cfg, audit)
            if detail is not None:
                read_errors += 1
                read_error_details.append(detail)
                if cfg.strict:
                    raise RuntimeError(detail["exception_repr"])
            elif seizure is not None:
                seizures.append(seizure)
            if completed == 1 or completed % 10 == 0 or completed == len(read_jobs):
                log(f"LZU: EDF read progress={completed}/{len(read_jobs)}, loaded_seizures={len(seizures)}, read_errors={read_errors}")
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_read_lzu_job, job, cfg, audit) for job in read_jobs]
            for completed, future in enumerate(as_completed(futures), start=1):
                seizure, detail = future.result()
                if detail is not None:
                    read_errors += 1
                    read_error_details.append(detail)
                    if cfg.strict:
                        raise RuntimeError(detail["exception_repr"])
                elif seizure is not None:
                    seizures.append(seizure)
                if completed == 1 or completed % 10 == 0 or completed == len(read_jobs):
                    log(f"LZU: EDF read progress={completed}/{len(read_jobs)}, loaded_seizures={len(seizures)}, read_errors={read_errors}")

    patients = build_patient_records(seizures)
    interictal_sources_by_subject = _lzu_standalone_interictal_sources_by_subject(edf_index)
    n_interictal_sources = sum(len(sources) for sources in interictal_sources_by_subject.values())
    if n_interictal_sources:
        log(
            f"LZU: indexed_standalone_interictal_sources={n_interictal_sources}, "
            f"subjects_with_interictal={len(interictal_sources_by_subject)}"
        )
    for patient in patients:
        subject_sources = list(interictal_sources_by_subject.get(subject_key(patient.subject_id), []))
        for seizure in patient.seizures:
            if subject_sources:
                setattr(seizure, "interictal_sources", subject_sources)
            for meta in seizure.channel_meta:
                meta.setdefault("n_interictal_runs_subject", len(subject_sources))
                if subject_sources:
                    meta.setdefault(
                        "interictal_source_paths",
                        ";".join(str(source.get("edf_path", "")) for source in subject_sources),
                    )
    kept = audit.validate_and_filter("lzu", patients, strict=cfg.strict)
    log(
        f"LZU: built_patients={len(patients)}, kept_trainable_patients={len(kept)}, "
        f"seizures={len(seizures)}, missing_annotation={missing_annotation}, "
        f"missing_edf={missing_edf}, read_errors={read_errors}"
    )
    _log_and_write_lzu_failure_details(cfg, missing_edf_details, read_error_details)
    return kept


def _read_lzu_job(
    job: Mapping[str, Any],
    cfg: DataInterfaceConfig,
    audit: ReadAudit,
) -> tuple[SeizureRecord | None, dict[str, Any] | None]:
    row_idx = int(job["row_idx"])
    subject_name = str(job["subject_name"])
    seizure_number = str(job["seizure_number"])
    edf_path = Path(job["edf_path"])
    try:
        time_info = _parse_lzu_time_row(job["row"])
        seizure = _read_lzu_seizure(
            edf_path,
            subject_name,
            seizure_number,
            job["annotation"],
            time_info,
            cfg,
            audit,
            row_idx,
        )
        return seizure, None
    except Exception as exc:
        detail = {
            "time_row_index": row_idx,
            "subject_id": subject_name,
            "seizure_id": seizure_number,
            "edf_path": str(edf_path),
            "exception_type": type(exc).__name__,
            "exception_repr": repr(exc),
        }
        audit.add_skipped_seizure(
            "lzu",
            subject_name,
            seizure_number,
            repr(exc),
            edf_path=str(edf_path),
            time_row_index=row_idx,
            exception_type=type(exc).__name__,
        )
        return None, detail


def _load_lzu_annotations(path: Path) -> dict[str, dict]:
    df = pd.read_excel(path)
    name_col = _first_matching_column(df.columns, exact_names=LZU_NAME_COLUMNS, contains=())
    if name_col is None:
        name_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
    ez_col = _first_matching_column(df.columns, exact_names=LZU_EZ_COLUMNS, contains=("ez channel",))
    bad_col = _first_matching_column(df.columns, exact_names=LZU_BAD_COLUMNS, contains=("deleted", "bad", "\u5220\u9664", "\u574f\u901a\u9053"))
    outcome_col = _first_matching_column(df.columns, exact_names=LZU_OUTCOME_COLUMNS, contains=("engel", "\u624b\u672f\u7ed3\u679c", "\u9884\u540e"))
    if ez_col is None:
        raise ValueError(
            f"Cannot find LZU EZ channel-number column in {path}. "
            f"Available columns: {[str(col) for col in df.columns]}"
        )

    annotations: dict[str, dict] = {}
    for _, row in df.iterrows():
        subject_name = clean_text(row.get(name_col))
        if not subject_name:
            continue
        outcome_raw = row.get(outcome_col) if outcome_col is not None else None
        annotations[subject_key(subject_name)] = {
            "subject_name": subject_name,
            "ez_channel_ids": _parse_int_set(row.get(ez_col)),
            "bad_channel_ids": _parse_int_set(row.get(bad_col)) if bad_col is not None else set(),
            "outcome_raw": outcome_raw,
            "is_successful": is_successful_surgery_value(outcome_raw) if outcome_col is not None else None,
            "source_row": {str(col): row.get(col) for col in df.columns},
        }
    return annotations


def _first_matching_column(columns: pd.Index, exact_names: tuple[str, ...], contains: tuple[str, ...]) -> Any | None:
    normalized = {str(col).strip().lower(): col for col in columns}
    for name in exact_names:
        col = normalized.get(name.lower())
        if col is not None:
            return col
    for col in columns:
        text = str(col).strip().lower()
        if any(token.lower() in text for token in contains):
            return col
    return None


def _load_lzu_seizure_times(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    df.columns = [str(col).strip() for col in df.columns]
    if df.empty:
        return df

    first = df.iloc[0]
    rename_map: dict[Any, str] = {}
    for col in df.columns:
        value = clean_text(first.get(col))
        if value in LZU_TIME_SUBHEADERS:
            rename_map[col] = value
    if rename_map:
        df = df.rename(columns=rename_map).iloc[1:].reset_index(drop=True)

    lower_cols = {str(col).strip().lower(): col for col in df.columns}
    name_col = "\u59d3\u540d" if "\u59d3\u540d" in df.columns else lower_cols.get("name")
    number_col = LZU_NUMBER_COLUMN if LZU_NUMBER_COLUMN in df.columns else lower_cols.get("number")
    anchor_col = LZU_ANCHOR_COLUMN if LZU_ANCHOR_COLUMN in df.columns else None
    if name_col is None:
        raise ValueError(
            f"Cannot find LZU seizure-time subject-name column in {path}. "
            f"Available columns: {[str(col) for col in df.columns]}"
        )
    if number_col is None:
        raise ValueError(
            f"Cannot find LZU seizure-time seizure-number column in {path}. "
            f"Available columns: {[str(col) for col in df.columns]}"
        )

    df["name"] = df[name_col].ffill().map(clean_text)
    df["number"] = df[number_col].map(_clean_lzu_identifier)
    if "\u7f16\u53f7" in df.columns:
        df["lzu_subject_number"] = df["\u7f16\u53f7"].ffill()
    if anchor_col is not None:
        df["lzu_recording_start_time"] = df[anchor_col]
    df = df[df["name"].ne("") & df["number"].ne("")].reset_index(drop=True)
    return df


def _clean_lzu_identifier(value: Any) -> str:
    numeric = safe_float(value)
    if numeric is not None and numeric.is_integer():
        return str(int(numeric))
    return clean_text(value)


def _parse_int_set(value: Any) -> set[int]:
    text = clean_text(value)
    if not text:
        return set()
    text = re.sub(r"[\u3001\uff0c,;]+", " ", text)
    values: set[int] = set()
    for token in text.split():
        range_match = re.fullmatch(r"(\d+)\s*[-~]\s*(\d+)", token)
        if range_match:
            start, end = int(range_match.group(1)), int(range_match.group(2))
            values.update(range(min(start, end), max(start, end) + 1))
            continue
        values.update(int(match.group(0)) for match in re.finditer(r"\d+", token))
    return values


def _build_lzu_edf_index(root: Path) -> dict[tuple[str, str], Path]:
    index: dict[tuple[str, str], Path] = {}
    if not root.exists():
        return index
    for edf_path in root.rglob("*.edf"):
        subject = _infer_lzu_subject_from_path(root, edf_path)
        index[(subject_key(subject), edf_path.stem.lower())] = edf_path
    return index


def _is_lzu_standalone_interictal_stem(stem: str) -> bool:
    text = clean_text(stem).lower()
    return any(token in text for token in ("iid", "bg", "hfo")) and "sz" not in text


def _lzu_standalone_interictal_sources_by_subject(edf_index: Mapping[tuple[str, str], Path]) -> dict[str, list[dict[str, Any]]]:
    sources_by_subject: dict[str, list[dict[str, Any]]] = {}
    for (subject_lookup, stem), edf_path in sorted(edf_index.items(), key=lambda item: (item[0][0], item[0][1])):
        if not _is_lzu_standalone_interictal_stem(stem):
            continue
        source = {
            "dataset_name": "lzu",
            "subject_id": subject_lookup,
            "edf_path": str(edf_path),
            "run_id": stem,
            "task": "interictal",
            "interictal_source_type": "lzu_iid_bg_hfo_edf",
        }
        sources_by_subject.setdefault(subject_lookup, []).append(source)
    return sources_by_subject


def _infer_lzu_subject_from_path(root: Path, edf_path: Path) -> str:
    try:
        rel_parts = edf_path.relative_to(root).parts
    except ValueError:
        rel_parts = edf_path.parts
    if len(rel_parts) >= 2:
        return rel_parts[0]
    return edf_path.parent.parent.name if edf_path.parent.name.lower() == "seeg" else edf_path.parent.name


def _resolve_lzu_edf_path(root: Path, subject_name: str, seizure_number: str, edf_index: Mapping[tuple[str, str], Path]) -> Path | None:
    subject_lookup = subject_key(subject_name)
    seizure_stems = _lzu_edf_stem_candidates(seizure_number)
    for candidate in _lzu_candidate_edf_paths(root, subject_name, seizure_stems, seizure_number):
        candidate_path = Path(candidate)
        if candidate_path.exists():
            return candidate_path
    for stem in seizure_stems:
        indexed = edf_index.get((subject_lookup, stem.lower()))
        if indexed is not None and indexed.exists():
            return indexed
    ranged = _resolve_lzu_range_edf_path(subject_lookup, seizure_number, edf_index)
    if ranged is not None and ranged.exists():
        return ranged
    return None


def _resolve_lzu_range_edf_path(
    subject_lookup: str,
    seizure_number: str,
    edf_index: Mapping[tuple[str, str], Path],
) -> Path | None:
    target = _lzu_single_sz_number(seizure_number)
    if target is None:
        return None
    matches: list[Path] = []
    for (indexed_subject, stem), edf_path in edf_index.items():
        if indexed_subject != subject_lookup:
            continue
        if _lzu_stem_contains_sz_number(stem, target):
            matches.append(edf_path)
    if not matches:
        return None
    return sorted(matches, key=lambda path: (len(path.stem), path.name.lower()))[0]


def _lzu_single_sz_number(value: str) -> int | None:
    text = clean_text(value)
    text = text.replace("\uFF08", "(").replace("\uFF09", ")").replace("\uFF0D", "-").replace("\u2014", "-")
    base = re.sub(r"\(.*$", "", re.sub(r"\s+", "", text)).strip()
    match = re.fullmatch(r"SZ0*(\d+)", base, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _lzu_stem_contains_sz_number(stem: str, target: int) -> bool:
    text = clean_text(stem)
    text = text.replace("\uFF0D", "-").replace("\u2014", "-").replace("~", "-")
    text = re.sub(r"\s+", "", text)
    range_match = re.fullmatch(r"SZ0*(\d+)-0*(\d+)", text, flags=re.IGNORECASE)
    if range_match:
        start, end = int(range_match.group(1)), int(range_match.group(2))
        return min(start, end) <= int(target) <= max(start, end)
    return False


def _lzu_candidate_edf_paths(
    root: Path,
    subject_name: str,
    seizure_stems: list[str],
    seizure_number: str | None = None,
) -> list[str]:
    paths: list[str] = []
    for stem in seizure_stems:
        paths.extend(
            [
                str(root / subject_name / "SEEG" / f"{stem}.edf"),
                str(root / subject_name / f"{stem}.edf"),
                str(root / subject_name.lower() / "SEEG" / f"{stem}.edf"),
                str(root / subject_name.lower() / f"{stem}.edf"),
            ]
        )
    target = _lzu_single_sz_number(seizure_number or "")
    if target is None:
        for stem in seizure_stems:
            target = _lzu_single_sz_number(stem)
            if target is not None:
                break
    if target is not None:
        for subject_variant in (subject_name, subject_name.lower()):
            subject_dir = root / subject_variant
            for parent in (subject_dir / "SEEG", subject_dir):
                if parent.exists():
                    for edf_path in sorted(parent.glob("*.edf")):
                        if _lzu_stem_contains_sz_number(edf_path.stem, target):
                            paths.append(str(edf_path))
    seen: set[str] = set()
    result: list[str] = []
    for path in paths:
        if path not in seen:
            seen.add(path)
            result.append(path)
    return result


def _log_and_write_lzu_failure_details(
    cfg: DataInterfaceConfig,
    missing_edf_details: list[dict[str, Any]],
    read_error_details: list[dict[str, Any]],
) -> None:
    if missing_edf_details:
        log(f"LZU missing EDF details: n={len(missing_edf_details)}")
        for item in missing_edf_details[:10]:
            log(
                "LZU missing_edf: "
                f"row={item.get('time_row_index')}, "
                f"subject={item.get('subject_id')}, "
                f"seizure={item.get('seizure_id')}, "
                f"stems={item.get('lzu_edf_stem_candidates')}, "
                f"candidate_paths={item.get('candidate_paths')}"
            )

    if read_error_details:
        log(f"LZU read error details: n={len(read_error_details)}")
        reason_counts = pd.Series([item["exception_repr"] for item in read_error_details]).value_counts()
        log("LZU read error reason counts:")
        for reason, count in reason_counts.items():
            log(f"  {count}: {reason}")
        for item in read_error_details[:10]:
            log(
                "LZU read_error: "
                f"row={item.get('time_row_index')}, "
                f"subject={item.get('subject_id')}, "
                f"seizure={item.get('seizure_id')}, "
                f"edf={item.get('edf_path')}, "
                f"exception_type={item.get('exception_type')}, "
                f"exception={item.get('exception_repr')}"
            )

    if cfg.write_read_audit:
        cfg.read_audit_dir.mkdir(parents=True, exist_ok=True)
        if missing_edf_details:
            pd.DataFrame(missing_edf_details).to_csv(
                cfg.read_audit_dir / "lzu_missing_edf_details.csv",
                index=False,
                encoding="utf-8-sig",
            )
        if read_error_details:
            pd.DataFrame(read_error_details).to_csv(
                cfg.read_audit_dir / "lzu_read_error_details.csv",
                index=False,
                encoding="utf-8-sig",
            )


def _lzu_edf_stem_candidates(seizure_number: str) -> list[str]:
    text = clean_text(seizure_number)
    if not text:
        return []

    text = (
        text.replace("\uFF08", "(")
        .replace("\uFF09", ")")
        .replace("\uFF0D", "-")
        .replace("\u2014", "-")
        .replace("~", "-")
    )
    text_no_space = re.sub(r"\s+", "", text)
    candidates: list[str] = []

    def add(value: str) -> None:
        value = clean_text(value)
        if not value:
            return
        candidates.append(value)
        candidates.append(value.upper())
        candidates.append(value.lower())

    add(text)
    add(text_no_space)
    base = re.sub(r"\(.*$", "", text_no_space).strip()
    add(base)
    base_no_quote = base.replace("'", "").replace("\u2019", "").replace("`", "")
    add(base_no_quote)

    range_match = re.match(r"^SZ(\d+)-(\d+)$", base_no_quote, flags=re.IGNORECASE)
    if range_match:
        start, end = int(range_match.group(1)), int(range_match.group(2))
        add(f"SZ{start}-{end}")
        for idx in range(min(start, end), max(start, end) + 1):
            add(f"SZ{idx}")

    single_match = re.match(r"^(SZ\d+)", base_no_quote, flags=re.IGNORECASE)
    if single_match:
        add(single_match.group(1).upper())

    if base_no_quote.upper().startswith("SZ"):
        add(base_no_quote)

    seen: set[str] = set()
    result: list[str] = []
    for item in candidates:
        item = clean_text(item)
        key = item.lower()
        if item and key not in seen:
            seen.add(key)
            result.append(item)
    return result


def _parse_lzu_time_row(row: pd.Series) -> LzuTimeInfo:
    anchor_col = LZU_ANCHOR_COLUMN if LZU_ANCHOR_COLUMN in row.index else "lzu_recording_start_time"
    anchor = _time_to_seconds_of_day_or_range_end(row.get(anchor_col)) if anchor_col in row.index else None

    onset_value = row.get(LZU_ONSET_COLUMN) if LZU_ONSET_COLUMN in row.index else None
    onset = _time_to_seconds_of_day_or_range_end(onset_value)
    onset_source_col = LZU_ONSET_COLUMN if onset is not None else None
    if onset is None and LZU_FALLBACK_ONSET_COLUMN in row.index:
        onset_value = row.get(LZU_FALLBACK_ONSET_COLUMN)
        onset = _time_to_seconds_of_day_or_range_end(onset_value)
        if onset is not None:
            onset_source_col = LZU_FALLBACK_ONSET_COLUMN
    if onset is None:
        raise ValueError(
            f"Cannot parse LZU seizure onset from {LZU_ONSET_COLUMN} or {LZU_FALLBACK_ONSET_COLUMN}: "
            f"{row.get(LZU_ONSET_COLUMN)!r}, {row.get(LZU_FALLBACK_ONSET_COLUMN)!r}"
        )

    offset_value = row.get(LZU_OFFSET_COLUMN) if LZU_OFFSET_COLUMN in row.index else None
    offset = _time_to_seconds_of_day_or_range_end(offset_value)
    source_columns = []
    if anchor_col in row.index:
        source_columns.append(anchor_col)
    source_columns.append(str(onset_source_col))
    if LZU_OFFSET_COLUMN in row.index:
        source_columns.append(LZU_OFFSET_COLUMN)
    return LzuTimeInfo(
        anchor_clock_sec=anchor,
        onset_value=onset_value,
        offset_value=offset_value,
        onset_clock_sec=onset,
        offset_clock_sec=offset,
        source_columns=source_columns,
    )


def _time_to_seconds_of_day_or_range_end(value: Any) -> float | None:
    sec = _time_to_seconds_of_day(value)
    if sec is not None:
        return sec
    text = clean_text(value).replace("\uff1a", ":")
    if not text:
        return None
    matches = re.findall(r"\d{1,2}:\d{1,2}(?::\d{1,2}(?:\.\d+)?)?", text)
    if not matches:
        return None
    return _time_to_seconds_of_day(matches[-1])


def _time_to_seconds_of_day(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, datetime):
        return float(value.hour * 3600 + value.minute * 60 + value.second + value.microsecond / 1e6)
    if isinstance(value, time):
        return float(value.hour * 3600 + value.minute * 60 + value.second + value.microsecond / 1e6)
    if isinstance(value, (int, float, np.integer, np.floating)):
        if not math.isfinite(float(value)):
            return None
        if 0.0 <= float(value) < 1.0:
            return float(value) * 24.0 * 3600.0
        return float(value)
    text = clean_text(value).replace("\uff1a", ":")
    match = re.match(r"^(\d{1,2}):(\d{1,2})(?::(\d{1,2}(?:\.\d+)?))?$", text)
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2))
        second = float(match.group(3) or 0.0)
        return float(hour * 3600 + minute * 60 + second)
    return safe_float(text)


def _read_lzu_seizure(
    edf_path: Path,
    subject_id: str,
    seizure_id: str,
    annotation: Mapping[str, Any],
    time_info: LzuTimeInfo,
    cfg: DataInterfaceConfig,
    audit: ReadAudit,
    row_idx: int,
) -> SeizureRecord | None:
    raw = read_raw_edf(edf_path, preload=False)
    try:
        raw_duration = float(raw.n_times) / float(raw.info["sfreq"])
        onset, offset, onset_source = _resolve_lzu_relative_bounds(raw, time_info)
        onset_valid = onset is not None and 0.0 <= onset <= raw_duration
        offset_valid = offset is None or 0.0 <= offset <= raw_duration
        if not onset_valid:
            audit.add_skipped_seizure(
                "lzu",
                subject_id,
                seizure_id,
                "onset_out_of_range",
                edf_path=str(edf_path),
                onset_sec=onset,
                offset_sec=offset,
                raw_duration_sec=raw_duration,
                onset_valid=False,
                offset_valid=offset_valid,
                onset_source=onset_source,
                time_source_columns=";".join(time_info.source_columns),
            )
            return None

        sorted_raw_names = sorted(list(raw.ch_names), key=natural_channel_sort_key)
        channel_entries = [(idx + 1, raw_name, normalize_channel_name(raw_name)) for idx, raw_name in enumerate(sorted_raw_names)]
        all_ids = {entry[0] for entry in channel_entries}
        bad_ids = set(annotation["bad_channel_ids"])
        ez_ids = set(annotation["ez_channel_ids"])
        bad_ez_overlap = sorted(bad_ids & ez_ids)
        unmatched_label_ids = sorted((ez_ids | bad_ids) - all_ids)
        selected_entries = [entry for entry in channel_entries if entry[0] not in bad_ids]
        if not selected_entries:
            audit.add_skipped_seizure("lzu", subject_id, seizure_id, "no_valid_channels_after_bad_removal", edf_path=str(edf_path))
            return None

        for raw_order, raw_name, norm_name in channel_entries:
            is_bad = raw_order in bad_ids
            is_ez = raw_order in ez_ids
            audit.add_channel(
                "lzu",
                subject_id,
                seizure_id,
                raw_channel_order=raw_order,
                raw_channel_name=raw_name,
                normalized_channel_name=norm_name,
                lzu_original_channel_id=raw_order,
                is_bad=is_bad,
                is_ez=is_ez,
                final_label=None if is_bad else (0.0 if is_ez else 1.0),
            )

        picked_raw_names = [entry[1] for entry in selected_entries]
        final_names = make_unique([entry[2] for entry in selected_entries])
        labels = np.asarray([0.0 if entry[0] in ez_ids else 1.0 for entry in selected_entries], dtype=np.float32)
        crop_start = crop_raw_to_preictal_context(raw, float(onset))
        data, sfreq, channel_names, sfreq_original = finalize_raw_data(raw, picked_raw_names, final_names, cfg)
    finally:
        close_raw(raw)

    adjusted_onset = float(onset) - float(crop_start)
    adjusted_offset = float(offset) - float(crop_start) if offset is not None else None
    channel_meta = []
    for entry, final_name in zip(selected_entries, channel_names):
        contact_group, contact_number = parse_contact_topology(final_name)
        channel_meta.append(
            {
                "dataset": "lzu",
                "source_path": str(edf_path),
                "time_row_index": row_idx,
                "channel_name_orig": entry[1],
                "channel_name_norm": final_name,
                "contact_group": contact_group,
                "contact_number": contact_number,
                "lzu_original_channel_id": entry[0],
                "is_bad": False,
                "is_ez_or_soz": int(entry[0] in ez_ids),
                "final_label": 0.0 if entry[0] in ez_ids else 1.0,
                "label_source": "lzu_annotation_xlsx_channel_number",
                "unmatched_label_ids": unmatched_label_ids,
                "bad_ez_overlap_ids": bad_ez_overlap,
                "n_bad_channels_source": len(bad_ids),
                "time_source_columns": list(time_info.source_columns),
                "onset_source": onset_source,
                "raw_duration_sec": raw_duration,
                "preictal_crop_start_sec": crop_start,
                "original_seizure_onset_sec": float(onset),
                "original_seizure_offset_sec": float(offset) if offset is not None else None,
                "sfreq_original": sfreq_original,
                "lzu_outcome_raw": annotation.get("outcome_raw"),
                "success_used": annotation.get("is_successful"),
            }
        )
    return SeizureRecord(
        subject_id=subject_id,
        seizure_id=f"{subject_id}__{seizure_id}",
        signal=data,
        sfreq=sfreq,
        channel_names=channel_names,
        seizure_onset_sec=adjusted_onset,
        seizure_offset_sec=adjusted_offset,
        labels=labels,
        channel_meta=channel_meta,
    )


def _resolve_lzu_relative_bounds(raw: Any, time_info: LzuTimeInfo) -> tuple[float, float | None, str]:
    raw_duration = float(raw.n_times) / float(raw.info["sfreq"])
    onset_clock = time_info.onset_clock_sec
    offset_clock = time_info.offset_clock_sec
    if onset_clock is None:
        onset_direct = safe_float(time_info.onset_value)
        offset_direct = safe_float(time_info.offset_value)
        if onset_direct is None:
            raise ValueError("Cannot parse LZU seizure onset.")
        return float(onset_direct), float(offset_direct) if offset_direct is not None else None, "numeric"
    if time_info.anchor_clock_sec is not None:
        onset = _clock_delta_seconds(time_info.anchor_clock_sec, onset_clock)
        offset = _clock_delta_seconds(time_info.anchor_clock_sec, offset_clock) if offset_clock is not None else None
        return onset, offset, "lzu_tail3_anchor"
    meas_date = raw.info.get("meas_date")
    if meas_date is not None:
        start_clock = float(meas_date.hour * 3600 + meas_date.minute * 60 + meas_date.second)
        onset = _clock_delta_seconds(start_clock, onset_clock)
        offset = _clock_delta_seconds(start_clock, offset_clock) if offset_clock is not None else None
        if 0.0 <= onset <= raw_duration:
            return onset, offset, "edf_meas_date"
    offset = _clock_delta_seconds(onset_clock, offset_clock) if offset_clock is not None else None
    return 0.0, offset, "onset_clock_as_segment_start"


def _clock_delta_seconds(start_sec: float, end_sec: float) -> float:
    delta = float(end_sec) - float(start_sec)
    if delta < 0.0:
        delta += 24.0 * 3600.0
    return delta
