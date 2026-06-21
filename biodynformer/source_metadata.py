from __future__ import annotations

import csv
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

SIGNAL_SUFFIXES = {".edf", ".npy", ".npz", ".csv", ".txt"}

DEFAULT_METADATA_FILES = {
    "lzu_quality": "slope_quality_report_en1修改.xlsx",
    "lzu_quality_failure": "slope_quality_report_en2-4.xlsx",
    "hup_quality": "slope_quality_report_HUP_S.xlsx",
    "hup_quality_failure": "slope_quality_report_HUP_F.xlsx",
    "multicenter_quality": "slope_quality_report_ds3029_S.xlsx",
    "multicenter_quality_failure": "slope_quality_report_ds003929_F.xlsx",
    "pediatric_quality": "儿科数据分类.xlsx",
    "lzu_outcome": "label-seeg_with_engel.xlsx",
    "lzu_times": "SEEG数据分析时间标签.xlsx",
    "multicenter_participants": "participants-muticenter.tsv",
    "pediatric_metadata": "pediatric_ez_channels_final.xlsx",
}

DEFAULT_QUALITY_FILE_KEYS = {
    "lzu": ("lzu_quality", "lzu_quality_failure"),
    "hup": ("hup_quality", "hup_quality_failure"),
    "multicenter": ("multicenter_quality", "multicenter_quality_failure"),
    "pediatric": ("pediatric_quality",),
}


@dataclass(frozen=True)
class QualitySummaryRow:
    center: str
    subject_id: str
    run_id: str
    quality_rating: str
    source_file_path: str


def clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip())


def read_quality_summary(path: str | Path, *, center: str) -> list[QualitySummaryRow]:
    workbook = pd.read_excel(path, sheet_name=None)
    required = {"患者ID", "发作名称", "质量评级"}
    for _, df in workbook.items():
        if required.issubset({str(col) for col in df.columns}):
            rows: list[QualitySummaryRow] = []
            for _, row in df.iterrows():
                subject_id = clean_text(row.get("患者ID"))
                run_id = clean_text(row.get("发作名称"))
                quality = clean_text(row.get("质量评级")).upper()
                if not subject_id or not run_id or not quality:
                    continue
                rows.append(
                    QualitySummaryRow(
                        center=center,
                        subject_id=subject_id,
                        run_id=run_id,
                        quality_rating=quality,
                        source_file_path=clean_text(row.get("文件路径")),
                    )
                )
            return rows
    available = {str(name): [str(col) for col in df.columns] for name, df in workbook.items()}
    raise ValueError(f"No quality summary sheet with columns {sorted(required)} in {path}; available={available}")


def read_lzu_outcomes(path: str | Path) -> dict[str, str]:
    df = pd.read_excel(path)
    name_col = _first_column(df.columns, ("姓名", "name", "subject_id", "patient_id"))
    outcome_col = _first_column(df.columns, ("Engel分级(S/F)", "Engel分级", "outcome", "engel"))
    if name_col is None or outcome_col is None:
        return {}
    return {
        clean_text(row.get(name_col)): clean_text(row.get(outcome_col)).upper()
        for _, row in df.iterrows()
        if clean_text(row.get(name_col)) and clean_text(row.get(outcome_col))
    }


def read_multicenter_outcomes(path: str | Path) -> dict[str, str]:
    df = pd.read_csv(path, sep="\t")
    if "participant_id" not in df.columns or "outcome" not in df.columns:
        return {}
    out: dict[str, str] = {}
    for _, row in df.iterrows():
        participant_id = clean_text(row.get("participant_id"))
        outcome = clean_text(row.get("outcome")).upper()
        if participant_id and outcome:
            out[participant_id] = outcome
            out[participant_id.replace("sub-", "")] = outcome
    return out


def read_hup_outcomes(path: str | Path) -> dict[str, str]:
    df = pd.read_csv(path, sep="\t")
    if "participant_id" not in df.columns or "outcome" not in df.columns:
        return {}
    out: dict[str, str] = {}
    for _, row in df.iterrows():
        participant_id = clean_text(row.get("participant_id"))
        outcome = clean_text(row.get("outcome")).upper()
        if participant_id and outcome:
            stripped = participant_id.replace("sub-", "")
            out[participant_id] = outcome
            out[stripped] = outcome
            out[stripped.upper()] = outcome
    return out


def read_pediatric_outcomes(path: str | Path) -> dict[str, str]:
    df = pd.read_excel(path, sheet_name="EZ_确定汇总")
    subject_col = _first_column(df.columns, ("subject_id", "被试id", "id"))
    outcome_col = _first_column(df.columns, ("surgery_result", "手术结果", "outcome"))
    if subject_col is None or outcome_col is None:
        return {}
    return {
        clean_text(row.get(subject_col)): clean_text(row.get(outcome_col))
        for _, row in df.iterrows()
        if clean_text(row.get(subject_col)) and clean_text(row.get(outcome_col))
    }


def signal_files_under(root: str | Path | None) -> list[Path]:
    if root is None:
        return []
    path = Path(root)
    if not path.exists():
        return []
    return [item for item in path.rglob("*") if item.is_file() and item.suffix.lower() in SIGNAL_SUFFIXES]


def audit_source_metadata(
    *,
    metadata_dir: str | Path,
    centers: Sequence[str] = ("lzu", "hup", "multicenter", "pediatric"),
    roots: dict[str, str | Path | None] | None = None,
    hup_participants_path: str | Path | None = None,
) -> dict[str, Any]:
    metadata_dir = Path(metadata_dir)
    roots = dict(roots or {})
    center_rows: dict[str, list[QualitySummaryRow]] = {}
    outcome_maps = _load_outcome_maps(metadata_dir, hup_participants_path=hup_participants_path)
    result: dict[str, Any] = {"metadata_dir": str(metadata_dir), "centers": {}, "can_build_feature_bank": True}
    for center in centers:
        center_key = str(center).strip().lower()
        quality_paths = _default_quality_paths(metadata_dir, center_key)
        rows: list[QualitySummaryRow] = []
        existing_quality_paths: list[Path] = []
        for quality_path in quality_paths:
            if quality_path.exists():
                existing_quality_paths.append(quality_path)
                rows.extend(read_quality_summary(quality_path, center=center_key))
        center_rows[center_key] = rows
        rating_counts = Counter(row.quality_rating for row in rows)
        subject_ids = {row.subject_id for row in rows}
        outcome_map = outcome_maps.get(center_key, {})
        canonical_outcomes = _canonical_outcome_map(outcome_map)
        outcome_counts = Counter(outcome for _, outcome in canonical_outcomes.values())
        quality_subject_keys = {_subject_match_key(subject_id) for subject_id in subject_ids}
        outcome_subject_keys = set(canonical_outcomes)
        quality_without_outcome = sorted(subject_id for subject_id in subject_ids if _subject_match_key(subject_id) not in outcome_subject_keys)
        outcome_without_quality = sorted(
            subject_id
            for match_key, (subject_id, _) in canonical_outcomes.items()
            if match_key not in quality_subject_keys
        )
        signal_count = len(signal_files_under(roots.get(center_key)))
        missing_signal = signal_count == 0
        if missing_signal:
            result["can_build_feature_bank"] = False
        result["centers"][center_key] = {
            "quality_reports": ";".join(str(path) for path in quality_paths),
            "quality_reports_found": ";".join(str(path) for path in existing_quality_paths),
            "quality_report_exists": bool(existing_quality_paths),
            "quality_rows": len(rows),
            "quality_subjects": len(subject_ids),
            "quality_ratings": dict(sorted(rating_counts.items())),
            "outcome_subjects": len(canonical_outcomes),
            "outcome_counts": dict(sorted(outcome_counts.items())),
            "quality_subjects_without_outcome": quality_without_outcome,
            "outcome_subjects_without_quality": outcome_without_quality,
            "signal_root": str(roots.get(center_key) or ""),
            "signal_files_found": signal_count,
            "blocker": "no_signal_files_found" if missing_signal else "",
        }
    return result


def _default_quality_paths(metadata_dir: Path, center: str) -> list[Path]:
    keys = DEFAULT_QUALITY_FILE_KEYS.get(center, (f"{center}_quality",))
    paths = []
    for key in keys:
        filename = DEFAULT_METADATA_FILES.get(key)
        if filename:
            paths.append(metadata_dir / filename)
    return paths


def write_audit_outputs(audit: dict[str, Any], output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with open(output / "source_metadata_audit.json", "w", encoding="utf-8") as fout:
        json.dump(audit, fout, ensure_ascii=False, indent=2)
    rows = []
    for center, payload in audit.get("centers", {}).items():
        row = {"center": center}
        row.update(payload)
        rows.append(row)
    if rows:
        with open(output / "source_metadata_audit.csv", "w", encoding="utf-8-sig", newline="") as fout:
            writer = csv.DictWriter(fout, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def _load_outcome_maps(metadata_dir: Path, *, hup_participants_path: str | Path | None = None) -> dict[str, dict[str, str]]:
    maps: dict[str, dict[str, str]] = {"lzu": {}, "hup": {}, "multicenter": {}, "pediatric": {}}
    lzu = metadata_dir / DEFAULT_METADATA_FILES["lzu_outcome"]
    if lzu.exists():
        maps["lzu"] = read_lzu_outcomes(lzu)
    if hup_participants_path is not None and Path(hup_participants_path).exists():
        maps["hup"] = read_hup_outcomes(hup_participants_path)
    multi = metadata_dir / DEFAULT_METADATA_FILES["multicenter_participants"]
    if multi.exists():
        maps["multicenter"] = read_multicenter_outcomes(multi)
    pediatric = metadata_dir / DEFAULT_METADATA_FILES["pediatric_metadata"]
    if pediatric.exists():
        maps["pediatric"] = read_pediatric_outcomes(pediatric)
    return maps


def _subject_match_key(subject_id: Any) -> str:
    text = clean_text(subject_id).lower()
    text = re.sub(r"^sub[-_]*", "", text)
    return text


def _canonical_outcome_subject(subject_id: Any) -> str:
    text = clean_text(subject_id)
    if text.lower().startswith("sub-"):
        return text.replace("sub-", "", 1)
    return text


def _canonical_outcome_map(outcome_map: dict[str, str]) -> dict[str, tuple[str, str]]:
    canonical: dict[str, tuple[str, str]] = {}
    for subject_id, outcome in outcome_map.items():
        match_key = _subject_match_key(subject_id)
        display = _canonical_outcome_subject(subject_id)
        if match_key not in canonical or str(subject_id).lower().startswith("sub-"):
            canonical[match_key] = (display, outcome)
    return canonical


def _first_column(columns: Iterable[Any], candidates: Sequence[str]) -> Any | None:
    normalized = {str(col).strip().lower(): col for col in columns}
    for candidate in candidates:
        key = str(candidate).strip().lower()
        if key in normalized:
            return normalized[key]
    return None


def rows_to_dicts(rows: Sequence[QualitySummaryRow]) -> list[dict[str, Any]]:
    return [asdict(row) for row in rows]
