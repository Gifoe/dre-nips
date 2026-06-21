from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from .source_adapters.common import normalize_channel_name


COL_SUBJECT = "\u60a3\u8005ID"
COL_RUN = "\u53d1\u4f5c\u540d\u79f0"
COL_RATING = "\u8d28\u91cf\u8bc4\u7ea7"
COL_FILE = "\u6587\u4ef6\u8def\u5f84"
COL_FILE_RATING = "\u6587\u4ef6\u8d28\u91cf\u8bc4\u7ea7"
COL_CHANNEL = "\u901a\u9053\u540d"
COL_IS_EZ = "\u662f\u5426EZ"
SHEET_SUMMARY = "\u6587\u4ef6\u6c47\u603b"
SHEET_CHANNELS = "\u95ee\u9898\u901a\u9053"

LZU_TIME_FILE = "SEEG\u6570\u636e\u5206\u6790\u65f6\u95f4\u6807\u7b7e.xlsx"
LZU_OUTCOME_FILE = "label-seeg_with_engel.xlsx"
MULTICENTER_OUTCOME_FILE = "participants-muticenter.tsv"
PEDIATRIC_CLASS_FILE = "\u513f\u79d1\u6570\u636e\u5206\u7c7b.xlsx"
PEDIATRIC_EZ_FILE = "pediatric_ez_channels_final.xlsx"

QUALITY_FILES = {
    "lzu": ("slope_quality_report_en1\u4fee\u6539.xlsx", "slope_quality_report_en2-4.xlsx"),
    "hup": ("slope_quality_report_HUP_S.xlsx", "slope_quality_report_HUP_F.xlsx"),
    "multicenter": ("slope_quality_report_ds3029_S.xlsx", "slope_quality_report_ds003929_F.xlsx"),
    "pediatric": (PEDIATRIC_CLASS_FILE,),
}

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

OUTPUT_COLUMNS = (
    *REQUIRED_MANIFEST_COLUMNS,
    "center",
    "source_result_path",
    "source_quality_file",
    "signal_path_status",
    "missing_fields",
    "manifest_status",
)

SUPPORTED_SIGNAL_SUFFIXES = {".edf", ".npy", ".npz", ".csv", ".txt"}


def generate_manifest_drafts(
    *,
    metadata_dir: str | Path,
    output_dir: str | Path | None = None,
    centers: Sequence[str] = ("lzu", "hup", "multicenter", "pediatric"),
    keep_ratings: Sequence[str] = ("GOOD", "REVIEW"),
    hup_participants_path: str | Path | None = None,
) -> dict[str, Any]:
    metadata = Path(metadata_dir)
    output = Path(output_dir) if output_dir is not None else metadata / "generated_manifests"
    output.mkdir(parents=True, exist_ok=True)

    keep = {str(rating).strip().upper() for rating in keep_ratings if str(rating).strip()}
    center_summaries: dict[str, Any] = {}
    combined_all: list[dict[str, str]] = []
    combined_keep: list[dict[str, str]] = []
    combined_strict: list[dict[str, str]] = []

    context = _load_cross_center_context(metadata, hup_participants_path=hup_participants_path)

    for center in [str(center).strip().lower() for center in centers]:
        quality_paths = [metadata / name for name in QUALITY_FILES.get(center, ())]
        rows = _build_center_rows(center, quality_paths, metadata=metadata, context=context)
        keep_rows = [row for row in rows if row["quality_rating"].upper() in keep]
        strict_rows = [row for row in rows if not row["missing_fields"]]

        _write_csv(output / f"{center}_all_draft_manifest.csv", rows)
        _write_csv(output / f"{center}_good_review_draft_manifest.csv", keep_rows)
        _write_csv(output / f"{center}_strict_pipeline_manifest.csv", strict_rows)

        combined_all.extend(rows)
        combined_keep.extend(keep_rows)
        combined_strict.extend(strict_rows)
        center_summaries[center] = _summarize_rows(rows, keep_rows, strict_rows, quality_paths)

    _write_csv(output / "four_center_all_draft_manifest.csv", combined_all)
    _write_csv(output / "four_center_good_review_draft_manifest.csv", combined_keep)
    _write_csv(output / "four_center_strict_pipeline_manifest.csv", combined_strict)

    summary = {
        "metadata_dir": str(metadata),
        "output_dir": str(output),
        "keep_ratings": sorted(keep),
        "centers": center_summaries,
        "combined": {
            "all_draft_rows": len(combined_all),
            "good_review_draft_rows": len(combined_keep),
            "strict_pipeline_rows": len(combined_strict),
            "missing_required_counts": _missing_counts(combined_all),
        },
    }
    with open(output / "manifest_generation_summary.json", "w", encoding="utf-8") as fout:
        json.dump(summary, fout, ensure_ascii=False, indent=2)
    return summary


def _build_center_rows(
    center: str,
    quality_paths: Sequence[Path],
    *,
    metadata: Path,
    context: Mapping[str, Any],
) -> list[dict[str, str]]:
    summaries = _read_quality_summaries(quality_paths)
    if summaries.empty:
        return []

    channel_index = _read_problem_channel_index(quality_paths)
    pediatric_channels = context.get("pediatric_channels", {})
    rows: list[dict[str, str]] = []
    for _, source in summaries.iterrows():
        subject_id = _clean(source.get(COL_SUBJECT))
        run_id = _clean(source.get(COL_RUN))
        source_result_path = _clean(source.get(COL_FILE))
        quality_rating = _clean(source.get(COL_RATING)).upper()
        source_quality_file = _clean(source.get("_source_quality_file"))

        channel_names = ""
        labels_ez = ""
        if center == "pediatric":
            channel_payload = pediatric_channels.get(_subject_key(subject_id), {})
            channel_names = ",".join(channel_payload.get("channel_names", []))
            labels_ez = ",".join(channel_payload.get("labels_ez", []))
        else:
            channel_payload = _lookup_channels(channel_index, subject_id, run_id, source_result_path)
            channel_names = ",".join(channel_payload.get("channel_names", []))
            labels_ez = ",".join(channel_payload.get("labels_ez", []))

        onset = _explicit_numeric(source, ("seizure_onset_sec", "onset_sec", "onset"))
        if onset == "" and center == "lzu":
            onset = _lookup_lzu_onset(context.get("lzu_onsets", {}), subject_id, run_id, source_result_path)

        outcome = _lookup_outcome(center, subject_id, source_quality_file, context)
        signal_path, signal_status = _candidate_signal_path(source_result_path, metadata)
        sfreq = _explicit_numeric(source, ("sfreq", "sampling_rate", "sample_rate"))

        row = {
            "subject_id": subject_id,
            "run_id": run_id,
            "signal_path": signal_path,
            "sfreq": sfreq,
            "seizure_onset_sec": onset,
            "channel_names": channel_names,
            "labels_ez": labels_ez,
            "outcome": outcome,
            "quality_rating": quality_rating,
            "center": center,
            "source_result_path": source_result_path,
            "source_quality_file": source_quality_file,
            "signal_path_status": signal_status,
            "missing_fields": "",
            "manifest_status": "",
        }
        missing = _required_missing(row)
        row["missing_fields"] = ",".join(missing)
        row["manifest_status"] = "pipeline_ready" if not missing else "draft_missing_required_fields"
        rows.append(row)
    return rows


def _read_quality_summaries(paths: Sequence[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        if not path.exists():
            continue
        workbook = pd.ExcelFile(path)
        sheet = SHEET_SUMMARY if SHEET_SUMMARY in workbook.sheet_names else "seizure_onsets"
        frame = pd.read_excel(path, sheet_name=sheet)
        frame["_source_quality_file"] = path.name
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _read_problem_channel_index(paths: Sequence[Path]) -> dict[str, Any]:
    by_exact: dict[tuple[str, str, str], dict[str, list[str]]] = {}
    by_run: defaultdict[tuple[str, str], list[dict[str, list[str]]]] = defaultdict(list)
    for path in paths:
        if not path.exists():
            continue
        workbook = pd.ExcelFile(path)
        if SHEET_CHANNELS not in workbook.sheet_names:
            continue
        frame = pd.read_excel(path, sheet_name=SHEET_CHANNELS)
        required = {COL_SUBJECT, COL_RUN, COL_CHANNEL, COL_IS_EZ}
        if not required.issubset(set(frame.columns)):
            continue
        for key, group in frame.groupby([COL_SUBJECT, COL_RUN, COL_FILE], dropna=False, sort=False):
            subject, run, source_path = key
            payload = _channel_payload(group)
            exact_key = (_subject_key(subject), _run_key(run), _path_key(source_path))
            by_exact[exact_key] = payload
            by_run[exact_key[:2]].append(payload)
    return {"by_exact": by_exact, "by_run": dict(by_run)}


def _channel_payload(group: pd.DataFrame) -> dict[str, list[str]]:
    channels: list[str] = []
    labels: list[str] = []
    for _, row in group.iterrows():
        channel = _clean(row.get(COL_CHANNEL))
        if not channel:
            continue
        channels.append(normalize_channel_name(channel))
        labels.append(_ez_label(row.get(COL_IS_EZ)))
    return {"channel_names": channels, "labels_ez": labels}


def _lookup_channels(index: Mapping[str, Any], subject_id: str, run_id: str, source_result_path: str) -> dict[str, list[str]]:
    exact_key = (_subject_key(subject_id), _run_key(run_id), _path_key(source_result_path))
    exact = index.get("by_exact", {}).get(exact_key)
    if exact:
        return exact
    candidates = index.get("by_run", {}).get(exact_key[:2], [])
    if len(candidates) == 1:
        return candidates[0]
    return {"channel_names": [], "labels_ez": []}


def _load_cross_center_context(metadata: Path, *, hup_participants_path: str | Path | None) -> dict[str, Any]:
    return {
        "lzu_outcomes": _read_lzu_outcomes(metadata / LZU_OUTCOME_FILE),
        "lzu_onsets": _read_lzu_onsets(metadata / LZU_TIME_FILE),
        "hup_outcomes": _read_hup_outcomes(Path(hup_participants_path)) if hup_participants_path else {},
        "multicenter_outcomes": _read_multicenter_outcomes(metadata / MULTICENTER_OUTCOME_FILE),
        "pediatric_outcomes": _read_pediatric_outcomes(metadata / PEDIATRIC_EZ_FILE),
        "pediatric_channels": _read_pediatric_channels(metadata / PEDIATRIC_EZ_FILE),
    }


def _read_lzu_outcomes(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    frame = pd.read_excel(path)
    subject_col = _first_existing(frame.columns, ("\u59d3\u540d", "subject_id", "patient_id"))
    outcome_col = _first_existing(frame.columns, ("Engel\u5206\u7ea7(S/F)", "outcome", "engel"))
    if subject_col is None or outcome_col is None:
        return {}
    return {_subject_key(row.get(subject_col)): _canonical_outcome(row.get(outcome_col)) for _, row in frame.iterrows()}


def _read_hup_outcomes(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    frame = pd.read_csv(path, sep="\t")
    if "participant_id" not in frame.columns or "outcome" not in frame.columns:
        return {}
    return {_subject_key(row.get("participant_id")): _canonical_outcome(row.get("outcome")) for _, row in frame.iterrows()}


def _read_multicenter_outcomes(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    frame = pd.read_csv(path, sep="\t")
    if "participant_id" not in frame.columns or "outcome" not in frame.columns:
        return {}
    return {_subject_key(row.get("participant_id")): _canonical_outcome(row.get("outcome")) for _, row in frame.iterrows()}


def _read_pediatric_outcomes(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    frame = pd.read_excel(path, sheet_name="EZ_\u786e\u5b9a\u6c47\u603b")
    if "subject_id" not in frame.columns:
        return {}
    outcome_col = _first_existing(frame.columns, ("surgery_result", "followup_result", "outcome"))
    if outcome_col is None:
        return {}
    return {_subject_key(row.get("subject_id")): _canonical_outcome(row.get(outcome_col)) for _, row in frame.iterrows()}


def _read_pediatric_channels(path: Path) -> dict[str, dict[str, list[str]]]:
    if not path.exists():
        return {}
    frame = pd.read_excel(path, sheet_name="channel_level_labels")
    required = {"subject_id", "channel_name_norm", "model_label_ez_excluding_bad"}
    if not required.issubset(set(frame.columns)):
        return {}
    if "usable_channel_mask" in frame.columns:
        frame = frame[frame["usable_channel_mask"].fillna(1).astype(int) == 1]
    if "channel_order" in frame.columns:
        frame = frame.sort_values(["subject_id", "channel_order"], kind="stable")
    out: dict[str, dict[str, list[str]]] = {}
    for subject, group in frame.groupby("subject_id", sort=False):
        channels = [normalize_channel_name(value) for value in group["channel_name_norm"].map(_clean) if value]
        labels = [_format_int_label(value) for value in group["model_label_ez_excluding_bad"]]
        out[_subject_key(subject)] = {"channel_names": channels, "labels_ez": labels}
    return out


def _read_lzu_onsets(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"by_exact": {}, "by_run": {}}
    frame = pd.read_excel(path, sheet_name="Sheet1")
    subject_col = _first_existing(frame.columns, ("\u59d3\u540d", "subject_id", "patient_id"))
    run_col = _first_existing(frame.columns, ("\u53d1\u4f5c\u7f16\u53f7", "run_id", "seizure_id"))
    start_col = _first_existing(frame.columns, ("\u8111\u7535\u56fe\u8bb0\u5f55\u5f00\u59cb\u65f6\u95f4", "recording_start"))
    onset_col = _first_existing(frame.columns, ("Unnamed: 13", "\u53d1\u4f5c\u671f\u5f00\u59cb", "seizure_onset"))
    if subject_col is None or run_col is None or start_col is None or onset_col is None:
        return {"by_exact": {}, "by_run": {}}
    frame[subject_col] = frame[subject_col].ffill()
    by_exact: dict[tuple[str, str, str], str] = {}
    by_run: defaultdict[tuple[str, str], list[str]] = defaultdict(list)
    for _, row in frame.iterrows():
        subject = _subject_key(row.get(subject_col))
        run = _lzu_run_key(row.get(run_col))
        start = _seconds_of_day(row.get(start_col))
        onset = _seconds_of_day(row.get(onset_col))
        if not subject or not run or start is None or onset is None:
            continue
        delta = onset - start
        if delta < 0:
            delta += 24 * 3600
        delta_text = f"{float(delta):.1f}"
        token = _time_token(onset)
        by_exact[(subject, run, token)] = delta_text
        by_run[(subject, run)].append(delta_text)
    return {"by_exact": by_exact, "by_run": dict(by_run)}


def _lookup_lzu_onset(index: Mapping[str, Any], subject_id: str, run_id: str, source_result_path: str) -> str:
    subject = _subject_key(subject_id)
    run = _lzu_run_key(run_id)
    token = _extract_time_token(source_result_path)
    if token:
        match = index.get("by_exact", {}).get((subject, run, token))
        if match:
            return match
    candidates = index.get("by_run", {}).get((subject, run), [])
    unique = sorted(set(candidates))
    return unique[0] if len(unique) == 1 else ""


def _lookup_outcome(center: str, subject_id: str, source_quality_file: str, context: Mapping[str, Any]) -> str:
    key = _subject_key(subject_id)
    if center == "lzu":
        return context.get("lzu_outcomes", {}).get(key, "")
    if center == "hup":
        mapped = context.get("hup_outcomes", {}).get(key, "")
        if mapped:
            return mapped
        if source_quality_file.endswith("_S.xlsx"):
            return "S"
        if source_quality_file.endswith("_F.xlsx"):
            return "F"
        return ""
    if center == "multicenter":
        return context.get("multicenter_outcomes", {}).get(key, "")
    if center == "pediatric":
        return context.get("pediatric_outcomes", {}).get(key, "")
    return ""


def _candidate_signal_path(source_result_path: str, metadata: Path) -> tuple[str, str]:
    if not source_result_path:
        return "", "missing_source_signal_path"
    candidate = Path(source_result_path)
    if not candidate.is_absolute():
        candidate = metadata / candidate
    suffix = candidate.suffix.lower()
    if suffix not in SUPPORTED_SIGNAL_SUFFIXES:
        return "", f"unsupported_source_suffix:{suffix or '<none>'}"
    if not candidate.exists():
        return str(candidate), "not_found"
    return str(candidate), "ok"


def _required_missing(row: Mapping[str, str]) -> list[str]:
    missing = [field for field in REQUIRED_MANIFEST_COLUMNS if not _clean(row.get(field))]
    signal_path = _clean(row.get("signal_path"))
    if signal_path:
        path = Path(signal_path)
        if path.suffix.lower() not in SUPPORTED_SIGNAL_SUFFIXES or not path.exists():
            missing.append("signal_path")
    return sorted(set(missing), key=lambda field: REQUIRED_MANIFEST_COLUMNS.index(field))


def _summarize_rows(
    rows: Sequence[Mapping[str, str]],
    keep_rows: Sequence[Mapping[str, str]],
    strict_rows: Sequence[Mapping[str, str]],
    quality_paths: Sequence[Path],
) -> dict[str, Any]:
    ratings = Counter(row.get("quality_rating", "") for row in rows)
    return {
        "quality_files": [str(path) for path in quality_paths],
        "quality_files_found": [str(path) for path in quality_paths if path.exists()],
        "all_draft_rows": len(rows),
        "good_review_draft_rows": len(keep_rows),
        "strict_pipeline_rows": len(strict_rows),
        "quality_ratings": dict(sorted(ratings.items())),
        "missing_required_counts": _missing_counts(rows),
        "signal_path_status_counts": dict(sorted(Counter(row.get("signal_path_status", "") for row in rows).items())),
    }


def _missing_counts(rows: Iterable[Mapping[str, str]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        for field in _clean(row.get("missing_fields")).split(","):
            if field:
                counts[field] += 1
    return dict(sorted(counts.items()))


def _write_csv(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=list(OUTPUT_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in OUTPUT_COLUMNS})


def _clean(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return "" if text.lower() == "nan" else re.sub(r"\s+", " ", text)


def _subject_key(value: Any) -> str:
    text = _clean(value).lower()
    text = re.sub(r"^sub[-_]*", "", text)
    return text


def _run_key(value: Any) -> str:
    return _clean(value).lower()


def _path_key(value: Any) -> str:
    return _clean(value).replace("/", "\\").lower()


def _lzu_run_key(value: Any) -> str:
    text = _clean(value).lower().replace("'", "")
    text = re.split(r"[\(\uff08]", text)[0].strip()
    text = re.sub(r"_?onset$", "", text)
    return text


def _ez_label(value: Any) -> str:
    text = _clean(value).upper()
    if text == "EZ":
        return "1"
    if text == "NEZ":
        return "0"
    return "-1"


def _format_int_label(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "-1"
    if math.isnan(numeric):
        return "-1"
    return str(int(numeric))


def _canonical_outcome(value: Any) -> str:
    text = _clean(value).upper()
    if not text:
        return ""
    if text in {"S", "SUCCESS", "SUCCESSFUL", "\u6210\u529f", "ENGELI", "ENGEL I", "I", "1", "TRUE"}:
        return "S"
    if text in {"F", "FAIL", "FAILED", "FAILURE", "\u5931\u8d25", "ENGELII", "ENGELIII", "ENGELIV", "II", "III", "IV", "0", "FALSE"}:
        return "F"
    if text == "NR":
        return "NR"
    return text


def _first_existing(columns: Iterable[Any], candidates: Sequence[str]) -> Any | None:
    lookup = {str(column).strip().lower(): column for column in columns}
    for candidate in candidates:
        key = candidate.strip().lower()
        if key in lookup:
            return lookup[key]
    return None


def _explicit_numeric(row: Mapping[str, Any], names: Sequence[str]) -> str:
    lowered = {str(key).strip().lower(): key for key in row.keys()}
    for name in names:
        key = lowered.get(name.strip().lower())
        if key is None:
            continue
        value = row.get(key)
        if _clean(value):
            try:
                return f"{float(value):.1f}"
            except (TypeError, ValueError):
                return _clean(value)
    return ""


def _seconds_of_day(value: Any) -> int | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    if isinstance(value, datetime):
        value = value.time()
    if isinstance(value, time):
        return value.hour * 3600 + value.minute * 60 + value.second
    text = _clean(value).replace("\uff1a", ":")
    match = re.search(r"(\d{1,2}):(\d{1,2})(?::(\d{1,2}))?", text)
    if not match:
        return None
    hour = int(match.group(1))
    minute = int(match.group(2))
    second = int(match.group(3) or 0)
    if hour > 23 or minute > 59 or second > 59:
        return None
    return hour * 3600 + minute * 60 + second


def _time_token(seconds: int) -> str:
    hour = seconds // 3600
    minute = (seconds % 3600) // 60
    second = seconds % 60
    return f"{hour:02d}-{minute:02d}-{second:02d}"


def _extract_time_token(value: Any) -> str:
    text = _path_key(value)
    matches = re.findall(r"(\d{1,2})-(\d{2})-(\d{2})(?=[^\\\/]*slope_results)", text)
    if not matches:
        return ""
    hour, minute, second = matches[-1]
    return f"{int(hour):02d}-{int(minute):02d}-{int(second):02d}"
