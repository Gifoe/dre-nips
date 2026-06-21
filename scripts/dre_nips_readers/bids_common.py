from __future__ import annotations

from pathlib import Path
import json
import re
from typing import Any, Mapping, Sequence

import pandas as pd

from .encoding import read_json_with_fallback, read_tsv_with_fallback
from .schemas import (
    DEFAULT_MULTICENTER_PARTICIPANTS,
    INTRACRANIAL_TYPES,
    as_binary,
    clean_text,
    is_successful_surgery_value,
    is_excluded_channel_name,
    normalize_channel_name,
    parse_contact_topology,
    safe_float,
    subject_key,
)


def load_participants(path: str | Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    df = read_tsv_with_fallback(path)
    if "participant_id" not in df.columns:
        return {}
    participants: dict[str, dict] = {}
    for _, row in df.iterrows():
        participant_id = clean_text(row.get("participant_id"))
        if participant_id and not participant_id.startswith("sub-"):
            participant_id = f"sub-{participant_id}"
        if participant_id:
            participants[participant_id] = {str(col): row.get(col) for col in df.columns}
    return participants


def find_participants_path(root: Path, sidecar_root: Path | None, dataset_name: str) -> Path | None:
    candidates: list[Path] = []
    if dataset_name.lower() == "multicenter":
        candidates.extend([root / "participants-muticenter.tsv", root / "participants.tsv", DEFAULT_MULTICENTER_PARTICIPANTS])
        if sidecar_root is not None:
            candidates.extend([sidecar_root / "participants-muticenter.tsv", sidecar_root / "participants.tsv"])
    else:
        candidates.extend([root / "participants.tsv", root / "participants-hup.tsv", root / "participants_HUP.tsv"])
    return next((candidate for candidate in candidates if candidate.exists()), None)


def is_successful_participant(participant_meta: Mapping[str, Any]) -> bool:
    if not participant_meta:
        return False
    outcome_success = is_successful_surgery_value(participant_meta.get("outcome"))
    if outcome_success is not None:
        return outcome_success
    for key in ("outcome_binary", "success", "surgery_success", "seizure_free"):
        binary = as_binary(participant_meta.get(key))
        if binary is not None:
            return bool(binary)
    engel_success = is_successful_surgery_value(
        participant_meta.get("engel_score", participant_meta.get("engel"))
    )
    if engel_success is not None:
        return engel_success
    ilae = safe_float(participant_meta.get("ilae_score", participant_meta.get("ilae")))
    if ilae is not None:
        return ilae == 1.0
    return False


def discover_edf_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    files = sorted(root.rglob("*_ieeg.edf"))
    return files if files else sorted(root.rglob("*.edf"))


def discover_bids_edfs_for_participants(
    root: Path,
    participants: Sequence[str] | None,
    subject_filter: set[str] | None,
) -> tuple[list[Path], dict[str, list[Path]]]:
    if participants is None:
        return discover_edf_files(root), {}
    if not root.exists():
        return [], {}

    subject_dirs = [path for path in root.iterdir() if path.is_dir() and path.name.startswith("sub-")]
    dir_by_key: dict[str, list[Path]] = {}
    for subject_dir in subject_dirs:
        for alias in participant_aliases(subject_dir.name):
            dir_by_key.setdefault(subject_key(alias), []).append(subject_dir)

    matched_dirs: dict[str, list[Path]] = {}
    edf_files: list[Path] = []
    seen_edfs: set[Path] = set()
    for participant_id in participants:
        if subject_filter is not None and not matches_subject_filter(participant_id, subject_filter):
            continue
        candidate_dirs: list[Path] = []
        for alias in participant_aliases(participant_id):
            candidate_dirs.extend(dir_by_key.get(subject_key(alias), []))
        unique_dirs = sorted(set(candidate_dirs), key=lambda path: path.name)
        if not unique_dirs:
            continue
        matched_dirs[participant_id] = unique_dirs
        for subject_dir in unique_dirs:
            for edf_path in discover_edf_files(subject_dir):
                if edf_path not in seen_edfs:
                    seen_edfs.add(edf_path)
                    edf_files.append(edf_path)
    return sorted(edf_files), matched_dirs


def subject_id_from_bids_path(edf_path: Path) -> str:
    for part in edf_path.parts:
        if part.startswith("sub-"):
            return part
    match = re.search(r"(sub-[A-Za-z0-9]+)", edf_path.name)
    return match.group(1) if match else edf_path.parent.name


def subject_id_from_bids_filename(edf_path: Path) -> str | None:
    match = re.match(r"(sub-[A-Za-z0-9]+)_", edf_path.name)
    return match.group(1) if match else None


def participant_aliases(participant_id: str) -> set[str]:
    raw = clean_text(participant_id)
    aliases = {raw}
    if not raw:
        return aliases
    body = raw[4:] if raw.startswith("sub-") else raw
    aliases.add(f"sub-{body}")
    lower = body.lower()
    mc_match = re.fullmatch(r"multicenterpt0*([0-9]+)", lower)
    if mc_match:
        number = int(mc_match.group(1))
        aliases.update({f"sub-pt{number}", f"sub-pt{number:02d}", f"sub-Multicenterpt{number}", f"sub-Multicenterpt{number:02d}"})
    pt_match = re.fullmatch(r"pt0*([0-9]+)", lower)
    if pt_match:
        number = int(pt_match.group(1))
        aliases.update({f"sub-pt{number}", f"sub-pt{number:02d}", f"sub-Multicenterpt{number}", f"sub-Multicenterpt{number:02d}"})
    hup_match = re.fullmatch(r"hup0*([0-9]+)", lower)
    if hup_match:
        number = int(hup_match.group(1))
        aliases.update({f"sub-HUP{number}", f"sub-HUP{number:03d}"})
    return {alias for alias in aliases if alias}


def matches_subject_filter(subject_id: str, subject_filter: set[str]) -> bool:
    for alias in participant_aliases(subject_id):
        if alias in subject_filter or subject_key(alias) in subject_filter:
            return True
    return subject_id in subject_filter or subject_key(subject_id) in subject_filter


def participant_id_for_edf(edf_path: Path, participants: Mapping[str, Mapping[str, Any]]) -> str:
    folder_subject_id = subject_id_from_bids_path(edf_path)
    filename_subject_id = subject_id_from_bids_filename(edf_path)
    for subject_id in [folder_subject_id, filename_subject_id]:
        if subject_id and subject_id in participants:
            return subject_id
    candidate_alias_keys = set()
    for subject_id in [folder_subject_id, filename_subject_id]:
        if subject_id:
            candidate_alias_keys.update(subject_key(alias) for alias in participant_aliases(subject_id))
    for participant_id in participants:
        if any(subject_key(alias) in candidate_alias_keys for alias in participant_aliases(participant_id)):
            return participant_id
    return folder_subject_id


def successful_participant_ids(participants: Mapping[str, Mapping[str, Any]]) -> list[str]:
    return sorted([subject_id for subject_id, meta in participants.items() if is_successful_participant(meta)])


def bids_run_metadata(edf_path: Path, root: Path) -> dict:
    stem = edf_path.name.replace("_ieeg.edf", "").replace(".edf", "")
    subject_id = subject_id_from_bids_path(edf_path)
    session = regex_group(stem, r"ses-([A-Za-z0-9]+)")
    task = regex_group(stem, r"task-([A-Za-z0-9]+)") or "unknown"
    acquisition = regex_group(stem, r"acq-([A-Za-z0-9]+)")
    run = regex_group(stem, r"run-([0-9]+)") or "01"
    run_id = f"{subject_id}__ses-{session}__task-{task}__run-{run}" if session else f"{subject_id}__task-{task}__run-{run}"
    try:
        relative_path = str(edf_path.relative_to(root))
    except ValueError:
        relative_path = str(edf_path)
    return {
        "subject_id": subject_id,
        "session": session,
        "task": task,
        "acquisition": acquisition,
        "run": run,
        "run_id": run_id,
        "relative_path": relative_path,
    }


def regex_group(text: str, pattern: str) -> str | None:
    match = re.search(pattern, text)
    return match.group(1) if match else None


def build_sidecar_index(sidecar_root: Path | None) -> dict[str, Path]:
    index: dict[str, Path] = {}
    if sidecar_root is None or not sidecar_root.exists():
        return index
    for path in sidecar_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".tsv", ".json"}:
            index.setdefault(path.name, path)
    return index


def resolve_bids_sidecars(edf_path: Path, sidecar_root: Path | None, sidecar_index: Mapping[str, Path]) -> dict[str, Path | None]:
    stem = edf_path.name.replace("_ieeg.edf", "").replace(".edf", "")
    names = {"channels": f"{stem}_channels.tsv", "events": f"{stem}_events.tsv", "json": f"{stem}_ieeg.json"}
    sidecars: dict[str, Path | None] = {}
    for key, name in names.items():
        local = edf_path.parent / name
        if local.exists():
            sidecars[key] = local
            continue
        run_matched = find_sidecar_by_run_identity(edf_path, sidecar_root, key)
        if run_matched is not None:
            sidecars[key] = run_matched
            continue
        indexed = sidecar_index.get(name)
        if indexed is not None and indexed.exists():
            sidecars[key] = indexed
            continue
        sidecars[key] = find_sidecar_by_relative_subject_path(edf_path, sidecar_root, name)
    return sidecars


def find_sidecar_by_run_identity(edf_path: Path, sidecar_root: Path | None, key: str) -> Path | None:
    suffix = {"channels": "channels.tsv", "events": "events.tsv", "json": "ieeg.json"}[key]
    stem = edf_path.name.replace("_ieeg.edf", "").replace(".edf", "")
    session = regex_group(stem, r"ses-([A-Za-z0-9]+)")
    task = regex_group(stem, r"task-([A-Za-z0-9]+)")
    acquisition = regex_group(stem, r"acq-([A-Za-z0-9]+)")
    run = regex_group(stem, r"run-([0-9]+)")
    if run is None:
        return None
    patterns = []
    if session and task and acquisition:
        patterns.append(f"*ses-{session}_task-{task}_acq-{acquisition}_run-{run}_{suffix}")
    if task and acquisition:
        patterns.append(f"*task-{task}_acq-{acquisition}_run-{run}_{suffix}")
    if task:
        patterns.append(f"*task-{task}*_run-{run}_{suffix}")
    patterns.append(f"*run-{run}_{suffix}")
    candidate_dirs = [edf_path.parent]
    if sidecar_root is not None and sidecar_root.exists():
        subject_id = subject_id_from_bids_path(edf_path)
        subject_idx = next((idx for idx, part in enumerate(edf_path.parts) if part == subject_id), None)
        if subject_idx is not None:
            candidate_dirs.append(sidecar_root / Path(*edf_path.parts[subject_idx:-1]))
    candidates: list[Path] = []
    for candidate_dir in candidate_dirs:
        if candidate_dir.exists():
            for pattern in patterns:
                candidates.extend(path for path in candidate_dir.glob(pattern) if path.is_file())
    return choose_best_sidecar_candidate(candidates, edf_path) if candidates else None


def choose_best_sidecar_candidate(candidates: Sequence[Path], edf_path: Path) -> Path:
    subject_id = subject_id_from_bids_path(edf_path)

    def score(path: Path) -> tuple[int, int, str]:
        name = path.name
        subject_score = 2 if name.startswith(f"{subject_id}_") else 1 if subject_id in name else 0
        same_dir_score = 1 if path.parent == edf_path.parent else 0
        return (-subject_score, -same_dir_score, name)

    return sorted(set(candidates), key=score)[0]


def find_sidecar_by_relative_subject_path(edf_path: Path, sidecar_root: Path | None, name: str) -> Path | None:
    if sidecar_root is None or not sidecar_root.exists():
        return None
    subject_id = subject_id_from_bids_path(edf_path)
    subject_idx = next((idx for idx, part in enumerate(edf_path.parts) if part == subject_id), None)
    if subject_idx is None:
        return None
    candidate = sidecar_root / Path(*edf_path.parts[subject_idx:-1]) / name
    return candidate if candidate.exists() else None


def read_json_metadata(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    return read_json_with_fallback(path)


def read_ictal_events_from_events(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    events = read_tsv_with_fallback(path)
    if events.empty or "onset" not in events.columns:
        return []
    events = events.copy()
    events["_onset_num"] = pd.to_numeric(events["onset"], errors="coerce")
    events = events[events["_onset_num"].notna()].sort_values("_onset_num").reset_index(drop=True)
    if events.empty:
        return []

    text = _event_text(events)
    onset_candidates = _onset_candidates(events, text)
    if not onset_candidates:
        return []
    offset_candidates = _offset_candidates(events, text)

    ictal_events: list[dict[str, Any]] = []
    for event_index, onset_candidate in enumerate(onset_candidates):
        onset = float(onset_candidate["onset"])
        next_onset = float(onset_candidates[event_index + 1]["onset"]) if event_index + 1 < len(onset_candidates) else None
        offset = _choose_offset(onset, next_onset, offset_candidates)
        ictal_events.append(
            {
                "event_index": event_index,
                "onset": onset,
                "offset": offset["onset"] if offset is not None else None,
                "event_trial_type_used": onset_candidate["text"],
                "offset_trial_type_used": offset["text"] if offset is not None else None,
                "onset_priority": onset_candidate["priority"],
                "offset_priority": offset["priority"] if offset is not None else None,
            }
        )
    return ictal_events


def read_ictal_bounds_from_events(path: Path | None) -> tuple[float | None, float | None]:
    events = read_ictal_events_from_events(path)
    if not events:
        return None, None
    return events[0]["onset"], events[0]["offset"]


def _event_text(events: pd.DataFrame) -> pd.Series:
    cols = [col for col in ["trial_type", "description", "event", "type", "name", "event_type"] if col in events.columns]
    if not cols:
        return pd.Series("", index=events.index)
    return events[cols].fillna("").astype(str).agg(" ".join, axis=1).str.strip().str.lower()


def _onset_candidates(events: pd.DataFrame, text: pd.Series) -> list[dict[str, Any]]:
    exact_onset = text.eq("onset")
    exact_start = text.isin({"seizure onset", "sz onset", "sz start", "seizure start"})
    contains_onset = text.str.contains("onset", na=False)
    fallback = text.isin({"sz", "seizure", "ictal"}) | text.str.contains(r"\b(?:sz|seizure|ictal)\b", regex=True, na=False)
    masks = [(1, exact_onset), (2, exact_start), (3, contains_onset)]
    if not any(mask.any() for _, mask in masks):
        masks.append((4, fallback))
    return _candidate_rows(events, text, masks)


def _offset_candidates(events: pd.DataFrame, text: pd.Series) -> list[dict[str, Any]]:
    exact_sz_end = text.eq("sz end")
    exact_seizure_end = text.eq("seizure end")
    exact_seizure_off = text.eq("seizure off")
    exact_offset = text.isin({"sz offset", "seizure offset", "offset", "end", "stop"})
    contains_end = text.str.contains(r"end|offset|off|stop", regex=True, na=False) & text.str.contains(
        r"sz|seiz|ictal|electrographic", regex=True, na=False
    )
    return _candidate_rows(
        events,
        text,
        [(1, exact_sz_end), (2, exact_seizure_end), (3, exact_seizure_off), (4, exact_offset), (5, contains_end)],
    )


def _candidate_rows(events: pd.DataFrame, text: pd.Series, masks: Sequence[tuple[int, pd.Series]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    for priority, mask in masks:
        for idx in events.index[mask]:
            if int(idx) in seen:
                continue
            seen.add(int(idx))
            rows.append({"row_index": int(idx), "onset": float(events.loc[idx, "_onset_num"]), "text": str(text.loc[idx]), "priority": priority})
    return sorted(rows, key=lambda row: (row["onset"], row["priority"], row["row_index"]))


def _choose_offset(onset: float, next_onset: float | None, candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    valid = [c for c in candidates if float(c["onset"]) >= onset and (next_onset is None or float(c["onset"]) < next_onset)]
    if not valid:
        return None
    return sorted(valid, key=lambda row: (int(row["priority"]), float(row["onset"]), int(row["row_index"])))[0]


def read_bids_channel_table(channels_path: Path, dataset_name: str, ez_definition: str) -> pd.DataFrame:
    channels = read_tsv_with_fallback(channels_path)
    if "name" not in channels.columns:
        raise ValueError(f"channels.tsv missing name column: {channels_path}")
    lower_cols = {str(col).lower(): col for col in channels.columns}
    if dataset_name == "multicenter" and "soz" not in lower_cols:
        raise ValueError(f"multicenter channels.tsv missing required soz column: {channels_path}")

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for order, row in channels.iterrows():
        orig = clean_text(row.get(lower_cols.get("name", "name")))
        norm = normalize_channel_name(orig)
        duplicate = norm in seen
        if norm:
            seen.add(norm)
        channel_type = clean_text(row.get(lower_cols.get("type", "type"), "SEEG")).upper() or "SEEG"
        status = clean_text(row.get(lower_cols.get("status", "status"), "good")).lower() or "good"
        status_desc = clean_text(row.get(lower_cols.get("status_description", "status_description"), "")).lower()
        good_value = _label_from_columns(row, ("good",))
        is_good = bool(good_value) if good_value is not None else status != "bad"
        is_valid = (
            bool(norm)
            and not duplicate
            and channel_type in INTRACRANIAL_TYPES
            and status != "bad"
            and is_good
            and not is_excluded_channel_name(orig)
        )

        soz = _label_from_columns(row, ("soz", "seizure_onset_zone", "onset_zone", "is_soz"))
        resection = _label_from_columns(row, ("resection", "resected", "is_resected"))
        status_ez = "soz" in status_desc or "seizure onset" in status_desc or "resect" in status_desc
        if dataset_name == "multicenter":
            is_ez = int(bool(soz == 1 or (ez_definition == "soz_or_resected" and resection == 1)))
            label_source = "channels.tsv:soz" if ez_definition != "soz_or_resected" else "channels.tsv:soz_or_resection"
        else:
            is_ez = int(bool(soz == 1 or resection == 1 or status_ez))
            label_source = "channels.tsv:soz/resection/status_description"
        final_label = 0.0 if is_ez else 1.0
        group, number = parse_contact_topology(norm)
        records.append(
            {
                "channel_name_orig": orig,
                "channel_name_norm": norm,
                "type": channel_type,
                "status": status,
                "good": good_value,
                "soz": soz,
                "resection": resection,
                "status_description": status_desc,
                "is_valid": int(is_valid),
                "is_ez_or_soz": int(is_ez),
                "final_label": final_label,
                "label_source": label_source,
                "contact_group": group,
                "contact_number": number,
                "channel_order": int(order),
            }
        )
    return pd.DataFrame(records)


def raw_norm_to_name(raw_names: Sequence[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for raw_name in raw_names:
        mapping.setdefault(normalize_channel_name(raw_name), raw_name)
    return mapping


def _label_from_columns(row: pd.Series, names: Sequence[str]) -> int | None:
    lower_to_col = {str(col).lower(): col for col in row.index}
    for name in names:
        col = lower_to_col.get(name.lower())
        if col is not None:
            value = as_binary(row.get(col))
            if value is not None:
                return value
    return None
