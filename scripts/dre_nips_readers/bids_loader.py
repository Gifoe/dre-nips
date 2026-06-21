from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .audit import ReadAudit
from .bids_common import (
    bids_run_metadata,
    build_sidecar_index,
    discover_bids_edfs_for_participants,
    find_participants_path,
    is_successful_participant,
    load_participants,
    matches_subject_filter,
    participant_id_for_edf,
    raw_norm_to_name,
    read_bids_channel_table,
    read_ictal_events_from_events,
    read_json_metadata,
    resolve_bids_sidecars,
    successful_participant_ids,
)
from .edf import close_raw, crop_raw_to_preictal_context, finalize_raw_data, read_raw_edf
from .schemas import (
    DataInterfaceConfig,
    PatientRecord,
    SeizureRecord,
    build_patient_records,
    clean_text,
    make_unique,
    resolve_cpu_workers,
    subject_filter_set,
)

try:
    from ..logging_utils import log
except Exception:
    def log(message: str) -> None:
        print(message)


def load_bids_patient_records(
    root: str | Path,
    participants_path: str | Path | None,
    sidecar_root: str | Path | None,
    dataset_name: str,
    cfg: DataInterfaceConfig,
    audit: ReadAudit | None = None,
) -> list[PatientRecord]:
    audit = audit or ReadAudit()
    dataset_name = dataset_name.lower()
    root = Path(root)
    sidecar_root_path = Path(sidecar_root) if sidecar_root is not None else None
    participants_path = participants_path or find_participants_path(root, sidecar_root_path, dataset_name)
    participants = load_participants(participants_path)
    subject_filter = subject_filter_set(cfg.subject_filter, add_sub_prefix=True)
    log(
        f"{dataset_name}: root={root}, participants={participants_path}, "
        f"participants_loaded={len(participants)}, sidecar_root={sidecar_root_path}"
    )

    success_filter_applies = dataset_name in {"hup", "multicenter"} and cfg.success_only
    success_participants = successful_participant_ids(participants) if success_filter_applies else []
    log(f"{dataset_name}: success_filter_applies={success_filter_applies}")
    if success_filter_applies:
        log(f"{dataset_name}: success_participants={len(success_participants)}")
        log(f"{dataset_name}: success_participants_sample={success_participants[:10]}")
    if success_filter_applies and not success_participants:
        if cfg.strict:
            raise FileNotFoundError(f"success_only=True but no successful participants were found for {dataset_name}.")
        return []

    discovery_participants = success_participants if success_filter_applies else sorted(participants) or None
    edf_files, matched_dirs = discover_bids_edfs_for_participants(
        root=root,
        participants=discovery_participants,
        subject_filter=subject_filter,
    )
    log(
        f"{dataset_name}: discovered_edfs={len(edf_files)}, "
        f"matched_participant_dirs={len(matched_dirs)}"
    )
    sidecar_index = build_sidecar_index(sidecar_root_path)
    log(f"{dataset_name}: sidecar_index_files={len(sidecar_index)}")
    seizures: list[SeizureRecord] = []
    per_subject_edfs: dict[str, set[str]] = {}
    per_subject_ictal_runs: dict[str, set[str]] = {}
    interictal_sources_by_subject: dict[str, list[dict[str, Any]]] = {}
    read_jobs: list[dict[str, Any]] = []

    for edf_idx, edf_path in enumerate(edf_files, start=1):
        if edf_idx == 1 or edf_idx % 25 == 0 or edf_idx == len(edf_files):
            log(f"{dataset_name}: scanning EDF {edf_idx}/{len(edf_files)}, loaded_seizures={len(seizures)}")
        subject_id = participant_id_for_edf(edf_path, participants)
        participant_meta = participants.get(subject_id, {})
        if subject_filter is not None and not matches_subject_filter(subject_id, subject_filter):
            continue
        if success_filter_applies and not is_successful_participant(participant_meta):
            continue

        run_meta = bids_run_metadata(edf_path, root)
        run_meta["subject_id"] = subject_id
        run_meta["folder_subject_id"] = run_meta.get("subject_id")
        task = str(run_meta.get("task", "")).lower()
        is_interictal = "interictal" in task or "interictal" in edf_path.name.lower()
        if is_interictal:
            sidecars = resolve_bids_sidecars(edf_path, sidecar_root_path, sidecar_index)
            if sidecars["channels"] is None:
                log(f"{dataset_name}: missing channels.tsv for interictal {edf_path}")
                audit.add(
                    dataset_name,
                    "interictal_source",
                    subject_id=subject_id,
                    skipped=True,
                    skip_reason="missing_channels_tsv",
                    edf_path=str(edf_path),
                )
                if cfg.strict:
                    raise FileNotFoundError(f"Missing channels.tsv for interictal {edf_path}")
                continue
            per_subject_edfs.setdefault(subject_id, set()).add(str(edf_path))
            source = {
                "dataset_name": dataset_name,
                "subject_id": subject_id,
                "edf_path": str(edf_path),
                "channels_path": str(sidecars["channels"]),
                "json_path": str(sidecars.get("json")) if sidecars.get("json") else None,
                "events_path": str(sidecars.get("events")) if sidecars.get("events") else None,
                "run_id": str(run_meta["run_id"]),
                "task": str(run_meta.get("task", "")),
                "participant_id": subject_id,
                "site": participant_meta.get("site"),
                "outcome": participant_meta.get("outcome"),
            }
            interictal_sources_by_subject.setdefault(subject_id, []).append(source)
            audit.add(
                dataset_name,
                "interictal_source",
                subject_id=subject_id,
                skipped=False,
                edf_path=str(edf_path),
                channels_path=str(sidecars["channels"]),
                run_id=str(run_meta["run_id"]),
            )
            continue
        if "ictal" not in task and "sz" not in edf_path.name.lower():
            continue

        sidecars = resolve_bids_sidecars(edf_path, sidecar_root_path, sidecar_index)
        if sidecars["channels"] is None:
            log(f"{dataset_name}: missing channels.tsv for {edf_path}")
            audit.add_skipped_seizure(dataset_name, subject_id, str(run_meta["run_id"]), "missing_channels_tsv", edf_path=str(edf_path))
            if cfg.strict:
                raise FileNotFoundError(f"Missing channels.tsv for {edf_path}")
            continue
        if sidecars["events"] is None:
            log(f"{dataset_name}: missing events.tsv for {edf_path}")
            audit.add_skipped_seizure(dataset_name, subject_id, str(run_meta["run_id"]), "missing_events_tsv", edf_path=str(edf_path), channels_path=str(sidecars["channels"]))
            if cfg.strict:
                raise FileNotFoundError(f"Missing events.tsv for {edf_path}")
            continue

        per_subject_edfs.setdefault(subject_id, set()).add(str(edf_path))
        ictal_events = read_ictal_events_from_events(sidecars["events"])
        if not ictal_events:
            log(f"{dataset_name}: no ictal onset event in {sidecars['events']}")
            audit.add_skipped_seizure(
                dataset_name,
                subject_id,
                str(run_meta["run_id"]),
                "no_ictal_onset_event",
                edf_path=str(edf_path),
                channels_path=str(sidecars["channels"]),
                events_path=str(sidecars["events"]),
            )
            continue
        per_subject_ictal_runs.setdefault(subject_id, set()).add(str(run_meta["run_id"]))
        log(f"{dataset_name}: {edf_path.name} ictal_events={len(ictal_events)}")

        for event in ictal_events:
            read_jobs.append(
                {
                    "edf_path": edf_path,
                    "sidecars": sidecars,
                    "run_meta": dict(run_meta),
                    "participant_meta": dict(participant_meta),
                    "subject_id": subject_id,
                    "dataset_name": dataset_name,
                    "event": dict(event),
                }
            )
            if cfg.debug_limit is not None and len(read_jobs) >= int(cfg.debug_limit):
                break
        if cfg.debug_limit is not None and len(read_jobs) >= int(cfg.debug_limit):
            break

    workers = resolve_cpu_workers(cfg, len(read_jobs))
    log(f"{dataset_name}: EDF read jobs={len(read_jobs)}, workers={workers}")
    read_errors = 0
    if workers <= 1:
        for completed, job in enumerate(read_jobs, start=1):
            seizure, detail = _read_bids_job(job, cfg, audit)
            if detail is not None:
                read_errors += 1
                if cfg.strict:
                    raise RuntimeError(detail["exception_repr"])
            elif seizure is not None:
                seizures.append(seizure)
                log(f"{dataset_name}: loaded {seizure.seizure_id}, channels={len(seizure.channel_names)}")
            if completed == 1 or completed % 10 == 0 or completed == len(read_jobs):
                log(f"{dataset_name}: EDF read progress={completed}/{len(read_jobs)}, loaded_seizures={len(seizures)}, read_errors={read_errors}")
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_read_bids_job, job, cfg, audit) for job in read_jobs]
            for completed, future in enumerate(as_completed(futures), start=1):
                seizure, detail = future.result()
                if detail is not None:
                    read_errors += 1
                    if cfg.strict:
                        raise RuntimeError(detail["exception_repr"])
                elif seizure is not None:
                    seizures.append(seizure)
                    log(f"{dataset_name}: loaded {seizure.seizure_id}, channels={len(seizure.channel_names)}")
                if completed == 1 or completed % 10 == 0 or completed == len(read_jobs):
                    log(f"{dataset_name}: EDF read progress={completed}/{len(read_jobs)}, loaded_seizures={len(seizures)}, read_errors={read_errors}")

    n_interictal_sources = sum(len(sources) for sources in interictal_sources_by_subject.values())
    if n_interictal_sources:
        log(
            f"{dataset_name}: indexed_interictal_sources={n_interictal_sources}, "
            f"subjects_with_interictal={len(interictal_sources_by_subject)}"
        )
    for seizure in seizures:
        sources = list(interictal_sources_by_subject.get(seizure.subject_id, []))
        if sources:
            setattr(seizure, "interictal_sources", sources)

    patients = build_patient_records(seizures)
    for patient in patients:
        subject_interictal_sources = list(interictal_sources_by_subject.get(patient.subject_id, []))
        for seizure in patient.seizures:
            if subject_interictal_sources:
                setattr(seizure, "interictal_sources", subject_interictal_sources)
            for meta in seizure.channel_meta:
                meta.setdefault("n_edf_subject", len(per_subject_edfs.get(patient.subject_id, set())))
                meta.setdefault("n_ictal_runs_subject", len(per_subject_ictal_runs.get(patient.subject_id, set())))
                meta.setdefault("n_interictal_runs_subject", len(subject_interictal_sources))
                if subject_interictal_sources:
                    meta.setdefault(
                        "interictal_source_paths",
                        ";".join(str(source.get("edf_path", "")) for source in subject_interictal_sources),
                    )
    kept = audit.validate_and_filter(dataset_name, patients, strict=cfg.strict)
    log(
        f"{dataset_name}: built_patients={len(patients)}, kept_trainable_patients={len(kept)}, "
        f"seizures={len(seizures)}, read_errors={read_errors}"
    )
    return kept


def _read_bids_job(
    job: Mapping[str, Any],
    cfg: DataInterfaceConfig,
    audit: ReadAudit,
) -> tuple[SeizureRecord | None, dict[str, Any] | None]:
    edf_path = Path(job["edf_path"])
    sidecars = job["sidecars"]
    run_meta = job["run_meta"]
    event = job["event"]
    subject_id = str(job["subject_id"])
    dataset_name = str(job["dataset_name"])
    try:
        seizure = _read_bids_seizure(
            edf_path=edf_path,
            sidecars=sidecars,
            run_meta=run_meta,
            participant_meta=job["participant_meta"],
            dataset_name=dataset_name,
            event=event,
            cfg=cfg,
            audit=audit,
        )
        return seizure, None
    except Exception as exc:
        detail = {
            "edf_path": str(edf_path),
            "channels_path": str(sidecars["channels"]) if sidecars.get("channels") else None,
            "events_path": str(sidecars["events"]) if sidecars.get("events") else None,
            "event_index": event.get("event_index"),
            "exception_type": type(exc).__name__,
            "exception_repr": repr(exc),
        }
        audit.add_skipped_seizure(
            dataset_name,
            subject_id,
            _event_seizure_id(run_meta, int(event["event_index"])),
            repr(exc),
            **detail,
        )
        return None, detail


def _read_bids_seizure(
    edf_path: Path,
    sidecars: Mapping[str, Path | None],
    run_meta: Mapping[str, Any],
    participant_meta: Mapping[str, Any],
    dataset_name: str,
    event: Mapping[str, Any],
    cfg: DataInterfaceConfig,
    audit: ReadAudit,
) -> SeizureRecord | None:
    channels_table = read_bids_channel_table(Path(sidecars["channels"]), dataset_name, cfg.ez_definition)
    subject_id = str(run_meta["subject_id"])
    run_id = str(run_meta["run_id"])
    seizure_id = _event_seizure_id(run_meta, int(event["event_index"]))
    for _, row in channels_table.iterrows():
        audit.add_channel(
            dataset_name,
            subject_id,
            seizure_id,
            channel_name_orig=row.get("channel_name_orig"),
            channel_name_norm=row.get("channel_name_norm"),
            type=row.get("type"),
            status=row.get("status"),
            good=row.get("good"),
            soz=row.get("soz"),
            resection=row.get("resection"),
            status_description=row.get("status_description"),
            is_valid=bool(row.get("is_valid")),
            is_ez_or_soz=bool(row.get("is_ez_or_soz")),
            final_label=row.get("final_label") if bool(row.get("is_valid")) else None,
        )

    valid_channels = channels_table[channels_table["is_valid"].eq(1)].copy()
    if valid_channels.empty:
        audit.add_skipped_seizure(dataset_name, subject_id, seizure_id, "no_valid_channels", edf_path=str(edf_path))
        return None

    raw = read_raw_edf(edf_path, preload=False)
    try:
        raw_duration = float(raw.n_times) / float(raw.info["sfreq"])
        onset = float(event["onset"])
        offset = float(event["offset"]) if event.get("offset") is not None else None
        onset_valid = 0.0 <= onset <= raw_duration
        offset_valid = offset is None or 0.0 <= offset <= raw_duration
        if not onset_valid:
            audit.add_skipped_seizure(
                dataset_name,
                subject_id,
                seizure_id,
                "onset_out_of_range",
                edf_path=str(edf_path),
                channels_path=str(sidecars["channels"]),
                events_path=str(sidecars["events"]),
                onset_sec=onset,
                offset_sec=offset,
                raw_duration_sec=raw_duration,
                onset_valid=onset_valid,
                offset_valid=offset_valid,
                event_index=event.get("event_index"),
                event_trial_type_used=event.get("event_trial_type_used"),
            )
            return None

        raw_names = raw_norm_to_name(raw.ch_names)
        picked_rows = []
        picked_raw_names = []
        unmatched_channels: list[str] = []
        for _, channel_row in valid_channels.iterrows():
            norm = str(channel_row["channel_name_norm"])
            raw_name = raw_names.get(norm)
            if raw_name is None:
                unmatched_channels.append(norm)
                continue
            picked_rows.append(channel_row)
            picked_raw_names.append(raw_name)
        if not picked_rows:
            audit.add_skipped_seizure(dataset_name, subject_id, seizure_id, "no_channels_matched_edf", edf_path=str(edf_path))
            return None
        picked_meta = pd.DataFrame(picked_rows).reset_index(drop=True)
        final_names = make_unique(picked_meta["channel_name_norm"].astype(str).tolist())
        crop_start = crop_raw_to_preictal_context(raw, float(event["onset"]))
        data, sfreq, channel_names, sfreq_original = finalize_raw_data(raw, picked_raw_names, final_names, cfg)
    finally:
        close_raw(raw)

    adjusted_onset = float(event["onset"]) - float(crop_start)
    adjusted_offset = float(event["offset"]) - float(crop_start) if event.get("offset") is not None else None
    json_meta = read_json_metadata(sidecars.get("json"))
    split = clean_text(participant_meta.get("split")) or None
    labels = picked_meta["final_label"].to_numpy(dtype=np.float32, copy=True)
    channel_meta = []
    for row, final_name in zip(picked_rows, channel_names):
        meta = dict(row)
        meta.update(
            {
                "channel_name_norm": final_name,
                "dataset": dataset_name,
                "source_path": str(edf_path),
                "run_id": run_id,
                "events_path": str(sidecars.get("events")) if sidecars.get("events") else None,
                "channels_path": str(sidecars.get("channels")) if sidecars.get("channels") else None,
                "site": participant_meta.get("site"),
                "participant_id": subject_id,
                "outcome": participant_meta.get("outcome"),
                "success_used": is_successful_participant(participant_meta)
                if dataset_name in {"hup", "multicenter"} else None,
                "engel_score": participant_meta.get("engel_score", participant_meta.get("engel")),
                "json_sampling_frequency": json_meta.get("SamplingFrequency"),
                "raw_duration_sec": raw_duration,
                "preictal_crop_start_sec": crop_start,
                "original_seizure_onset_sec": float(event["onset"]),
                "original_seizure_offset_sec": float(event["offset"]) if event.get("offset") is not None else None,
                "sfreq_original": sfreq_original,
                "event_index": int(event["event_index"]),
                "event_trial_type_used": event.get("event_trial_type_used"),
                "offset_trial_type_used": event.get("offset_trial_type_used"),
                "unmatched_channels": unmatched_channels,
            }
        )
        channel_meta.append(meta)

    return SeizureRecord(
        subject_id=subject_id,
        seizure_id=seizure_id,
        signal=data,
        sfreq=sfreq,
        channel_names=channel_names,
        seizure_onset_sec=adjusted_onset,
        seizure_offset_sec=adjusted_offset,
        labels=labels,
        channel_meta=channel_meta,
        split=split,
    )


def _event_seizure_id(run_meta: Mapping[str, Any], event_index: int) -> str:
    return f"{run_meta['run_id']}__event-{event_index:02d}"
