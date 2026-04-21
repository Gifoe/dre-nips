import numpy as np
import pandas as pd


def _safe_float(value):
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(value_f):
        return None
    return value_f


def _is_ictal_run(task: str, phase_group: str) -> bool:
    phase_group_lower = str(phase_group).lower()
    if "interictal" in phase_group_lower:
        return False
    if "ictal" in phase_group_lower:
        return True

    task_lower = str(task).lower()
    return "ictal" in task_lower and "interictal" not in task_lower


def assign_analysis_phases(windows_df: pd.DataFrame, run_info) -> pd.DataFrame:
    windows_df = windows_df.copy()
    phase_group = str(run_info.get("phase_group", "")).strip().lower()
    raw_phase = windows_df["phase"].astype(str).str.strip().str.lower()

    analysis_phase = np.full(len(windows_df), "other", dtype=object)
    if phase_group == "interictal":
        analysis_phase[:] = "interictal"
    else:
        analysis_phase[raw_phase.eq("ictal").to_numpy()] = "ictal"
        analysis_phase[raw_phase.eq("interictal").to_numpy()] = "interictal"

        if phase_group == "ictal":
            has_explicit_ictal = np.any(analysis_phase == "ictal")
            all_unknown_ictal = len(windows_df) > 0 and raw_phase.isin(["unknown_ictal"]).all()
            if not has_explicit_ictal and all_unknown_ictal:
                analysis_phase[:] = "ictal"

    windows_df["analysis_phase"] = analysis_phase
    return windows_df


def _compute_window_artifact_stats(
    win_data,
    sfreq,
    *,
    subseg_sec=1.0,
    amplitude_uv_threshold=2000.0,
    bad_subseg_channel_ratio_threshold=0.25,
    bad_channel_subseg_ratio_threshold=0.5,
):
    n_channels, n_samples = win_data.shape
    if n_channels == 0 or n_samples == 0:
        return {"bad_segment_ratio": 1.0, "bad_channel_ratio": 1.0, "n_subsegments": 0}

    subseg_len = max(1, int(round(float(subseg_sec) * float(sfreq))))
    amp_threshold = float(amplitude_uv_threshold) * 1e-6

    bad_segment_flags = []
    bad_channel_hits = np.zeros(n_channels, dtype=np.int32)

    for start in range(0, n_samples, subseg_len):
        end = min(start + subseg_len, n_samples)
        segment = win_data[:, start:end]
        if segment.shape[1] == 0:
            continue

        amplitude_range = np.max(segment, axis=1) - np.min(segment, axis=1)
        flatline = np.all(np.isclose(segment, segment[:, :1], atol=1e-12), axis=1)
        bad_channels = (amplitude_range > amp_threshold) | flatline

        bad_channel_hits += bad_channels.astype(np.int32)
        bad_segment_flags.append(bad_channels.mean() > float(bad_subseg_channel_ratio_threshold))

    n_subsegments = len(bad_segment_flags)
    if n_subsegments == 0:
        return {"bad_segment_ratio": 1.0, "bad_channel_ratio": 1.0, "n_subsegments": 0}

    bad_segment_ratio = float(np.mean(bad_segment_flags))
    channel_bad_subseg_ratio = bad_channel_hits / float(n_subsegments)
    bad_channel_ratio = float(np.mean(channel_bad_subseg_ratio > float(bad_channel_subseg_ratio_threshold)))

    return {
        "bad_segment_ratio": bad_segment_ratio,
        "bad_channel_ratio": bad_channel_ratio,
        "n_subsegments": n_subsegments,
    }


def create_time_windows(
    raw,
    data,
    run_info,
    win_len_sec=8.0,
    step_sec=4.0,
    artifact_subseg_sec=1.0,
    bad_subseg_channel_ratio_threshold=0.25,
    bad_segment_ratio_threshold=0.40,
    bad_channel_ratio_threshold=0.20,
    bad_channel_subseg_ratio_threshold=0.50,
):
    sfreq = raw.info["sfreq"]
    total_samples = raw.n_times

    windows = []
    subject_id = run_info.get("subject_id", "unknown")
    run_id = run_info["run_id"]
    task = run_info["task"]
    phase_group = run_info.get("phase_group", "unknown")
    onset = _safe_float(run_info.get("seizure_onset"))
    offset = _safe_float(run_info.get("seizure_offset"))

    win_len_samples = int(win_len_sec * sfreq)
    step_samples = int(step_sec * sfreq)

    start_idx = 0
    window_index = 0
    while start_idx + win_len_samples <= total_samples:
        end_idx = start_idx + win_len_samples
        start_t = start_idx / sfreq
        end_t = end_idx / sfreq

        phase = "interictal"
        if _is_ictal_run(task, phase_group):
            if onset is not None and offset is not None:
                if end_t <= onset:
                    phase = "preictal" if (onset - end_t) <= 60 else "interictal"
                elif start_t >= onset and end_t <= offset:
                    phase = "ictal"
                elif start_t >= offset:
                    phase = "postictal" if (start_t - offset) <= 60 else "interictal"
                else:
                    phase = "transition"
            else:
                phase = "unknown_ictal"

        win_data = data[:, start_idx:end_idx]
        artifact_stats = _compute_window_artifact_stats(
            win_data,
            sfreq,
            subseg_sec=artifact_subseg_sec,
            bad_subseg_channel_ratio_threshold=bad_subseg_channel_ratio_threshold,
            bad_channel_subseg_ratio_threshold=bad_channel_subseg_ratio_threshold,
        )
        unusable = (
            artifact_stats["bad_segment_ratio"] > float(bad_segment_ratio_threshold)
            and artifact_stats["bad_channel_ratio"] > float(bad_channel_ratio_threshold)
        )

        windows.append(
            {
                "subject_id": subject_id,
                "run_id": run_id,
                "task": task,
                "phase_group": phase_group,
                "window_index": window_index,
                "start_sec": start_t,
                "end_sec": end_t,
                "phase": phase,
                "unusable_mask": unusable,
                "bad_segment_ratio": artifact_stats["bad_segment_ratio"],
                "bad_channel_ratio": artifact_stats["bad_channel_ratio"],
                "artifact_subsegments": artifact_stats["n_subsegments"],
                "start_sample": start_idx,
                "end_sample": end_idx,
            }
        )

        start_idx += step_samples
        window_index += 1

    windows_df = pd.DataFrame(windows)
    return assign_analysis_phases(windows_df, run_info)
