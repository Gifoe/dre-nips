import re

import pandas as pd

from module2_preprocessing import normalize_channel_name


def _has_non_na_status_label(status_description: str) -> bool:
    normalized = str(status_description).strip().lower()
    return normalized not in {"", "n/a", "na", "nan", "none", "unknown"}


def parse_contact_topology(channel_name_norm: str):
    match = re.match(r"^([A-Z]+)(\d+)$", channel_name_norm)
    if match:
        return match.group(1), int(match.group(2))
    return channel_name_norm, None


def parse_channel_labels(channels_path, ez_definition="any_non_na_status"):
    channels_df = pd.read_csv(channels_path, sep="\t")
    meta_records = []
    seen_norms = set()

    for _, row in channels_df.iterrows():
        channel_name_orig = str(row["name"])
        channel_name_norm = normalize_channel_name(channel_name_orig)

        if channel_name_norm in seen_norms:
            print(f"Warning: Duplicate normalized channel {channel_name_norm} in {channels_path}.")
            continue
        seen_norms.add(channel_name_norm)

        channel_type = str(row["type"]).upper()
        status = str(row.get("status", "good")).strip().lower()
        status_desc = str(row.get("status_description", "")).strip().lower()

        is_soz = 1 if "soz" in status_desc else 0
        is_resected = 1 if "resect" in status_desc else 0
        has_non_na_label = 1 if _has_non_na_status_label(status_desc) else 0

        if ez_definition in {"any_non_na_status", "non_na"}:
            is_ez = has_non_na_label
        elif ez_definition == "soz_or_resected":
            is_ez = 1 if (is_soz or is_resected) else 0
        elif ez_definition == "soz_only":
            is_ez = 1 if is_soz else 0
        else:
            is_ez = 0

        is_valid = 1 if status != "bad" and channel_type in ["SEEG", "ECOG", "STEREOEEG"] else 0
        contact_group, contact_number = parse_contact_topology(channel_name_norm)

        meta_records.append(
            {
                "channel_name_orig": channel_name_orig,
                "channel_name_norm": channel_name_norm,
                "status_description": status_desc,
                "is_soz": is_soz,
                "is_resected": is_resected,
                "has_non_na_label": has_non_na_label,
                "is_ez": is_ez,
                "is_valid": is_valid,
                "type": channel_type,
                "contact_group": contact_group,
                "contact_number": contact_number,
                "channel_order": len(meta_records),
            }
        )

    return pd.DataFrame(meta_records)
