from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biodynformer.source_metadata import audit_source_metadata, write_audit_outputs


def _centers(value: str) -> list[str]:
    return [part.strip().lower() for part in value.replace(";", ",").split(",") if part.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit four-center metadata files before BioDynFormer feature extraction.")
    parser.add_argument("--metadata-dir", type=Path, default=Path(r"C:\Users\gifoe\Downloads\all_seeg_data"))
    parser.add_argument("--centers", type=_centers, default=["lzu", "hup", "multicenter", "pediatric"])
    parser.add_argument("--output-dir", type=Path, default=Path(r"D:\nips-temp\biodynformer_source_audit"))
    parser.add_argument("--lzu-root", type=Path, default=None)
    parser.add_argument("--hup-root", type=Path, default=None)
    parser.add_argument("--multicenter-root", type=Path, default=None)
    parser.add_argument("--pediatric-root", type=Path, default=None)
    parser.add_argument("--hup-participants-path", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    audit = audit_source_metadata(
        metadata_dir=args.metadata_dir,
        centers=args.centers,
        roots={
            "lzu": args.lzu_root,
            "hup": args.hup_root,
            "multicenter": args.multicenter_root,
            "pediatric": args.pediatric_root,
        },
        hup_participants_path=args.hup_participants_path,
    )
    write_audit_outputs(audit, args.output_dir)
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
    if not audit["can_build_feature_bank"]:
        print(
            "BLOCKED: metadata files were parsed, but at least one requested center has no signal files under its root. "
            "Provide EDF/NPY/NPZ roots before building a preictal feature bank.",
            file=sys.stderr,
            flush=True,
        )


if __name__ == "__main__":
    main()
