from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biodynformer.manifest_drafts import generate_manifest_drafts


def _csv(value: str) -> list[str]:
    return [part.strip().lower() for part in value.replace(";", ",").split(",") if part.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate auditable draft manifests from BioDynFormer metadata spreadsheets."
    )
    parser.add_argument("--metadata-dir", type=Path, default=Path(r"C:\Users\gifoe\Downloads\all_seeg_data"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--centers", type=_csv, default=["lzu", "hup", "multicenter", "pediatric"])
    parser.add_argument("--quality-keep-ratings", type=_csv, default=["good", "review"])
    parser.add_argument("--hup-participants-path", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = generate_manifest_drafts(
        metadata_dir=args.metadata_dir,
        output_dir=args.output_dir,
        centers=args.centers,
        keep_ratings=[rating.upper() for rating in args.quality_keep_ratings],
        hup_participants_path=args.hup_participants_path,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
