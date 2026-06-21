from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biodynformer.feature_bank import build_feature_bank_from_records, load_patient_records_pickle, load_records_json
from biodynformer.source_adapters import load_four_center_records


def _parse_centers(value: str) -> list[str]:
    return [part.strip().lower() for part in value.replace(";", ",").split(",") if part.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build one reusable preictal BioDynFormer feature bank from local source_adapters.")
    parser.add_argument(
        "--source",
        choices=["four-center-raw", "records-json", "patient-records-pkl"],
        default="four-center-raw",
        help="Default uses biodynformer.source_adapters in this folder; no external GitHub code is required.",
    )
    parser.add_argument("--records-json", type=Path, default=None)
    parser.add_argument("--patient-records-pkl", type=Path, default=None)
    parser.add_argument("--centers", type=_parse_centers, default=["lzu", "hup", "multicenter", "pediatric"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--quality-filter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--quality-keep-ratings", type=str, default="GOOD,REVIEW")
    parser.add_argument("--quality-drop-ratings", type=str, default="POOR")
    parser.add_argument("--quality-missing-policy", choices=["drop", "keep"], default="drop")
    parser.add_argument("--strict-quality-reports", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-missing-quality-report", action="store_true")
    parser.add_argument("--success-only", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--lzu-root", type=Path, default=None)
    parser.add_argument("--hup-root", type=Path, default=None)
    parser.add_argument("--multicenter-root", type=Path, default=None)
    parser.add_argument("--pediatric-root", type=Path, default=None)
    parser.add_argument("--lzu-manifest", type=Path, default=None)
    parser.add_argument("--hup-manifest", type=Path, default=None)
    parser.add_argument("--multicenter-manifest", type=Path, default=None)
    parser.add_argument("--pediatric-manifest", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.allow_missing_quality_report:
        args.strict_quality_reports = False
    if args.source == "records-json" or args.records_json is not None:
        if args.records_json is None:
            raise SystemExit("--records-json is required for source=records-json")
        records = load_records_json(args.records_json)
    elif args.source == "patient-records-pkl" or args.patient_records_pkl is not None:
        if args.patient_records_pkl is None:
            raise SystemExit("--patient-records-pkl is required for source=patient-records-pkl")
        records = load_patient_records_pickle(args.patient_records_pkl)
    else:
        records = load_four_center_records(
            centers=args.centers,
            manifest_paths={
                "lzu": args.lzu_manifest,
                "hup": args.hup_manifest,
                "multicenter": args.multicenter_manifest,
                "pediatric": args.pediatric_manifest,
            },
            roots={
                "lzu": args.lzu_root,
                "hup": args.hup_root,
                "multicenter": args.multicenter_root,
                "pediatric": args.pediatric_root,
            },
        )
    summary = build_feature_bank_from_records(
        records,
        output_dir=args.output_dir,
        quality_filter=args.quality_filter,
        keep_ratings=args.quality_keep_ratings,
        drop_ratings=args.quality_drop_ratings,
        missing_policy=args.quality_missing_policy,
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
