from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biodynformer.pipeline import ManifestInputError, run_full_pipeline


def _csv(value: str) -> list[str]:
    return [part.strip().lower() for part in value.replace(";", ",").split(",") if part.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="audit, build, audit, and run BioDynFormer experiments.")
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--hup-participants-path", type=Path, default=None)
    parser.add_argument("--centers", type=_csv, default=["lzu", "hup", "multicenter", "pediatric"])

    parser.add_argument("--lzu-root", type=Path, default=None)
    parser.add_argument("--hup-root", type=Path, default=None)
    parser.add_argument("--multicenter-root", type=Path, default=None)
    parser.add_argument("--pediatric-root", type=Path, default=None)
    parser.add_argument("--lzu-manifest", type=Path, default=None)
    parser.add_argument("--hup-manifest", type=Path, default=None)
    parser.add_argument("--multicenter-manifest", type=Path, default=None)
    parser.add_argument("--pediatric-manifest", type=Path, default=None)

    parser.add_argument("--source-audit-output-dir", type=Path, default=Path(r"D:\nips-temp\biodynformer_source_audit"))
    parser.add_argument("--feature-bank-output-dir", "--feature-bank", dest="feature_bank_output_dir", type=Path, default=Path(r"D:\nips-temp\biodynformer_preictal_feature_bank"))
    parser.add_argument("--runs-output-dir", "--output-dir", dest="runs_output_dir", type=Path, default=Path(r"D:\nips-temp\biodynformer_runs"))

    parser.add_argument("--rebuild-feature-bank", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--quality-filter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--quality-keep-ratings", type=str, default="GOOD,REVIEW")
    parser.add_argument("--quality-drop-ratings", type=str, default="POOR")
    parser.add_argument("--quality-missing-policy", choices=["drop", "keep"], default="drop")

    parser.add_argument("--versions", type=_csv, default=["v1", "v2", "final"])
    parser.add_argument("--tasks", type=_csv, default=["task1", "task2"])
    parser.add_argument("--run-5fold", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-loco", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=200)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        summary = run_full_pipeline(
            metadata_dir=args.metadata_dir,
            source_audit_output_dir=args.source_audit_output_dir,
            feature_bank_output_dir=args.feature_bank_output_dir,
            runs_output_dir=args.runs_output_dir,
            centers=args.centers,
            roots={
                "lzu": args.lzu_root,
                "hup": args.hup_root,
                "multicenter": args.multicenter_root,
                "pediatric": args.pediatric_root,
            },
            manifest_paths={
                "lzu": args.lzu_manifest,
                "hup": args.hup_manifest,
                "multicenter": args.multicenter_manifest,
                "pediatric": args.pediatric_manifest,
            },
            hup_participants_path=args.hup_participants_path,
            rebuild_feature_bank=args.rebuild_feature_bank,
            quality_filter=args.quality_filter,
            quality_keep_ratings=args.quality_keep_ratings,
            quality_drop_ratings=args.quality_drop_ratings,
            quality_missing_policy=args.quality_missing_policy,
            versions=args.versions,
            tasks=args.tasks,
            run_5fold=args.run_5fold,
            run_loco=args.run_loco,
            resume=args.resume,
            n_splits=args.n_splits,
            seed=args.seed,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
        )
    except ManifestInputError as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2) from exc
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
