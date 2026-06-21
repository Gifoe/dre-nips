from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biodynformer.orchestrator import run_all_versions


def _csv(value: str) -> list[str]:
    return [part.strip().lower() for part in value.replace(";", ",").split(",") if part.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run v1, v2, and final BioDynFormer experiments from one feature bank.")
    parser.add_argument("--feature-bank", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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
    results = run_all_versions(
        feature_bank=args.feature_bank,
        output_dir=args.output_dir,
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
    print(json.dumps({"new_runs": len(results), "output_dir": str(args.output_dir)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
