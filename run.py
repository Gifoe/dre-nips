from __future__ import annotations

import argparse
from pathlib import Path
import sys


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from exp.exp_ez_localization import Exp_EZLocalization
from report_threshold import summarize_cv_results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TeChEZ patient-level EZ localization experiment")
    parser.add_argument("--dataset_dir", type=str, default=".")
    parser.add_argument("--participants_path", type=str, default=None)
    parser.add_argument("--subject_filter", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=str(CURRENT_DIR / "outputs"))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--model", type=str, default="TeChEZ")

    parser.add_argument("--split_strategy", type=str, default="5fold", choices=["5fold", "lopo"])
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--target_sfreq", type=float, default=512.0)
    parser.add_argument("--win_len_sec", type=float, default=15.0)
    parser.add_argument("--step_sec", type=float, default=5.0)
    parser.add_argument("--ez_definition", type=str, default="soz_or_resected")
    parser.add_argument("--success_only", action="store_true", default=True)
    parser.add_argument("--force_rebuild_cache", action="store_true")

    parser.add_argument("--feature_dim", type=int, default=14)
    parser.add_argument("--conn_dim", type=int, default=7)
    parser.add_argument("--edge_dim", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--temporal_kernel_size", type=int, default=3)
    parser.add_argument("--comparator_layers", type=int, default=2)

    parser.add_argument("--focal_alpha", type=float, default=0.75)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--rank_lambda", type=float, default=0.2)
    parser.add_argument("--rank_margin", type=float, default=0.4)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    exp = Exp_EZLocalization(args)
    predictions = exp.run()
    summary = summarize_cv_results(predictions, args.output_dir)
    print(f"TeChEZ completed. Summary written to: {Path(args.output_dir) / 'summary_metrics_threshold.json'}")
    print(summary)


if __name__ == "__main__":
    main()
