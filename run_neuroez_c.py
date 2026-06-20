from __future__ import annotations

import argparse
import json
from pathlib import Path

from neuroez_c.config import DEFAULT_WINDOW_CACHE, apply_pruned_defaults


def _str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value_norm = str(value).strip().lower()
    if value_norm in {"1", "true", "yes", "y", "on"}:
        return True
    if value_norm in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run B0-Pruned-EZBackbone on an existing NeuroEZ window cache.")
    parser.add_argument("--dataset_dir", type=str, default=str(Path(__file__).resolve().parent))
    parser.add_argument("--output_dir", type=str, default=r"D:\nips-temp\b0_pruned_results\manual")
    parser.add_argument("--window_cache_path", type=str, default=DEFAULT_WINDOW_CACHE)
    parser.add_argument("--sample_cache_path", type=str, default=None)

    parser.add_argument("--split_strategy", type=str, default="5fold", choices=["5fold", "lopo"])
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--random_seed", type=int, default=42)

    parser.add_argument("--positive_label", type=str, default="nez", choices=["nez", "ez"])
    parser.add_argument("--self_compare_eps", type=float, default=1e-5)
    parser.add_argument("--b0_feature_parts", type=str, default="abs,delta,zdelta,ratio")
    parser.add_argument("--b0_feature_groups", type=str, default="spectral_classical", choices=["spectral_classical", "gamma_line_length", "all"])

    parser.add_argument("--model_dim", type=int, default=32)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.40)
    parser.add_argument("--use_channel_attention", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_patient_relative_z", type=_str_to_bool, nargs="?", const=True, default=True)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--min_epochs_before_early_stop", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--patient_batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--class_weight_mode", type=str, default="ez_negative", choices=["ez_negative", "none"])
    parser.add_argument("--ez_negative_weight", type=str, default="2")
    parser.add_argument("--ez_negative_weight_cap", type=float, default=20.0)
    parser.add_argument("--early_stop_metric", type=str, default="patient_macro_f1")

    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--drop_high_ez_fraction_lzu", type=_str_to_bool, nargs="?", const=True, default=True)
    parser.add_argument("--lzu_max_ez_fraction", type=float, default=0.40)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--log_interval", type=int, default=1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    apply_pruned_defaults(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "run_args_b0_pruned.json", "w", encoding="utf-8") as fout:
        json.dump(vars(args), fout, indent=2, ensure_ascii=False, sort_keys=True)

    from exp_ez_hybrid import Exp_EZHybridLocalization

    Exp_EZHybridLocalization(args).run()


if __name__ == "__main__":
    main()
