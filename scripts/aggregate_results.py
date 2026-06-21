from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biodynformer.orchestrator import aggregate_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate BioDynFormer run metrics.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(aggregate_results(args.output_dir), flush=True)


if __name__ == "__main__":
    main()
