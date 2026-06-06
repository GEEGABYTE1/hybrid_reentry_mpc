from __future__ import annotations

import argparse

from reentry_mpc.phase12 import run_phase12_learning_augmented_mpc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Phase 12 consolidated learning-augmented MPC benchmark."
    )
    parser.add_argument(
        "--config",
        default="configs/phase12_learning_augmented_mpc.yaml",
        help="Path to Phase 12 YAML config.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/phase12_learning_augmented_mpc",
        help="Directory for Phase 12 artifacts.",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Print one progress line per rollout.",
    )
    args = parser.parse_args()
    artifacts = run_phase12_learning_augmented_mpc(
        config_path=args.config, output_dir=args.output_dir, progress=args.progress
    )
    print(f"Saved Phase 12 summary to {artifacts['summary_csv']}")


if __name__ == "__main__":
    main()
