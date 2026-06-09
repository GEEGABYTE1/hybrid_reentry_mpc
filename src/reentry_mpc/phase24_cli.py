from __future__ import annotations

import argparse

from reentry_mpc.phase24 import run_phase24_hybrid_imitation_mpc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Phase 24 hybrid oracle-imitation plus slack-MPC benchmark."
    )
    parser.add_argument(
        "--config",
        default="configs/phase24_hybrid_imitation_mpc.yaml",
        help="Path to Phase 24 YAML config.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/phase24_hybrid_imitation_mpc",
        help="Directory for Phase 24 artifacts.",
    )
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    artifacts = run_phase24_hybrid_imitation_mpc(
        config_path=args.config,
        output_dir=args.output_dir,
        progress=args.progress,
    )
    print(f"Saved Phase 24 comparison to {artifacts['comparison_csv']}")


if __name__ == "__main__":
    main()
