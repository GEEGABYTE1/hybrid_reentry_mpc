from __future__ import annotations

import argparse

from reentry_mpc.phase20 import run_phase20_full_oracle_imitation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Phase 20 full oracle-imitation benchmark."
    )
    parser.add_argument(
        "--config",
        default="configs/phase20_full_oracle_imitation.yaml",
        help="Path to Phase 20 YAML config.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/phase20_full_oracle_imitation",
        help="Directory for Phase 20 artifacts.",
    )
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    artifacts = run_phase20_full_oracle_imitation(
        config_path=args.config,
        output_dir=args.output_dir,
        progress=args.progress,
    )
    print(f"Saved Phase 20 comparison to {artifacts['comparison_csv']}")


if __name__ == "__main__":
    main()
