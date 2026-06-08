from __future__ import annotations

import argparse

from reentry_mpc.phase19 import run_phase19_oracle_imitation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Phase 19 oracle-imitation safety policy."
    )
    parser.add_argument(
        "--config",
        default="configs/phase19_oracle_imitation.yaml",
        help="Path to Phase 19 YAML config.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/phase19_oracle_imitation",
        help="Directory for Phase 19 artifacts.",
    )
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    artifacts = run_phase19_oracle_imitation(
        config_path=args.config,
        output_dir=args.output_dir,
        progress=args.progress,
    )
    print(f"Saved Phase 19 summary to {artifacts['summary_csv']}")


if __name__ == "__main__":
    main()
