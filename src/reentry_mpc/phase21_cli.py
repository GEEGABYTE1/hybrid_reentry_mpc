from __future__ import annotations

import argparse

from reentry_mpc.phase21 import run_phase21_missed_case_autopsy


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Phase 21 missed oracle-feasible case autopsy."
    )
    parser.add_argument(
        "--config",
        default="configs/phase21_missed_case_autopsy.yaml",
        help="Path to Phase 21 YAML config.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/phase21_missed_case_autopsy",
        help="Directory for Phase 21 artifacts.",
    )
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    artifacts = run_phase21_missed_case_autopsy(
        config_path=args.config,
        output_dir=args.output_dir,
        progress=args.progress,
    )
    print(f"Saved Phase 21 transfer comparison to {artifacts['comparison_csv']}")


if __name__ == "__main__":
    main()
