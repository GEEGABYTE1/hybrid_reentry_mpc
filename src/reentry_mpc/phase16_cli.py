from __future__ import annotations

import argparse

from reentry_mpc.phase16 import run_phase16_success_recovery


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Phase 16 success-recovery NMPC benchmark."
    )
    parser.add_argument(
        "--config",
        default="configs/phase16_success_recovery.yaml",
        help="Path to Phase 16 YAML config.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/phase16_success_recovery",
        help="Directory for Phase 16 artifacts.",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Print one progress line per rollout.",
    )
    args = parser.parse_args()
    artifacts = run_phase16_success_recovery(
        config_path=args.config,
        output_dir=args.output_dir,
        progress=args.progress,
    )
    print(f"Saved Phase 16 summary to {artifacts['summary_csv']}")


if __name__ == "__main__":
    main()
