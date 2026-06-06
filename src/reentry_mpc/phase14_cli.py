from __future__ import annotations

import argparse

from reentry_mpc.phase14 import run_phase14_realtime_timing


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run real-time feasibility and timing analysis."
    )
    parser.add_argument(
        "--config",
        default="configs/phase14_realtime_timing.yaml",
        help="Path to Phase 14 timing config.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/phase14_realtime_timing",
        help="Directory for timing artifacts.",
    )
    args = parser.parse_args()
    artifacts = run_phase14_realtime_timing(
        config_path=args.config,
        output_dir=args.output_dir,
    )
    print(f"Saved onboard feasibility table to {artifacts['summary_csv']}")


if __name__ == "__main__":
    main()
