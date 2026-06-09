from __future__ import annotations

import argparse

from reentry_mpc.phase22 import run_phase22_actuator_aware_slack_mpc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Phase 22 actuator-aware slack-MPC benchmark."
    )
    parser.add_argument(
        "--config",
        default="configs/phase22_actuator_aware_slack_mpc.yaml",
        help="Path to Phase 22 YAML config.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/phase22_actuator_aware_slack_mpc",
        help="Directory for Phase 22 artifacts.",
    )
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    artifacts = run_phase22_actuator_aware_slack_mpc(
        config_path=args.config,
        output_dir=args.output_dir,
        progress=args.progress,
    )
    print(f"Saved Phase 22 comparison to {artifacts['vs_phase20_csv']}")


if __name__ == "__main__":
    main()
