from __future__ import annotations

import argparse

from reentry_mpc.phase18 import run_phase18_slack_oracle


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Phase 18 slack-maximizing feasibility oracle."
    )
    parser.add_argument(
        "--config",
        default="configs/phase18_slack_oracle.yaml",
        help="Path to Phase 18 YAML config.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/phase18_slack_oracle",
        help="Directory for Phase 18 artifacts.",
    )
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    artifacts = run_phase18_slack_oracle(
        config_path=args.config,
        output_dir=args.output_dir,
        progress=args.progress,
    )
    print(f"Saved Phase 18 oracle summary to {artifacts['summary_csv']}")


if __name__ == "__main__":
    main()
