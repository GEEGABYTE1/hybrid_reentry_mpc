from __future__ import annotations

import argparse

from reentry_mpc.phase7 import run_phase7_scenario_mpc


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 7 scenario NMPC benchmark.")
    parser.add_argument(
        "--config", default="configs/phase7_scenario_mpc.yaml", help="Phase 7 config."
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/phase7_scenario_mpc",
        help="Directory for Phase 7 artifacts.",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Print one progress line per tier/scenario rollout.",
    )
    args = parser.parse_args()
    artifacts = run_phase7_scenario_mpc(
        config_path=args.config,
        output_dir=args.output_dir,
        progress=args.progress,
    )
    print(
        "phase7_ok "
        f"summary={artifacts['summary_csv']} "
        f"comparison={artifacts['comparison_csv']}"
    )


if __name__ == "__main__":
    main()
