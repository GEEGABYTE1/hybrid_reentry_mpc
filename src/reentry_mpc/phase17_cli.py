from __future__ import annotations

import argparse

from reentry_mpc.phase17 import run_phase17_feasibility_safety


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Phase 17 feasibility and actuator-aware safety NMPC."
    )
    parser.add_argument(
        "--config",
        default="configs/phase17_feasibility_safety.yaml",
        help="Path to Phase 17 YAML config.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/phase17_feasibility_safety",
        help="Directory for Phase 17 artifacts.",
    )
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    artifacts = run_phase17_feasibility_safety(
        config_path=args.config,
        output_dir=args.output_dir,
        progress=args.progress,
    )
    print(f"Saved Phase 17 summary to {artifacts['summary_csv']}")


if __name__ == "__main__":
    main()
