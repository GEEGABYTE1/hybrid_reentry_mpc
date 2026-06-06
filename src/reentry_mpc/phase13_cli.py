from __future__ import annotations

import argparse

from reentry_mpc.phase13 import run_phase13_feasibility_diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Phase 13 feasibility diagnostics from Phase 12 artifacts."
    )
    parser.add_argument(
        "--config",
        default="configs/phase13_feasibility_diagnostics.yaml",
        help="Path to Phase 13 YAML config.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/phase13_feasibility_diagnostics",
        help="Directory for Phase 13 artifacts.",
    )
    args = parser.parse_args()
    artifacts = run_phase13_feasibility_diagnostics(
        config_path=args.config,
        output_dir=args.output_dir,
    )
    print(f"Saved Phase 13 diagnostics to {artifacts['diagnostics_csv']}")


if __name__ == "__main__":
    main()
