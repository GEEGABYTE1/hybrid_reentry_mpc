from __future__ import annotations

import argparse

from reentry_mpc.phase23 import run_phase23_feasibility_ceiling_audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Phase 23 actuator-consistent feasibility-ceiling audit."
    )
    parser.add_argument(
        "--config",
        default="configs/phase23_feasibility_ceiling_audit.yaml",
        help="Path to Phase 23 YAML config.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/phase23_feasibility_ceiling_audit",
        help="Directory for Phase 23 artifacts.",
    )
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    artifacts = run_phase23_feasibility_ceiling_audit(
        config_path=args.config,
        output_dir=args.output_dir,
        progress=args.progress,
    )
    print(f"Saved Phase 23 ceiling comparison to {artifacts['comparison_csv']}")


if __name__ == "__main__":
    main()
