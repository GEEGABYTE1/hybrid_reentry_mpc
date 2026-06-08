from __future__ import annotations

import argparse

from reentry_mpc.phase15 import run_phase15_fault_injection


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Phase 15 fault injection and fallback benchmark."
    )
    parser.add_argument(
        "--config",
        default="configs/phase15_fault_injection.yaml",
        help="Path to Phase 15 fault injection config.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/phase15_fault_injection",
        help="Directory for Phase 15 artifacts.",
    )
    args = parser.parse_args()
    artifacts = run_phase15_fault_injection(
        config_path=args.config,
        output_dir=args.output_dir,
    )
    print(f"Saved fault metrics table to {artifacts['summary_csv']}")


if __name__ == "__main__":
    main()
