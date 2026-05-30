
from __future__ import annotations

import argparse

from reentry_mpc.phase5 import run_phase5_monte_carlo


def main() -> int:


    parser = argparse.ArgumentParser(description="Run Phase 5 Monte Carlo benchmark.")
    parser.add_argument("--config", default="configs/phase5_monte_carlo.yaml")
    parser.add_argument("--output-dir", default="outputs/phase5_monte_carlo")
    args = parser.parse_args()

    artifacts = run_phase5_monte_carlo(args.config, args.output_dir)
    print(
        "phase5_ok "
        f"summary={artifacts['summary_csv']} "
        f"rollouts={artifacts['combined_rollouts_csv']}"
    )
    return 0
