from __future__ import annotations

import argparse

from reentry_mpc.phase4 import run_phase4_nmpc


def main() -> int:

    parser = argparse.ArgumentParser(description="Run nominal nonlinear MPC artifacts.")
    parser.add_argument("--config", default="configs/phase4_nmpc.yaml")
    parser.add_argument("--output-dir", default="outputs/phase4_nmpc")
    args = parser.parse_args()

    artifacts = run_phase4_nmpc(args.config, args.output_dir)
    print(
        "phase4_ok "
        f"comparison={artifacts['comparison_csv']} "
        f"solver_log={artifacts['solver_log_csv']}"
    )
    return 0
