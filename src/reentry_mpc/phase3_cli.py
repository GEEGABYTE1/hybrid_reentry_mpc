
from __future__ import annotations

import argparse

from reentry_mpc.phase3 import run_phase3_baselines


def main() -> int:

    parser = argparse.ArgumentParser(
        description="Run PID and gain-scheduled LQR baseline rollouts."
    )
    parser.add_argument("--config", default="configs/phase3_baselines.yaml")
    parser.add_argument("--output-dir", default="outputs/phase3_baselines")
    args = parser.parse_args()

    artifacts = run_phase3_baselines(args.config, args.output_dir)
    print(
        "phase3_ok "
        f"metrics={artifacts['metrics_csv']} "
        f"summary={artifacts['summary_md']}"
    )
    return 0
