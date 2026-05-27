
from __future__ import annotations

import argparse

from reentry_mpc.phase1 import run_phase1_open_loop


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run reduced-order longitudinal reentry simulator artifacts."
    )
    parser.add_argument("--config", default="configs/phase1_open_loop.yaml")
    parser.add_argument("--output-dir", default="outputs/phase1_simulator")
    args = parser.parse_args()

    artifacts = run_phase1_open_loop(args.config, args.output_dir)
    print(
        "phase1_ok "
        f"trajectory={artifacts['trajectory_csv']} "
        f"metrics={artifacts['metrics_json']}"
    )
    return 0
