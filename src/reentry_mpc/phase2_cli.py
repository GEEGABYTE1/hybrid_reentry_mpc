
from __future__ import annotations

import argparse

from reentry_mpc.phase2 import run_phase2_reference


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate configurable reentry reference profile and corridor."
    )
    parser.add_argument("--config", default="configs/phase2_reference.yaml")
    parser.add_argument("--output-dir", default="outputs/phase2_reference")
    args = parser.parse_args()

    artifacts = run_phase2_reference(args.config, args.output_dir)
    print(
        "phase2_ok "
        f"reference={artifacts['reference_csv']} "
        f"corridor={artifacts['corridor_json']}"
    )
    return 0
