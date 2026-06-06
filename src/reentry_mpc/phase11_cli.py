from __future__ import annotations

import argparse

from reentry_mpc.phase11 import run_phase11_residual_mpc


def main() -> None:
    parser = argparse.ArgumentParser(description="Run horizon-embedded residual MPC.")
    parser.add_argument("--config", default="configs/phase11_residual_mpc.yaml")
    parser.add_argument("--output-dir", default="outputs/phase11_residual_mpc")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    artifacts = run_phase11_residual_mpc(
        args.config, args.output_dir, progress=args.progress
    )
    print(
        "phase11_residual_mpc_ok "
        + " ".join(
            f"{name}={path}"
            for name, path in sorted(artifacts.items())
            if name not in {"summary", "rollouts", "comparison"}
        )
    )


if __name__ == "__main__":
    main()
