from __future__ import annotations

import argparse

from reentry_mpc.phase9 import run_phase9_residual_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the residual dynamics model.")
    parser.add_argument("--config", default="configs/phase9_residual_model.yaml")
    args = parser.parse_args()
    artifacts = run_phase9_residual_model(args.config)
    print(
        "phase9_residual_model_ok "
        + " ".join(
            f"{name}={path}"
            for name, path in sorted(artifacts.items())
            if name != "metrics"
        )
    )


if __name__ == "__main__":
    main()
