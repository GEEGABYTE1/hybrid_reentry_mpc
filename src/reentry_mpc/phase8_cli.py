from __future__ import annotations

import argparse

from reentry_mpc.phase8 import run_phase8_residual_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the residual dynamics dataset."
    )
    parser.add_argument("--config", default="configs/phase8_residual_dataset.yaml")
    parser.add_argument("--output-dir", default="outputs/phase8_residual_dataset")
    args = parser.parse_args()
    artifacts = run_phase8_residual_dataset(args.config, args.output_dir)
    print(
        "phase8_residual_dataset_ok "
        + " ".join(
            f"{name}={path}"
            for name, path in sorted(artifacts.items())
            if name != "metadata"
        )
    )


if __name__ == "__main__":
    main()
