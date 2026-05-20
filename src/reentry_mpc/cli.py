"""Command-line entry points."""

from __future__ import annotations

import argparse
from pathlib import Path

from reentry_mpc.pipeline import run_smoke_experiment


def build_parser() -> argparse.ArgumentParser:
    """Build the smoke-test CLI parser."""

    parser = argparse.ArgumentParser(
        description="Run deterministic reentry MPC smoke artifacts."
    )
    parser.add_argument(
        "--config", default="configs/smoke.yaml", help="YAML config path."
    )
    parser.add_argument("--output-dir", default="outputs", help="Artifact output root.")
    return parser


def main() -> int:
    """Run the CLI."""

    args = build_parser().parse_args()
    artifacts = run_smoke_experiment(args.config, args.output_dir)
    summary_path = Path(artifacts["summary_csv"])
    figure_path = Path(artifacts["tracking_figure"])
    print(f"smoke_ok summary={summary_path} figure={figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
