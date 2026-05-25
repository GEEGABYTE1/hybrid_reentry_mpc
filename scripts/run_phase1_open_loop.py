
#generating phase 1 reduced-order simulator artifacts.
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reentry_mpc.phase1 import run_phase1_open_loop  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
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


if __name__ == "__main__":
    raise SystemExit(main())
