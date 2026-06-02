"""Learning-augmented MPC research utilities for reentry attitude control."""

from reentry_mpc.phase1 import run_phase1_open_loop
from reentry_mpc.phase2 import run_phase2_reference
from reentry_mpc.phase3 import run_phase3_baselines
from reentry_mpc.phase4 import run_phase4_nmpc
from reentry_mpc.phase5 import run_phase5_monte_carlo
from reentry_mpc.pipeline import run_smoke_experiment

__all__ = [
    "run_phase1_open_loop",
    "run_phase2_reference",
    "run_phase3_baselines",
    "run_phase4_nmpc",
    "run_phase5_monte_carlo",
    "run_smoke_experiment",
]
