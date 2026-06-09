"""Learning-augmented MPC research utilities for reentry attitude control."""

from reentry_mpc.phase1 import run_phase1_open_loop
from reentry_mpc.phase2 import run_phase2_reference
from reentry_mpc.phase3 import run_phase3_baselines
from reentry_mpc.phase4 import run_phase4_nmpc
from reentry_mpc.phase5 import run_phase5_monte_carlo
from reentry_mpc.phase6 import run_phase6_robust_mpc
from reentry_mpc.phase7 import run_phase7_scenario_mpc
from reentry_mpc.phase8 import run_phase8_residual_dataset
from reentry_mpc.phase9 import run_phase9_residual_model
from reentry_mpc.phase10 import run_phase10_learned_residual_mpc
from reentry_mpc.phase11 import run_phase11_residual_mpc
from reentry_mpc.phase12 import run_phase12_learning_augmented_mpc
from reentry_mpc.phase13 import run_phase13_feasibility_diagnostics
from reentry_mpc.phase14 import run_phase14_realtime_timing
from reentry_mpc.phase15 import run_phase15_fault_injection
from reentry_mpc.pipeline import run_smoke_experiment

__all__ = [
    "run_phase1_open_loop",
    "run_phase2_reference",
    "run_phase3_baselines",
    "run_phase4_nmpc",
    "run_phase5_monte_carlo",
    "run_phase6_robust_mpc",
    "run_phase7_scenario_mpc",
    "run_phase8_residual_dataset",
    "run_phase9_residual_model",
    "run_phase10_learned_residual_mpc",
    "run_phase11_residual_mpc",
    "run_phase12_learning_augmented_mpc",
    "run_phase13_feasibility_diagnostics",
    "run_phase14_realtime_timing",
    "run_phase15_fault_injection",
    "run_smoke_experiment",
]
