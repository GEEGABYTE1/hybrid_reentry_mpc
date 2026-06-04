from pathlib import Path

import numpy as np

from reentry_mpc.longitudinal import load_phase1_config
from reentry_mpc.nmpc import NmpcConfig, apply_flap_limits, solve_nmpc_step
from reentry_mpc.phase2 import build_reference_profile, load_phase2_config
from reentry_mpc.phase4 import load_phase4_config, run_phase4_nmpc


def test_nmpc_one_step_returns_finite_control_and_log() -> None:
    config = load_phase4_config("configs/phase4_nmpc.yaml")
    plant = load_phase1_config(config.phase1_config)
    reference = build_reference_profile(load_phase2_config(config.phase2_config)).iloc[
        ::4
    ]
    reference = reference.reset_index(drop=True)
    first = reference.iloc[0]
    state = np.array(
        [first["alpha_ref_rad"], first["q_ref_radps"], first["theta_ref_rad"]],
        dtype=float,
    )
    small_config = NmpcConfig(
        horizon_steps=3,
        dt=config.nmpc.dt,
        weights=config.nmpc.weights,
        solver=config.nmpc.solver,
    )

    control, log = solve_nmpc_step(
        state=state,
        previous_flap_rad=0.0,
        horizon=reference.head(4),
        vehicle=plant.vehicle,
        aero=plant.aero,
        config=small_config,
    )

    assert np.isfinite(control)
    assert log["solver_status"] == "Solve_Succeeded"
    assert log["solve_time_s"] >= 0.0
    assert "objective_value" in log


def test_apply_flap_limits_respects_angle_and_rate_bounds() -> None:
    reference = build_reference_profile(
        load_phase2_config("configs/phase2_reference.yaml")
    )
    row = reference.iloc[0]

    limited, flap_saturated, rate_saturated = apply_flap_limits(
        raw_command=10.0,
        previous_flap_rad=0.0,
        row=row,
        dt=0.5,
    )

    assert limited <= row["flap_max_rad"]
    assert flap_saturated or rate_saturated


def test_phase4_runner_writes_requested_artifacts(tmp_path: Path) -> None:
    artifacts = run_phase4_nmpc(
        config_path="configs/phase4_nmpc.yaml",
        output_dir=tmp_path,
    )

    assert artifacts["rollout_csv"].exists()
    assert artifacts["solver_log_csv"].exists()
    assert artifacts["comparison_csv"].exists()
    assert artifacts["tracking_figure"].exists()
    assert artifacts["flap_figure"].exists()
    assert artifacts["constraint_activity_figure"].exists()
    assert artifacts["solve_time_figure"].exists()
    assert {"pid", "gain_scheduled_lqr", "nominal_nmpc"}.issubset(
        set(artifacts["comparison"]["controller"])
    )
    assert {
        "solver_status",
        "solve_time_s",
        "objective_value",
        "first_control_action_rad",
        "max_alpha_violation_rad",
        "alpha_constraint_active",
    }.issubset(artifacts["solver_log"].columns)
