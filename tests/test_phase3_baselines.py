from pathlib import Path

import numpy as np

from reentry_mpc.baseline_controllers import PIDController, build_lqr_controller
from reentry_mpc.baseline_metrics import summarize_all
from reentry_mpc.baseline_rollout import rollout_controller
from reentry_mpc.linearization import finite_difference_linearization
from reentry_mpc.longitudinal import load_phase1_config
from reentry_mpc.phase2 import build_reference_profile, load_phase2_config
from reentry_mpc.phase3 import load_phase3_config, run_phase3_baselines


def test_linearization_shapes_are_valid() -> None:
    plant = load_phase1_config("configs/phase1_open_loop.yaml")
    reference = build_reference_profile(
        load_phase2_config("configs/phase2_reference.yaml")
    )
    row = reference.iloc[10]
    schedule = {
        "altitude_m": float(row["altitude_m"]),
        "velocity_mps": float(row["velocity_mps"]),
        "mach": float(row["mach"]),
        "density_kgm3": float(row["density_kgm3"]),
        "dynamic_pressure_pa": float(row["dynamic_pressure_pa"]),
    }

    model = finite_difference_linearization(
        state=np.zeros(3),
        delta_flap_rad=0.0,
        schedule=schedule,
        vehicle=plant.vehicle,
        aero=plant.aero,
        dt=0.5,
    )

    assert model.a_discrete.shape == (3, 3)
    assert model.b_discrete.shape == (3, 1)
    assert np.isfinite(model.a_discrete).all()
    assert np.isfinite(model.b_discrete).all()


def test_pid_rollout_and_metrics_contract() -> None:
    plant = load_phase1_config("configs/phase1_open_loop.yaml")
    reference = build_reference_profile(
        load_phase2_config("configs/phase2_reference.yaml")
    ).head(20)
    controller = PIDController(
        kp_alpha=1.0,
        ki_alpha=0.0,
        kd_alpha=0.0,
        kp_q=5.0,
        anti_windup_limit=1.0,
    )
    initial = np.array(
        [
            reference.iloc[0]["alpha_ref_rad"],
            reference.iloc[0]["q_ref_radps"],
            reference.iloc[0]["theta_ref_rad"],
        ]
    )

    rollout = rollout_controller(
        controller_name="pid",
        controller=controller,
        reference_profile=reference,
        vehicle=plant.vehicle,
        aero=plant.aero,
        initial_state=initial,
    )
    metrics = summarize_all(
        {"pid": rollout},
        {
            "rms_alpha_error_rad": 1.0,
            "max_alpha_error_rad": 1.0,
            "corridor_violation_count": 100.0,
        },
    )

    assert not rollout.empty
    assert {"alpha_error_rad", "flap_saturated", "corridor_violation"}.issubset(
        rollout.columns
    )
    assert metrics.loc[0, "controller"] == "pid"


def test_lqr_controller_builds_scheduled_gains() -> None:
    phase3 = load_phase3_config("configs/phase3_baselines.yaml")
    plant = load_phase1_config("configs/phase1_open_loop.yaml")

    controller = build_lqr_controller(
        schedule_points=phase3.lqr["schedule_points"],
        q_weights={key: float(value) for key, value in phase3.lqr["q_weights"].items()},
        r_weight=float(phase3.lqr["r_weight"]),
        dt=0.5,
        vehicle=plant.vehicle,
        aero=plant.aero,
    )

    assert len(controller.scheduled_gains) == 3
    assert controller.scheduled_gains[0].gain.shape == (1, 3)


def test_phase3_runner_writes_requested_artifacts(tmp_path: Path) -> None:
    artifacts = run_phase3_baselines(
        config_path="configs/phase3_baselines.yaml",
        output_dir=tmp_path,
    )

    assert artifacts["metrics_csv"].exists()
    assert artifacts["summary_md"].exists()
    assert artifacts["tracking_figure"].exists()
    assert artifacts["flap_figure"].exists()
    assert set(artifacts["metrics"]["controller"]) == {"pid", "gain_scheduled_lqr"}
