from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from reentry_mpc.learning_augmented_mpc import (
    biased_nmpc_derivatives_numeric,
    build_horizon_residual_biases,
)
from reentry_mpc.longitudinal import load_phase1_config
from reentry_mpc.nmpc import nmpc_derivatives_numeric
from reentry_mpc.phase2 import build_reference_profile, load_phase2_config
from reentry_mpc.phase4 import _downsample_reference_profile
from reentry_mpc.phase6 import TighteningMargins, tighten_reference_profile
from reentry_mpc.phase10 import load_residual_model
from reentry_mpc.phase12 import (
    load_phase12_config,
    run_phase12_learning_augmented_mpc,
)


def test_horizon_residual_inference_shape_and_time() -> None:
    config = load_phase12_config("configs/phase12_learning_augmented_mpc.yaml")
    phase2_config = load_phase2_config(config.phase2_config)
    reference = build_reference_profile(phase2_config)
    horizon = _downsample_reference_profile(reference, config.nmpc.dt).iloc[
        : config.nmpc.horizon_steps + 1
    ]
    loaded = load_residual_model(config.residual_model_checkpoint)
    state = (
        horizon[["alpha_ref_rad", "q_ref_radps", "theta_ref_rad"]].iloc[0].to_numpy()
    )

    biases, inference_time = build_horizon_residual_biases(
        loaded_model=loaded,
        state=state,
        previous_flap_rad=0.0,
        horizon=horizon,
        horizon_steps=config.nmpc.horizon_steps,
    )

    assert biases.shape == (config.nmpc.horizon_steps,)
    assert np.all(np.isfinite(biases))
    assert inference_time >= 0.0


def test_zero_bias_numeric_dynamics_match_nominal() -> None:
    plant = load_phase1_config("configs/phase1_open_loop.yaml")
    reference = build_reference_profile(
        load_phase2_config("configs/phase2_reference.yaml")
    )
    row = reference.iloc[5]
    state = np.array([row["alpha_ref_rad"], row["q_ref_radps"], row["theta_ref_rad"]])

    nominal = nmpc_derivatives_numeric(
        state=state,
        delta_flap_rad=0.01,
        row=row,
        vehicle=plant.vehicle,
        aero=plant.aero,
    )
    biased = biased_nmpc_derivatives_numeric(
        state=state,
        delta_flap_rad=0.01,
        row=row,
        vehicle=plant.vehicle,
        aero=plant.aero,
        residual_q_dot_bias=0.0,
    )

    np.testing.assert_allclose(biased, nominal)


def test_tightening_changes_planning_corridor_not_original_reference() -> None:
    reference = build_reference_profile(
        load_phase2_config("configs/phase2_reference.yaml")
    )
    tightened = tighten_reference_profile(
        reference,
        TighteningMargins(alpha_margin_rad=0.035, q_margin_radps=0.010),
    )

    assert np.all(tightened["alpha_min_rad"] > reference["alpha_min_rad"])
    assert np.all(tightened["alpha_max_rad"] < reference["alpha_max_rad"])
    assert np.all(reference["alpha_min_rad"] < reference["alpha_ref_rad"])
    assert np.all(reference["alpha_max_rad"] > reference["alpha_ref_rad"])


def test_tiny_phase12_run_writes_artifacts(tmp_path) -> None:
    phase5_raw = yaml.safe_load(
        Path("configs/phase5_monte_carlo.yaml").read_text(encoding="utf-8")
    )
    for tier in phase5_raw["tiers"]:
        tier["scenario_count"] = 1
    phase5_path = tmp_path / "tiny_phase5.yaml"
    phase5_path.write_text(yaml.safe_dump(phase5_raw), encoding="utf-8")

    phase12_raw = yaml.safe_load(
        Path("configs/phase12_learning_augmented_mpc.yaml").read_text(encoding="utf-8")
    )
    phase12_raw["phase5_config"] = str(phase5_path)
    phase12_path = tmp_path / "tiny_phase12.yaml"
    phase12_path.write_text(yaml.safe_dump(phase12_raw), encoding="utf-8")

    artifacts = run_phase12_learning_augmented_mpc(
        config_path=phase12_path,
        output_dir=tmp_path,
        progress=False,
    )
    summary = artifacts["summary"]
    rollouts = artifacts["rollouts"]

    assert set(summary["controller"]) == {
        "nominal_nmpc",
        "residual_corrected_nmpc",
        "residual_corrected_tightened_nmpc",
    }
    assert (tmp_path / "phase12_summary_table.csv").exists()
    assert (tmp_path / "phase8_summary_table.csv").exists()
    assert (tmp_path / "phase12_rollouts.csv").exists()
    assert (tmp_path / "moderate/scenario_000/nominal_nmpc/trajectory.csv").exists()
    assert (tmp_path / "moderate/scenario_000/nominal_nmpc/metrics.json").exists()
    assert np.all(rollouts["solve_time_s"] >= 0.0)
    assert np.all(rollouts["nn_inference_time_s"] >= 0.0)
    assert np.all(rollouts["total_loop_time_s"] >= 0.0)
