from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from reentry_mpc.longitudinal import load_phase1_config
from reentry_mpc.phase2 import build_reference_profile, load_phase2_config
from reentry_mpc.phase5 import load_phase5_config
from reentry_mpc.phase17 import _maybe_truncate_reference
from reentry_mpc.phase22 import (
    compare_phase22_to_phase20,
    load_phase22_config,
    rollout_online_slack_mpc,
    run_phase22_actuator_aware_slack_mpc,
    solve_online_slack_mpc_step,
)
from reentry_mpc.uncertainty import initialize_actuator, sample_scenario


def test_phase22_config_loads_variants_and_slack_weights() -> None:
    config = load_phase22_config("configs/phase22_actuator_aware_slack_mpc.yaml")

    assert len(config.variants) == 3
    for variant in config.variants:
        assert variant.weights.alpha_slack > variant.weights.alpha_center
        assert variant.weights.alpha_slack > variant.weights.command
        assert variant.weights.q_slack > variant.weights.q_center


def test_solve_online_slack_mpc_step_returns_finite_command() -> None:
    config = load_phase22_config("configs/phase22_actuator_aware_slack_mpc.yaml")
    phase5 = load_phase5_config(config.phase5_config)
    plant = load_phase1_config(config.phase1_config)
    reference = _maybe_truncate_reference(
        build_reference_profile(load_phase2_config(config.phase2_config)),
        2.0,
    )
    scenario = sample_scenario(
        scenario_id=0,
        seed=phase5.seed,
        ranges=phase5.tiers[0].uncertainty_ranges,
    )
    first = reference.iloc[0]
    state = np.array(
        [
            first["alpha_ref_rad"] + scenario.initial_error.alpha_rad,
            first["q_ref_radps"] + scenario.initial_error.q_radps,
            first["theta_ref_rad"] + scenario.initial_error.theta_rad,
        ],
        dtype=float,
    )
    variant = config.variants[0]
    actuator = initialize_actuator(scenario, 0.5)
    command, log = solve_online_slack_mpc_step(
        state=state,
        applied_flap_rad=actuator.previous_applied_rad,
        previous_raw_flap_rad=0.0,
        horizon=reference.head(variant.horizon_steps + 1),
        vehicle=plant.vehicle,
        aero=plant.aero,
        variant=variant,
        solver=config.solver,
        actuator_tau_s=max(
            0.5 * variant.control_dt_s,
            scenario.actuator_lag_s + 0.5 * scenario.actuator_delay_s,
        ),
    )

    assert np.isfinite(command)
    assert log["solve_time_s"] >= 0.0
    assert log["predicted_max_alpha_slack_rad"] >= 0.0
    assert "first_raw_flap_rad" in log


def test_rollout_rows_include_slack_and_actuator_fields() -> None:
    config = load_phase22_config("configs/phase22_actuator_aware_slack_mpc.yaml")
    phase5 = load_phase5_config(config.phase5_config)
    plant = load_phase1_config(config.phase1_config)
    reference = _maybe_truncate_reference(
        build_reference_profile(load_phase2_config(config.phase2_config)),
        2.0,
    )
    scenario = sample_scenario(
        scenario_id=0,
        seed=phase5.seed,
        ranges=phase5.tiers[0].uncertainty_ranges,
    )
    variant = config.variants[0]
    rollout = rollout_online_slack_mpc(
        tier_name=phase5.tiers[0].name,
        variant=variant,
        scenario=scenario,
        reference_profile=reference,
        plant_config=plant,
        solver=config.solver,
        thresholds=phase5.failure_thresholds,
    )

    assert "predicted_max_alpha_slack_rad" in rollout.columns
    assert "predicted_max_q_slack_radps" in rollout.columns
    assert "actuator_tau_prediction_s" in rollout.columns
    assert "failure_label" not in rollout.columns


def test_phase22_tiny_run_writes_artifacts(tmp_path: Path) -> None:
    raw = yaml.safe_load(
        Path("configs/phase22_actuator_aware_slack_mpc.yaml").read_text(
            encoding="utf-8"
        )
    )
    raw["scenario_count_per_tier"] = 1
    raw["max_time_s"] = 2.0
    raw["variants"] = [raw["variants"][0]]
    raw["variants"][0]["horizon_steps"] = 2
    raw["solver"]["max_iter"] = 25
    config_path = tmp_path / "phase22_tiny.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    artifacts = run_phase22_actuator_aware_slack_mpc(
        config_path=config_path,
        output_dir=tmp_path / "phase22",
    )

    for key in [
        "summary_csv",
        "rollouts_csv",
        "comparison_csv",
        "vs_phase20_csv",
        "success_png",
        "vs_phase20_png",
        "envelope_png",
        "failure_png",
        "solve_time_png",
        "predicted_slack_png",
    ]:
        assert Path(artifacts[key]).exists()
    assert len(artifacts["summary"]) == 2
    assert "strict_success_delta_count" in artifacts["vs_phase20"].columns


def test_compare_phase22_to_phase20_uses_strict_labels() -> None:
    import pandas as pd

    config = load_phase22_config("configs/phase22_actuator_aware_slack_mpc.yaml")
    summary = pd.DataFrame(
        {
            "tier": ["moderate"],
            "scenario_id": [0],
            "controller": ["online_slack_mpc"],
            "failure_label": ["alpha_corridor_violation"],
            "controlled_recovery": [True],
            "max_alpha_corridor_miss_rad": [0.01],
        }
    )

    comparison = compare_phase22_to_phase20(config=config, summary=summary)

    assert comparison["phase22_strict_success_count"].iloc[0] == 0
