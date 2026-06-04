import json
from pathlib import Path

import numpy as np

from reentry_mpc.phase2 import (
    build_corridor_config,
    build_reference_profile,
    load_phase2_config,
    run_phase2_reference,
)


def test_phase2_reference_profile_contains_required_schedules() -> None:
    config = load_phase2_config("configs/phase2_reference.yaml")
    profile = build_reference_profile(config)

    required_columns = {
        "altitude_m",
        "velocity_mps",
        "mach",
        "alpha_ref_rad",
        "theta_ref_rad",
        "q_ref_radps",
        "dynamic_pressure_pa",
        "heating_proxy",
    }
    assert required_columns.issubset(profile.columns)
    assert profile["altitude_m"].is_monotonic_decreasing
    assert profile["velocity_mps"].is_monotonic_decreasing
    assert np.isfinite(profile[list(required_columns)]).all().all()


def test_alpha_reference_stays_inside_corridor() -> None:
    config = load_phase2_config("configs/phase2_reference.yaml")
    profile = build_reference_profile(config)

    assert (profile["alpha_ref_rad"] >= profile["alpha_min_rad"]).all()
    assert (profile["alpha_ref_rad"] <= profile["alpha_max_rad"]).all()
    assert (profile["alpha_min_rad"] >= config.constraints.alpha_min_abs_rad).all()
    assert (profile["alpha_max_rad"] <= config.constraints.alpha_max_abs_rad).all()


def test_flap_and_rate_constraints_are_written_to_profile() -> None:
    config = load_phase2_config("configs/phase2_reference.yaml")
    profile = build_reference_profile(config)

    assert (profile["flap_min_rad"] == config.constraints.flap_min_rad).all()
    assert (profile["flap_max_rad"] == config.constraints.flap_max_rad).all()
    assert (
        profile["flap_rate_min_radps"] == config.constraints.flap_rate_min_radps
    ).all()
    assert (
        profile["flap_rate_max_radps"] == config.constraints.flap_rate_max_radps
    ).all()


def test_corridor_config_json_contains_diagnostics() -> None:
    config = load_phase2_config("configs/phase2_reference.yaml")
    profile = build_reference_profile(config)
    corridor = build_corridor_config(profile, config)

    assert "constraints" in corridor
    assert "diagnostics" in corridor
    assert corridor["diagnostics"]["max_dynamic_pressure_pa"] > 0.0
    assert corridor["diagnostics"]["max_heating_proxy"] > 0.0


def test_phase2_runner_writes_requested_artifacts(tmp_path: Path) -> None:
    artifacts = run_phase2_reference(
        config_path="configs/phase2_reference.yaml",
        output_dir=tmp_path,
    )

    assert artifacts["reference_csv"].exists()
    assert artifacts["corridor_json"].exists()
    assert artifacts["mach_vs_altitude"].exists()
    assert artifacts["dynamic_pressure_vs_time"].exists()
    assert artifacts["alpha_reference_corridor"].exists()
    assert artifacts["flap_authority_vs_time"].exists()

    with artifacts["corridor_json"].open("r", encoding="utf-8") as handle:
        corridor = json.load(handle)
    assert corridor["constraints"]["flap_min_rad"] < 0.0
    assert corridor["constraints"]["flap_max_rad"] > 0.0
