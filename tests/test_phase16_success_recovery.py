from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from reentry_mpc.phase16 import (
    Phase16Variant,
    _planning_profile,
    load_phase16_config,
    run_phase16_success_recovery,
    summarize_corridor_diagnostics,
)


def test_phase16_config_loads_variants() -> None:
    config = load_phase16_config("configs/phase16_success_recovery.yaml")

    assert len(config.controller_variants) == 6
    assert config.controller_variants[0].name == "nominal_nmpc_2s"
    assert config.controller_variants[-1].planning_alpha_buffer_rad > 0.0
    assert config.sweep_scenario_count_per_tier <= config.full_scenario_count_per_tier


def test_planning_buffer_does_not_mutate_evaluation_corridor() -> None:
    reference = pd.DataFrame(
        {
            "alpha_min_rad": [0.10],
            "alpha_max_rad": [0.30],
        }
    )
    variant = Phase16Variant(
        name="buffered",
        control_dt_s=0.5,
        horizon_steps=3,
        alpha_weight_scale=1.0,
        q_weight_scale=1.0,
        terminal_alpha_weight_scale=1.0,
        state_slack_scale=1.0,
        control_weight_scale=1.0,
        flap_rate_weight_scale=1.0,
        planning_alpha_buffer_rad=0.02,
    )

    planning = _planning_profile(reference, variant)

    assert np.isclose(reference["alpha_min_rad"].iloc[0], 0.10)
    assert np.isclose(reference["alpha_max_rad"].iloc[0], 0.30)
    assert np.isclose(planning["alpha_min_rad"].iloc[0], 0.12)
    assert np.isclose(planning["alpha_max_rad"].iloc[0], 0.28)


def test_corridor_diagnostics_detect_side_and_timing() -> None:
    rollout = pd.DataFrame(
        {
            "time_s": [0.0, 0.5, 1.0],
            "alpha_rad": [0.20, 0.34, 0.18],
            "alpha_min_rad": [0.10, 0.10, 0.10],
            "alpha_max_rad": [0.30, 0.30, 0.30],
            "q_radps": [0.0, 0.0, 0.0],
            "q_min_radps": [-0.08, -0.08, -0.08],
            "q_max_radps": [0.08, 0.08, 0.08],
            "solver_status": ["held", "Solve_Succeeded", "held"],
            "solve_time_s": [0.0, 0.01, 0.0],
        }
    )

    diagnostics = summarize_corridor_diagnostics(
        rollout=rollout, tolerance=0.001, control_dt_s=0.5
    )

    assert np.isclose(diagnostics["first_alpha_violation_time_s"], 0.5)
    assert diagnostics["first_alpha_violation_side"] == "high"
    assert diagnostics["first_alpha_violation_before_three_updates"]
    assert diagnostics["max_alpha_corridor_miss_rad"] > 0.0


def test_phase16_tiny_artifact_run(tmp_path) -> None:
    raw = yaml.safe_load(
        Path("configs/phase16_success_recovery.yaml").read_text(encoding="utf-8")
    )
    raw["sweep_scenario_count_per_tier"] = 1
    raw["max_time_s"] = 2.0
    raw["controller_variants"] = [
        raw["controller_variants"][0],
        raw["controller_variants"][1],
    ]
    raw["controller_variants"][0]["horizon_steps"] = 2
    raw["controller_variants"][1]["horizon_steps"] = 2
    raw["nmpc_base"]["solver"]["max_iter"] = 25
    config_path = tmp_path / "phase16_tiny.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    artifacts = run_phase16_success_recovery(
        config_path=config_path, output_dir=tmp_path
    )

    assert artifacts["summary_csv"].exists()
    assert artifacts["rollouts_csv"].exists()
    assert artifacts["comparison_csv"].exists()
    assert artifacts["success_png"].exists()
    assert len(artifacts["summary"]) == 4
    assert set(artifacts["summary"]["controller"]) == {
        "nominal_nmpc_2s",
        "fast_nmpc_1s",
    }
