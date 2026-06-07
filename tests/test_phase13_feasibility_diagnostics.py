from __future__ import annotations

import numpy as np
import pandas as pd

from reentry_mpc.phase13 import (
    compute_feasibility_diagnostics,
    compute_slack_summary,
    run_phase13_feasibility_diagnostics,
)


def test_feasibility_diagnostics_compute_alpha_slack() -> None:
    summary = pd.DataFrame(
        [
            {
                "tier": "moderate",
                "controller": "nominal_nmpc",
                "scenario_id": 0,
                "failure_label": "alpha_corridor_violation",
                "rms_alpha_error_rad": 0.02,
                "max_alpha_error_rad": 0.04,
            }
        ]
    )
    rollouts = pd.DataFrame(
        {
            "tier": ["moderate", "moderate"],
            "controller": ["nominal_nmpc", "nominal_nmpc"],
            "scenario_id": [0, 0],
            "time_s": [0.0, 0.5],
            "alpha_rad": [0.10, 0.13],
            "alpha_min_rad": [0.08, 0.08],
            "alpha_max_rad": [0.12, 0.12],
            "q_radps": [0.0, 0.0],
            "q_min_radps": [-0.1, -0.1],
            "q_max_radps": [0.1, 0.1],
            "flap_saturated": [False, True],
            "flap_rate_saturated": [False, False],
            "initial_alpha_error_rad": [0.0, 0.0],
            "initial_q_error_radps": [0.0, 0.0],
            "actuator_lag_s": [0.1, 0.1],
            "actuator_delay_s": [0.5, 0.5],
            "density_scale": [1.0, 1.0],
            "cm_delta_scale": [1.0, 1.0],
            "external_disturbance_moment_nm": [0.0, 0.0],
        }
    )

    diagnostics = compute_feasibility_diagnostics(summary=summary, rollouts=rollouts)

    assert np.isclose(diagnostics["needed_alpha_corridor_expansion_rad"].iloc[0], 0.01)
    assert np.isclose(diagnostics["first_alpha_violation_time_s"].iloc[0], 0.5)
    assert np.isclose(diagnostics["flap_saturation_fraction"].iloc[0], 0.5)


def test_slack_summary_bins_are_created() -> None:
    diagnostics = pd.DataFrame(
        {
            "tier": ["moderate", "moderate"],
            "controller": ["nominal_nmpc", "nominal_nmpc"],
            "needed_alpha_corridor_expansion_rad": [0.0, 0.003],
        }
    )

    summary = compute_slack_summary(
        diagnostics=diagnostics,
        bins=[0.0, 0.0025, 0.005],
    )

    assert int(summary["rollout_count"].sum()) == 2
    assert "alpha_slack_bin_rad" in summary.columns


def test_phase13_artifact_run(tmp_path) -> None:
    artifacts = run_phase13_feasibility_diagnostics(
        config_path="configs/phase13_feasibility_diagnostics.yaml",
        output_dir=tmp_path,
    )

    assert artifacts["diagnostics_csv"].exists()
    assert artifacts["slack_summary_csv"].exists()
    assert artifacts["failure_timing_csv"].exists()
    assert artifacts["slack_histogram_png"].exists()
    diagnostics = artifacts["diagnostics"]
    assert "needed_alpha_corridor_expansion_rad" in diagnostics.columns
    assert diagnostics["needed_alpha_corridor_expansion_rad"].ge(0.0).all()
