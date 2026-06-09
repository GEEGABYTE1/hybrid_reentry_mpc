from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from reentry_mpc.phase19 import (
    OraclePolicyConfig,
    build_policy_training_frame,
    fit_oracle_policy,
    load_phase19_config,
    predict_oracle_policy,
    run_phase19_oracle_imitation,
)


def _policy_config() -> OraclePolicyConfig:
    return OraclePolicyConfig(
        ridge_lambda=1.0e-3,
        use_only_feasible_or_near_feasible=True,
        feature_columns=[
            "alpha_error_to_center_rad",
            "q_radps",
            "alpha_margin_low_rad",
            "alpha_margin_high_rad",
            "actuator_tau_s",
        ],
        safety_blend_gain=5.0,
        safety_margin_rad=0.02,
        command_clip_rad=0.35,
    )


def _oracle_summary() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tier": ["moderate", "stress"],
            "scenario_id": [0, 0],
            "oracle_near_feasible": [True, True],
            "oracle_feasible": [True, False],
            "oracle_max_alpha_miss_rad": [0.0, 0.012],
        }
    )


def _oracle_trajectories() -> pd.DataFrame:
    rows = []
    for tier in ["moderate", "stress"]:
        for idx, alpha in enumerate([0.09, 0.12, 0.16, 0.18]):
            rows.append(
                {
                    "tier": tier,
                    "scenario_id": 0,
                    "time_s": float(idx),
                    "alpha_rad": alpha,
                    "q_radps": 0.01 * idx,
                    "alpha_min_rad": 0.05,
                    "alpha_max_rad": 0.19,
                    "delta_flap_raw_rad": 0.18 + 0.04 * idx,
                    "actuator_tau_prediction_s": 0.5 if tier == "moderate" else 1.0,
                }
            )
    return pd.DataFrame(rows)


def test_phase19_config_loads_policy_features() -> None:
    config = load_phase19_config("configs/phase19_oracle_imitation.yaml")
    assert config.policy.feature_columns == [
        "alpha_error_to_center_rad",
        "q_radps",
        "alpha_margin_low_rad",
        "alpha_margin_high_rad",
        "actuator_tau_s",
    ]
    assert config.scenario_count_per_tier > 0


def test_policy_training_frame_has_center_and_margin_features() -> None:
    frame = build_policy_training_frame(
        oracle_summary=_oracle_summary(),
        oracle_trajectories=_oracle_trajectories(),
        use_only_feasible_or_near_feasible=True,
    )
    expected = {
        "alpha_error_to_center_rad",
        "alpha_margin_low_rad",
        "alpha_margin_high_rad",
        "actuator_tau_s",
    }
    assert expected.issubset(frame.columns)
    first = frame.iloc[0]
    assert np.isclose(first["alpha_error_to_center_rad"], -0.03)
    assert np.isclose(first["alpha_margin_low_rad"], 0.04)


def test_fit_oracle_policy_is_deterministic_and_predicts_finite_command() -> None:
    config = _policy_config()
    policy_a, training_a = fit_oracle_policy(
        oracle_summary=_oracle_summary(),
        oracle_trajectories=_oracle_trajectories(),
        config=config,
    )
    policy_b, training_b = fit_oracle_policy(
        oracle_summary=_oracle_summary(),
        oracle_trajectories=_oracle_trajectories(),
        config=config,
    )
    assert training_a.equals(training_b)
    np.testing.assert_allclose(policy_a.coefficients, policy_b.coefficients)
    command = predict_oracle_policy(
        policy_a,
        {
            "alpha_error_to_center_rad": 0.01,
            "q_radps": 0.02,
            "alpha_margin_low_rad": 0.08,
            "alpha_margin_high_rad": 0.06,
            "actuator_tau_s": 0.5,
        },
    )
    assert np.isfinite(command)


def test_phase19_tiny_run_writes_artifacts(tmp_path: Path) -> None:
    oracle_summary = tmp_path / "oracle_summary.csv"
    oracle_trajectories = tmp_path / "oracle_trajectories.csv"
    _oracle_summary().to_csv(oracle_summary, index=False)
    _oracle_trajectories().to_csv(oracle_trajectories, index=False)
    config_path = tmp_path / "phase19.yaml"
    config = yaml.safe_load(Path("configs/phase19_oracle_imitation.yaml").read_text())
    config.update(
        {
            "oracle_summary": str(oracle_summary),
            "oracle_trajectories": str(oracle_trajectories),
            "scenario_count_per_tier": 1,
            "max_time_s": 3.0,
        }
    )
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    artifacts = run_phase19_oracle_imitation(
        config_path=config_path,
        output_dir=tmp_path / "phase19",
    )
    assert Path(artifacts["summary_csv"]).exists()
    assert Path(artifacts["comparison_csv"]).exists()
    assert Path(artifacts["success_png"]).exists()
    summary = pd.read_csv(artifacts["summary_csv"])
    assert set(summary["tier"]) == {"moderate", "stress"}
    assert summary["trajectory_csv"].map(Path).map(Path.exists).all()
