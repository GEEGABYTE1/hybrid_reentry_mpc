from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from reentry_mpc.longitudinal import load_phase1_config
from reentry_mpc.phase2 import build_reference_profile, load_phase2_config
from reentry_mpc.phase5 import load_phase5_config
from reentry_mpc.phase17 import _maybe_truncate_reference
from reentry_mpc.phase19 import fit_oracle_policy
from reentry_mpc.phase24 import (
    compare_phase24_to_phase22,
    compare_phase24_to_phase23,
    load_phase24_config,
    policy_command_from_state,
    rollout_hybrid_imitation_slack_mpc,
    run_phase24_hybrid_imitation_mpc,
)
from reentry_mpc.uncertainty import sample_scenario


def test_phase24_config_loads_variants_and_policy() -> None:
    config = load_phase24_config("configs/phase24_hybrid_imitation_mpc.yaml")

    assert len(config.variants) == 3
    assert config.phase23_oracle_summary.exists()
    assert config.phase23_oracle_trajectories.exists()
    assert config.policy.feature_columns


def test_policy_command_from_state_is_finite() -> None:
    config = load_phase24_config("configs/phase24_hybrid_imitation_mpc.yaml")
    oracle_summary = pd.read_csv(config.phase23_oracle_summary)
    oracle_trajectories = pd.read_csv(config.phase23_oracle_trajectories)
    policy, _training = fit_oracle_policy(
        oracle_summary=oracle_summary,
        oracle_trajectories=oracle_trajectories,
        config=config.policy,
    )
    reference = _maybe_truncate_reference(
        build_reference_profile(load_phase2_config(config.phase2_config)),
        2.0,
    )
    row = reference.iloc[0]
    command = policy_command_from_state(
        policy=policy,
        policy_config=config.policy,
        state=np.array(
            [row["alpha_ref_rad"], row["q_ref_radps"], row["theta_ref_rad"]]
        ),
        row=row,
        actuator_tau_s=0.7,
        prev_row=None,
        dt=0.5,
    )

    assert np.isfinite(command)
    assert abs(command) <= config.policy.command_clip_rad


def test_hybrid_rollout_contains_policy_and_mpc_fields() -> None:
    config = load_phase24_config("configs/phase24_hybrid_imitation_mpc.yaml")
    phase5 = load_phase5_config(config.phase5_config)
    plant = load_phase1_config(config.phase1_config)
    reference = _maybe_truncate_reference(
        build_reference_profile(load_phase2_config(config.phase2_config)),
        2.0,
    )
    oracle_summary = pd.read_csv(config.phase23_oracle_summary)
    oracle_trajectories = pd.read_csv(config.phase23_oracle_trajectories)
    policy, _training = fit_oracle_policy(
        oracle_summary=oracle_summary,
        oracle_trajectories=oracle_trajectories,
        config=config.policy,
    )
    scenario = sample_scenario(
        scenario_id=0,
        seed=phase5.seed,
        ranges=phase5.tiers[0].uncertainty_ranges,
    )
    variant = config.variants[0]
    rollout = rollout_hybrid_imitation_slack_mpc(
        tier_name=phase5.tiers[0].name,
        variant=variant,
        scenario=scenario,
        reference_profile=reference,
        plant_config=plant,
        solver=config.solver,
        thresholds=phase5.failure_thresholds,
        policy=policy,
        policy_config=config.policy,
    )

    assert "policy_delta_flap_raw_rad" in rollout.columns
    assert "optimized_delta_flap_raw_rad" in rollout.columns
    assert "policy_mpc_command_gap_rad" in rollout.columns


def test_phase24_tiny_run_writes_artifacts(tmp_path: Path) -> None:
    raw = yaml.safe_load(
        Path("configs/phase24_hybrid_imitation_mpc.yaml").read_text(encoding="utf-8")
    )
    raw["scenario_count_per_tier"] = 1
    raw["max_time_s"] = 2.0
    raw["variants"] = [raw["variants"][0]]
    raw["variants"][0]["horizon_steps"] = 2
    raw["solver"]["max_iter"] = 25
    config_path = tmp_path / "phase24_tiny.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    artifacts = run_phase24_hybrid_imitation_mpc(
        config_path=config_path,
        output_dir=tmp_path / "phase24",
    )

    for key in [
        "summary_csv",
        "rollouts_csv",
        "comparison_csv",
        "vs_phase22_csv",
        "ceiling_gap_csv",
        "policy_json",
        "training_csv",
        "success_png",
        "vs_phase22_png",
        "ceiling_gap_png",
        "timing_png",
        "command_gap_png",
        "envelope_png",
    ]:
        assert Path(artifacts[key]).exists()
    assert len(artifacts["summary"]) == 2


def test_phase24_comparisons_do_not_redefine_success() -> None:
    config = load_phase24_config("configs/phase24_hybrid_imitation_mpc.yaml")
    summary = pd.DataFrame(
        {
            "tier": ["moderate"],
            "scenario_id": [0],
            "controller": ["hybrid"],
            "failure_label": ["alpha_corridor_violation"],
            "controlled_recovery": [True],
            "oracle_feasible": [True],
            "oracle_near_feasible": [True],
            "rms_alpha_error_rad": [0.0],
            "max_alpha_corridor_miss_rad": [0.01],
            "mean_solve_time_s": [0.01],
            "p95_solve_time_s": [0.02],
            "mean_policy_inference_time_s": [0.0001],
            "mean_policy_mpc_command_gap_rad": [0.0],
        }
    )

    vs_phase22 = compare_phase24_to_phase22(config=config, summary=summary)
    ceiling_gap = compare_phase24_to_phase23(config=config, summary=summary)

    assert vs_phase22["strict_success_count"].iloc[0] == 0
    assert ceiling_gap["strict_gap_to_audited_ceiling"].iloc[0] < 0
