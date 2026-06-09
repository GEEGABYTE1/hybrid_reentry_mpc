from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from reentry_mpc.phase17 import (
    Phase17Variant,
    _governed_reference,
    load_phase17_config,
    run_phase17_feasibility_safety,
    summarize_controlled_recovery,
)


def test_phase17_config_loads_variants() -> None:
    config = load_phase17_config("configs/phase17_feasibility_safety.yaml")

    assert len(config.variants) == 2
    assert config.variants[0].horizon_steps > 0
    assert config.recovery_thresholds.max_alpha_miss_rad > 0.0


def test_reference_governor_moves_alpha_reference_to_corridor_center() -> None:
    reference = pd.DataFrame(
        {
            "alpha_ref_rad": [0.20],
            "q_ref_radps": [0.03],
            "alpha_min_rad": [0.10],
            "alpha_max_rad": [0.30],
        }
    )
    variant = Phase17Variant(
        name="governed",
        control_dt_s=0.5,
        horizon_steps=3,
        use_reference_governor=True,
        alpha_buffer_rad=0.01,
        tracking_weight=1.0,
        q_weight=1.0,
        theta_weight=1.0,
        center_weight=1.0,
        terminal_center_weight=1.0,
        slack_weight=1.0,
        command_weight=1.0,
        command_rate_weight=1.0,
    )

    governed = _governed_reference(reference, variant)

    assert np.isclose(governed["alpha_ref_rad"].iloc[0], 0.20)
    assert np.isclose(governed["q_ref_radps"].iloc[0], 0.0)


def test_controlled_recovery_allows_small_bounded_miss() -> None:
    rollout = pd.DataFrame(
        {
            "alpha_rad": [0.2, 0.301],
            "alpha_min_rad": [0.1, 0.1],
            "alpha_max_rad": [0.3, 0.3],
            "q_radps": [0.0, 0.02],
            "q_min_radps": [-0.08, -0.08],
            "q_max_radps": [0.08, 0.08],
            "solver_failure": [False, False],
        }
    )
    config = load_phase17_config("configs/phase17_feasibility_safety.yaml")

    recovery = summarize_controlled_recovery(
        rollout=rollout, thresholds=config.recovery_thresholds
    )

    assert recovery["controlled_recovery"]
    assert recovery["controlled_recovery_max_alpha_miss_rad"] > 0.0


def test_phase17_tiny_artifact_run(tmp_path) -> None:
    raw = yaml.safe_load(
        Path("configs/phase17_feasibility_safety.yaml").read_text(encoding="utf-8")
    )
    raw["scenario_count_per_tier"] = 1
    raw["max_time_s"] = 2.0
    raw["variants"] = [raw["variants"][0]]
    raw["variants"][0]["horizon_steps"] = 2
    raw["solver"]["max_iter"] = 25
    config_path = tmp_path / "phase17_tiny.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    artifacts = run_phase17_feasibility_safety(
        config_path=config_path, output_dir=tmp_path
    )

    assert artifacts["summary_csv"].exists()
    assert artifacts["feasibility_csv"].exists()
    assert artifacts["comparison_csv"].exists()
    assert artifacts["success_png"].exists()
    assert len(artifacts["summary"]) == 2
