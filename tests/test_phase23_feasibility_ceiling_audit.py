from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from reentry_mpc.longitudinal import load_phase1_config
from reentry_mpc.phase2 import build_reference_profile, load_phase2_config
from reentry_mpc.phase5 import load_phase5_config
from reentry_mpc.phase17 import _maybe_truncate_reference
from reentry_mpc.phase18 import _downsample_for_oracle
from reentry_mpc.phase23 import (
    compare_phase23_ceiling,
    load_phase23_config,
    run_phase23_feasibility_ceiling_audit,
    solve_truth_consistent_slack_oracle,
)
from reentry_mpc.uncertainty import sample_scenario


def test_phase23_config_loads_oracle_settings() -> None:
    config = load_phase23_config("configs/phase23_feasibility_ceiling_audit.yaml")

    assert config.scenario_count_per_tier == 30
    assert config.oracle.alpha_slack_weight > config.oracle.command_weight
    assert config.phase20_output_dir.exists()
    assert config.phase22_output_dir.exists()


def test_truth_consistent_oracle_uses_uncertainty_fields() -> None:
    config = load_phase23_config("configs/phase23_feasibility_ceiling_audit.yaml")
    phase5 = load_phase5_config(config.phase5_config)
    plant = load_phase1_config(config.phase1_config)
    reference = _downsample_for_oracle(
        _maybe_truncate_reference(
            build_reference_profile(load_phase2_config(config.phase2_config)),
            3.0,
        ),
        config.oracle.dt_s,
    )
    scenario = sample_scenario(
        scenario_id=0,
        seed=phase5.seed,
        ranges=phase5.tiers[0].uncertainty_ranges,
    )
    trajectory, metrics = solve_truth_consistent_slack_oracle(
        tier_name=phase5.tiers[0].name,
        scenario=scenario,
        reference=reference,
        vehicle=plant.vehicle,
        aero=plant.aero,
        settings=config.oracle,
        solver=config.solver,
        classification=config.classification,
        tolerance=phase5.failure_thresholds["corridor_tolerance_rad"],
    )

    assert np.isclose(trajectory["density_scale"].iloc[0], scenario.density_scale)
    assert np.isclose(
        trajectory["external_disturbance_moment_nm"].iloc[0],
        scenario.external_disturbance_moment_nm,
    )
    assert metrics["solve_time_s"] >= 0.0
    assert "oracle_feasible" in metrics


def test_compare_phase23_ceiling_reports_gaps() -> None:
    config = load_phase23_config("configs/phase23_feasibility_ceiling_audit.yaml")
    ceiling = pd.DataFrame(
        {
            "tier": ["moderate", "stress"],
            "scenario_count": [30, 30],
            "feasible_count": [14, 12],
            "feasible_rate": [14 / 30, 12 / 30],
            "near_feasible_count": [22, 15],
            "near_feasible_rate": [22 / 30, 15 / 30],
            "median_max_alpha_miss_rad": [0.001, 0.02],
            "max_alpha_miss_rad": [0.03, 0.08],
        }
    )

    comparison = compare_phase23_ceiling(config=config, ceiling=ceiling)

    assert "phase23_minus_phase20_oracle_count" in comparison.columns
    assert "phase23_minus_phase22_online_count" in comparison.columns


def test_phase23_tiny_run_writes_artifacts(tmp_path: Path) -> None:
    raw = yaml.safe_load(
        Path("configs/phase23_feasibility_ceiling_audit.yaml").read_text(
            encoding="utf-8"
        )
    )
    raw["scenario_count_per_tier"] = 1
    raw["max_time_s"] = 3.0
    raw["oracle"]["horizon_steps"] = 3
    raw["solver"]["max_iter"] = 50
    config_path = tmp_path / "phase23_tiny.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    artifacts = run_phase23_feasibility_ceiling_audit(
        config_path=config_path,
        output_dir=tmp_path / "phase23",
    )

    for key in [
        "summary_csv",
        "trajectories_csv",
        "ceiling_csv",
        "comparison_csv",
        "ceiling_png",
        "comparison_png",
        "miss_png",
        "trajectory_png",
    ]:
        assert Path(artifacts[key]).exists()
    assert len(artifacts["summary"]) == 2
