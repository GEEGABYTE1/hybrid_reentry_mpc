from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from reentry_mpc.longitudinal import load_phase1_config
from reentry_mpc.phase2 import build_reference_profile, load_phase2_config
from reentry_mpc.phase4 import _downsample_reference_profile
from reentry_mpc.phase7 import load_phase7_config, run_phase7_scenario_mpc
from reentry_mpc.scenario_mpc import solve_scenario_mpc_step


def test_phase7_config_loads_design_scenarios() -> None:
    config = load_phase7_config("configs/phase7_scenario_mpc.yaml")

    assert config.controller_name == "scenario_nmpc"
    assert len(config.scenario_mpc.design_scenarios) >= 2
    assert config.scenario_mpc.nmpc.horizon_steps == 6


def test_scenario_mpc_step_returns_finite_control() -> None:
    config = load_phase7_config("configs/phase7_scenario_mpc.yaml")
    plant = load_phase1_config("configs/phase1_open_loop.yaml")
    reference = build_reference_profile(
        load_phase2_config("configs/phase2_reference.yaml")
    )
    nmpc_reference = _downsample_reference_profile(
        reference, config.scenario_mpc.nmpc.dt
    )
    first = nmpc_reference.iloc[0]
    state = np.array(
        [first["alpha_ref_rad"], first["q_ref_radps"], first["theta_ref_rad"]],
        dtype=float,
    )

    control, log = solve_scenario_mpc_step(
        state=state,
        previous_flap_rad=0.0,
        horizon=nmpc_reference.iloc[: config.scenario_mpc.nmpc.horizon_steps + 1],
        vehicle=plant.vehicle,
        aero=plant.aero,
        config=config.scenario_mpc,
    )

    assert np.isfinite(control)
    assert first["flap_min_rad"] - 1.0e-6 <= control <= first["flap_max_rad"] + 1.0e-6
    assert log["design_scenario_count"] == len(config.scenario_mpc.design_scenarios)


def test_phase7_runner_writes_requested_artifacts(tmp_path: Path) -> None:
    config_path = _write_small_phase7_config(tmp_path)

    artifacts = run_phase7_scenario_mpc(
        config_path=config_path, output_dir=tmp_path / "phase7"
    )
    summary = artifacts["summary"]
    comparison = artifacts["comparison"]

    assert artifacts["summary_csv"].exists()
    assert artifacts["rollouts_csv"].exists()
    assert artifacts["comparison_csv"].exists()
    assert artifacts["success_rates_figure"].exists()
    assert artifacts["comparison_success_rates_figure"].exists()
    assert artifacts["alpha_error_envelopes_figure"].exists()
    assert artifacts["failure_mode_figure"].exists()
    assert artifacts["solve_time_figure"].exists()
    assert len(summary) == 2
    assert set(summary["tier"]) == {"moderate", "stress"}
    assert set(summary["controller"]) == {"scenario_nmpc"}
    assert "scenario_nmpc" in set(comparison["controller"])
    for path in summary["trajectory_csv"]:
        assert Path(path).exists()
    for path in summary["metrics_json"]:
        assert Path(path).exists()


def _write_small_phase7_config(tmp_path: Path) -> Path:
    phase5_raw = yaml.safe_load(Path("configs/phase5_monte_carlo.yaml").read_text())
    for tier in phase5_raw["tiers"]:
        tier["scenario_count"] = 1
    phase5_config_path = tmp_path / "phase5_small.yaml"
    phase5_config_path.write_text(
        yaml.safe_dump(phase5_raw, sort_keys=False), encoding="utf-8"
    )

    baseline = pd.DataFrame(
        [
            {
                "tier": "moderate",
                "controller": "tightened_nmpc",
                "scenario_id": 0,
                "failure_label": "alpha_corridor_violation",
            },
            {
                "tier": "stress",
                "controller": "tightened_nmpc",
                "scenario_id": 0,
                "failure_label": "unstable_response",
            },
        ]
    )
    baseline_path = tmp_path / "phase6_summary.csv"
    baseline.to_csv(baseline_path, index=False)

    phase7_raw = yaml.safe_load(Path("configs/phase7_scenario_mpc.yaml").read_text())
    phase7_raw["phase5_config"] = str(phase5_config_path)
    phase7_raw["phase6_summary"] = str(baseline_path)
    phase7_raw["scenario_mpc"]["max_scenarios_per_tier"] = 1
    phase7_raw["scenario_mpc"]["horizon_steps"] = 3
    phase7_raw["scenario_mpc"]["design_scenarios"] = phase7_raw["scenario_mpc"][
        "design_scenarios"
    ][:2]
    config_path = tmp_path / "phase7_small.yaml"
    config_path.write_text(
        yaml.safe_dump(phase7_raw, sort_keys=False), encoding="utf-8"
    )
    return config_path
