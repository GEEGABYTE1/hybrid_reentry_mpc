from dataclasses import replace
from pathlib import Path

import yaml

from reentry_mpc.phase2 import build_reference_profile, load_phase2_config
from reentry_mpc.phase5 import run_phase5_monte_carlo, summarize_monte_carlo_rollout
from reentry_mpc.uncertainty import actuator_step, initialize_actuator, sample_scenario


def test_scenario_sampling_is_deterministic() -> None:
    ranges = _load_phase5_ranges()

    first = sample_scenario(scenario_id=0, seed=123, ranges=ranges)
    second = sample_scenario(scenario_id=0, seed=123, ranges=ranges)

    assert first == second
    assert first.to_flat_dict() == second.to_flat_dict()


def test_actuator_delay_and_lag_are_reproducible() -> None:
    ranges = _load_phase5_ranges()
    scenario = sample_scenario(scenario_id=0, seed=123, ranges=ranges)
    scenario = replace(scenario, actuator_lag_s=0.0, actuator_delay_s=1.0)
    reference = build_reference_profile(
        load_phase2_config("configs/phase2_reference.yaml")
    )
    actuator = initialize_actuator(scenario, dt=0.5)

    first, first_log = actuator_step(
        raw_command=0.25,
        actuator=actuator,
        scenario=scenario,
        row=reference.iloc[0],
        dt=0.5,
    )
    second, second_log = actuator_step(
        raw_command=0.25,
        actuator=actuator,
        scenario=scenario,
        row=reference.iloc[1],
        dt=0.5,
    )
    third, third_log = actuator_step(
        raw_command=0.25,
        actuator=actuator,
        scenario=scenario,
        row=reference.iloc[2],
        dt=0.5,
    )

    assert first == 0.0
    assert second == 0.0
    assert first_log["delta_flap_delayed_rad"] == 0.0
    assert second_log["delta_flap_delayed_rad"] == 0.0
    assert third_log["delta_flap_delayed_rad"] == 0.25
    assert third > 0.0


def test_phase5_runner_writes_requested_artifacts(tmp_path: Path) -> None:
    config_path = _write_small_config(tmp_path, scenario_count=2)

    artifacts = run_phase5_monte_carlo(config_path=config_path, output_dir=tmp_path)
    summary = artifacts["summary"]

    assert artifacts["summary_csv"].exists()
    assert artifacts["combined_rollouts_csv"].exists()
    assert artifacts["success_rates_figure"].exists()
    assert artifacts["alpha_error_envelopes_figure"].exists()
    assert artifacts["failure_mode_figure"].exists()
    assert artifacts["rms_error_histogram_figure"].exists()
    assert artifacts["worst_case_replay_figure"].exists()
    assert len(summary) == 6
    assert {"pid", "gain_scheduled_lqr", "nominal_nmpc"} == set(summary["controller"])
    assert set(summary["tier"]) == {"moderate"}
    for path in summary["trajectory_csv"]:
        assert Path(path).exists()
        assert "moderate" in Path(path).parts
    for path in summary["metrics_json"]:
        assert Path(path).exists()
        assert "moderate" in Path(path).parts


def test_failure_label_precedence_prefers_solver_failure(tmp_path: Path) -> None:
    config_path = _write_small_config(tmp_path, scenario_count=1)
    artifacts = run_phase5_monte_carlo(config_path=config_path, output_dir=tmp_path)
    rollout = artifacts["rollouts"].copy()
    rollout["solver_failure"] = True
    scenario = sample_scenario(scenario_id=0, seed=55, ranges=_load_phase5_ranges())
    thresholds = yaml.safe_load(Path(config_path).read_text())["failure_thresholds"]

    metrics = summarize_monte_carlo_rollout(
        rollout=rollout[rollout["controller"] == "pid"],
        tier_name="moderate",
        controller_name="pid",
        scenario=scenario,
        thresholds={key: float(value) for key, value in thresholds.items()},
    )

    assert metrics["failure_label"] == "solver_failure"


def _load_phase5_ranges() -> dict:
    raw = yaml.safe_load(Path("configs/phase5_monte_carlo.yaml").read_text())
    return raw["tiers"][0]["uncertainty_ranges"]


def _write_small_config(tmp_path: Path, scenario_count: int) -> Path:
    raw = yaml.safe_load(Path("configs/phase5_monte_carlo.yaml").read_text())
    raw["tiers"] = [raw["tiers"][0]]
    raw["tiers"][0]["scenario_count"] = scenario_count
    config_path = tmp_path / "phase5_small.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return config_path
