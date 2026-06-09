from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from reentry_mpc.phase20 import (
    build_ceiling_gap,
    build_failure_diagnostics,
    load_phase20_config,
    run_phase20_full_oracle_imitation,
    summarize_phase20_comparison,
)


def test_phase20_config_loads_full_benchmark_defaults() -> None:
    config = load_phase20_config("configs/phase20_full_oracle_imitation.yaml")

    assert config.scenario_count_per_tier == 30
    assert len(config.variants) == 4
    assert config.feature_columns == [
        "alpha_error_to_center_rad",
        "q_radps",
        "alpha_margin_low_rad",
        "alpha_margin_high_rad",
        "alpha_min_rate_radps",
        "alpha_max_rate_radps",
        "mach_scaled",
        "dynamic_pressure_scaled",
        "actuator_tau_s",
    ]


def test_failure_diagnostics_classify_oracle_feasibility_and_misses() -> None:
    summary = pd.DataFrame(
        {
            "tier": ["moderate", "moderate", "stress"],
            "controller": ["ridge", "ridge", "ridge"],
            "scenario_id": [0, 1, 0],
            "failure_label": ["success", "alpha_corridor_violation", "success"],
            "max_alpha_corridor_miss_rad": [0.0, 0.02, 0.0],
            "oracle_max_alpha_miss_rad": [0.0, 0.005, 0.04],
            "oracle_feasible": [True, True, False],
            "oracle_near_feasible": [True, True, False],
        }
    )

    diagnostics = build_failure_diagnostics(summary)
    failed = diagnostics[diagnostics["scenario_id"].eq(1)].iloc[0]

    assert failed["oracle_feasibility_class"] == "strict_feasible"
    assert bool(failed["missed_oracle_feasible"])
    assert failed["online_minus_oracle_alpha_miss_rad"] == 0.015


def test_ceiling_gap_counts_missed_feasible_scenarios() -> None:
    summary = pd.DataFrame(
        {
            "tier": ["moderate", "moderate"],
            "controller": ["ridge", "ridge"],
            "scenario_id": [0, 1],
            "failure_label": ["success", "alpha_corridor_violation"],
            "controlled_recovery": [True, True],
            "oracle_feasible": [True, True],
            "oracle_near_feasible": [True, True],
            "online_minus_oracle_alpha_miss_rad": [0.0, 0.01],
            "max_alpha_corridor_miss_rad": [0.0, 0.01],
            "raw_applied_flap_gap_mean_rad": [0.02, 0.03],
        }
    )
    comparison = summarize_phase20_comparison(summary)
    ceiling = pd.DataFrame(
        {
            "tier": ["moderate"],
            "feasible_count": [2],
            "feasible_rate": [1.0],
            "near_feasible_count": [2],
            "near_feasible_rate": [1.0],
        }
    )

    gap = build_ceiling_gap(comparison, ceiling).iloc[0]

    assert gap["strict_success_count"] == 1
    assert gap["missed_feasible_scenarios_count"] == 1
    assert gap["online_success_beyond_oracle_count"] == 0
    assert gap["strict_success_gap_vs_oracle_rate"] == -0.5


def test_phase20_tiny_run_writes_required_artifacts(tmp_path: Path) -> None:
    raw = yaml.safe_load(
        Path("configs/phase20_full_oracle_imitation.yaml").read_text(encoding="utf-8")
    )
    raw["scenario_count_per_tier"] = 1
    raw["max_time_s"] = 3.0
    raw["oracle"]["horizon_steps"] = 3
    raw["solver"]["max_iter"] = 40
    raw["policy"]["variants"] = raw["policy"]["variants"][:2]
    config_path = tmp_path / "phase20_tiny.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    artifacts = run_phase20_full_oracle_imitation(
        config_path=config_path,
        output_dir=tmp_path / "phase20",
    )

    for key in [
        "summary_csv",
        "comparison_csv",
        "ceiling_gap_csv",
        "failure_diagnostics_csv",
        "success_png",
        "gap_png",
        "first_violation_png",
        "flap_lag_png",
        "envelope_png",
    ]:
        assert Path(artifacts[key]).exists()
    summary = pd.read_csv(artifacts["summary_csv"])
    assert set(summary["tier"]) == {"moderate", "stress"}
    assert set(summary["controller"]) == {"ridge_no_safety", "ridge_safety"}
