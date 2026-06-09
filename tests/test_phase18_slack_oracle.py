from __future__ import annotations

from pathlib import Path

import yaml

from reentry_mpc.phase18 import (
    load_phase18_config,
    run_phase18_slack_oracle,
    summarize_feasibility_ceiling,
)


def test_phase18_config_loads_oracle_settings() -> None:
    config = load_phase18_config("configs/phase18_slack_oracle.yaml")

    assert config.oracle.horizon_steps > 0
    assert config.oracle.alpha_slack_weight > config.oracle.command_weight
    assert config.classification.near_feasible_alpha_miss_rad > (
        config.classification.feasible_alpha_miss_rad
    )


def test_feasibility_ceiling_summarizes_rates() -> None:
    import pandas as pd

    summary = pd.DataFrame(
        {
            "tier": ["moderate", "moderate", "stress"],
            "scenario_id": [0, 1, 0],
            "oracle_feasible": [True, False, False],
            "oracle_near_feasible": [True, True, False],
            "oracle_max_alpha_miss_rad": [0.0, 0.01, 0.08],
        }
    )

    ceiling = summarize_feasibility_ceiling(summary)
    moderate = ceiling[ceiling["tier"].eq("moderate")].iloc[0]

    assert moderate["feasible_rate"] == 0.5
    assert moderate["near_feasible_rate"] == 1.0


def test_phase18_tiny_artifact_run(tmp_path) -> None:
    raw = yaml.safe_load(
        Path("configs/phase18_slack_oracle.yaml").read_text(encoding="utf-8")
    )
    raw["scenario_count_per_tier"] = 1
    raw["max_time_s"] = 3.0
    raw["oracle"]["horizon_steps"] = 3
    raw["solver"]["max_iter"] = 40
    config_path = tmp_path / "phase18_tiny.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    artifacts = run_phase18_slack_oracle(
        config_path=config_path,
        output_dir=tmp_path,
    )

    assert artifacts["summary_csv"].exists()
    assert artifacts["trajectories_csv"].exists()
    assert artifacts["ceiling_csv"].exists()
    assert artifacts["ceiling_png"].exists()
    assert len(artifacts["summary"]) == 2
