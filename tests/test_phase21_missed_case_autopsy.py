from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from reentry_mpc.phase21 import (
    build_transfer_row,
    load_phase21_config,
    run_phase21_missed_case_autopsy,
    select_missed_oracle_feasible_cases,
)


def test_phase21_config_loads_sources() -> None:
    config = load_phase21_config("configs/phase21_missed_case_autopsy.yaml")

    assert config.baseline_controller == "ridge_safety"
    assert config.phase20_output_dir.exists()


def test_select_missed_oracle_feasible_cases_finds_phase20_misses() -> None:
    config = load_phase21_config("configs/phase21_missed_case_autopsy.yaml")
    missed = select_missed_oracle_feasible_cases(config)

    assert len(missed) >= 1
    assert missed["missed_oracle_feasible"].astype(bool).all()
    assert set(missed["controller"]) == {"ridge_safety"}


def test_build_transfer_row_detects_recovered_success() -> None:
    row = build_transfer_row(
        baseline_metrics={
            "failure_label": "alpha_corridor_violation",
            "max_alpha_corridor_miss_rad": 0.02,
            "first_alpha_violation_side": "high",
            "first_alpha_violation_time_s": 10.0,
        },
        replay_metrics={
            "failure_label": "success",
            "max_alpha_corridor_miss_rad": 0.0,
            "first_alpha_violation_side": "none",
            "first_alpha_violation_time_s": None,
        },
    )

    assert row["strict_success_recovered"]
    assert row["replay_minus_baseline_alpha_miss_rad"] == -0.02


def test_phase21_tiny_run_writes_artifacts(tmp_path: Path) -> None:
    raw = yaml.safe_load(
        Path("configs/phase21_missed_case_autopsy.yaml").read_text(encoding="utf-8")
    )
    raw["max_cases"] = 1
    config_path = tmp_path / "phase21_tiny.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    artifacts = run_phase21_missed_case_autopsy(
        config_path=config_path,
        output_dir=tmp_path / "phase21",
    )

    for key in [
        "summary_csv",
        "rollouts_csv",
        "comparison_csv",
        "missed_cases_csv",
        "transfer_png",
        "alpha_png",
        "command_png",
    ]:
        assert Path(artifacts[key]).exists()
    summary = pd.read_csv(artifacts["summary_csv"])
    assert len(summary) == 1
    assert "strict_success_recovered" in summary.columns
