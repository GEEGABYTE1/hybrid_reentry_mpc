from pathlib import Path

import pandas as pd
import yaml

from reentry_mpc.phase2 import build_reference_profile, load_phase2_config
from reentry_mpc.phase6 import (
    TighteningMargins,
    run_phase6_robust_mpc,
    tighten_reference_profile,
)


def test_tighten_reference_profile_preserves_original_profile() -> None:
    reference = build_reference_profile(
        load_phase2_config("configs/phase2_reference.yaml")
    )
    margins = TighteningMargins(alpha_margin_rad=0.004, q_margin_radps=0.003)

    tightened = tighten_reference_profile(reference, margins)

    assert (tightened["alpha_min_rad"] == reference["alpha_min_rad"] + 0.004).all()
    assert (tightened["alpha_max_rad"] == reference["alpha_max_rad"] - 0.004).all()
    assert (tightened["q_min_radps"] == reference["q_min_radps"] + 0.003).all()
    assert (tightened["q_max_radps"] == reference["q_max_radps"] - 0.003).all()
    assert not reference.equals(tightened)


def test_phase6_runner_writes_requested_artifacts(tmp_path: Path) -> None:
    config_path = _write_small_phase6_config(tmp_path)

    artifacts = run_phase6_robust_mpc(
        config_path=config_path, output_dir=tmp_path / "phase6"
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
    assert artifacts["worst_case_replay_figure"].exists()
    assert len(summary) == 2
    assert set(summary["tier"]) == {"moderate", "stress"}
    assert set(summary["controller"]) == {"tightened_nmpc"}
    assert "nominal_nmpc" in set(comparison["controller"])
    assert "tightened_nmpc" in set(comparison["controller"])
    for path in summary["trajectory_csv"]:
        assert Path(path).exists()
    for path in summary["metrics_json"]:
        assert Path(path).exists()


def _write_small_phase6_config(tmp_path: Path) -> Path:
    phase5_raw = yaml.safe_load(Path("configs/phase5_monte_carlo.yaml").read_text())
    for tier in phase5_raw["tiers"]:
        tier["scenario_count"] = 1
    phase5_config_path = tmp_path / "phase5_small.yaml"
    phase5_config_path.write_text(
        yaml.safe_dump(phase5_raw, sort_keys=False), encoding="utf-8"
    )

    phase5_summary = pd.DataFrame(
        [
            {
                "tier": "moderate",
                "controller": "nominal_nmpc",
                "scenario_id": 0,
                "failure_label": "alpha_corridor_violation",
            },
            {
                "tier": "stress",
                "controller": "nominal_nmpc",
                "scenario_id": 0,
                "failure_label": "unstable_response",
            },
        ]
    )
    phase5_summary_path = tmp_path / "phase5_summary.csv"
    phase5_summary.to_csv(phase5_summary_path, index=False)

    phase6_raw = yaml.safe_load(Path("configs/phase6_robust_mpc.yaml").read_text())
    phase6_raw["phase5_config"] = str(phase5_config_path)
    phase6_raw["phase5_summary"] = str(phase5_summary_path)
    config_path = tmp_path / "phase6_small.yaml"
    config_path.write_text(
        yaml.safe_dump(phase6_raw, sort_keys=False), encoding="utf-8"
    )
    return config_path
