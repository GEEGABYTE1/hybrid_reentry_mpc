from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from reentry_mpc.phase15 import (
    FaultCase,
    _fallback_counts,
    _fault_density_scale,
    load_phase15_config,
    run_phase15_fault_injection,
)


def test_fault_config_loads_all_faults() -> None:
    config = load_phase15_config("configs/phase15_fault_injection.yaml")

    assert len(config.faults) == 6
    assert "residual_nmpc_with_fallback" in config.controllers
    assert config.nmpc.horizon_steps > 0


def test_fault_density_scale_only_applies_when_active() -> None:
    fault = FaultCase(
        name="sudden_density_jump",
        trigger_time_s=10.0,
        parameters={"density_scale": 1.45},
    )

    assert np.isclose(_fault_density_scale(fault, False), 1.0)
    assert np.isclose(_fault_density_scale(fault, True), 1.45)


def test_fallback_counts_from_rollout() -> None:
    rollout = pd.DataFrame(
        {
            "fallback_action": [
                "none",
                "previous_feasible_control",
                "constraint_tightening",
                "lqr_safe_mode",
            ],
            "fault_active": [False, True, True, True],
        }
    )

    counts = _fallback_counts(rollout)

    assert counts["previous_feasible_control_count"] == 1
    assert counts["constraint_tightening_count"] == 1
    assert counts["lqr_safe_mode_count"] == 1
    assert counts["fault_active_count"] == 3


def test_phase15_tiny_artifact_run(tmp_path) -> None:
    raw = yaml.safe_load(
        Path("configs/phase15_fault_injection.yaml").read_text(encoding="utf-8")
    )
    raw["faults"] = raw["faults"][:1]
    raw["controllers"] = ["residual_nmpc_with_fallback"]
    raw["nmpc"]["horizon_steps"] = 3
    config_path = tmp_path / "phase15_tiny.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    artifacts = run_phase15_fault_injection(
        config_path=config_path,
        output_dir=tmp_path,
    )

    assert artifacts["summary_csv"].exists()
    assert artifacts["limitations_md"].exists()
    assert artifacts["success_png"].exists()
    assert len(artifacts["summary"]) == 1
