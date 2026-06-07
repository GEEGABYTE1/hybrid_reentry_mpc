from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from reentry_mpc.phase2 import build_reference_profile, load_phase2_config
from reentry_mpc.phase14 import (
    _interpolated_horizon,
    run_phase14_realtime_timing,
    summarize_realtime_timings,
)


def test_interpolated_horizon_uses_requested_dt() -> None:
    reference = build_reference_profile(
        load_phase2_config("configs/phase2_reference.yaml")
    )
    horizon = _interpolated_horizon(reference, start_time=1.0, rows=4, dt=0.05)

    np.testing.assert_allclose(np.diff(horizon["time_s"]), [0.05, 0.05, 0.05])
    assert {"alpha_ref_rad", "dynamic_pressure_pa", "mach"}.issubset(horizon.columns)


def test_summarize_realtime_timings_budget_flags() -> None:
    raw = pd.DataFrame(
        {
            "controller": ["pid", "pid"],
            "horizon_steps": [5, 5],
            "control_frequency_hz": [10.0, 10.0],
            "warm_start": [False, False],
            "sample_idx": [0, 1],
            "solve_time_ms": [0.1, 0.2],
            "nn_inference_time_ms": [0.0, 0.0],
            "total_loop_time_ms": [0.2, 0.3],
            "solver_failed": [False, False],
        }
    )
    phase5 = pd.DataFrame(
        {
            "controller": ["pid", "pid"],
            "failure_label": ["success", "alpha_corridor_violation"],
        }
    )
    summary = summarize_realtime_timings(
        raw_timings=raw,
        budgets_ms={"10 Hz": 100.0, "20 Hz": 50.0, "50 Hz": 20.0},
        phase5_summary=phase5,
        phase12_summary=pd.DataFrame(),
    )

    assert bool(summary["meets_50hz_budget_p95"].iloc[0])
    assert np.isclose(summary["success_rate"].iloc[0], 0.5)


def test_phase14_tiny_artifact_run(tmp_path) -> None:
    raw = yaml.safe_load(
        Path("configs/phase14_realtime_timing.yaml").read_text(encoding="utf-8")
    )
    raw["horizon_lengths"] = [3]
    raw["control_frequencies_hz"] = [10.0]
    raw["sample_count"] = 1
    config_path = tmp_path / "phase14_tiny.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    artifacts = run_phase14_realtime_timing(
        config_path=config_path,
        output_dir=tmp_path,
    )

    assert artifacts["summary_csv"].exists()
    assert artifacts["summary_md"].exists()
    assert artifacts["histogram_png"].exists()
    assert set(artifacts["summary"]["controller"]) == {
        "pid",
        "gain_scheduled_lqr",
        "nominal_nmpc",
        "residual_corrected_nmpc",
        "residual_corrected_tightened_nmpc",
    }
