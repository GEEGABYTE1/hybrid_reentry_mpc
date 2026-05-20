"""End-to-end smoke experiment pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from reentry_mpc.artifacts import (
    append_blog_log,
    ensure_output_dirs,
    write_figures,
    write_metrics,
)
from reentry_mpc.config import load_config
from reentry_mpc.metrics import summarize_trajectory
from reentry_mpc.simulation import ControllerSpec, run_attitude_simulation


def run_smoke_experiment(
    config_path: str | Path = "configs/smoke.yaml",
    output_dir: str | Path = "outputs",
) -> dict[str, Path | pd.DataFrame]:
    """Run baseline and learning-augmented controllers and save artifacts."""

    config = load_config(config_path)
    artifact_dirs = ensure_output_dirs(output_dir)
    controllers = [
        ControllerSpec(
            name="baseline_pd",
            kp=config.baseline_kp,
            kd=config.baseline_kd,
        ),
        ControllerSpec(
            name="learning_augmented_pd",
            kp=config.baseline_kp,
            kd=config.baseline_kd,
            learned_bias_gain=config.learned_bias_gain,
        ),
    ]

    trajectories = [
        run_attitude_simulation(config, controller) for controller in controllers
    ]
    trajectory = pd.concat(trajectories, ignore_index=True)
    summary = summarize_trajectory(trajectory)

    metric_paths = write_metrics(trajectory, summary, artifact_dirs)
    figure_paths = write_figures(trajectory, artifact_dirs)
    log_path = append_blog_log(
        artifact_dirs, phase="smoke", seed=config.seed, metrics=summary
    )

    return {
        "trajectory": trajectory,
        "summary": summary,
        "blog_log": log_path,
        **metric_paths,
        **figure_paths,
    }
