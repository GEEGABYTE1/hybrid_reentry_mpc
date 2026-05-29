
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from reentry_mpc.artifacts import plt
from reentry_mpc.baseline_controllers import (
    GainScheduledLQRController,
    PIDController,
    build_lqr_controller,
)
from reentry_mpc.baseline_metrics import summarize_all
from reentry_mpc.baseline_rollout import rollout_controller
from reentry_mpc.longitudinal import Phase1Config, load_phase1_config
from reentry_mpc.phase2 import build_reference_profile, load_phase2_config


@dataclass(frozen=True)
class Phase3Config:
    seed: int
    phase1_config: Path
    phase2_config: Path
    initial_error: np.ndarray
    pid: dict[str, float]
    lqr: dict[str, Any]
    success_thresholds: dict[str, float]


def load_phase3_config(path: str | Path) -> Phase3Config:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)
    return Phase3Config(
        seed=int(raw["seed"]),
        phase1_config=Path(raw["phase1_config"]),
        phase2_config=Path(raw["phase2_config"]),
        initial_error=np.array(
            [
                float(raw["initial_error"]["alpha_rad"]),
                float(raw["initial_error"]["q_radps"]),
                float(raw["initial_error"]["theta_rad"]),
            ],
            dtype=float,
        ),
        pid={key: float(value) for key, value in raw["pid"].items()},
        lqr=raw["lqr"],
        success_thresholds={
            key: float(value) for key, value in raw["success_thresholds"].items()
        },
    )


def build_controllers(
    *, config: Phase3Config, plant_config: Phase1Config, dt: float
) -> dict[str, PIDController | GainScheduledLQRController]:
    pid_controller = PIDController(**config.pid)
    lqr_controller = build_lqr_controller(
        schedule_points=config.lqr["schedule_points"],
        q_weights={key: float(value) for key, value in config.lqr["q_weights"].items()},
        r_weight=float(config.lqr["r_weight"]),
        dt=dt,
        vehicle=plant_config.vehicle,
        aero=plant_config.aero,
    )
    return {
        "pid": pid_controller,
        "gain_scheduled_lqr": lqr_controller,
    }


def run_phase3_baselines(
    config_path: str | Path = "configs/phase3_baselines.yaml",
    output_dir: str | Path = "outputs/phase3_baselines",
) -> dict[str, Path | pd.DataFrame]:
    config = load_phase3_config(config_path)
    plant_config = load_phase1_config(config.phase1_config)
    reference_config = load_phase2_config(config.phase2_config)
    reference_profile = build_reference_profile(reference_config)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    first_reference = reference_profile.iloc[0]
    initial_state = (
        np.array(
            [
                first_reference["alpha_ref_rad"],
                first_reference["q_ref_radps"],
                first_reference["theta_ref_rad"],
            ],
            dtype=float,
        )
        + config.initial_error
    )
    dt = float(
        reference_profile["time_s"].iloc[1] - reference_profile["time_s"].iloc[0]
    )
    controllers = build_controllers(config=config, plant_config=plant_config, dt=dt)

    rollouts = {
        name: rollout_controller(
            controller_name=name,
            controller=controller,
            reference_profile=reference_profile,
            vehicle=plant_config.vehicle,
            aero=plant_config.aero,
            initial_state=initial_state,
        )
        for name, controller in controllers.items()
    }
    combined_rollout = pd.concat(rollouts.values(), ignore_index=True)
    metrics = summarize_all(rollouts, config.success_thresholds)

    rollout_path = output_path / "baseline_rollouts.csv"
    metrics_path = output_path / "baseline_metrics_table.csv"
    summary_path = output_path / "baseline_metrics_summary.md"
    combined_rollout.to_csv(rollout_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    summary_path.write_text(_metrics_summary_markdown(metrics), encoding="utf-8")

    figure_paths = write_phase3_figures(combined_rollout, output_path)
    return {
        "rollout_csv": rollout_path,
        "metrics_csv": metrics_path,
        "summary_md": summary_path,
        "rollouts": combined_rollout,
        "metrics": metrics,
        **figure_paths,
    }


def write_phase3_figures(rollouts: pd.DataFrame, output_dir: Path) -> dict[str, Path]:
    tracking_path = output_dir / "pid_vs_lqr_tracking.png"
    flap_path = output_dir / "pid_vs_lqr_flap_commands.png"

    fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.4), sharex=True)
    for controller, frame in rollouts.groupby("controller", sort=True):
        axes[0].plot(frame["time_s"], frame["alpha_rad"], label=controller)
        axes[1].plot(frame["time_s"], frame["q_radps"], label=controller)
    reference = rollouts.groupby("time_s", as_index=False).first()
    axes[0].plot(
        reference["time_s"],
        reference["alpha_ref_rad"],
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="alpha_ref",
    )
    axes[0].fill_between(
        reference["time_s"],
        reference["alpha_min_rad"],
        reference["alpha_max_rad"],
        color="gray",
        alpha=0.2,
        label="alpha corridor",
    )
    axes[1].plot(
        reference["time_s"],
        reference["q_ref_radps"],
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="q_ref",
    )
    axes[0].set_ylabel("Alpha (rad)")
    axes[1].set_ylabel("Pitch rate (rad/s)")
    axes[1].set_xlabel("Time (s)")
    axes[0].legend(loc="best")
    axes[1].legend(loc="best")
    fig.tight_layout()
    fig.savefig(tracking_path, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    for controller, frame in rollouts.groupby("controller", sort=True):
        ax.plot(frame["time_s"], frame["delta_flap_rad"], label=controller)
    ax.fill_between(
        reference["time_s"],
        reference["flap_min_rad"],
        reference["flap_max_rad"],
        color="gray",
        alpha=0.18,
        label="flap limits",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Flap command (rad)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(flap_path, dpi=160)
    plt.close(fig)

    return {
        "tracking_figure": tracking_path,
        "flap_figure": flap_path,
    }


def _metrics_summary_markdown(metrics: pd.DataFrame) -> str:
    rows = [
        "# Phase 3 Baseline Metrics Summary",
        "",
        (
            "PID and gain-scheduled LQR were evaluated on the shared Phase 2 "
            "reference/corridor profile."
        ),
        "",
        _dataframe_to_markdown(metrics),
        "",
        "Success labels use the thresholds in `configs/phase3_baselines.yaml`.",
        "",
    ]
    return "\n".join(rows)


def _dataframe_to_markdown(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _column in columns) + " |",
    ]
    for _idx, row in frame.iterrows():
        values = [_format_metric_value(row[column]) for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _format_metric_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)
