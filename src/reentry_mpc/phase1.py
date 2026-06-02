from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from reentry_mpc.artifacts import plt
from reentry_mpc.longitudinal import (
    Phase1Config,
    load_phase1_config,
    simulate_open_loop,
)


def summarize_open_loop(
    trajectory: pd.DataFrame,
    config: Phase1Config,
) -> dict[str, float]:
    return {
        "seed": float(config.seed),
        "duration_s": float(config.duration_s),
        "dt": float(config.dt),
        "max_abs_alpha_rad": float(trajectory["alpha_rad"].abs().max()),
        "max_abs_q_radps": float(trajectory["q_radps"].abs().max()),
        "max_abs_theta_rad": float(trajectory["theta_rad"].abs().max()),
        "max_dynamic_pressure_pa": float(trajectory["dynamic_pressure_pa"].max()),
        "max_mach": float(trajectory["mach"].max()),
        "min_density_kgm3": float(trajectory["density_kgm3"].min()),
        "max_density_kgm3": float(trajectory["density_kgm3"].max()),
        "min_flap_effectiveness": float(trajectory["flap_effectiveness"].min()),
        "max_flap_effectiveness": float(trajectory["flap_effectiveness"].max()),
        "all_states_finite": bool(
            np.isfinite(trajectory[["alpha_rad", "q_radps", "theta_rad"]]).all().all()
        ),
    }


def write_phase1_figures(trajectory: pd.DataFrame, output_dir: Path) -> dict[str, Path]:
    response_path = output_dir / "alpha_q_theta_response.png"
    qbar_path = output_dir / "dynamic_pressure_profile.png"
    effectiveness_path = output_dir / "control_effectiveness_profile.png"

    fig, axes = plt.subplots(3, 1, figsize=(8.5, 7.0), sharex=True)
    axes[0].plot(trajectory["time_s"], trajectory["alpha_rad"])
    axes[1].plot(trajectory["time_s"], trajectory["q_radps"])
    axes[2].plot(trajectory["time_s"], trajectory["theta_rad"])
    axes[0].set_ylabel("alpha (rad)")
    axes[1].set_ylabel("q (rad/s)")
    axes[2].set_ylabel("theta (rad)")
    axes[2].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(response_path, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.plot(trajectory["time_s"], trajectory["dynamic_pressure_pa"] / 1000.0)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Dynamic pressure (kPa)")
    fig.tight_layout()
    fig.savefig(qbar_path, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.plot(trajectory["mach"], trajectory["flap_effectiveness"])
    ax.set_xlabel("Mach")
    ax.set_ylabel("Flap effectiveness multiplier")
    ax.invert_xaxis()
    fig.tight_layout()
    fig.savefig(effectiveness_path, dpi=160)
    plt.close(fig)

    return {
        "response_figure": response_path,
        "dynamic_pressure_figure": qbar_path,
        "effectiveness_figure": effectiveness_path,
    }


def run_phase1_open_loop(
    config_path: str | Path = "configs/phase1_open_loop.yaml",
    output_dir: str | Path = "outputs/phase1_simulator",
) -> dict[str, Path | pd.DataFrame | dict[str, float]]:

    config = load_phase1_config(config_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    trajectory = simulate_open_loop(config)
    metrics = summarize_open_loop(trajectory, config)

    trajectory_path = output_path / "open_loop_trajectory.csv"
    metrics_path = output_path / "open_loop_metrics.json"
    trajectory.to_csv(trajectory_path, index=False)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")

    figure_paths = write_phase1_figures(trajectory, output_path)
    return {
        "trajectory": trajectory,
        "metrics": metrics,
        "trajectory_csv": trajectory_path,
        "metrics_json": metrics_path,
        **figure_paths,
    }
