
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from reentry_mpc.artifacts import plt
from reentry_mpc.baseline_metrics import summarize_closed_loop
from reentry_mpc.longitudinal import load_phase1_config, scheduled_pitching_moment
from reentry_mpc.nmpc import (
    NmpcConfig,
    NmpcSolverOptions,
    NmpcWeights,
    apply_flap_limits,
    rk4_step_numeric,
    solve_nmpc_step,
)
from reentry_mpc.phase2 import build_reference_profile, load_phase2_config


@dataclass(frozen=True)
class Phase4Config:
 
    seed: int
    phase1_config: Path
    phase2_config: Path
    phase3_metrics: Path
    initial_error: np.ndarray
    control_dt_s: float
    corridor_tolerance_rad: float
    nmpc: NmpcConfig
    success_thresholds: dict[str, float]


def load_phase4_config(path: str | Path) -> Phase4Config:

    with Path(path).open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)
    weights = NmpcWeights(
        **{key: float(value) for key, value in raw["weights"].items()}
    )
    solver = NmpcSolverOptions(
        max_iter=int(raw["solver"]["max_iter"]),
        acceptable_tol=float(raw["solver"]["acceptable_tol"]),
        print_level=int(raw["solver"]["print_level"]),
    )
    return Phase4Config(
        seed=int(raw["seed"]),
        phase1_config=Path(raw["phase1_config"]),
        phase2_config=Path(raw["phase2_config"]),
        phase3_metrics=Path(raw["phase3_metrics"]),
        initial_error=np.array(
            [
                float(raw["initial_error"]["alpha_rad"]),
                float(raw["initial_error"]["q_radps"]),
                float(raw["initial_error"]["theta_rad"]),
            ],
            dtype=float,
        ),
        control_dt_s=float(raw["control_dt_s"]),
        corridor_tolerance_rad=float(raw.get("corridor_tolerance_rad", 0.0)),
        nmpc=NmpcConfig(
            horizon_steps=int(raw["horizon_steps"]),
            dt=float(raw["control_dt_s"]),
            weights=weights,
            solver=solver,
        ),
        success_thresholds={
            key: float(value) for key, value in raw["success_thresholds"].items()
        },
    )


def run_phase4_nmpc(
    config_path: str | Path = "configs/phase4_nmpc.yaml",
    output_dir: str | Path = "outputs/phase4_nmpc",
) -> dict[str, Path | pd.DataFrame]:
    """Run nominal NMPC and save artifacts."""

    config = load_phase4_config(config_path)
    plant_config = load_phase1_config(config.phase1_config)
    reference_config = load_phase2_config(config.phase2_config)
    reference_profile = _downsample_reference_profile(
        build_reference_profile(reference_config), config.control_dt_s
    )
    first_reference = reference_profile.iloc[0]
    state = (
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

    previous_flap = 0.0
    rollout_rows: list[dict[str, Any]] = []
    solver_rows: list[dict[str, Any]] = []
    for idx, row in reference_profile.iterrows():
        horizon = reference_profile.iloc[idx : idx + config.nmpc.horizon_steps + 1]
        raw_control, step_log = solve_nmpc_step(
            state=state,
            previous_flap_rad=previous_flap,
            horizon=horizon,
            vehicle=plant_config.vehicle,
            aero=plant_config.aero,
            config=config.nmpc,
        )
        applied_control, flap_saturated, rate_saturated = apply_flap_limits(
            raw_command=raw_control,
            previous_flap_rad=previous_flap,
            row=row,
            dt=config.control_dt_s,
        )
        moment_nm, cm, effectiveness = scheduled_pitching_moment(
            state=state,
            delta_flap_rad=applied_control,
            schedule=_schedule_from_row(row),
            vehicle=plant_config.vehicle,
            aero=plant_config.aero,
        )
        corridor_violation = bool(
            state[0] < row["alpha_min_rad"] - config.corridor_tolerance_rad
            or state[0] > row["alpha_max_rad"] + config.corridor_tolerance_rad
            or state[1] < row["q_min_radps"] - config.corridor_tolerance_rad
            or state[1] > row["q_max_radps"] + config.corridor_tolerance_rad
        )
        rollout_rows.append(
            {
                "controller": "nominal_nmpc",
                "time_s": float(row["time_s"]),
                "alpha_rad": float(state[0]),
                "q_radps": float(state[1]),
                "theta_rad": float(state[2]),
                "alpha_ref_rad": float(row["alpha_ref_rad"]),
                "q_ref_radps": float(row["q_ref_radps"]),
                "theta_ref_rad": float(row["theta_ref_rad"]),
                "alpha_min_rad": float(row["alpha_min_rad"]),
                "alpha_max_rad": float(row["alpha_max_rad"]),
                "q_min_radps": float(row["q_min_radps"]),
                "q_max_radps": float(row["q_max_radps"]),
                "flap_min_rad": float(row["flap_min_rad"]),
                "flap_max_rad": float(row["flap_max_rad"]),
                "flap_rate_min_radps": float(row["flap_rate_min_radps"]),
                "flap_rate_max_radps": float(row["flap_rate_max_radps"]),
                "delta_flap_raw_rad": raw_control,
                "delta_flap_rad": applied_control,
                "delta_flap_rate_radps": (applied_control - previous_flap)
                / config.control_dt_s,
                "flap_saturated": flap_saturated,
                "flap_rate_saturated": rate_saturated,
                "corridor_violation": corridor_violation,
                "alpha_error_rad": float(state[0] - row["alpha_ref_rad"]),
                "q_error_rad": float(state[1] - row["q_ref_radps"]),
                "theta_error_rad": float(state[2] - row["theta_ref_rad"]),
                "pitching_moment_nm": moment_nm,
                "cm": cm,
                "flap_effectiveness": effectiveness,
                **_schedule_from_row(row),
            }
        )
        solver_rows.append({"time_s": float(row["time_s"]), **step_log})
        previous_flap = applied_control
        state = rk4_step_numeric(
            state=state,
            delta_flap_rad=applied_control,
            row=row,
            vehicle=plant_config.vehicle,
            aero=plant_config.aero,
            dt=config.control_dt_s,
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    rollout = pd.DataFrame(rollout_rows)
    solver_log = pd.DataFrame(solver_rows)
    nmpc_metrics = pd.DataFrame(
        [
            summarize_closed_loop(
                rollout,
                controller="nominal_nmpc",
                success_thresholds=config.success_thresholds,
            )
        ]
    )
    phase3_metrics = pd.read_csv(config.phase3_metrics)
    comparison = pd.concat([phase3_metrics, nmpc_metrics], ignore_index=True)

    rollout_path = output_path / "nmpc_rollout.csv"
    solver_log_path = output_path / "nmpc_solver_log.csv"
    comparison_path = output_path / "phase4_comparison_table.csv"
    rollout.to_csv(rollout_path, index=False)
    solver_log.to_csv(solver_log_path, index=False)
    comparison.to_csv(comparison_path, index=False)
    figure_paths = write_phase4_figures(rollout, solver_log, output_path)
    return {
        "rollout_csv": rollout_path,
        "solver_log_csv": solver_log_path,
        "comparison_csv": comparison_path,
        "rollout": rollout,
        "solver_log": solver_log,
        "comparison": comparison,
        **figure_paths,
    }


def write_phase4_figures(
    rollout: pd.DataFrame, solver_log: pd.DataFrame, output_dir: Path
) -> dict[str, Path]:

    tracking_path = output_dir / "nmpc_tracking_nominal.png"
    flap_path = output_dir / "nmpc_flap_command.png"
    activity_path = output_dir / "nmpc_constraint_activity.png"
    solve_time_path = output_dir / "nmpc_solve_time.png"

    fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.4), sharex=True)
    axes[0].fill_between(
        rollout["time_s"],
        rollout["alpha_min_rad"],
        rollout["alpha_max_rad"],
        color="gray",
        alpha=0.22,
        label="alpha corridor",
    )
    axes[0].plot(rollout["time_s"], rollout["alpha_rad"], label="NMPC alpha")
    axes[0].plot(
        rollout["time_s"],
        rollout["alpha_ref_rad"],
        color="black",
        linestyle="--",
        label="alpha_ref",
    )
    axes[1].plot(rollout["time_s"], rollout["q_radps"], label="NMPC q")
    axes[1].plot(
        rollout["time_s"],
        rollout["q_ref_radps"],
        color="black",
        linestyle="--",
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
    ax.fill_between(
        rollout["time_s"],
        rollout["flap_min_rad"],
        rollout["flap_max_rad"],
        color="gray",
        alpha=0.18,
        label="flap bounds",
    )
    ax.plot(rollout["time_s"], rollout["delta_flap_rad"], label="applied flap")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Flap command (rad)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(flap_path, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.plot(
        solver_log["time_s"],
        solver_log["alpha_constraint_active"].astype(int),
        label="alpha active",
    )
    ax.plot(
        solver_log["time_s"],
        solver_log["q_constraint_active"].astype(int),
        label="q active",
    )
    ax.plot(
        solver_log["time_s"],
        solver_log["flap_saturated"].astype(int),
        label="flap saturated",
    )
    ax.plot(
        solver_log["time_s"],
        solver_log["flap_rate_saturated"].astype(int),
        label="rate saturated",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Activity flag")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(activity_path, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    ax.plot(solver_log["time_s"], solver_log["solve_time_s"])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Solve time (s)")
    fig.tight_layout()
    fig.savefig(solve_time_path, dpi=160)
    plt.close(fig)

    return {
        "tracking_figure": tracking_path,
        "flap_figure": flap_path,
        "constraint_activity_figure": activity_path,
        "solve_time_figure": solve_time_path,
    }


def _downsample_reference_profile(
    profile: pd.DataFrame, control_dt_s: float
) -> pd.DataFrame:
    base_dt = float(profile["time_s"].iloc[1] - profile["time_s"].iloc[0])
    stride = max(1, int(round(control_dt_s / base_dt)))
    return profile.iloc[::stride].reset_index(drop=True)


def _schedule_from_row(row: pd.Series) -> dict[str, float]:
    return {
        "altitude_m": float(row["altitude_m"]),
        "velocity_mps": float(row["velocity_mps"]),
        "mach": float(row["mach"]),
        "density_kgm3": float(row["density_kgm3"]),
        "dynamic_pressure_pa": float(row["dynamic_pressure_pa"]),
    }
