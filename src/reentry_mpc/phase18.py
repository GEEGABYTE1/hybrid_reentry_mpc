from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import casadi as ca
import numpy as np
import pandas as pd
import yaml

from reentry_mpc.artifacts import plt
from reentry_mpc.longitudinal import (
    AeroParams,
    VehicleParams,
    flap_effectiveness,
    load_phase1_config,
)
from reentry_mpc.nmpc import NmpcSolverOptions
from reentry_mpc.phase2 import build_reference_profile, load_phase2_config
from reentry_mpc.phase5 import load_phase5_config
from reentry_mpc.phase16 import summarize_corridor_diagnostics
from reentry_mpc.phase17 import (
    _add_command_constraints,
    _add_safety_constraints,
    _augmented_initial_guess,
    _maybe_truncate_reference,
    _pad_horizon,
    _reference_dt,
)
from reentry_mpc.uncertainty import UncertaintyScenario, sample_scenario


@dataclass(frozen=True)
class SlackOracleSettings:
    dt_s: float
    horizon_steps: int
    actuator_lag_scale: float
    delay_as_lag_scale: float
    alpha_slack_weight: float
    q_slack_weight: float
    command_weight: float
    command_rate_weight: float
    q_center_weight: float
    alpha_center_weight: float


@dataclass(frozen=True)
class OracleClassification:
    feasible_alpha_miss_rad: float
    near_feasible_alpha_miss_rad: float


@dataclass(frozen=True)
class Phase18Config:
    seed: int
    phase5_config: Path
    phase1_config: Path
    phase2_config: Path
    scenario_count_per_tier: int
    max_time_s: float | None
    oracle: SlackOracleSettings
    solver: NmpcSolverOptions
    classification: OracleClassification
    plot_settings: dict[str, Any]


def load_phase18_config(path: str | Path) -> Phase18Config:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)
    max_time = raw.get("max_time_s")
    oracle = raw["oracle"]
    classification = raw["classification"]
    return Phase18Config(
        seed=int(raw["seed"]),
        phase5_config=Path(raw["phase5_config"]),
        phase1_config=Path(raw["phase1_config"]),
        phase2_config=Path(raw["phase2_config"]),
        scenario_count_per_tier=int(raw["scenario_count_per_tier"]),
        max_time_s=None if max_time is None else float(max_time),
        oracle=SlackOracleSettings(
            dt_s=float(oracle["dt_s"]),
            horizon_steps=int(oracle["horizon_steps"]),
            actuator_lag_scale=float(oracle["actuator_lag_scale"]),
            delay_as_lag_scale=float(oracle["delay_as_lag_scale"]),
            alpha_slack_weight=float(oracle["alpha_slack_weight"]),
            q_slack_weight=float(oracle["q_slack_weight"]),
            command_weight=float(oracle["command_weight"]),
            command_rate_weight=float(oracle["command_rate_weight"]),
            q_center_weight=float(oracle["q_center_weight"]),
            alpha_center_weight=float(oracle["alpha_center_weight"]),
        ),
        solver=NmpcSolverOptions(
            max_iter=int(raw["solver"]["max_iter"]),
            acceptable_tol=float(raw["solver"]["acceptable_tol"]),
            print_level=int(raw["solver"]["print_level"]),
        ),
        classification=OracleClassification(
            feasible_alpha_miss_rad=float(classification["feasible_alpha_miss_rad"]),
            near_feasible_alpha_miss_rad=float(
                classification["near_feasible_alpha_miss_rad"]
            ),
        ),
        plot_settings=dict(raw.get("plot_settings", {})),
    )


def run_phase18_slack_oracle(
    config_path: str | Path = "configs/phase18_slack_oracle.yaml",
    output_dir: str | Path = "outputs/phase18_slack_oracle",
    *,
    progress: bool = False,
) -> dict[str, Path | pd.DataFrame]:
    config = load_phase18_config(config_path)
    phase5_config = load_phase5_config(config.phase5_config)
    plant = load_phase1_config(config.phase1_config)
    reference = _downsample_for_oracle(
        _maybe_truncate_reference(
            build_reference_profile(load_phase2_config(config.phase2_config)),
            config.max_time_s,
        ),
        config.oracle.dt_s,
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    trajectory_frames: list[pd.DataFrame] = []
    for tier_idx, tier in enumerate(phase5_config.tiers):
        scenario_count = min(config.scenario_count_per_tier, tier.scenario_count)
        for scenario_id in range(scenario_count):
            scenario = sample_scenario(
                scenario_id=scenario_id,
                seed=phase5_config.seed + tier_idx * 10_000 + scenario_id,
                ranges=tier.uncertainty_ranges,
            )
            if progress:
                print(
                    f"phase18_oracle tier={tier.name} scenario={scenario_id:03d}",
                    flush=True,
                )
            trajectory, metrics = solve_slack_oracle(
                tier_name=tier.name,
                scenario=scenario,
                reference=reference,
                vehicle=plant.vehicle,
                aero=plant.aero,
                settings=config.oracle,
                solver=config.solver,
                classification=config.classification,
                tolerance=phase5_config.failure_thresholds["corridor_tolerance_rad"],
            )
            run_dir = output_path / tier.name / f"scenario_{scenario_id:03d}"
            run_dir.mkdir(parents=True, exist_ok=True)
            trajectory_path = run_dir / "oracle_trajectory.csv"
            metrics_path = run_dir / "oracle_metrics.json"
            trajectory.to_csv(trajectory_path, index=False)
            metrics_path.write_text(
                json.dumps(
                    {
                        **metrics,
                        "trajectory_csv": str(trajectory_path),
                        "uncertainty_parameters": scenario.to_nested_dict(),
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            summary_rows.append(
                {
                    "tier": tier.name,
                    **scenario.to_flat_dict(),
                    **metrics,
                    "trajectory_csv": str(trajectory_path),
                    "metrics_json": str(metrics_path),
                }
            )
            trajectory_frames.append(trajectory)
    summary = pd.DataFrame(summary_rows)
    trajectories = pd.concat(trajectory_frames, ignore_index=True)
    ceiling = summarize_feasibility_ceiling(summary)
    summary_path = output_path / "slack_oracle_summary.csv"
    trajectories_path = output_path / "slack_oracle_trajectories.csv"
    ceiling_path = output_path / "feasibility_ceiling_summary.csv"
    summary.to_csv(summary_path, index=False)
    trajectories.to_csv(trajectories_path, index=False)
    ceiling.to_csv(ceiling_path, index=False)
    figure_paths = write_phase18_figures(
        summary=summary,
        trajectories=trajectories,
        ceiling=ceiling,
        output_dir=output_path,
    )
    return {
        "summary_csv": summary_path,
        "trajectories_csv": trajectories_path,
        "ceiling_csv": ceiling_path,
        "summary": summary,
        "trajectories": trajectories,
        "ceiling": ceiling,
        **figure_paths,
    }


def solve_slack_oracle(
    *,
    tier_name: str,
    scenario: UncertaintyScenario,
    reference: pd.DataFrame,
    vehicle: VehicleParams,
    aero: AeroParams,
    settings: SlackOracleSettings,
    solver: NmpcSolverOptions,
    classification: OracleClassification,
    tolerance: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    horizon = _pad_horizon(reference, settings.horizon_steps + 1)
    first = horizon.iloc[0]
    initial_state = np.array(
        [
            first["alpha_ref_rad"] + scenario.initial_error.alpha_rad,
            first["q_ref_radps"] + scenario.initial_error.q_radps,
            first["theta_ref_rad"] + scenario.initial_error.theta_rad,
        ],
        dtype=float,
    )
    actuator_tau = max(
        0.5 * settings.dt_s,
        settings.actuator_lag_scale * scenario.actuator_lag_s
        + settings.delay_as_lag_scale * scenario.actuator_delay_s,
    )
    opti = ca.Opti()
    x_var = opti.variable(4, settings.horizon_steps + 1)
    u_var = opti.variable(1, settings.horizon_steps)
    alpha_slack = opti.variable(2, settings.horizon_steps + 1)
    q_slack = opti.variable(2, settings.horizon_steps + 1)
    opti.subject_to(x_var[:, 0] == np.array([*initial_state, 0.0], dtype=float))
    opti.subject_to(ca.vec(alpha_slack) >= 0)
    opti.subject_to(ca.vec(q_slack) >= 0)
    objective = 0
    for k_idx in range(settings.horizon_steps):
        row = horizon.iloc[k_idx]
        center = 0.5 * (float(row["alpha_min_rad"]) + float(row["alpha_max_rad"]))
        objective += settings.alpha_center_weight * (x_var[0, k_idx] - center) ** 2
        objective += settings.q_center_weight * x_var[1, k_idx] ** 2
        objective += settings.alpha_slack_weight * (
            alpha_slack[0, k_idx] ** 2 + alpha_slack[1, k_idx] ** 2
        )
        objective += settings.q_slack_weight * (
            q_slack[0, k_idx] ** 2 + q_slack[1, k_idx] ** 2
        )
        objective += settings.command_weight * u_var[0, k_idx] ** 2
        previous_u = 0.0 if k_idx == 0 else u_var[0, k_idx - 1]
        delta_u = u_var[0, k_idx] - previous_u
        objective += settings.command_rate_weight * delta_u**2
        opti.subject_to(
            x_var[:, k_idx + 1]
            == _rk4_oracle_symbolic(
                x_var[:, k_idx],
                u_var[0, k_idx],
                row,
                vehicle,
                aero,
                settings.dt_s,
                actuator_tau,
            )
        )
        _add_safety_constraints(
            opti,
            x_var[:, k_idx],
            row,
            alpha_slack[:, k_idx],
            q_slack[:, k_idx],
            0.0,
        )
        _add_command_constraints(opti, u_var[0, k_idx], delta_u, row, settings.dt_s)
    terminal = horizon.iloc[settings.horizon_steps]
    terminal_center = 0.5 * (
        float(terminal["alpha_min_rad"]) + float(terminal["alpha_max_rad"])
    )
    objective += (
        settings.alpha_center_weight
        * (x_var[0, settings.horizon_steps] - terminal_center) ** 2
    )
    objective += settings.alpha_slack_weight * (
        alpha_slack[0, settings.horizon_steps] ** 2
        + alpha_slack[1, settings.horizon_steps] ** 2
    )
    objective += settings.q_slack_weight * (
        q_slack[0, settings.horizon_steps] ** 2
        + q_slack[1, settings.horizon_steps] ** 2
    )
    _add_safety_constraints(
        opti,
        x_var[:, settings.horizon_steps],
        terminal,
        alpha_slack[:, settings.horizon_steps],
        q_slack[:, settings.horizon_steps],
        0.0,
    )
    opti.minimize(objective)
    opti.set_initial(
        x_var,
        _augmented_initial_guess(initial_state, 0.0, horizon, settings.horizon_steps),
    )
    opti.set_initial(u_var, 0.0)
    opti.solver(
        "ipopt",
        {"print_time": False, "error_on_fail": False},
        {
            "max_iter": solver.max_iter,
            "acceptable_tol": solver.acceptable_tol,
            "print_level": solver.print_level,
            "sb": "yes",
        },
    )
    start = time.perf_counter()
    solution = opti.solve()
    solve_time = time.perf_counter() - start
    status = opti.stats().get("return_status", "unknown")
    predicted_state = np.array(solution.value(x_var), dtype=float)
    predicted_command = np.array(solution.value(u_var), dtype=float).reshape(-1)
    trajectory = _oracle_trajectory_frame(
        tier_name=tier_name,
        scenario=scenario,
        reference=horizon,
        predicted_state=predicted_state,
        predicted_command=predicted_command,
        actuator_tau=actuator_tau,
    )
    diagnostics = summarize_corridor_diagnostics(
        rollout=trajectory,
        tolerance=tolerance,
        control_dt_s=settings.dt_s,
    )
    max_alpha_miss = float(diagnostics["max_alpha_corridor_miss_rad"])
    feasible = bool(max_alpha_miss <= classification.feasible_alpha_miss_rad)
    near_feasible = bool(max_alpha_miss <= classification.near_feasible_alpha_miss_rad)
    metrics = {
        "controller": "slack_oracle",
        "scenario_id": scenario.scenario_id,
        "seed": scenario.seed,
        "solver_status": status,
        "solve_time_s": solve_time,
        "oracle_feasible": feasible,
        "oracle_near_feasible": near_feasible,
        "oracle_max_alpha_miss_rad": max_alpha_miss,
        "oracle_max_q_miss_radps": float(diagnostics["max_q_corridor_miss_radps"]),
        "oracle_alpha_violation_duration_s": float(
            diagnostics["alpha_violation_duration_s"]
        ),
        "first_alpha_violation_time_s": diagnostics["first_alpha_violation_time_s"],
        "first_alpha_violation_side": diagnostics["first_alpha_violation_side"],
        "actuator_tau_prediction_s": actuator_tau,
        "objective_value": float(solution.value(objective)),
    }
    return trajectory, metrics


def summarize_feasibility_ceiling(summary: pd.DataFrame) -> pd.DataFrame:
    return (
        summary.groupby("tier", as_index=False)
        .agg(
            scenario_count=("scenario_id", "size"),
            feasible_count=("oracle_feasible", "sum"),
            feasible_rate=("oracle_feasible", "mean"),
            near_feasible_count=("oracle_near_feasible", "sum"),
            near_feasible_rate=("oracle_near_feasible", "mean"),
            median_max_alpha_miss_rad=("oracle_max_alpha_miss_rad", "median"),
            max_alpha_miss_rad=("oracle_max_alpha_miss_rad", "max"),
        )
        .sort_values("tier")
    )


def write_phase18_figures(
    *,
    summary: pd.DataFrame,
    trajectories: pd.DataFrame,
    ceiling: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    paths = {
        "ceiling_png": output_dir / "oracle_feasibility_ceiling.png",
        "miss_png": output_dir / "oracle_alpha_miss_distribution.png",
        "trajectories_png": output_dir / "oracle_worst_case_trajectories.png",
    }
    _plot_ceiling(ceiling, paths["ceiling_png"])
    _plot_miss(summary, paths["miss_png"])
    _plot_worst_trajectories(summary, trajectories, paths["trajectories_png"])
    return paths


def _oracle_trajectory_frame(
    *,
    tier_name: str,
    scenario: UncertaintyScenario,
    reference: pd.DataFrame,
    predicted_state: np.ndarray,
    predicted_command: np.ndarray,
    actuator_tau: float,
) -> pd.DataFrame:
    rows = []
    for idx in range(predicted_state.shape[1]):
        row = reference.iloc[idx]
        command_idx = min(idx, len(predicted_command) - 1)
        alpha = float(predicted_state[0, idx])
        q_radps = float(predicted_state[1, idx])
        theta = float(predicted_state[2, idx])
        rows.append(
            {
                "tier": tier_name,
                "scenario_id": scenario.scenario_id,
                "seed": scenario.seed,
                "controller": "slack_oracle",
                "time_s": float(row["time_s"]),
                "alpha_rad": alpha,
                "q_radps": q_radps,
                "theta_rad": theta,
                "alpha_ref_rad": float(row["alpha_ref_rad"]),
                "q_ref_radps": float(row["q_ref_radps"]),
                "theta_ref_rad": float(row["theta_ref_rad"]),
                "alpha_min_rad": float(row["alpha_min_rad"]),
                "alpha_max_rad": float(row["alpha_max_rad"]),
                "q_min_radps": float(row["q_min_radps"]),
                "q_max_radps": float(row["q_max_radps"]),
                "alpha_error_rad": alpha - float(row["alpha_ref_rad"]),
                "q_error_rad": q_radps - float(row["q_ref_radps"]),
                "theta_error_rad": theta - float(row["theta_ref_rad"]),
                "mach": float(row["mach"]),
                "altitude_m": float(row["altitude_m"]),
                "dynamic_pressure_pa": float(row["dynamic_pressure_pa"]),
                "delta_flap_rad": float(predicted_state[3, idx]),
                "delta_flap_raw_rad": float(predicted_command[command_idx]),
                "delta_flap_rate_radps": 0.0,
                "flap_saturated": False,
                "flap_rate_saturated": False,
                "solver_status": "oracle_solution",
                "solve_time_s": 0.0,
                "solver_failure": False,
                "actuator_tau_prediction_s": actuator_tau,
                "corridor_violation": bool(
                    alpha < float(row["alpha_min_rad"])
                    or alpha > float(row["alpha_max_rad"])
                    or q_radps < float(row["q_min_radps"])
                    or q_radps > float(row["q_max_radps"])
                ),
            }
        )
    return pd.DataFrame(rows)


def _rk4_oracle_symbolic(
    state: ca.MX,
    command: ca.MX,
    row: pd.Series,
    vehicle: VehicleParams,
    aero: AeroParams,
    dt: float,
    actuator_tau_s: float,
) -> ca.MX:
    k1 = _dynamics_oracle_symbolic(state, command, row, vehicle, aero, actuator_tau_s)
    k2 = _dynamics_oracle_symbolic(
        state + 0.5 * dt * k1, command, row, vehicle, aero, actuator_tau_s
    )
    k3 = _dynamics_oracle_symbolic(
        state + 0.5 * dt * k2, command, row, vehicle, aero, actuator_tau_s
    )
    k4 = _dynamics_oracle_symbolic(
        state + dt * k3, command, row, vehicle, aero, actuator_tau_s
    )
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _dynamics_oracle_symbolic(
    state: ca.MX,
    command: ca.MX,
    row: pd.Series,
    vehicle: VehicleParams,
    aero: AeroParams,
    actuator_tau_s: float,
) -> ca.MX:
    alpha = state[0]
    pitch_rate = state[1]
    flap = state[3]
    effectiveness = flap_effectiveness(
        float(row["mach"]), float(row["altitude_m"]), aero
    )
    alpha_slope = aero.cm_alpha + aero.cm_alpha_mach_slope * (float(row["mach"]) - 15.0)
    q_hat = (
        pitch_rate
        * vehicle.reference_length_m
        / max(2.0 * float(row["velocity_mps"]), 1.0)
    )
    cm = (
        aero.cm0
        + alpha_slope * alpha
        + aero.cm_q * q_hat
        + aero.cm_delta * effectiveness * flap
    )
    moment = (
        float(row["dynamic_pressure_pa"])
        * vehicle.reference_area_m2
        * vehicle.reference_length_m
        * cm
    )
    q_dot = moment / vehicle.pitch_inertia_kgm2
    alpha_dot = pitch_rate - 0.22 * alpha
    theta_dot = pitch_rate
    flap_dot = (command - flap) / max(actuator_tau_s, 1.0e-3)
    return ca.vertcat(alpha_dot, q_dot, theta_dot, flap_dot)


def _downsample_for_oracle(reference: pd.DataFrame, dt_s: float) -> pd.DataFrame:
    step = max(1, int(round(dt_s / _reference_dt(reference))))
    return reference.iloc[::step].reset_index(drop=True)


def _plot_ceiling(ceiling: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    x = np.arange(len(ceiling))
    ax.bar(x - 0.18, ceiling["feasible_rate"], width=0.36, label="strict feasible")
    ax.bar(x + 0.18, ceiling["near_feasible_rate"], width=0.36, label="near feasible")
    ax.set_xticks(x, ceiling["tier"])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("oracle ceiling rate")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_miss(summary: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    data = [
        frame["oracle_max_alpha_miss_rad"].to_numpy(dtype=float)
        for _tier, frame in summary.groupby("tier")
    ]
    labels = [tier for tier, _frame in summary.groupby("tier")]
    ax.boxplot(data, tick_labels=labels, showfliers=True)
    ax.set_ylabel("oracle max alpha miss [rad]")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_worst_trajectories(
    summary: pd.DataFrame, trajectories: pd.DataFrame, path: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), sharey=True)
    for ax, (tier, tier_summary) in zip(axes, summary.groupby("tier"), strict=False):
        worst = tier_summary.sort_values(
            "oracle_max_alpha_miss_rad", ascending=False
        ).head(2)
        tier_traj = trajectories[trajectories["tier"].eq(tier)]
        for row in worst.itertuples():
            traj = tier_traj[tier_traj["scenario_id"].eq(row.scenario_id)]
            ax.fill_between(
                traj["time_s"],
                traj["alpha_min_rad"],
                traj["alpha_max_rad"],
                color="gray",
                alpha=0.12,
            )
            ax.plot(
                traj["time_s"], traj["alpha_rad"], label=f"scenario {row.scenario_id}"
            )
        ax.set_title(tier)
        ax.set_xlabel("time [s]")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("alpha [rad]")
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
