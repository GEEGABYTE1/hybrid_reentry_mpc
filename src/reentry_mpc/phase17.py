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
from reentry_mpc.phase5 import (
    _has_corridor_violation,
    _schedule_from_row,
    _uncertain_moment_for_log,
    load_phase5_config,
    summarize_monte_carlo_rollout,
)
from reentry_mpc.phase16 import summarize_corridor_diagnostics
from reentry_mpc.uncertainty import (
    UncertaintyScenario,
    actuator_step,
    initialize_actuator,
    noisy_measurement,
    perturb_aero,
    sample_scenario,
    uncertain_rk4_step,
)


@dataclass(frozen=True)
class Phase17Variant:
    name: str
    control_dt_s: float
    horizon_steps: int
    use_reference_governor: bool
    alpha_buffer_rad: float
    tracking_weight: float
    q_weight: float
    theta_weight: float
    center_weight: float
    terminal_center_weight: float
    slack_weight: float
    command_weight: float
    command_rate_weight: float


@dataclass(frozen=True)
class ControlledRecoveryThresholds:
    max_alpha_miss_rad: float
    max_q_miss_radps: float
    max_alpha_abs_rad: float
    max_q_abs_radps: float


@dataclass(frozen=True)
class Phase17Config:
    seed: int
    phase5_config: Path
    phase1_config: Path
    phase2_config: Path
    scenario_count_per_tier: int
    max_time_s: float | None
    early_window_s: float
    variants: list[Phase17Variant]
    solver: NmpcSolverOptions
    recovery_thresholds: ControlledRecoveryThresholds
    plot_settings: dict[str, Any]


def load_phase17_config(path: str | Path) -> Phase17Config:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)
    max_time = raw.get("max_time_s")
    thresholds = raw["failure_thresholds"]
    return Phase17Config(
        seed=int(raw["seed"]),
        phase5_config=Path(raw["phase5_config"]),
        phase1_config=Path(raw["phase1_config"]),
        phase2_config=Path(raw["phase2_config"]),
        scenario_count_per_tier=int(raw["scenario_count_per_tier"]),
        max_time_s=None if max_time is None else float(max_time),
        early_window_s=float(raw["early_window_s"]),
        variants=[
            Phase17Variant(
                name=str(item["name"]),
                control_dt_s=float(item["control_dt_s"]),
                horizon_steps=int(item["horizon_steps"]),
                use_reference_governor=bool(item["use_reference_governor"]),
                alpha_buffer_rad=float(item["alpha_buffer_rad"]),
                tracking_weight=float(item["tracking_weight"]),
                q_weight=float(item["q_weight"]),
                theta_weight=float(item["theta_weight"]),
                center_weight=float(item["center_weight"]),
                terminal_center_weight=float(item["terminal_center_weight"]),
                slack_weight=float(item["slack_weight"]),
                command_weight=float(item["command_weight"]),
                command_rate_weight=float(item["command_rate_weight"]),
            )
            for item in raw["variants"]
        ],
        solver=NmpcSolverOptions(
            max_iter=int(raw["solver"]["max_iter"]),
            acceptable_tol=float(raw["solver"]["acceptable_tol"]),
            print_level=int(raw["solver"]["print_level"]),
        ),
        recovery_thresholds=ControlledRecoveryThresholds(
            max_alpha_miss_rad=float(
                thresholds["controlled_recovery_max_alpha_miss_rad"]
            ),
            max_q_miss_radps=float(thresholds["controlled_recovery_max_q_miss_radps"]),
            max_alpha_abs_rad=float(
                thresholds["controlled_recovery_max_alpha_abs_rad"]
            ),
            max_q_abs_radps=float(thresholds["controlled_recovery_max_q_abs_radps"]),
        ),
        plot_settings=dict(raw.get("plot_settings", {})),
    )


def run_phase17_feasibility_safety(
    config_path: str | Path = "configs/phase17_feasibility_safety.yaml",
    output_dir: str | Path = "outputs/phase17_feasibility_safety",
    *,
    progress: bool = False,
) -> dict[str, Path | pd.DataFrame]:
    config = load_phase17_config(config_path)
    phase5_config = load_phase5_config(config.phase5_config)
    plant = load_phase1_config(config.phase1_config)
    reference = _maybe_truncate_reference(
        build_reference_profile(load_phase2_config(config.phase2_config)),
        config.max_time_s,
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    feasibility_rows: list[dict[str, Any]] = []
    rollout_frames: list[pd.DataFrame] = []
    for tier_idx, tier in enumerate(phase5_config.tiers):
        scenario_count = min(config.scenario_count_per_tier, tier.scenario_count)
        for scenario_id in range(scenario_count):
            scenario = sample_scenario(
                scenario_id=scenario_id,
                seed=phase5_config.seed + tier_idx * 10_000 + scenario_id,
                ranges=tier.uncertainty_ranges,
            )
            feasibility = diagnose_initial_feasibility(
                tier_name=tier.name,
                scenario=scenario,
                reference_profile=reference,
                plant_config=plant,
                early_window_s=config.early_window_s,
                tolerance=phase5_config.failure_thresholds["corridor_tolerance_rad"],
            )
            feasibility_rows.append(feasibility)
            for variant in config.variants:
                if progress:
                    print(
                        "phase17_rollout "
                        f"tier={tier.name} scenario={scenario_id:03d} "
                        f"controller={variant.name}",
                        flush=True,
                    )
                rollout = rollout_actuator_aware_safety_nmpc(
                    tier_name=tier.name,
                    variant=variant,
                    scenario=scenario,
                    reference_profile=reference,
                    plant_config=plant,
                    solver=config.solver,
                    thresholds=phase5_config.failure_thresholds,
                )
                metrics = summarize_monte_carlo_rollout(
                    rollout=rollout,
                    tier_name=tier.name,
                    controller_name=variant.name,
                    scenario=scenario,
                    thresholds=phase5_config.failure_thresholds,
                )
                diagnostics = summarize_corridor_diagnostics(
                    rollout=rollout,
                    tolerance=phase5_config.failure_thresholds[
                        "corridor_tolerance_rad"
                    ],
                    control_dt_s=variant.control_dt_s,
                )
                recovery = summarize_controlled_recovery(
                    rollout=rollout, thresholds=config.recovery_thresholds
                )
                run_dir = (
                    output_path
                    / tier.name
                    / f"scenario_{scenario_id:03d}"
                    / variant.name
                )
                run_dir.mkdir(parents=True, exist_ok=True)
                trajectory_path = run_dir / "trajectory.csv"
                metrics_path = run_dir / "metrics.json"
                rollout.to_csv(trajectory_path, index=False)
                payload = {
                    **metrics,
                    **diagnostics,
                    **recovery,
                    **feasibility,
                    "variant": variant.__dict__,
                    "trajectory_csv": str(trajectory_path),
                    "uncertainty_parameters": scenario.to_nested_dict(),
                }
                metrics_path.write_text(
                    json.dumps(payload, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                summary_rows.append(
                    {
                        "tier": tier.name,
                        **scenario.to_flat_dict(),
                        **metrics,
                        **diagnostics,
                        **recovery,
                        **feasibility,
                        "trajectory_csv": str(trajectory_path),
                        "metrics_json": str(metrics_path),
                    }
                )
                rollout_frames.append(rollout)

    summary = pd.DataFrame(summary_rows)
    feasibility_summary = pd.DataFrame(feasibility_rows)
    rollouts = pd.concat(rollout_frames, ignore_index=True)
    comparison = summarize_phase17_comparison(summary)
    summary_path = output_path / "phase17_summary.csv"
    rollouts_path = output_path / "phase17_rollouts.csv"
    comparison_path = output_path / "phase17_variant_comparison.csv"
    feasibility_path = output_path / "phase17_feasibility_ceiling.csv"
    summary.to_csv(summary_path, index=False)
    rollouts.to_csv(rollouts_path, index=False)
    comparison.to_csv(comparison_path, index=False)
    feasibility_summary.to_csv(feasibility_path, index=False)
    figure_paths = write_phase17_figures(
        summary=summary,
        rollouts=rollouts,
        feasibility=feasibility_summary,
        output_dir=output_path,
        plot_settings=config.plot_settings,
    )
    return {
        "summary_csv": summary_path,
        "rollouts_csv": rollouts_path,
        "comparison_csv": comparison_path,
        "feasibility_csv": feasibility_path,
        "summary": summary,
        "rollouts": rollouts,
        "comparison": comparison,
        "feasibility": feasibility_summary,
        **figure_paths,
    }


def diagnose_initial_feasibility(
    *,
    tier_name: str,
    scenario: UncertaintyScenario,
    reference_profile: pd.DataFrame,
    plant_config: Any,
    early_window_s: float,
    tolerance: float,
) -> dict[str, Any]:
    window = reference_profile[reference_profile["time_s"] <= early_window_s]
    first = window.iloc[0]
    initial_alpha = float(first["alpha_ref_rad"] + scenario.initial_error.alpha_rad)
    initial_q = float(first["q_ref_radps"] + scenario.initial_error.q_radps)
    initial_alpha_dot = initial_q - 0.22 * initial_alpha
    initial_high_time = _time_to_bound(
        value=initial_alpha,
        rate=initial_alpha_dot,
        bound=float(first["alpha_max_rad"]) + tolerance,
    )
    initial_low_time = _time_to_bound(
        value=initial_alpha,
        rate=initial_alpha_dot,
        bound=float(first["alpha_min_rad"]) - tolerance,
    )
    max_delay = scenario.actuator_delay_s + scenario.actuator_lag_s
    open_loop_miss = _emergency_miss(
        scenario=scenario,
        reference_profile=window,
        plant_config=plant_config,
        command=0.0,
        tolerance=tolerance,
    )
    high_recovery_miss = _emergency_miss(
        scenario=scenario,
        reference_profile=window,
        plant_config=plant_config,
        command=float(first["flap_max_rad"]),
        tolerance=tolerance,
    )
    low_recovery_miss = _emergency_miss(
        scenario=scenario,
        reference_profile=window,
        plant_config=plant_config,
        command=float(first["flap_min_rad"]),
        tolerance=tolerance,
    )
    best_emergency_miss = min(open_loop_miss, high_recovery_miss, low_recovery_miss)
    earliest_boundary_time = min(initial_high_time, initial_low_time)
    delay_limited = bool(
        np.isfinite(earliest_boundary_time)
        and earliest_boundary_time < max(0.5, max_delay)
    )
    return {
        "tier": tier_name,
        "scenario_id": scenario.scenario_id,
        "seed": scenario.seed,
        "initial_alpha_rad": initial_alpha,
        "initial_q_radps": initial_q,
        "initial_alpha_dot_radps": initial_alpha_dot,
        "earliest_linear_boundary_time_s": float(earliest_boundary_time),
        "actuator_delay_plus_lag_s": float(max_delay),
        "delay_limited_initial_condition": delay_limited,
        "early_open_loop_alpha_miss_rad": open_loop_miss,
        "early_best_emergency_alpha_miss_rad": best_emergency_miss,
        "early_feasible_under_emergency": bool(best_emergency_miss <= tolerance),
    }


def rollout_actuator_aware_safety_nmpc(
    *,
    tier_name: str,
    variant: Phase17Variant,
    scenario: UncertaintyScenario,
    reference_profile: pd.DataFrame,
    plant_config: Any,
    solver: NmpcSolverOptions,
    thresholds: dict[str, float],
) -> pd.DataFrame:
    rng = np.random.default_rng(scenario.seed + _variant_seed_offset(variant.name))
    perturbed_aero = perturb_aero(plant_config.aero, scenario)
    planning_reference = _governed_reference(reference_profile, variant)
    control_times = planning_reference.iloc[
        :: max(1, int(round(variant.control_dt_s / _reference_dt(reference_profile))))
    ].reset_index(drop=True)
    first = reference_profile.iloc[0]
    state = np.array(
        [
            first["alpha_ref_rad"] + scenario.initial_error.alpha_rad,
            first["q_ref_radps"] + scenario.initial_error.q_radps,
            first["theta_ref_rad"] + scenario.initial_error.theta_rad,
        ],
        dtype=float,
    )
    dt = _reference_dt(reference_profile)
    actuator = initialize_actuator(scenario, dt)
    rows: list[dict[str, Any]] = []
    last_raw = 0.0
    last_log = _empty_solver_log()
    solver_failure_seen = False
    tau_eff = max(
        0.5 * variant.control_dt_s,
        scenario.actuator_lag_s + 0.5 * scenario.actuator_delay_s,
    )
    for _idx, row in reference_profile.iterrows():
        measured_state = noisy_measurement(state, scenario, rng)
        solver_status = "held"
        solve_time = 0.0
        if _is_update_time(float(row["time_s"]), variant.control_dt_s):
            nmpc_idx = min(
                int(round(float(row["time_s"]) / variant.control_dt_s)),
                len(control_times) - 1,
            )
            horizon = control_times.iloc[
                nmpc_idx : nmpc_idx + variant.horizon_steps + 1
            ]
            try:
                last_raw, last_log = solve_actuator_aware_safety_step(
                    state=measured_state,
                    applied_flap_rad=actuator.previous_applied_rad,
                    previous_raw_flap_rad=last_raw,
                    horizon=horizon,
                    vehicle=plant_config.vehicle,
                    aero=plant_config.aero,
                    variant=variant,
                    solver=solver,
                    actuator_tau_s=tau_eff,
                )
                solver_status = str(last_log["solver_status"])
                solve_time = float(last_log["solve_time_s"])
                if solver_status != "Solve_Succeeded":
                    solver_failure_seen = True
            except RuntimeError:
                last_log = _empty_solver_log()
                solver_status = "RuntimeError"
                solver_failure_seen = True
        applied, actuator_log = actuator_step(
            raw_command=float(last_raw),
            actuator=actuator,
            scenario=scenario,
            row=row,
            dt=dt,
        )
        moment_nm, cm, effectiveness = _uncertain_moment_for_log(
            state=state,
            delta_flap_rad=applied,
            row=row,
            vehicle=plant_config.vehicle,
            aero=perturbed_aero,
            scenario=scenario,
        )
        rows.append(
            {
                "scenario_id": scenario.scenario_id,
                "tier": tier_name,
                "seed": scenario.seed,
                "controller": variant.name,
                "time_s": float(row["time_s"]),
                "alpha_rad": float(state[0]),
                "q_radps": float(state[1]),
                "theta_rad": float(state[2]),
                "measured_alpha_rad": float(measured_state[0]),
                "measured_q_radps": float(measured_state[1]),
                "measured_theta_rad": float(measured_state[2]),
                "alpha_ref_rad": float(row["alpha_ref_rad"]),
                "q_ref_radps": float(row["q_ref_radps"]),
                "theta_ref_rad": float(row["theta_ref_rad"]),
                "alpha_min_rad": float(row["alpha_min_rad"]),
                "alpha_max_rad": float(row["alpha_max_rad"]),
                "q_min_radps": float(row["q_min_radps"]),
                "q_max_radps": float(row["q_max_radps"]),
                "alpha_error_rad": float(state[0] - row["alpha_ref_rad"]),
                "q_error_rad": float(state[1] - row["q_ref_radps"]),
                "theta_error_rad": float(state[2] - row["theta_ref_rad"]),
                "solver_status": solver_status,
                "solve_time_s": solve_time,
                "solver_failure": solver_failure_seen,
                "actuator_tau_prediction_s": float(tau_eff),
                "predicted_min_alpha_margin_rad": float(
                    last_log["predicted_min_alpha_margin_rad"]
                ),
                "predicted_max_alpha_slack_rad": float(
                    last_log["predicted_max_alpha_slack_rad"]
                ),
                "pitching_moment_nm": moment_nm,
                "cm": cm,
                "flap_effectiveness": effectiveness,
                **actuator_log,
                **_schedule_from_row(row, scenario),
                **scenario.to_flat_dict(),
                "corridor_violation": _has_corridor_violation(
                    state=state,
                    row=row,
                    tolerance=thresholds["corridor_tolerance_rad"],
                ),
            }
        )
        state = uncertain_rk4_step(
            state=state,
            delta_flap_rad=applied,
            row=row,
            vehicle=plant_config.vehicle,
            aero=perturbed_aero,
            scenario=scenario,
            dt=dt,
        )
    return pd.DataFrame(rows)


def solve_actuator_aware_safety_step(
    *,
    state: np.ndarray,
    applied_flap_rad: float,
    previous_raw_flap_rad: float,
    horizon: pd.DataFrame,
    vehicle: VehicleParams,
    aero: AeroParams,
    variant: Phase17Variant,
    solver: NmpcSolverOptions,
    actuator_tau_s: float,
) -> tuple[float, dict[str, Any]]:
    horizon = _pad_horizon(horizon, variant.horizon_steps + 1)
    opti = ca.Opti()
    x_var = opti.variable(4, variant.horizon_steps + 1)
    u_var = opti.variable(1, variant.horizon_steps)
    alpha_slack = opti.variable(2, variant.horizon_steps + 1)
    q_slack = opti.variable(2, variant.horizon_steps + 1)
    opti.subject_to(x_var[:, 0] == np.array([*state, applied_flap_rad], dtype=float))
    opti.subject_to(ca.vec(alpha_slack) >= 0)
    opti.subject_to(ca.vec(q_slack) >= 0)
    objective = 0
    for k_idx in range(variant.horizon_steps):
        row = horizon.iloc[k_idx]
        alpha_center = 0.5 * (float(row["alpha_min_rad"]) + float(row["alpha_max_rad"]))
        ref_alpha = float(row["alpha_ref_rad"])
        ref_q = float(row["q_ref_radps"])
        ref_theta = float(row["theta_ref_rad"])
        objective += variant.tracking_weight * (x_var[0, k_idx] - ref_alpha) ** 2
        objective += variant.q_weight * (x_var[1, k_idx] - ref_q) ** 2
        objective += variant.theta_weight * (x_var[2, k_idx] - ref_theta) ** 2
        objective += variant.center_weight * (x_var[0, k_idx] - alpha_center) ** 2
        objective += variant.command_weight * u_var[0, k_idx] ** 2
        previous_u = previous_raw_flap_rad if k_idx == 0 else u_var[0, k_idx - 1]
        delta_u = u_var[0, k_idx] - previous_u
        objective += variant.command_rate_weight * delta_u**2
        objective += variant.slack_weight * (
            alpha_slack[0, k_idx] ** 2
            + alpha_slack[1, k_idx] ** 2
            + q_slack[0, k_idx] ** 2
            + q_slack[1, k_idx] ** 2
        )
        opti.subject_to(
            x_var[:, k_idx + 1]
            == _rk4_augmented_symbolic(
                x_var[:, k_idx],
                u_var[0, k_idx],
                row,
                vehicle,
                aero,
                variant.control_dt_s,
                actuator_tau_s,
            )
        )
        _add_safety_constraints(
            opti,
            x_var[:, k_idx],
            row,
            alpha_slack[:, k_idx],
            q_slack[:, k_idx],
            variant.alpha_buffer_rad,
        )
        _add_command_constraints(
            opti, u_var[0, k_idx], delta_u, row, variant.control_dt_s
        )
    terminal = horizon.iloc[variant.horizon_steps]
    terminal_center = 0.5 * (
        float(terminal["alpha_min_rad"]) + float(terminal["alpha_max_rad"])
    )
    objective += (
        variant.terminal_center_weight
        * (x_var[0, variant.horizon_steps] - terminal_center) ** 2
    )
    objective += variant.slack_weight * (
        alpha_slack[0, variant.horizon_steps] ** 2
        + alpha_slack[1, variant.horizon_steps] ** 2
        + q_slack[0, variant.horizon_steps] ** 2
        + q_slack[1, variant.horizon_steps] ** 2
    )
    _add_safety_constraints(
        opti,
        x_var[:, variant.horizon_steps],
        terminal,
        alpha_slack[:, variant.horizon_steps],
        q_slack[:, variant.horizon_steps],
        variant.alpha_buffer_rad,
    )
    opti.minimize(objective)
    opti.set_initial(
        x_var,
        _augmented_initial_guess(
            state, applied_flap_rad, horizon, variant.horizon_steps
        ),
    )
    opti.set_initial(u_var, previous_raw_flap_rad)
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
    predicted_state = np.array(solution.value(x_var), dtype=float)
    alpha_min = horizon["alpha_min_rad"].iloc[: predicted_state.shape[1]].to_numpy()
    alpha_max = horizon["alpha_max_rad"].iloc[: predicted_state.shape[1]].to_numpy()
    alpha_values = predicted_state[0, :]
    margins = np.minimum(alpha_values - alpha_min, alpha_max - alpha_values)
    alpha_slack_value = np.array(solution.value(alpha_slack), dtype=float)
    first_control = float(solution.value(u_var[0, 0]))
    return first_control, {
        "solver_status": opti.stats().get("return_status", "unknown"),
        "solve_time_s": solve_time,
        "objective_value": float(solution.value(objective)),
        "predicted_min_alpha_margin_rad": float(np.min(margins)),
        "predicted_max_alpha_slack_rad": float(np.max(alpha_slack_value)),
    }


def summarize_controlled_recovery(
    *, rollout: pd.DataFrame, thresholds: ControlledRecoveryThresholds
) -> dict[str, Any]:
    alpha_low = rollout["alpha_min_rad"] - rollout["alpha_rad"]
    alpha_high = rollout["alpha_rad"] - rollout["alpha_max_rad"]
    q_low = rollout["q_min_radps"] - rollout["q_radps"]
    q_high = rollout["q_radps"] - rollout["q_max_radps"]
    max_alpha_miss = float(
        np.maximum.reduce([alpha_low, alpha_high, np.zeros(len(rollout))]).max()
    )
    max_q_miss = float(np.maximum.reduce([q_low, q_high, np.zeros(len(rollout))]).max())
    stable_enough = bool(
        (rollout["alpha_rad"].abs() <= thresholds.max_alpha_abs_rad).all()
        and (rollout["q_radps"].abs() <= thresholds.max_q_abs_radps).all()
    )
    controlled = bool(
        stable_enough
        and max_alpha_miss <= thresholds.max_alpha_miss_rad
        and max_q_miss <= thresholds.max_q_miss_radps
        and not rollout["solver_failure"].any()
    )
    return {
        "controlled_recovery": controlled,
        "controlled_recovery_max_alpha_miss_rad": max_alpha_miss,
        "controlled_recovery_max_q_miss_radps": max_q_miss,
    }


def summarize_phase17_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        summary.assign(
            strict_success=summary["failure_label"].eq("success"),
            controlled=summary["controlled_recovery"].astype(bool),
        )
        .groupby(["tier", "controller"], as_index=False)
        .agg(
            scenario_count=("scenario_id", "size"),
            strict_success_count=("strict_success", "sum"),
            strict_success_rate=("strict_success", "mean"),
            controlled_recovery_count=("controlled", "sum"),
            controlled_recovery_rate=("controlled", "mean"),
            mean_rms_alpha_error_rad=("rms_alpha_error_rad", "mean"),
            median_max_alpha_miss_rad=("max_alpha_corridor_miss_rad", "median"),
            mean_early_best_emergency_miss_rad=(
                "early_best_emergency_alpha_miss_rad",
                "mean",
            ),
            delay_limited_fraction=("delay_limited_initial_condition", "mean"),
        )
    )
    return grouped


def write_phase17_figures(
    *,
    summary: pd.DataFrame,
    rollouts: pd.DataFrame,
    feasibility: pd.DataFrame,
    output_dir: Path,
    plot_settings: dict[str, Any],
) -> dict[str, Path]:
    paths = {
        "success_png": output_dir / "strict_vs_controlled_success.png",
        "feasibility_png": output_dir / "feasibility_ceiling.png",
        "miss_png": output_dir / "alpha_miss_by_feasibility.png",
        "envelope_png": output_dir / "safety_nmpc_alpha_envelopes.png",
    }
    _plot_strict_vs_controlled(summary, paths["success_png"])
    _plot_feasibility(feasibility, paths["feasibility_png"])
    _plot_miss(summary, paths["miss_png"])
    _plot_envelopes(
        rollouts,
        paths["envelope_png"],
        plot_settings.get("envelope_percentiles", [5.0, 95.0]),
    )
    return paths


def _dynamics_augmented_symbolic(
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


def _rk4_augmented_symbolic(
    state: ca.MX,
    command: ca.MX,
    row: pd.Series,
    vehicle: VehicleParams,
    aero: AeroParams,
    dt: float,
    actuator_tau_s: float,
) -> ca.MX:
    k1 = _dynamics_augmented_symbolic(
        state, command, row, vehicle, aero, actuator_tau_s
    )
    k2 = _dynamics_augmented_symbolic(
        state + 0.5 * dt * k1, command, row, vehicle, aero, actuator_tau_s
    )
    k3 = _dynamics_augmented_symbolic(
        state + 0.5 * dt * k2, command, row, vehicle, aero, actuator_tau_s
    )
    k4 = _dynamics_augmented_symbolic(
        state + dt * k3, command, row, vehicle, aero, actuator_tau_s
    )
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _add_safety_constraints(
    opti: ca.Opti,
    state: ca.MX,
    row: pd.Series,
    alpha_slack: ca.MX,
    q_slack: ca.MX,
    alpha_buffer_rad: float,
) -> None:
    opti.subject_to(
        state[0] + alpha_slack[0] >= float(row["alpha_min_rad"]) + alpha_buffer_rad
    )
    opti.subject_to(
        state[0] - alpha_slack[1] <= float(row["alpha_max_rad"]) - alpha_buffer_rad
    )
    opti.subject_to(state[1] + q_slack[0] >= float(row["q_min_radps"]))
    opti.subject_to(state[1] - q_slack[1] <= float(row["q_max_radps"]))


def _add_command_constraints(
    opti: ca.Opti, command: ca.MX, delta_u: ca.MX, row: pd.Series, dt: float
) -> None:
    opti.subject_to(command >= float(row["flap_min_rad"]))
    opti.subject_to(command <= float(row["flap_max_rad"]))
    opti.subject_to(delta_u >= float(row["flap_rate_min_radps"]) * dt)
    opti.subject_to(delta_u <= float(row["flap_rate_max_radps"]) * dt)


def _governed_reference(
    reference_profile: pd.DataFrame, variant: Phase17Variant
) -> pd.DataFrame:
    governed = reference_profile.copy()
    if variant.use_reference_governor:
        center = 0.5 * (governed["alpha_min_rad"] + governed["alpha_max_rad"])
        governed["alpha_ref_rad"] = center
        governed["q_ref_radps"] = 0.0
    return governed


def _emergency_miss(
    *,
    scenario: UncertaintyScenario,
    reference_profile: pd.DataFrame,
    plant_config: Any,
    command: float,
    tolerance: float,
) -> float:
    first = reference_profile.iloc[0]
    state = np.array(
        [
            first["alpha_ref_rad"] + scenario.initial_error.alpha_rad,
            first["q_ref_radps"] + scenario.initial_error.q_radps,
            first["theta_ref_rad"] + scenario.initial_error.theta_rad,
        ],
        dtype=float,
    )
    actuator = initialize_actuator(scenario, _reference_dt(reference_profile))
    aero = perturb_aero(plant_config.aero, scenario)
    misses = []
    dt = _reference_dt(reference_profile)
    for _idx, row in reference_profile.iterrows():
        applied, _log = actuator_step(
            raw_command=command,
            actuator=actuator,
            scenario=scenario,
            row=row,
            dt=dt,
        )
        misses.append(
            max(
                float(row["alpha_min_rad"]) - state[0] - tolerance,
                state[0] - float(row["alpha_max_rad"]) - tolerance,
                0.0,
            )
        )
        state = uncertain_rk4_step(
            state=state,
            delta_flap_rad=applied,
            row=row,
            vehicle=plant_config.vehicle,
            aero=aero,
            scenario=scenario,
            dt=dt,
        )
    return float(max(misses))


def _time_to_bound(value: float, rate: float, bound: float) -> float:
    if np.isclose(rate, 0.0):
        return float("inf")
    time_s = (bound - value) / rate
    return float(time_s) if time_s >= 0.0 else float("inf")


def _pad_horizon(horizon: pd.DataFrame, required_rows: int) -> pd.DataFrame:
    if len(horizon) >= required_rows:
        return horizon.iloc[:required_rows].reset_index(drop=True)
    padding = [horizon.iloc[[-1]]] * (required_rows - len(horizon))
    return pd.concat([horizon, *padding], ignore_index=True)


def _augmented_initial_guess(
    state: np.ndarray, applied_flap_rad: float, horizon: pd.DataFrame, steps: int
) -> np.ndarray:
    guess = np.zeros((4, steps + 1))
    guess[:, 0] = [*state, applied_flap_rad]
    for idx in range(1, steps + 1):
        row = horizon.iloc[idx]
        guess[:, idx] = [
            row["alpha_ref_rad"],
            row["q_ref_radps"],
            row["theta_ref_rad"],
            applied_flap_rad,
        ]
    return guess


def _reference_dt(reference_profile: pd.DataFrame) -> float:
    return float(
        reference_profile["time_s"].iloc[1] - reference_profile["time_s"].iloc[0]
    )


def _maybe_truncate_reference(
    reference_profile: pd.DataFrame, max_time_s: float | None
) -> pd.DataFrame:
    if max_time_s is None:
        return reference_profile
    return reference_profile[reference_profile["time_s"] <= max_time_s].reset_index(
        drop=True
    )


def _is_update_time(time_s: float, control_dt_s: float) -> bool:
    remainder = np.mod(time_s, control_dt_s)
    return bool(np.isclose(remainder, 0.0) or np.isclose(remainder, control_dt_s))


def _variant_seed_offset(name: str) -> int:
    return 17_000 + sum(ord(char) for char in name)


def _empty_solver_log() -> dict[str, float | str]:
    return {
        "solver_status": "not_solved",
        "solve_time_s": 0.0,
        "objective_value": 0.0,
        "predicted_min_alpha_margin_rad": 0.0,
        "predicted_max_alpha_slack_rad": 0.0,
    }


def _plot_strict_vs_controlled(summary: pd.DataFrame, path: Path) -> None:
    grouped = (
        summary.assign(strict=summary["failure_label"].eq("success"))
        .groupby(["tier", "controller"])
        .agg(strict=("strict", "mean"), controlled=("controlled_recovery", "mean"))
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    x = np.arange(len(grouped))
    ax.bar(x - 0.18, grouped["strict"], width=0.36, label="strict success")
    ax.bar(x + 0.18, grouped["controlled"], width=0.36, label="controlled recovery")
    ax.set_xticks(
        x,
        [f"{r.tier}\n{r.controller}" for r in grouped.itertuples()],
        rotation=20,
        ha="right",
    )
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("rate")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_feasibility(feasibility: pd.DataFrame, path: Path) -> None:
    grouped = feasibility.groupby("tier")["early_feasible_under_emergency"].mean()
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    grouped.plot(kind="bar", ax=ax)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("early emergency-feasible fraction")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_miss(summary: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    for controller, frame in summary.groupby("controller"):
        ax.scatter(
            frame["early_best_emergency_alpha_miss_rad"],
            frame["max_alpha_corridor_miss_rad"],
            label=controller,
            alpha=0.75,
        )
    ax.set_xlabel("early best emergency alpha miss [rad]")
    ax.set_ylabel("closed-loop max alpha miss [rad]")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_envelopes(
    rollouts: pd.DataFrame, path: Path, percentiles: list[float]
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, (tier, tier_data) in zip(axes, rollouts.groupby("tier"), strict=False):
        for controller, frame in tier_data.groupby("controller"):
            pivot = frame.pivot_table(
                index="time_s", columns="scenario_id", values="alpha_error_rad"
            )
            median = pivot.median(axis=1)
            low = pivot.quantile(percentiles[0] / 100.0, axis=1)
            high = pivot.quantile(percentiles[1] / 100.0, axis=1)
            ax.plot(median.index, median, label=controller)
            ax.fill_between(median.index, low, high, alpha=0.12)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title(tier)
        ax.set_xlabel("time [s]")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("alpha error [rad]")
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
