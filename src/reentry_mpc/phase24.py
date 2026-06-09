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
from reentry_mpc.longitudinal import AeroParams, VehicleParams, load_phase1_config
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
from reentry_mpc.phase17 import (
    ControlledRecoveryThresholds,
    _add_command_constraints,
    _add_safety_constraints,
    _augmented_initial_guess,
    _is_update_time,
    _maybe_truncate_reference,
    _pad_horizon,
    _reference_dt,
    _rk4_augmented_symbolic,
    summarize_controlled_recovery,
)
from reentry_mpc.phase19 import (
    FittedOraclePolicy,
    OraclePolicyConfig,
    _feature_dict_from_state,
    _policy_to_json,
    _safety_blend_command,
    fit_oracle_policy,
    predict_oracle_policy,
)
from reentry_mpc.phase22 import SlackMpcWeights
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
class Phase24Variant:
    name: str
    control_dt_s: float
    horizon_steps: int
    alpha_buffer_rad: float
    terminal_center_weight: float
    command_prior_weight: float
    blend_initial_command: bool
    weights: SlackMpcWeights


@dataclass(frozen=True)
class Phase24Config:
    seed: int
    phase5_config: Path
    phase1_config: Path
    phase2_config: Path
    phase23_oracle_summary: Path
    phase23_oracle_trajectories: Path
    phase22_output_dir: Path
    phase23_output_dir: Path
    scenario_count_per_tier: int
    max_time_s: float | None
    policy: OraclePolicyConfig
    variants: list[Phase24Variant]
    solver: NmpcSolverOptions
    recovery_thresholds: ControlledRecoveryThresholds
    plot_settings: dict[str, Any]


def load_phase24_config(path: str | Path) -> Phase24Config:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)
    thresholds = raw["failure_thresholds"]
    policy = raw["policy"]
    max_time = raw.get("max_time_s")
    return Phase24Config(
        seed=int(raw["seed"]),
        phase5_config=Path(raw["phase5_config"]),
        phase1_config=Path(raw["phase1_config"]),
        phase2_config=Path(raw["phase2_config"]),
        phase23_oracle_summary=Path(raw["phase23_oracle_summary"]),
        phase23_oracle_trajectories=Path(raw["phase23_oracle_trajectories"]),
        phase22_output_dir=Path(raw["phase22_output_dir"]),
        phase23_output_dir=Path(raw["phase23_output_dir"]),
        scenario_count_per_tier=int(raw["scenario_count_per_tier"]),
        max_time_s=None if max_time is None else float(max_time),
        policy=OraclePolicyConfig(
            ridge_lambda=float(policy["ridge_lambda"]),
            use_only_feasible_or_near_feasible=bool(
                policy["use_only_feasible_or_near_feasible"]
            ),
            feature_columns=[str(value) for value in policy["feature_columns"]],
            safety_blend_gain=float(policy["safety_blend_gain"]),
            safety_margin_rad=float(policy["safety_margin_rad"]),
            command_clip_rad=float(policy["command_clip_rad"]),
        ),
        variants=[
            Phase24Variant(
                name=str(item["name"]),
                control_dt_s=float(item["control_dt_s"]),
                horizon_steps=int(item["horizon_steps"]),
                alpha_buffer_rad=float(item["alpha_buffer_rad"]),
                terminal_center_weight=float(item["terminal_center_weight"]),
                command_prior_weight=float(item["command_prior_weight"]),
                blend_initial_command=bool(item["blend_initial_command"]),
                weights=SlackMpcWeights(
                    alpha_slack=float(item["weights"]["alpha_slack"]),
                    q_slack=float(item["weights"]["q_slack"]),
                    alpha_center=float(item["weights"]["alpha_center"]),
                    q_center=float(item["weights"]["q_center"]),
                    theta_tracking=float(item["weights"]["theta_tracking"]),
                    command=float(item["weights"]["command"]),
                    command_rate=float(item["weights"]["command_rate"]),
                ),
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


def run_phase24_hybrid_imitation_mpc(
    config_path: str | Path = "configs/phase24_hybrid_imitation_mpc.yaml",
    output_dir: str | Path = "outputs/phase24_hybrid_imitation_mpc",
    *,
    progress: bool = False,
) -> dict[str, Path | pd.DataFrame]:
    config = load_phase24_config(config_path)
    phase5_config = load_phase5_config(config.phase5_config)
    plant = load_phase1_config(config.phase1_config)
    reference = _maybe_truncate_reference(
        build_reference_profile(load_phase2_config(config.phase2_config)),
        config.max_time_s,
    )
    oracle_summary = pd.read_csv(config.phase23_oracle_summary)
    oracle_trajectories = pd.read_csv(config.phase23_oracle_trajectories)
    policy, training_frame = fit_oracle_policy(
        oracle_summary=oracle_summary,
        oracle_trajectories=oracle_trajectories,
        config=config.policy,
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    rollout_frames: list[pd.DataFrame] = []
    for tier_idx, tier in enumerate(phase5_config.tiers):
        scenario_count = min(config.scenario_count_per_tier, tier.scenario_count)
        for scenario_id in range(scenario_count):
            scenario = sample_scenario(
                scenario_id=scenario_id,
                seed=phase5_config.seed + tier_idx * 10_000 + scenario_id,
                ranges=tier.uncertainty_ranges,
            )
            for variant in config.variants:
                if progress:
                    print(
                        "phase24_rollout "
                        f"tier={tier.name} scenario={scenario_id:03d} "
                        f"controller={variant.name}",
                        flush=True,
                    )
                rollout = rollout_hybrid_imitation_slack_mpc(
                    tier_name=tier.name,
                    variant=variant,
                    scenario=scenario,
                    reference_profile=reference,
                    plant_config=plant,
                    solver=config.solver,
                    thresholds=phase5_config.failure_thresholds,
                    policy=policy,
                    policy_config=config.policy,
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
                oracle_row = oracle_summary[
                    oracle_summary["tier"].eq(tier.name)
                    & oracle_summary["scenario_id"].eq(scenario_id)
                ]
                oracle_metrics = _oracle_metrics(oracle_row)
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
                    **oracle_metrics,
                    "variant": _variant_payload(variant),
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
                        **oracle_metrics,
                        "mean_solve_time_s": float(rollout["solve_time_s"].mean()),
                        "p95_solve_time_s": float(
                            rollout["solve_time_s"].quantile(0.95)
                        ),
                        "mean_policy_inference_time_s": float(
                            rollout["policy_inference_time_s"].mean()
                        ),
                        "mean_policy_mpc_command_gap_rad": float(
                            rollout["policy_mpc_command_gap_rad"].abs().mean()
                        ),
                        "trajectory_csv": str(trajectory_path),
                        "metrics_json": str(metrics_path),
                    }
                )
                rollout_frames.append(rollout)
    summary = pd.DataFrame(summary_rows)
    rollouts = pd.concat(rollout_frames, ignore_index=True)
    comparison = summarize_phase24_comparison(summary)
    vs_phase22 = compare_phase24_to_phase22(config=config, summary=summary)
    ceiling_gap = compare_phase24_to_phase23(config=config, summary=summary)
    paths = _write_phase24_tables(
        output_path=output_path,
        summary=summary,
        rollouts=rollouts,
        comparison=comparison,
        vs_phase22=vs_phase22,
        ceiling_gap=ceiling_gap,
        policy=policy,
        training_frame=training_frame,
    )
    figure_paths = write_phase24_figures(
        summary=summary,
        rollouts=rollouts,
        comparison=comparison,
        vs_phase22=vs_phase22,
        ceiling_gap=ceiling_gap,
        output_dir=output_path,
        plot_settings=config.plot_settings,
    )
    return {
        **paths,
        **figure_paths,
        "summary": summary,
        "rollouts": rollouts,
        "comparison": comparison,
        "vs_phase22": vs_phase22,
        "ceiling_gap": ceiling_gap,
    }


def rollout_hybrid_imitation_slack_mpc(
    *,
    tier_name: str,
    variant: Phase24Variant,
    scenario: UncertaintyScenario,
    reference_profile: pd.DataFrame,
    plant_config: Any,
    solver: NmpcSolverOptions,
    thresholds: dict[str, float],
    policy: FittedOraclePolicy,
    policy_config: OraclePolicyConfig,
) -> pd.DataFrame:
    rng = np.random.default_rng(scenario.seed + _variant_seed_offset(variant.name))
    perturbed_aero = perturb_aero(plant_config.aero, scenario)
    control_times = reference_profile.iloc[
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
    prev_row: pd.Series | None = None
    for _idx, row in reference_profile.iterrows():
        measured_state = noisy_measurement(state, scenario, rng)
        policy_start = time.perf_counter()
        policy_command = policy_command_from_state(
            policy=policy,
            policy_config=policy_config,
            state=measured_state,
            row=row,
            actuator_tau_s=tau_eff,
            prev_row=prev_row,
            dt=dt,
        )
        policy_time = time.perf_counter() - policy_start
        prev_row = row
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
            policy_sequence = _policy_sequence_for_horizon(
                policy=policy,
                policy_config=policy_config,
                state=measured_state,
                horizon=horizon,
                actuator_tau_s=tau_eff,
                dt=variant.control_dt_s,
            )
            try:
                optimized_raw, last_log = solve_hybrid_slack_mpc_step(
                    state=measured_state,
                    applied_flap_rad=actuator.previous_applied_rad,
                    previous_raw_flap_rad=last_raw,
                    horizon=horizon,
                    vehicle=plant_config.vehicle,
                    aero=plant_config.aero,
                    variant=variant,
                    solver=solver,
                    actuator_tau_s=tau_eff,
                    policy_command_sequence=policy_sequence,
                )
                solver_status = str(last_log["solver_status"])
                solve_time = float(last_log["solve_time_s"])
                last_raw = (
                    0.5 * optimized_raw + 0.5 * policy_command
                    if variant.blend_initial_command
                    else optimized_raw
                )
                if solver_status not in {
                    "Solve_Succeeded",
                    "Solved_To_Acceptable_Level",
                }:
                    solver_failure_seen = True
            except RuntimeError:
                last_log = _empty_solver_log()
                solver_status = "RuntimeError"
                solver_failure_seen = True
                last_raw = policy_command
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
                "policy_inference_time_s": policy_time,
                "policy_delta_flap_raw_rad": float(policy_command),
                "optimized_delta_flap_raw_rad": float(
                    last_log["optimized_raw_flap_rad"]
                ),
                "policy_mpc_command_gap_rad": float(last_raw - float(policy_command)),
                "actuator_tau_prediction_s": float(tau_eff),
                "objective_value": float(last_log["objective_value"]),
                "predicted_min_alpha_margin_rad": float(
                    last_log["predicted_min_alpha_margin_rad"]
                ),
                "predicted_max_alpha_slack_rad": float(
                    last_log["predicted_max_alpha_slack_rad"]
                ),
                "predicted_max_q_slack_radps": float(
                    last_log["predicted_max_q_slack_radps"]
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


def solve_hybrid_slack_mpc_step(
    *,
    state: np.ndarray,
    applied_flap_rad: float,
    previous_raw_flap_rad: float,
    horizon: pd.DataFrame,
    vehicle: VehicleParams,
    aero: AeroParams,
    variant: Phase24Variant,
    solver: NmpcSolverOptions,
    actuator_tau_s: float,
    policy_command_sequence: np.ndarray,
) -> tuple[float, dict[str, Any]]:
    horizon = _pad_horizon(horizon, variant.horizon_steps + 1)
    policy_sequence = np.resize(policy_command_sequence, variant.horizon_steps)
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
        objective += variant.weights.alpha_slack * (
            alpha_slack[0, k_idx] ** 2 + alpha_slack[1, k_idx] ** 2
        )
        objective += variant.weights.q_slack * (
            q_slack[0, k_idx] ** 2 + q_slack[1, k_idx] ** 2
        )
        objective += (
            variant.weights.alpha_center * (x_var[0, k_idx] - alpha_center) ** 2
        )
        objective += variant.weights.q_center * x_var[1, k_idx] ** 2
        objective += (
            variant.weights.theta_tracking
            * (x_var[2, k_idx] - float(row["theta_ref_rad"])) ** 2
        )
        objective += variant.weights.command * u_var[0, k_idx] ** 2
        objective += (
            variant.command_prior_weight
            * (u_var[0, k_idx] - float(policy_sequence[k_idx])) ** 2
        )
        previous_u = previous_raw_flap_rad if k_idx == 0 else u_var[0, k_idx - 1]
        delta_u = u_var[0, k_idx] - previous_u
        objective += variant.weights.command_rate * delta_u**2
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
    objective += variant.weights.alpha_slack * (
        alpha_slack[0, variant.horizon_steps] ** 2
        + alpha_slack[1, variant.horizon_steps] ** 2
    )
    objective += variant.weights.q_slack * (
        q_slack[0, variant.horizon_steps] ** 2 + q_slack[1, variant.horizon_steps] ** 2
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
    opti.set_initial(u_var, policy_sequence.reshape(1, -1))
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
    margins = np.minimum(
        predicted_state[0, :] - alpha_min, alpha_max - predicted_state[0, :]
    )
    alpha_slack_value = np.array(solution.value(alpha_slack), dtype=float)
    q_slack_value = np.array(solution.value(q_slack), dtype=float)
    optimized = float(solution.value(u_var[0, 0]))
    return optimized, {
        "solver_status": opti.stats().get("return_status", "unknown"),
        "solve_time_s": solve_time,
        "objective_value": float(solution.value(objective)),
        "predicted_max_alpha_slack_rad": float(np.max(alpha_slack_value)),
        "predicted_max_q_slack_radps": float(np.max(q_slack_value)),
        "predicted_min_alpha_margin_rad": float(np.min(margins)),
        "optimized_raw_flap_rad": optimized,
    }


def policy_command_from_state(
    *,
    policy: FittedOraclePolicy,
    policy_config: OraclePolicyConfig,
    state: np.ndarray,
    row: pd.Series,
    actuator_tau_s: float,
    prev_row: pd.Series | None,
    dt: float,
) -> float:
    features = _feature_dict_from_state(
        state=state,
        row=row,
        actuator_tau_s=actuator_tau_s,
        prev_row=prev_row,
        dt=dt,
    )
    learned = predict_oracle_policy(policy, features)
    blended = _safety_blend_command(
        learned_command=learned,
        state=state,
        row=row,
        config=policy_config,
    )
    return float(
        np.clip(
            blended, -policy_config.command_clip_rad, policy_config.command_clip_rad
        )
    )


def summarize_phase24_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    return (
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
            oracle_feasible_count=("oracle_feasible", "sum"),
            oracle_near_feasible_count=("oracle_near_feasible", "sum"),
            mean_rms_alpha_error_rad=("rms_alpha_error_rad", "mean"),
            median_max_alpha_miss_rad=("max_alpha_corridor_miss_rad", "median"),
            mean_solve_time_s=("mean_solve_time_s", "mean"),
            p95_solve_time_s=("p95_solve_time_s", "mean"),
            mean_policy_inference_time_s=("mean_policy_inference_time_s", "mean"),
            mean_policy_mpc_command_gap_rad=(
                "mean_policy_mpc_command_gap_rad",
                "mean",
            ),
        )
        .sort_values(["tier", "controller"])
    )


def compare_phase24_to_phase22(
    *, config: Phase24Config, summary: pd.DataFrame
) -> pd.DataFrame:
    phase22 = pd.read_csv(config.phase22_output_dir / "phase22_variant_comparison.csv")
    baseline = phase22[phase22["controller"].eq("online_slack_mpc")][
        ["tier", "strict_success_count", "controlled_recovery_count"]
    ].rename(
        columns={
            "strict_success_count": "phase22_success_count",
            "controlled_recovery_count": "phase22_controlled_count",
        }
    )
    current = summarize_phase24_comparison(summary)
    merged = current.merge(baseline, on="tier", how="left")
    merged["strict_success_delta_vs_phase22"] = (
        merged["strict_success_count"] - merged["phase22_success_count"]
    )
    merged["controlled_delta_vs_phase22"] = (
        merged["controlled_recovery_count"] - merged["phase22_controlled_count"]
    )
    return merged


def compare_phase24_to_phase23(
    *, config: Phase24Config, summary: pd.DataFrame
) -> pd.DataFrame:
    ceiling = pd.read_csv(config.phase23_output_dir / "phase23_feasibility_ceiling.csv")
    current = summarize_phase24_comparison(summary)
    merged = current.merge(
        ceiling[["tier", "feasible_count", "near_feasible_count"]],
        on="tier",
        how="left",
    )
    merged["strict_gap_to_audited_ceiling"] = (
        merged["strict_success_count"] - merged["feasible_count"]
    )
    merged["controlled_gap_to_near_ceiling"] = (
        merged["controlled_recovery_count"] - merged["near_feasible_count"]
    )
    return merged


def write_phase24_figures(
    *,
    summary: pd.DataFrame,
    rollouts: pd.DataFrame,
    comparison: pd.DataFrame,
    vs_phase22: pd.DataFrame,
    ceiling_gap: pd.DataFrame,
    output_dir: Path,
    plot_settings: dict[str, Any],
) -> dict[str, Path]:
    paths = {
        "success_png": output_dir / "hybrid_mpc_success_rates.png",
        "vs_phase22_png": output_dir / "hybrid_vs_slack_mpc.png",
        "ceiling_gap_png": output_dir / "hybrid_ceiling_gap.png",
        "timing_png": output_dir / "hybrid_mpc_timing.png",
        "command_gap_png": output_dir / "policy_vs_mpc_command_gap.png",
        "envelope_png": output_dir / "hybrid_mpc_alpha_envelopes.png",
    }
    _plot_success(comparison, paths["success_png"])
    _plot_delta(vs_phase22, paths["vs_phase22_png"])
    _plot_ceiling_gap(ceiling_gap, paths["ceiling_gap_png"])
    _plot_timing(comparison, paths["timing_png"])
    _plot_command_gap(rollouts, paths["command_gap_png"])
    _plot_envelopes(
        rollouts,
        paths["envelope_png"],
        plot_settings.get("envelope_percentiles", [5.0, 95.0]),
    )
    return paths


def _policy_sequence_for_horizon(
    *,
    policy: FittedOraclePolicy,
    policy_config: OraclePolicyConfig,
    state: np.ndarray,
    horizon: pd.DataFrame,
    actuator_tau_s: float,
    dt: float,
) -> np.ndarray:
    commands = []
    prev_row: pd.Series | None = None
    for _idx, row in horizon.iloc[:-1].iterrows():
        commands.append(
            policy_command_from_state(
                policy=policy,
                policy_config=policy_config,
                state=state,
                row=row,
                actuator_tau_s=actuator_tau_s,
                prev_row=prev_row,
                dt=dt,
            )
        )
        prev_row = row
    return np.array(commands, dtype=float)


def _write_phase24_tables(
    *,
    output_path: Path,
    summary: pd.DataFrame,
    rollouts: pd.DataFrame,
    comparison: pd.DataFrame,
    vs_phase22: pd.DataFrame,
    ceiling_gap: pd.DataFrame,
    policy: FittedOraclePolicy,
    training_frame: pd.DataFrame,
) -> dict[str, Path]:
    summary_path = output_path / "phase24_summary.csv"
    rollouts_path = output_path / "phase24_rollouts.csv"
    comparison_path = output_path / "phase24_variant_comparison.csv"
    vs_phase22_path = output_path / "phase24_vs_phase22.csv"
    ceiling_gap_path = output_path / "phase24_ceiling_gap.csv"
    policy_path = output_path / "hybrid_oracle_policy.json"
    training_path = output_path / "hybrid_policy_training_data.csv"
    summary.to_csv(summary_path, index=False)
    rollouts.to_csv(rollouts_path, index=False)
    comparison.to_csv(comparison_path, index=False)
    vs_phase22.to_csv(vs_phase22_path, index=False)
    ceiling_gap.to_csv(ceiling_gap_path, index=False)
    training_frame.to_csv(training_path, index=False)
    policy_path.write_text(
        json.dumps(_policy_to_json(policy), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "summary_csv": summary_path,
        "rollouts_csv": rollouts_path,
        "comparison_csv": comparison_path,
        "vs_phase22_csv": vs_phase22_path,
        "ceiling_gap_csv": ceiling_gap_path,
        "policy_json": policy_path,
        "training_csv": training_path,
    }


def _oracle_metrics(row: pd.DataFrame) -> dict[str, Any]:
    if row.empty:
        return {}
    return {
        "oracle_feasible": bool(row["oracle_feasible"].iloc[0]),
        "oracle_near_feasible": bool(row["oracle_near_feasible"].iloc[0]),
        "oracle_max_alpha_miss_rad": float(row["oracle_max_alpha_miss_rad"].iloc[0]),
    }


def _variant_payload(variant: Phase24Variant) -> dict[str, Any]:
    return {
        "name": variant.name,
        "control_dt_s": variant.control_dt_s,
        "horizon_steps": variant.horizon_steps,
        "alpha_buffer_rad": variant.alpha_buffer_rad,
        "terminal_center_weight": variant.terminal_center_weight,
        "command_prior_weight": variant.command_prior_weight,
        "blend_initial_command": variant.blend_initial_command,
        "weights": variant.weights.__dict__,
    }


def _variant_seed_offset(name: str) -> int:
    return 24_000 + sum(ord(char) for char in name)


def _empty_solver_log() -> dict[str, float | str]:
    return {
        "solver_status": "not_solved",
        "solve_time_s": 0.0,
        "objective_value": 0.0,
        "predicted_max_alpha_slack_rad": 0.0,
        "predicted_max_q_slack_radps": 0.0,
        "predicted_min_alpha_margin_rad": 0.0,
        "optimized_raw_flap_rad": 0.0,
    }


def _plot_success(comparison: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    x = np.arange(len(comparison))
    ax.bar(x - 0.18, comparison["strict_success_rate"], width=0.36, label="strict")
    ax.bar(
        x + 0.18,
        comparison["controlled_recovery_rate"],
        width=0.36,
        label="controlled",
    )
    ax.set_xticks(
        x,
        [f"{row.tier}\n{row.controller}" for row in comparison.itertuples()],
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


def _plot_delta(vs_phase22: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.0, 4.6))
    labels = [f"{row.tier}\n{row.controller}" for row in vs_phase22.itertuples()]
    x = np.arange(len(vs_phase22))
    ax.bar(x, vs_phase22["strict_success_delta_vs_phase22"])
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel("strict success delta vs Phase 22")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_ceiling_gap(ceiling_gap: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.0, 4.6))
    labels = [f"{row.tier}\n{row.controller}" for row in ceiling_gap.itertuples()]
    x = np.arange(len(ceiling_gap))
    ax.bar(x, ceiling_gap["strict_gap_to_audited_ceiling"])
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel("strict success minus audited ceiling")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_timing(comparison: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.0, 4.6))
    x = np.arange(len(comparison))
    ax.bar(x - 0.18, comparison["mean_solve_time_s"], width=0.36, label="MPC solve")
    ax.bar(
        x + 0.18,
        comparison["mean_policy_inference_time_s"],
        width=0.36,
        label="policy inference",
    )
    ax.set_xticks(
        x,
        [f"{row.tier}\n{row.controller}" for row in comparison.itertuples()],
        rotation=20,
        ha="right",
    )
    ax.set_ylabel("mean time [s]")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_command_gap(rollouts: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    for controller, frame in rollouts.groupby("controller"):
        solved = frame[frame["solve_time_s"] > 0.0]
        ax.hist(
            solved["policy_mpc_command_gap_rad"],
            bins=40,
            alpha=0.35,
            label=controller,
        )
    ax.set_xlabel("applied raw command minus policy command [rad]")
    ax.set_ylabel("control updates")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_envelopes(
    rollouts: pd.DataFrame, path: Path, percentiles: list[float]
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), sharey=True)
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
