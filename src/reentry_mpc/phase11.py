from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from reentry_mpc.artifacts import plt
from reentry_mpc.longitudinal import load_phase1_config
from reentry_mpc.nmpc import NmpcConfig, NmpcSolverOptions, NmpcWeights
from reentry_mpc.phase2 import build_reference_profile, load_phase2_config
from reentry_mpc.phase5 import (
    _controller_seed_offset,
    _has_corridor_violation,
    _is_nmpc_update_time,
    _schedule_from_row,
    _uncertain_moment_for_log,
    load_phase5_config,
    summarize_monte_carlo_rollout,
)
from reentry_mpc.phase7 import (
    _filter_baseline_to_phase7_subset,
    _is_successful_solver_status,
    _plot_failures,
    _plot_solve_time,
    _plot_success,
)
from reentry_mpc.residual_mpc import (
    ResidualSurrogate,
    fit_residual_surrogate,
    predict_residual_qdot_numpy,
    save_residual_surrogate,
    solve_residual_mpc_step,
)
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
class ResidualSurrogateConfig:
    ridge_lambda: float
    feature_mode: str


@dataclass(frozen=True)
class Phase11Config:
    seed: int
    phase5_config: Path
    phase1_config: Path
    phase2_config: Path
    phase7_summary: Path
    residual_dataset_dir: Path
    controller_name: str
    max_scenarios_per_tier: int
    residual_gains: list[float]
    surrogate: ResidualSurrogateConfig
    nmpc: NmpcConfig
    plot_settings: dict[str, Any]


def load_phase11_config(path: str | Path) -> Phase11Config:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)
    weights = NmpcWeights(
        **{key: float(value) for key, value in raw["nmpc"]["weights"].items()}
    )
    solver = NmpcSolverOptions(
        max_iter=int(raw["nmpc"]["solver"]["max_iter"]),
        acceptable_tol=float(raw["nmpc"]["solver"]["acceptable_tol"]),
        print_level=int(raw["nmpc"]["solver"]["print_level"]),
    )
    return Phase11Config(
        seed=int(raw["seed"]),
        phase5_config=Path(raw["phase5_config"]),
        phase1_config=Path(raw["phase1_config"]),
        phase2_config=Path(raw["phase2_config"]),
        phase7_summary=Path(raw["phase7_summary"]),
        residual_dataset_dir=Path(raw["residual_dataset_dir"]),
        controller_name=str(raw["controller_name"]),
        max_scenarios_per_tier=int(raw["max_scenarios_per_tier"]),
        residual_gains=[float(value) for value in raw["residual_gains"]],
        surrogate=ResidualSurrogateConfig(
            ridge_lambda=float(raw["surrogate"]["ridge_lambda"]),
            feature_mode=str(raw["surrogate"]["feature_mode"]),
        ),
        nmpc=NmpcConfig(
            horizon_steps=int(raw["nmpc"]["horizon_steps"]),
            dt=float(raw["nmpc"]["control_dt_s"]),
            weights=weights,
            solver=solver,
        ),
        plot_settings=dict(raw.get("plot_settings", {})),
    )


def run_phase11_residual_mpc(
    config_path: str | Path = "configs/phase11_residual_mpc.yaml",
    output_dir: str | Path = "outputs/phase11_residual_mpc",
    *,
    progress: bool = False,
) -> dict[str, Path | pd.DataFrame]:
    config = load_phase11_config(config_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    surrogate = fit_residual_surrogate(
        dataset_dir=config.residual_dataset_dir,
        ridge_lambda=config.surrogate.ridge_lambda,
        feature_mode=config.surrogate.feature_mode,
    )
    surrogate_path = save_residual_surrogate(
        surrogate, output_path / "residual_surrogate.json"
    )
    phase5_config = load_phase5_config(config.phase5_config)
    plant_config = load_phase1_config(config.phase1_config)
    phase2_config = load_phase2_config(config.phase2_config)
    reference_profile = build_reference_profile(phase2_config)
    nmpc_reference = _downsample_reference_profile(reference_profile, config.nmpc.dt)

    summary_rows: list[dict[str, Any]] = []
    rollouts: list[pd.DataFrame] = []
    for tier_idx, tier in enumerate(phase5_config.tiers):
        scenario_count = min(tier.scenario_count, config.max_scenarios_per_tier)
        for scenario_id in range(scenario_count):
            scenario_seed = phase5_config.seed + tier_idx * 10_000 + scenario_id
            scenario = sample_scenario(
                scenario_id=scenario_id,
                seed=scenario_seed,
                ranges=tier.uncertainty_ranges,
            )
            for residual_gain in config.residual_gains:
                controller_name = _controller_name(
                    config.controller_name, residual_gain
                )
                if progress:
                    print(
                        "phase11_rollout "
                        f"tier={tier.name} scenario={scenario_id:03d} "
                        f"gain={residual_gain:g}",
                        flush=True,
                    )
                rollout = rollout_residual_mpc(
                    tier_name=tier.name,
                    controller_name=controller_name,
                    scenario=scenario,
                    reference_profile=reference_profile,
                    nmpc_reference=nmpc_reference,
                    plant_config=plant_config,
                    nmpc_config=config.nmpc,
                    surrogate=surrogate,
                    residual_gain=residual_gain,
                    thresholds=phase5_config.failure_thresholds,
                )
                metrics = summarize_monte_carlo_rollout(
                    rollout=rollout,
                    tier_name=tier.name,
                    controller_name=controller_name,
                    scenario=scenario,
                    thresholds=phase5_config.failure_thresholds,
                )
                run_dir = (
                    output_path
                    / tier.name
                    / f"scenario_{scenario_id:03d}"
                    / f"gain_{residual_gain:g}"
                )
                run_dir.mkdir(parents=True, exist_ok=True)
                trajectory_path = run_dir / "trajectory.csv"
                metrics_path = run_dir / "metrics.json"
                rollout.to_csv(trajectory_path, index=False)
                metrics_payload = {
                    **metrics,
                    "trajectory_csv": str(trajectory_path),
                    "uncertainty_parameters": scenario.to_nested_dict(),
                    "residual_gain": residual_gain,
                    "residual_surrogate": str(surrogate_path),
                }
                metrics_path.write_text(
                    json.dumps(metrics_payload, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                summary_rows.append(
                    {
                        "tier": tier.name,
                        **scenario.to_flat_dict(),
                        **metrics,
                        "residual_gain": residual_gain,
                        "trajectory_csv": str(trajectory_path),
                        "metrics_json": str(metrics_path),
                        "residual_surrogate": str(surrogate_path),
                    }
                )
                rollouts.append(rollout)

    summary = pd.DataFrame(summary_rows)
    combined_rollouts = pd.concat(rollouts, ignore_index=True)
    prior = _filter_baseline_to_phase7_subset(
        _load_prior(config.phase7_summary), summary
    )
    comparison = pd.concat([prior, summary], ignore_index=True, sort=False)
    summary_path = output_path / "phase11_summary.csv"
    rollouts_path = output_path / "phase11_rollouts.csv"
    comparison_path = output_path / "phase11_vs_prior_summary.csv"
    summary.to_csv(summary_path, index=False)
    combined_rollouts.to_csv(rollouts_path, index=False)
    comparison.to_csv(comparison_path, index=False)
    figure_paths = write_phase11_figures(
        summary=summary,
        rollouts=combined_rollouts,
        comparison=comparison,
        output_dir=output_path,
        plot_settings=config.plot_settings,
    )
    return {
        "surrogate_json": surrogate_path,
        "summary_csv": summary_path,
        "rollouts_csv": rollouts_path,
        "comparison_csv": comparison_path,
        "summary": summary,
        "rollouts": combined_rollouts,
        "comparison": comparison,
        **figure_paths,
    }


def rollout_residual_mpc(
    *,
    tier_name: str,
    controller_name: str,
    scenario: UncertaintyScenario,
    reference_profile: pd.DataFrame,
    nmpc_reference: pd.DataFrame,
    plant_config: Any,
    nmpc_config: NmpcConfig,
    surrogate: ResidualSurrogate,
    residual_gain: float,
    thresholds: dict[str, float],
) -> pd.DataFrame:
    rng = np.random.default_rng(
        scenario.seed + _controller_seed_offset(controller_name)
    )
    perturbed_aero = perturb_aero(plant_config.aero, scenario)
    first_reference = reference_profile.iloc[0]
    state = np.array(
        [
            first_reference["alpha_ref_rad"] + scenario.initial_error.alpha_rad,
            first_reference["q_ref_radps"] + scenario.initial_error.q_radps,
            first_reference["theta_ref_rad"] + scenario.initial_error.theta_rad,
        ],
        dtype=float,
    )
    dt = float(
        reference_profile["time_s"].iloc[1] - reference_profile["time_s"].iloc[0]
    )
    actuator = initialize_actuator(scenario, dt)
    rows: list[dict[str, Any]] = []
    last_raw = 0.0
    solver_failure_seen = False
    latest_residual_qdot = 0.0
    for _idx, row in reference_profile.iterrows():
        measured_state = noisy_measurement(state, scenario, rng)
        solver_status = "not_applicable"
        solve_time = 0.0
        objective_value = np.nan
        if _is_nmpc_update_time(float(row["time_s"]), nmpc_config.dt):
            nmpc_idx = min(
                int(round(float(row["time_s"]) / nmpc_config.dt)),
                len(nmpc_reference) - 1,
            )
            horizon = nmpc_reference.iloc[
                nmpc_idx : nmpc_idx + nmpc_config.horizon_steps + 1
            ]
            try:
                last_raw, step_log = solve_residual_mpc_step(
                    state=measured_state,
                    previous_flap_rad=actuator.previous_applied_rad,
                    horizon=horizon,
                    vehicle=plant_config.vehicle,
                    aero=plant_config.aero,
                    config=nmpc_config,
                    surrogate=surrogate,
                    residual_gain=residual_gain,
                )
                solver_status = str(step_log["solver_status"])
                solve_time = float(step_log["solve_time_s"])
                objective_value = float(step_log["objective_value"])
                latest_residual_qdot = float(step_log["predicted_residual_q_dot_first"])
                if not _is_successful_solver_status(solver_status):
                    solver_failure_seen = True
            except RuntimeError:
                solver_status = "RuntimeError"
                solver_failure_seen = True

        applied_control, actuator_log = actuator_step(
            raw_command=float(last_raw),
            actuator=actuator,
            scenario=scenario,
            row=row,
            dt=dt,
        )
        if solve_time == 0.0:
            latest_residual_qdot = _predict_current_residual_qdot(
                state=measured_state,
                delta_flap_rad=applied_control,
                row=row,
                surrogate=surrogate,
            )
        moment_nm, cm, effectiveness = _uncertain_moment_for_log(
            state=state,
            delta_flap_rad=applied_control,
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
                "controller": controller_name,
                "residual_gain": float(residual_gain),
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
                "predicted_residual_q_dot": float(latest_residual_qdot),
                "applied_residual_q_dot": float(residual_gain * latest_residual_qdot),
                "solver_status": solver_status,
                "solve_time_s": solve_time,
                "objective_value": objective_value,
                "solver_failure": solver_failure_seen,
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
            delta_flap_rad=applied_control,
            row=row,
            vehicle=plant_config.vehicle,
            aero=perturbed_aero,
            scenario=scenario,
            dt=dt,
        )
    return pd.DataFrame(rows)


def write_phase11_figures(
    *,
    summary: pd.DataFrame,
    rollouts: pd.DataFrame,
    comparison: pd.DataFrame,
    output_dir: Path,
    plot_settings: dict[str, Any],
) -> dict[str, Path]:
    gain_path = output_dir / "success_rate_vs_residual_gain.png"
    envelope_path = output_dir / "residual_mpc_alpha_error_envelopes.png"
    failure_path = output_dir / "residual_mpc_failure_modes.png"
    correction_path = output_dir / "residual_correction_over_time.png"
    comparison_path = output_dir / "phase11_vs_prior_success_rates.png"
    worst_path = output_dir / "residual_mpc_worst_case_replay.png"
    solve_time_path = output_dir / "residual_mpc_solve_time.png"
    _plot_gain_success(summary, gain_path)
    _plot_alpha_envelopes(rollouts, envelope_path, plot_settings)
    _plot_failures(summary, failure_path)
    _plot_residual_corrections(rollouts, correction_path)
    _plot_success(comparison, comparison_path)
    _plot_worst_case(summary, rollouts, worst_path)
    _plot_solve_time(rollouts, solve_time_path)
    return {
        "success_rate_vs_gain_figure": gain_path,
        "alpha_error_envelopes_figure": envelope_path,
        "failure_mode_figure": failure_path,
        "residual_correction_figure": correction_path,
        "comparison_success_rates_figure": comparison_path,
        "worst_case_replay_figure": worst_path,
        "solve_time_figure": solve_time_path,
    }


def _downsample_reference_profile(profile: pd.DataFrame, dt: float) -> pd.DataFrame:
    step = max(
        1, int(round(dt / float(profile["time_s"].iloc[1] - profile["time_s"].iloc[0])))
    )
    return profile.iloc[::step].reset_index(drop=True)


def _controller_name(base_name: str, residual_gain: float) -> str:
    return f"{base_name}_gain_{residual_gain:g}"


def _predict_current_residual_qdot(
    *,
    state: np.ndarray,
    delta_flap_rad: float,
    row: pd.Series,
    surrogate: ResidualSurrogate,
) -> float:
    features = np.array(
        [
            state[0],
            state[1],
            state[2],
            delta_flap_rad,
            float(row["mach"]),
            float(row["altitude_m"]),
            float(row["velocity_mps"]),
            float(row["density_kgm3"]),
            float(row["dynamic_pressure_pa"]),
        ],
        dtype=float,
    )
    return float(predict_residual_qdot_numpy(features=features, surrogate=surrogate)[0])


def _plot_gain_success(summary: pd.DataFrame, path: Path) -> None:
    rates = (
        summary.assign(success=summary["failure_label"].eq("success"))
        .groupby(["tier", "residual_gain"])["success"]
        .mean()
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for tier, frame in rates.groupby("tier", sort=True):
        ax.plot(frame["residual_gain"], frame["success"], marker="o", label=tier)
    ax.set_xlabel("Residual gain")
    ax.set_ylabel("Success rate")
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_alpha_envelopes(
    rollouts: pd.DataFrame, path: Path, plot_settings: dict[str, Any]
) -> None:
    percentiles = plot_settings.get("envelope_percentiles", [5.0, 95.0])
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    for (tier, gain), frame in rollouts.groupby(["tier", "residual_gain"], sort=True):
        grouped = frame.groupby("time_s")["alpha_error_rad"]
        median = grouped.median()
        low = grouped.quantile(float(percentiles[0]) / 100.0)
        high = grouped.quantile(float(percentiles[1]) / 100.0)
        ax.plot(median.index, median.to_numpy(), label=f"{tier}/gain={gain:g}")
        ax.fill_between(median.index, low.to_numpy(), high.to_numpy(), alpha=0.10)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Alpha error (rad)")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_residual_corrections(rollouts: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    solve_rows = rollouts[rollouts["solve_time_s"] > 0.0]
    for gain, frame in solve_rows.groupby("residual_gain", sort=True):
        grouped = frame.groupby("time_s")["applied_residual_q_dot"].median()
        ax.plot(grouped.index, grouped.to_numpy(), label=f"gain={gain:g}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Applied q_dot residual (rad/s^2)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_worst_case(summary: pd.DataFrame, rollouts: pd.DataFrame, path: Path) -> None:
    worst_row = (
        summary.assign(non_success=summary["failure_label"].ne("success"))
        .sort_values(["non_success", "max_alpha_error_rad"], ascending=[False, False])
        .iloc[0]
    )
    worst = rollouts[
        (rollouts["tier"] == worst_row["tier"])
        & (rollouts["scenario_id"] == int(worst_row["scenario_id"]))
        & (rollouts["controller"] == worst_row["controller"])
    ]
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.4), sharex=True)
    axes[0].fill_between(
        worst["time_s"],
        worst["alpha_min_rad"],
        worst["alpha_max_rad"],
        color="gray",
        alpha=0.22,
        label="alpha corridor",
    )
    axes[0].plot(worst["time_s"], worst["alpha_rad"], label="alpha")
    axes[0].plot(worst["time_s"], worst["alpha_ref_rad"], linestyle="--", label="ref")
    axes[1].plot(worst["time_s"], worst["delta_flap_rad"], label="applied flap")
    axes[0].set_ylabel("Alpha (rad)")
    axes[1].set_ylabel("Flap (rad)")
    axes[1].set_xlabel("Time (s)")
    axes[0].set_title(
        "Worst residual-MPC replay: "
        f"{worst_row['tier']}/{worst_row['controller']} "
        f"scenario {int(worst_row['scenario_id'])}"
    )
    axes[0].legend(loc="best")
    axes[1].legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _load_prior(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)
