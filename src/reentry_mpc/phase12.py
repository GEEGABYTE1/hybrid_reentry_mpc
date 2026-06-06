from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from reentry_mpc.artifacts import plt
from reentry_mpc.learning_augmented_mpc import (
    build_horizon_residual_biases,
    solve_horizon_biased_nmpc_step,
)
from reentry_mpc.longitudinal import load_phase1_config
from reentry_mpc.nmpc import NmpcConfig, NmpcSolverOptions, NmpcWeights
from reentry_mpc.phase2 import build_reference_profile, load_phase2_config
from reentry_mpc.phase4 import _downsample_reference_profile
from reentry_mpc.phase5 import (
    _controller_seed_offset,
    _has_corridor_violation,
    _is_nmpc_update_time,
    _schedule_from_row,
    _uncertain_moment_for_log,
    load_phase5_config,
    summarize_monte_carlo_rollout,
)
from reentry_mpc.phase6 import TighteningMargins, tighten_reference_profile
from reentry_mpc.phase10 import LoadedResidualModel, load_residual_model
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
class Phase12ControllerVariant:
    name: str
    residual_gain: float
    use_residual_model: bool
    use_tightened_corridor: bool


@dataclass(frozen=True)
class Phase12Config:
    seed: int
    phase5_config: Path
    phase1_config: Path
    phase2_config: Path
    residual_model_checkpoint: Path
    controller_variants: list[Phase12ControllerVariant]
    tightening: TighteningMargins
    nmpc: NmpcConfig
    plot_settings: dict[str, Any]


def load_phase12_config(path: str | Path) -> Phase12Config:
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
    return Phase12Config(
        seed=int(raw["seed"]),
        phase5_config=Path(raw["phase5_config"]),
        phase1_config=Path(raw["phase1_config"]),
        phase2_config=Path(raw["phase2_config"]),
        residual_model_checkpoint=Path(raw["residual_model_checkpoint"]),
        controller_variants=[
            Phase12ControllerVariant(
                name=str(item["name"]),
                residual_gain=float(item["residual_gain"]),
                use_residual_model=bool(item["use_residual_model"]),
                use_tightened_corridor=bool(item["use_tightened_corridor"]),
            )
            for item in raw["controller_variants"]
        ],
        tightening=TighteningMargins(
            alpha_margin_rad=float(raw["tightening"]["alpha_margin_rad"]),
            q_margin_radps=float(raw["tightening"]["q_margin_radps"]),
        ),
        nmpc=NmpcConfig(
            horizon_steps=int(raw["nmpc"]["horizon_steps"]),
            dt=float(raw["nmpc"]["control_dt_s"]),
            weights=weights,
            solver=solver,
        ),
        plot_settings=dict(raw.get("plot_settings", {})),
    )


def run_phase12_learning_augmented_mpc(
    config_path: str | Path = "configs/phase12_learning_augmented_mpc.yaml",
    output_dir: str | Path = "outputs/phase12_learning_augmented_mpc",
    *,
    progress: bool = False,
) -> dict[str, Path | pd.DataFrame]:
    config = load_phase12_config(config_path)
    phase5_config = load_phase5_config(config.phase5_config)
    plant_config = load_phase1_config(config.phase1_config)
    phase2_config = load_phase2_config(config.phase2_config)
    reference_profile = build_reference_profile(phase2_config)
    nmpc_reference = _downsample_reference_profile(reference_profile, config.nmpc.dt)
    tightened_reference = tighten_reference_profile(
        reference_profile, config.tightening
    )
    tightened_nmpc_reference = _downsample_reference_profile(
        tightened_reference, config.nmpc.dt
    )
    residual_model = load_residual_model(config.residual_model_checkpoint)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    rollouts: list[pd.DataFrame] = []
    for tier_idx, tier in enumerate(phase5_config.tiers):
        for scenario_id in range(tier.scenario_count):
            scenario_seed = phase5_config.seed + tier_idx * 10_000 + scenario_id
            scenario = sample_scenario(
                scenario_id=scenario_id,
                seed=scenario_seed,
                ranges=tier.uncertainty_ranges,
            )
            for variant in config.controller_variants:
                if progress:
                    print(
                        "phase12_rollout "
                        f"tier={tier.name} scenario={scenario_id:03d} "
                        f"controller={variant.name}",
                        flush=True,
                    )
                planning_reference = (
                    tightened_nmpc_reference
                    if variant.use_tightened_corridor
                    else nmpc_reference
                )
                rollout = rollout_learning_augmented_nmpc(
                    tier_name=tier.name,
                    variant=variant,
                    scenario=scenario,
                    reference_profile=reference_profile,
                    planning_reference=planning_reference,
                    plant_config=plant_config,
                    nmpc_config=config.nmpc,
                    residual_model=residual_model,
                    thresholds=phase5_config.failure_thresholds,
                )
                metrics = summarize_monte_carlo_rollout(
                    rollout=rollout,
                    tier_name=tier.name,
                    controller_name=variant.name,
                    scenario=scenario,
                    thresholds=phase5_config.failure_thresholds,
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
                metrics_payload = {
                    **metrics,
                    "trajectory_csv": str(trajectory_path),
                    "uncertainty_parameters": scenario.to_nested_dict(),
                    "residual_model_checkpoint": str(config.residual_model_checkpoint),
                    "residual_gain": variant.residual_gain,
                    "use_residual_model": variant.use_residual_model,
                    "use_tightened_corridor": variant.use_tightened_corridor,
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
                        "residual_gain": variant.residual_gain,
                        "use_residual_model": variant.use_residual_model,
                        "use_tightened_corridor": variant.use_tightened_corridor,
                        "trajectory_csv": str(trajectory_path),
                        "metrics_json": str(metrics_path),
                    }
                )
                rollouts.append(rollout)

    summary = pd.DataFrame(summary_rows)
    combined_rollouts = pd.concat(rollouts, ignore_index=True)
    summary_path = output_path / "phase12_summary_table.csv"
    phase8_alias_path = output_path / "phase8_summary_table.csv"
    rollouts_path = output_path / "phase12_rollouts.csv"
    summary.to_csv(summary_path, index=False)
    summary.to_csv(phase8_alias_path, index=False)
    combined_rollouts.to_csv(rollouts_path, index=False)
    figure_paths = write_phase12_figures(
        summary=summary,
        rollouts=combined_rollouts,
        output_dir=output_path,
        plot_settings=config.plot_settings,
    )
    return {
        "summary_csv": summary_path,
        "phase8_summary_csv": phase8_alias_path,
        "rollouts_csv": rollouts_path,
        "summary": summary,
        "rollouts": combined_rollouts,
        **figure_paths,
    }


def rollout_learning_augmented_nmpc(
    *,
    tier_name: str,
    variant: Phase12ControllerVariant,
    scenario: UncertaintyScenario,
    reference_profile: pd.DataFrame,
    planning_reference: pd.DataFrame,
    plant_config: Any,
    nmpc_config: NmpcConfig,
    residual_model: LoadedResidualModel,
    thresholds: dict[str, float],
) -> pd.DataFrame:
    rng = np.random.default_rng(scenario.seed + _controller_seed_offset("nominal_nmpc"))
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
    last_step_log = _empty_step_log()
    solver_failure_seen = False
    for _idx, row in reference_profile.iterrows():
        loop_start = time.perf_counter()
        measured_state = noisy_measurement(state, scenario, rng)
        schedule = _schedule_from_row(row, scenario)
        solver_status = "held"
        solve_time = 0.0
        nn_time = 0.0
        residual_biases = np.zeros(nmpc_config.horizon_steps, dtype=float)
        if _is_nmpc_update_time(float(row["time_s"]), nmpc_config.dt):
            nmpc_idx = min(
                int(round(float(row["time_s"]) / nmpc_config.dt)),
                len(planning_reference) - 1,
            )
            horizon = planning_reference.iloc[
                nmpc_idx : nmpc_idx + nmpc_config.horizon_steps + 1
            ]
            if variant.use_residual_model:
                residual_biases, nn_time = build_horizon_residual_biases(
                    loaded_model=residual_model,
                    state=measured_state,
                    previous_flap_rad=actuator.previous_applied_rad,
                    horizon=horizon,
                    horizon_steps=nmpc_config.horizon_steps,
                )
                residual_biases = residual_biases * variant.residual_gain
            try:
                last_raw, last_step_log = solve_horizon_biased_nmpc_step(
                    state=measured_state,
                    previous_flap_rad=actuator.previous_applied_rad,
                    horizon=horizon,
                    vehicle=plant_config.vehicle,
                    aero=plant_config.aero,
                    config=nmpc_config,
                    residual_q_dot_biases=residual_biases,
                )
                solver_status = str(last_step_log["solver_status"])
                solve_time = float(last_step_log["solve_time_s"])
                if solver_status != "Solve_Succeeded":
                    solver_failure_seen = True
            except RuntimeError:
                last_step_log = _empty_step_log()
                solver_status = "RuntimeError"
                solver_failure_seen = True
        raw_command = last_raw

        applied_control, actuator_log = actuator_step(
            raw_command=float(raw_command),
            actuator=actuator,
            scenario=scenario,
            row=row,
            dt=dt,
        )
        moment_nm, cm, effectiveness = _uncertain_moment_for_log(
            state=state,
            delta_flap_rad=applied_control,
            row=row,
            vehicle=plant_config.vehicle,
            aero=perturbed_aero,
            scenario=scenario,
        )
        alpha_error = float(state[0] - row["alpha_ref_rad"])
        q_error = float(state[1] - row["q_ref_radps"])
        theta_error = float(state[2] - row["theta_ref_rad"])
        total_loop_time = time.perf_counter() - loop_start
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
                "alpha_error_rad": alpha_error,
                "q_error_rad": q_error,
                "theta_error_rad": theta_error,
                "solver_status": solver_status,
                "solve_time_s": solve_time,
                "nn_inference_time_s": float(nn_time),
                "total_loop_time_s": float(total_loop_time),
                "solver_failure": solver_failure_seen,
                "residual_gain": float(variant.residual_gain),
                "use_residual_model": variant.use_residual_model,
                "use_tightened_corridor": variant.use_tightened_corridor,
                "predicted_residual_q_dot_first": float(
                    last_step_log["predicted_residual_q_dot_first"]
                ),
                "predicted_residual_q_dot_mean": float(
                    last_step_log["predicted_residual_q_dot_mean"]
                ),
                "predicted_residual_q_dot_max_abs": float(
                    last_step_log["predicted_residual_q_dot_max_abs"]
                ),
                "residual_correction_abs": abs(
                    float(last_step_log["predicted_residual_q_dot_first"])
                ),
                "pitching_moment_nm": moment_nm,
                "cm": cm,
                "flap_effectiveness": effectiveness,
                **actuator_log,
                **schedule,
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


def write_phase12_figures(
    *,
    summary: pd.DataFrame,
    rollouts: pd.DataFrame,
    output_dir: Path,
    plot_settings: dict[str, Any],
) -> dict[str, Path]:
    paths = {
        "envelopes_png": output_dir / "nominal_vs_residual_mpc_envelopes.png",
        "success_png": output_dir / "residual_mpc_success_rates.png",
        "failures_png": output_dir / "residual_mpc_failure_modes.png",
        "correction_png": output_dir / "residual_correction_magnitude.png",
        "timing_png": output_dir / "residual_mpc_timing.png",
    }
    percentiles = plot_settings.get("envelope_percentiles", [5.0, 95.0])
    _plot_alpha_envelopes(rollouts, paths["envelopes_png"], percentiles)
    _plot_success(summary, paths["success_png"])
    _plot_failures(summary, paths["failures_png"])
    _plot_corrections(rollouts, paths["correction_png"])
    _plot_timing(rollouts, paths["timing_png"])
    return paths


def _plot_alpha_envelopes(
    rollouts: pd.DataFrame, path: Path, percentiles: list[float]
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, (tier, tier_data) in zip(axes, rollouts.groupby("tier"), strict=False):
        for controller, data in tier_data.groupby("controller"):
            pivot = data.pivot_table(
                index="time_s", columns="scenario_id", values="alpha_error_rad"
            )
            lower = pivot.quantile(percentiles[0] / 100.0, axis=1)
            upper = pivot.quantile(percentiles[1] / 100.0, axis=1)
            median = pivot.median(axis=1)
            ax.plot(median.index, median, label=controller)
            ax.fill_between(median.index, lower, upper, alpha=0.14)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title(tier)
        ax.set_xlabel("time [s]")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("alpha error [rad]")
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_success(summary: pd.DataFrame, path: Path) -> None:
    grouped = (
        summary.assign(success=summary["failure_label"].eq("success"))
        .groupby(["tier", "controller"])["success"]
        .mean()
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    for idx, (tier, data) in enumerate(grouped.groupby("tier")):
        x = np.arange(len(data)) + idx * 0.36
        ax.bar(x, data["success"], width=0.34, label=tier)
        ax.set_xticks(x, data["controller"], rotation=20, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("success rate")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_failures(summary: pd.DataFrame, path: Path) -> None:
    counts = (
        summary.groupby(["tier", "controller", "failure_label"])
        .size()
        .reset_index(name="count")
    )
    labels = sorted(counts["failure_label"].unique())
    groups = counts[["tier", "controller"]].drop_duplicates().reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    bottoms = np.zeros(len(groups))
    x = np.arange(len(groups))
    for label in labels:
        values = []
        for _, group in groups.iterrows():
            match = counts[
                (counts["tier"] == group["tier"])
                & (counts["controller"] == group["controller"])
                & (counts["failure_label"] == label)
            ]
            values.append(0 if match.empty else int(match["count"].iloc[0]))
        ax.bar(x, values, bottom=bottoms, label=label)
        bottoms += np.asarray(values)
    ax.set_xticks(
        x,
        [f"{row.tier}\n{row.controller}" for row in groups.itertuples()],
        rotation=20,
        ha="right",
    )
    ax.set_ylabel("rollout count")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_corrections(rollouts: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    data = rollouts[rollouts["use_residual_model"]]
    if data.empty:
        ax.text(0.5, 0.5, "No residual-corrected rollouts", ha="center")
    else:
        for controller, subset in data.groupby("controller"):
            profile = subset.groupby("time_s")["residual_correction_abs"].median()
            ax.plot(profile.index, profile, label=controller)
        ax.set_ylabel("|scheduled residual q-dot| [rad/s^2]")
        ax.set_xlabel("time [s]")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_timing(rollouts: pd.DataFrame, path: Path) -> None:
    update_rows = rollouts[rollouts["solver_status"].ne("held")]
    timing = (
        update_rows.groupby(["tier", "controller"])[
            ["solve_time_s", "nn_inference_time_s", "total_loop_time_s"]
        ]
        .median()
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(timing))
    width = 0.25
    ax.bar(x - width, timing["solve_time_s"], width=width, label="solver")
    ax.bar(x, timing["nn_inference_time_s"], width=width, label="NN inference")
    ax.bar(x + width, timing["total_loop_time_s"], width=width, label="loop")
    ax.set_xticks(
        x,
        [f"{row.tier}\n{row.controller}" for row in timing.itertuples()],
        rotation=20,
        ha="right",
    )
    ax.set_ylabel("median time [s]")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _empty_step_log() -> dict[str, float]:
    return {
        "predicted_residual_q_dot_first": 0.0,
        "predicted_residual_q_dot_mean": 0.0,
        "predicted_residual_q_dot_max_abs": 0.0,
    }
