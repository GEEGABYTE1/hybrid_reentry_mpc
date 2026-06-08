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
from reentry_mpc.nmpc import (
    NmpcConfig,
    NmpcSolverOptions,
    NmpcWeights,
    solve_nmpc_step,
)
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
class Phase16Variant:
    name: str
    control_dt_s: float
    horizon_steps: int
    alpha_weight_scale: float
    q_weight_scale: float
    terminal_alpha_weight_scale: float
    state_slack_scale: float
    control_weight_scale: float
    flap_rate_weight_scale: float
    planning_alpha_buffer_rad: float


@dataclass(frozen=True)
class Phase16Config:
    seed: int
    phase5_config: Path
    phase1_config: Path
    phase2_config: Path
    sweep_scenario_count_per_tier: int
    full_scenario_count_per_tier: int
    run_full_best_variants: bool
    max_time_s: float | None
    controller_variants: list[Phase16Variant]
    base_weights: NmpcWeights
    solver: NmpcSolverOptions
    plot_settings: dict[str, Any]


def load_phase16_config(path: str | Path) -> Phase16Config:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)
    base = raw["nmpc_base"]
    max_time = raw.get("max_time_s")
    return Phase16Config(
        seed=int(raw["seed"]),
        phase5_config=Path(raw["phase5_config"]),
        phase1_config=Path(raw["phase1_config"]),
        phase2_config=Path(raw["phase2_config"]),
        sweep_scenario_count_per_tier=int(raw["sweep_scenario_count_per_tier"]),
        full_scenario_count_per_tier=int(raw["full_scenario_count_per_tier"]),
        run_full_best_variants=bool(raw["run_full_best_variants"]),
        max_time_s=None if max_time is None else float(max_time),
        controller_variants=[
            Phase16Variant(
                name=str(item["name"]),
                control_dt_s=float(item["control_dt_s"]),
                horizon_steps=int(item["horizon_steps"]),
                alpha_weight_scale=float(item["alpha_weight_scale"]),
                q_weight_scale=float(item["q_weight_scale"]),
                terminal_alpha_weight_scale=float(item["terminal_alpha_weight_scale"]),
                state_slack_scale=float(item["state_slack_scale"]),
                control_weight_scale=float(item["control_weight_scale"]),
                flap_rate_weight_scale=float(item["flap_rate_weight_scale"]),
                planning_alpha_buffer_rad=float(item["planning_alpha_buffer_rad"]),
            )
            for item in raw["controller_variants"]
        ],
        base_weights=NmpcWeights(
            **{key: float(value) for key, value in base["weights"].items()}
        ),
        solver=NmpcSolverOptions(
            max_iter=int(base["solver"]["max_iter"]),
            acceptable_tol=float(base["solver"]["acceptable_tol"]),
            print_level=int(base["solver"]["print_level"]),
        ),
        plot_settings=dict(raw.get("plot_settings", {})),
    )


def run_phase16_success_recovery(
    config_path: str | Path = "configs/phase16_success_recovery.yaml",
    output_dir: str | Path = "outputs/phase16_success_recovery",
    *,
    progress: bool = False,
) -> dict[str, Path | pd.DataFrame]:
    config = load_phase16_config(config_path)
    phase5_config = load_phase5_config(config.phase5_config)
    plant_config = load_phase1_config(config.phase1_config)
    phase2_config = load_phase2_config(config.phase2_config)
    reference_profile = _maybe_truncate_reference(
        build_reference_profile(phase2_config), config.max_time_s
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    rollout_frames: list[pd.DataFrame] = []
    for tier_idx, tier in enumerate(phase5_config.tiers):
        scenario_count = min(tier.scenario_count, config.sweep_scenario_count_per_tier)
        for scenario_id in range(scenario_count):
            scenario_seed = phase5_config.seed + tier_idx * 10_000 + scenario_id
            scenario = sample_scenario(
                scenario_id=scenario_id,
                seed=scenario_seed,
                ranges=tier.uncertainty_ranges,
            )
            for variant in config.controller_variants:
                if progress:
                    print(
                        "phase16_rollout "
                        f"tier={tier.name} scenario={scenario_id:03d} "
                        f"controller={variant.name}",
                        flush=True,
                    )
                rollout = rollout_success_recovery_nmpc(
                    tier_name=tier.name,
                    variant=variant,
                    scenario=scenario,
                    reference_profile=reference_profile,
                    plant_config=plant_config,
                    base_weights=config.base_weights,
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
                    **diagnostics,
                    "variant": variant.__dict__,
                    "trajectory_csv": str(trajectory_path),
                    "uncertainty_parameters": scenario.to_nested_dict(),
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
                        **diagnostics,
                        "control_dt_s": variant.control_dt_s,
                        "horizon_steps": variant.horizon_steps,
                        "planning_alpha_buffer_rad": (
                            variant.planning_alpha_buffer_rad
                        ),
                        "scenario_count_requested": scenario_count,
                        "reference_duration_s": float(
                            reference_profile["time_s"].max()
                        ),
                        "trajectory_csv": str(trajectory_path),
                        "metrics_json": str(metrics_path),
                    }
                )
                rollout_frames.append(rollout)

    summary = pd.DataFrame(summary_rows)
    rollouts = pd.concat(rollout_frames, ignore_index=True)
    comparison = summarize_variant_comparison(summary)
    summary_path = output_path / "phase16_summary.csv"
    rollouts_path = output_path / "phase16_rollouts.csv"
    comparison_path = output_path / "phase16_variant_comparison.csv"
    summary.to_csv(summary_path, index=False)
    rollouts.to_csv(rollouts_path, index=False)
    comparison.to_csv(comparison_path, index=False)
    figure_paths = write_phase16_figures(
        summary=summary,
        rollouts=rollouts,
        output_dir=output_path,
        plot_settings=config.plot_settings,
    )
    return {
        "summary_csv": summary_path,
        "rollouts_csv": rollouts_path,
        "comparison_csv": comparison_path,
        "summary": summary,
        "rollouts": rollouts,
        "comparison": comparison,
        **figure_paths,
    }


def rollout_success_recovery_nmpc(
    *,
    tier_name: str,
    variant: Phase16Variant,
    scenario: UncertaintyScenario,
    reference_profile: pd.DataFrame,
    plant_config: Any,
    base_weights: NmpcWeights,
    solver: NmpcSolverOptions,
    thresholds: dict[str, float],
) -> pd.DataFrame:
    rng = np.random.default_rng(scenario.seed + _controller_seed_offset(variant.name))
    perturbed_aero = perturb_aero(plant_config.aero, scenario)
    planning_profile = _planning_profile(reference_profile, variant)
    nmpc_reference = _downsample_reference_profile(
        planning_profile, variant.control_dt_s
    )
    nmpc_config = NmpcConfig(
        horizon_steps=variant.horizon_steps,
        dt=variant.control_dt_s,
        weights=_scaled_weights(base_weights, variant),
        solver=solver,
    )
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
        measured_state = noisy_measurement(state, scenario, rng)
        schedule = _schedule_from_row(row, scenario)
        solver_status = "held"
        solve_time = 0.0
        if _is_nmpc_update_time(float(row["time_s"]), variant.control_dt_s):
            nmpc_idx = min(
                int(round(float(row["time_s"]) / variant.control_dt_s)),
                len(nmpc_reference) - 1,
            )
            horizon = nmpc_reference.iloc[
                nmpc_idx : nmpc_idx + variant.horizon_steps + 1
            ]
            try:
                last_raw, last_step_log = solve_nmpc_step(
                    state=measured_state,
                    previous_flap_rad=actuator.previous_applied_rad,
                    horizon=horizon,
                    vehicle=plant_config.vehicle,
                    aero=plant_config.aero,
                    config=nmpc_config,
                )
                solver_status = str(last_step_log["solver_status"])
                solve_time = float(last_step_log["solve_time_s"])
                if solver_status != "Solve_Succeeded":
                    solver_failure_seen = True
            except RuntimeError:
                last_step_log = _empty_step_log()
                solver_status = "RuntimeError"
                solver_failure_seen = True

        applied_control, actuator_log = actuator_step(
            raw_command=float(last_raw),
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
                "solver_failure": solver_failure_seen,
                "objective_value": float(last_step_log["objective_value"]),
                "max_predicted_alpha_violation_rad": float(
                    last_step_log["max_alpha_violation_rad"]
                ),
                "max_predicted_q_violation_radps": float(
                    last_step_log["max_q_violation_radps"]
                ),
                "predicted_alpha_constraint_active": bool(
                    last_step_log["alpha_constraint_active"]
                ),
                "predicted_q_constraint_active": bool(
                    last_step_log["q_constraint_active"]
                ),
                "control_dt_s": variant.control_dt_s,
                "horizon_steps": variant.horizon_steps,
                "planning_alpha_buffer_rad": variant.planning_alpha_buffer_rad,
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


def summarize_corridor_diagnostics(
    *, rollout: pd.DataFrame, tolerance: float, control_dt_s: float
) -> dict[str, Any]:
    lower_miss = rollout["alpha_min_rad"] - rollout["alpha_rad"]
    upper_miss = rollout["alpha_rad"] - rollout["alpha_max_rad"]
    lower_violation = lower_miss > tolerance
    upper_violation = upper_miss > tolerance
    alpha_violation = lower_violation | upper_violation
    q_lower = rollout["q_min_radps"] - rollout["q_radps"]
    q_upper = rollout["q_radps"] - rollout["q_max_radps"]
    q_violation = (q_lower > tolerance) | (q_upper > tolerance)
    if alpha_violation.any():
        first_idx = alpha_violation.idxmax()
        first_time = float(rollout.loc[first_idx, "time_s"])
        side = "high" if bool(upper_violation.loc[first_idx]) else "low"
    else:
        first_time = np.nan
        side = "none"
    alpha_miss = np.maximum.reduce(
        [
            lower_miss.to_numpy(dtype=float),
            upper_miss.to_numpy(dtype=float),
            np.zeros(len(rollout), dtype=float),
        ]
    )
    q_miss = np.maximum.reduce(
        [
            q_lower.to_numpy(dtype=float),
            q_upper.to_numpy(dtype=float),
            np.zeros(len(rollout), dtype=float),
        ]
    )
    row_dt = float(np.median(np.diff(rollout["time_s"]))) if len(rollout) > 1 else 0.0
    return {
        "first_alpha_violation_time_s": first_time,
        "first_alpha_violation_side": side,
        "max_alpha_corridor_miss_rad": float(np.max(alpha_miss)),
        "max_q_corridor_miss_radps": float(np.max(q_miss)),
        "alpha_violation_duration_s": float(alpha_violation.sum() * row_dt),
        "q_violation_duration_s": float(q_violation.sum() * row_dt),
        "first_alpha_violation_before_three_updates": bool(
            np.isfinite(first_time) and first_time < 3.0 * control_dt_s
        ),
        "mean_solve_time_s": float(
            rollout.loc[rollout["solver_status"].ne("held"), "solve_time_s"].mean()
        ),
        "p95_solve_time_s": float(
            rollout.loc[rollout["solver_status"].ne("held"), "solve_time_s"].quantile(
                0.95
            )
        ),
    }


def summarize_variant_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        summary.assign(success=summary["failure_label"].eq("success"))
        .groupby(["tier", "controller"], as_index=False)
        .agg(
            scenario_count=("scenario_id", "size"),
            success_count=("success", "sum"),
            success_rate=("success", "mean"),
            mean_rms_alpha_error_rad=("rms_alpha_error_rad", "mean"),
            median_max_alpha_miss_rad=("max_alpha_corridor_miss_rad", "median"),
            mean_alpha_violation_count=("alpha_violation_count", "mean"),
            mean_first_alpha_violation_time_s=(
                "first_alpha_violation_time_s",
                "mean",
            ),
            p95_solve_time_s=("p95_solve_time_s", "mean"),
        )
    )
    nominal = grouped[grouped["controller"].eq("nominal_nmpc_2s")][
        ["tier", "success_rate"]
    ].rename(columns={"success_rate": "nominal_success_rate"})
    grouped = grouped.merge(nominal, on="tier", how="left")
    grouped["success_rate_delta_vs_nominal"] = (
        grouped["success_rate"] - grouped["nominal_success_rate"]
    )
    return grouped


def write_phase16_figures(
    *,
    summary: pd.DataFrame,
    rollouts: pd.DataFrame,
    output_dir: Path,
    plot_settings: dict[str, Any],
) -> dict[str, Path]:
    paths = {
        "success_png": output_dir / "success_rate_by_variant.png",
        "failure_png": output_dir / "failure_mode_by_variant.png",
        "miss_png": output_dir / "alpha_corridor_miss_distribution.png",
        "first_violation_png": output_dir / "first_violation_time.png",
        "envelopes_png": output_dir / "alpha_envelopes_success_recovery.png",
        "timing_png": output_dir / "solve_time_vs_control_dt.png",
    }
    _plot_success(summary, paths["success_png"])
    _plot_failures(summary, paths["failure_png"])
    _plot_miss_distribution(summary, paths["miss_png"])
    _plot_first_violation(summary, paths["first_violation_png"])
    _plot_alpha_envelopes(
        rollouts,
        paths["envelopes_png"],
        plot_settings.get("envelope_percentiles", [5.0, 95.0]),
    )
    _plot_timing(summary, paths["timing_png"])
    return paths


def _scaled_weights(base: NmpcWeights, variant: Phase16Variant) -> NmpcWeights:
    return NmpcWeights(
        alpha=base.alpha * variant.alpha_weight_scale,
        q=base.q * variant.q_weight_scale,
        theta=base.theta,
        control=base.control * variant.control_weight_scale,
        flap_rate=base.flap_rate * variant.flap_rate_weight_scale,
        terminal_alpha=base.terminal_alpha * variant.terminal_alpha_weight_scale,
        state_slack=base.state_slack * variant.state_slack_scale,
    )


def _planning_profile(
    reference_profile: pd.DataFrame, variant: Phase16Variant
) -> pd.DataFrame:
    planning = reference_profile.copy()
    if variant.planning_alpha_buffer_rad > 0.0:
        planning["alpha_min_rad"] = (
            planning["alpha_min_rad"] + variant.planning_alpha_buffer_rad
        )
        planning["alpha_max_rad"] = (
            planning["alpha_max_rad"] - variant.planning_alpha_buffer_rad
        )
    return planning


def _maybe_truncate_reference(
    reference_profile: pd.DataFrame, max_time_s: float | None
) -> pd.DataFrame:
    if max_time_s is None:
        return reference_profile
    return reference_profile[reference_profile["time_s"] <= max_time_s].reset_index(
        drop=True
    )


def _empty_step_log() -> dict[str, Any]:
    return {
        "objective_value": 0.0,
        "max_alpha_violation_rad": 0.0,
        "max_q_violation_radps": 0.0,
        "alpha_constraint_active": False,
        "q_constraint_active": False,
    }


def _plot_success(summary: pd.DataFrame, path: Path) -> None:
    rates = (
        summary.assign(success=summary["failure_label"].eq("success"))
        .groupby(["tier", "controller"])["success"]
        .mean()
        .unstack("controller")
    )
    fig, ax = plt.subplots(figsize=(10, 4.8))
    rates.plot(kind="bar", ax=ax)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("success rate")
    ax.set_xlabel("uncertainty tier")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_failures(summary: pd.DataFrame, path: Path) -> None:
    counts = pd.crosstab(
        [summary["tier"], summary["controller"]], summary["failure_label"]
    )
    fig, ax = plt.subplots(figsize=(11, 5))
    counts.plot(kind="bar", stacked=True, ax=ax)
    ax.set_ylabel("scenario count")
    ax.set_xlabel("tier / controller")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_miss_distribution(summary: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    labels = []
    values = []
    for (tier, controller), frame in summary.groupby(["tier", "controller"]):
        labels.append(f"{tier}\n{controller}")
        values.append(frame["max_alpha_corridor_miss_rad"].to_numpy(dtype=float))
    ax.boxplot(values, tick_labels=labels, showfliers=False)
    ax.set_ylabel("max alpha corridor miss [rad]")
    ax.tick_params(axis="x", labelrotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_first_violation(summary: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for (tier, controller), frame in summary.groupby(["tier", "controller"]):
        times = frame["first_alpha_violation_time_s"].dropna()
        if not times.empty:
            ax.scatter(
                [f"{tier}\n{controller}"] * len(times),
                times,
                s=16,
                alpha=0.7,
            )
    ax.set_ylabel("first alpha violation time [s]")
    ax.tick_params(axis="x", labelrotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_alpha_envelopes(
    rollouts: pd.DataFrame, path: Path, percentiles: list[float]
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for ax, (tier, tier_data) in zip(axes, rollouts.groupby("tier"), strict=False):
        for controller, data in tier_data.groupby("controller"):
            pivot = data.pivot_table(
                index="time_s", columns="scenario_id", values="alpha_error_rad"
            )
            median = pivot.median(axis=1)
            lower = pivot.quantile(float(percentiles[0]) / 100.0, axis=1)
            upper = pivot.quantile(float(percentiles[1]) / 100.0, axis=1)
            ax.plot(median.index, median, label=controller)
            ax.fill_between(median.index, lower, upper, alpha=0.10)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title(tier)
        ax.set_xlabel("time [s]")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("alpha error [rad]")
    axes[-1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_timing(summary: pd.DataFrame, path: Path) -> None:
    timing = (
        summary.groupby(["controller", "control_dt_s"], as_index=False)[
            "p95_solve_time_s"
        ]
        .mean()
        .sort_values(["control_dt_s", "controller"])
    )
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    for controller, frame in timing.groupby("controller"):
        ax.plot(
            frame["control_dt_s"],
            frame["p95_solve_time_s"] * 1000.0,
            marker="o",
            label=controller,
        )
    ax.set_xlabel("control update period [s]")
    ax.set_ylabel("mean per-rollout p95 solve time [ms]")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
