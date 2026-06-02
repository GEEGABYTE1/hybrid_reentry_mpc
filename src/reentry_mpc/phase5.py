from __future__ import annotations

import json
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
)
from reentry_mpc.baseline_metrics import rms
from reentry_mpc.longitudinal import (
    AeroParams,
    Phase1Config,
    VehicleParams,
    load_phase1_config,
    scheduled_pitching_moment,
)
from reentry_mpc.nmpc import solve_nmpc_step
from reentry_mpc.phase2 import build_reference_profile, load_phase2_config
from reentry_mpc.phase3 import build_controllers, load_phase3_config
from reentry_mpc.phase4 import _downsample_reference_profile, load_phase4_config
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
class MonteCarloTier:
    """Named uncertainty tier for Phase 5."""

    name: str
    scenario_count: int
    uncertainty_ranges: dict[str, Any]


@dataclass(frozen=True)
class Phase5Config:
    """Configuration for the Monte Carlo uncertainty benchmark."""

    seed: int
    phase1_config: Path
    phase2_config: Path
    phase3_config: Path
    phase4_config: Path
    controller_names: list[str]
    tiers: list[MonteCarloTier]
    failure_thresholds: dict[str, float]
    plot_settings: dict[str, Any]


def load_phase5_config(path: str | Path) -> Phase5Config:
    """Load Phase 5 config."""

    with Path(path).open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)
    if "tiers" in raw:
        tiers = [
            MonteCarloTier(
                name=str(tier["name"]),
                scenario_count=int(tier["scenario_count"]),
                uncertainty_ranges=dict(tier["uncertainty_ranges"]),
            )
            for tier in raw["tiers"]
        ]
    else:
        tiers = [
            MonteCarloTier(
                name=str(raw.get("tier_name", "moderate")),
                scenario_count=int(raw["scenario_count"]),
                uncertainty_ranges=dict(raw["uncertainty_ranges"]),
            )
        ]
    return Phase5Config(
        seed=int(raw["seed"]),
        phase1_config=Path(raw["phase1_config"]),
        phase2_config=Path(raw["phase2_config"]),
        phase3_config=Path(raw["phase3_config"]),
        phase4_config=Path(raw["phase4_config"]),
        controller_names=list(raw["controller_names"]),
        tiers=tiers,
        failure_thresholds={
            key: float(value) for key, value in raw["failure_thresholds"].items()
        },
        plot_settings=dict(raw.get("plot_settings", {})),
    )


def run_phase5_monte_carlo(
    config_path: str | Path = "configs/phase5_monte_carlo.yaml",
    output_dir: str | Path = "outputs/phase5_monte_carlo",
) -> dict[str, Path | pd.DataFrame]:
    """Run the Phase 5 Monte Carlo benchmark and save artifacts."""

    config = load_phase5_config(config_path)
    plant_config = load_phase1_config(config.phase1_config)
    phase2_config = load_phase2_config(config.phase2_config)
    phase3_config = load_phase3_config(config.phase3_config)
    phase4_config = load_phase4_config(config.phase4_config)
    reference_profile = build_reference_profile(phase2_config)
    full_dt = float(
        reference_profile["time_s"].iloc[1] - reference_profile["time_s"].iloc[0]
    )
    nmpc_reference = _downsample_reference_profile(
        reference_profile, phase4_config.control_dt_s
    )
    controllers = build_controllers(
        config=phase3_config, plant_config=plant_config, dt=full_dt
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    rollouts: list[pd.DataFrame] = []
    for tier_idx, tier in enumerate(config.tiers):
        for scenario_id in range(tier.scenario_count):
            scenario_seed = config.seed + tier_idx * 10_000 + scenario_id
            scenario = sample_scenario(
                scenario_id=scenario_id,
                seed=scenario_seed,
                ranges=tier.uncertainty_ranges,
            )
            for controller_name in config.controller_names:
                rollout = _rollout_one_controller(
                    tier_name=tier.name,
                    controller_name=controller_name,
                    scenario=scenario,
                    controller=controllers.get(controller_name),
                    reference_profile=reference_profile,
                    nmpc_reference=nmpc_reference,
                    plant_config=plant_config,
                    phase4_config=phase4_config,
                    thresholds=config.failure_thresholds,
                )
                metrics = summarize_monte_carlo_rollout(
                    rollout=rollout,
                    tier_name=tier.name,
                    controller_name=controller_name,
                    scenario=scenario,
                    thresholds=config.failure_thresholds,
                )
                run_dir = (
                    output_path
                    / tier.name
                    / f"scenario_{scenario_id:03d}"
                    / controller_name
                )
                run_dir.mkdir(parents=True, exist_ok=True)
                trajectory_path = run_dir / "trajectory.csv"
                metrics_path = run_dir / "metrics.json"
                rollout.to_csv(trajectory_path, index=False)
                metrics_payload = {
                    **metrics,
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
                        "trajectory_csv": str(trajectory_path),
                        "metrics_json": str(metrics_path),
                    }
                )
                rollouts.append(rollout)

    summary = pd.DataFrame(summary_rows)
    combined_rollouts = pd.concat(rollouts, ignore_index=True)
    summary_path = output_path / "monte_carlo_summary.csv"
    combined_path = output_path / "monte_carlo_rollouts.csv"
    summary.to_csv(summary_path, index=False)
    combined_rollouts.to_csv(combined_path, index=False)
    figure_paths = write_phase5_figures(
        summary=summary,
        rollouts=combined_rollouts,
        output_dir=output_path,
        plot_settings=config.plot_settings,
    )
    return {
        "summary_csv": summary_path,
        "combined_rollouts_csv": combined_path,
        "summary": summary,
        "rollouts": combined_rollouts,
        **figure_paths,
    }


def summarize_monte_carlo_rollout(
    *,
    rollout: pd.DataFrame,
    tier_name: str,
    controller_name: str,
    scenario: UncertaintyScenario,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    """Compute per-rollout metrics and the ordered failure label."""

    alpha_error_abs = rollout["alpha_error_rad"].abs()
    q_error = rollout["q_error_rad"]
    control = rollout["delta_flap_rad"]
    alpha_low = rollout["alpha_min_rad"] - rollout["alpha_rad"]
    alpha_high = rollout["alpha_rad"] - rollout["alpha_max_rad"]
    q_low = rollout["q_min_radps"] - rollout["q_radps"]
    q_high = rollout["q_radps"] - rollout["q_max_radps"]
    tol = thresholds["corridor_tolerance_rad"]
    alpha_violation_count = int(((alpha_low > tol) | (alpha_high > tol)).sum())
    q_violation_count = int(((q_low > tol) | (q_high > tol)).sum())
    nonfinite = not np.isfinite(
        rollout[["alpha_rad", "q_radps", "theta_rad"]].to_numpy(dtype=float)
    ).all()
    unstable = bool(
        nonfinite
        or (rollout["alpha_rad"].abs() > thresholds["unstable_alpha_abs_rad"]).any()
        or (rollout["q_radps"].abs() > thresholds["unstable_q_abs_radps"]).any()
        or (rollout["theta_rad"].abs() > thresholds["unstable_theta_abs_rad"]).any()
    )
    solver_failure = bool(rollout["solver_failure"].any())
    flap_sat_fraction = float(rollout["flap_saturated"].mean())
    rate_sat_fraction = float(rollout["flap_rate_saturated"].mean())
    rms_alpha_error = rms(rollout["alpha_error_rad"])
    max_alpha_error = float(alpha_error_abs.max())
    rms_pitch_rate_error = rms(q_error)
    control_effort = float(np.trapezoid(np.abs(control), rollout["time_s"]))
    flap_saturation_failure = bool(
        (
            rms_alpha_error > thresholds["rms_alpha_error_rad"]
            or max_alpha_error > thresholds["max_alpha_error_rad"]
        )
        and (
            flap_sat_fraction > thresholds["flap_saturation_fraction"]
            or rate_sat_fraction > thresholds["flap_rate_saturation_fraction"]
        )
    )
    if solver_failure:
        failure_label = "solver_failure"
    elif unstable:
        failure_label = "unstable_response"
    elif q_violation_count > 0:
        failure_label = "pitch_rate_violation"
    elif alpha_violation_count > 0:
        failure_label = "alpha_corridor_violation"
    elif flap_saturation_failure:
        failure_label = "flap_saturation_failure"
    else:
        tracking_success = (
            rms_alpha_error <= thresholds["rms_alpha_error_rad"]
            and max_alpha_error <= thresholds["max_alpha_error_rad"]
        )
        failure_label = "success" if tracking_success else "alpha_corridor_violation"

    return {
        "tier": tier_name,
        "controller": controller_name,
        "scenario_id": scenario.scenario_id,
        "seed": scenario.seed,
        "rms_alpha_error_rad": rms_alpha_error,
        "max_alpha_error_rad": max_alpha_error,
        "rms_pitch_rate_error_radps": rms_pitch_rate_error,
        "control_effort_abs_rad_s": control_effort,
        "flap_saturation_fraction": flap_sat_fraction,
        "flap_rate_saturation_fraction": rate_sat_fraction,
        "alpha_violation_count": alpha_violation_count,
        "pitch_rate_violation_count": q_violation_count,
        "solver_failure": solver_failure,
        "unstable_response": unstable,
        "failure_label": failure_label,
    }


def write_phase5_figures(
    *,
    summary: pd.DataFrame,
    rollouts: pd.DataFrame,
    output_dir: Path,
    plot_settings: dict[str, Any],
) -> dict[str, Path]:
    """Write Phase 5 benchmark figures."""

    success_path = output_dir / "monte_carlo_success_rates.png"
    envelope_path = output_dir / "alpha_error_envelopes.png"
    failure_path = output_dir / "failure_mode_stacked_bar.png"
    histogram_path = output_dir / "rms_error_histogram.png"
    worst_path = output_dir / "worst_case_replay.png"

    success_rates = (
        summary.assign(success=summary["failure_label"].eq("success"))
        .groupby(["tier", "controller"], as_index=True)["success"]
        .mean()
        .sort_index()
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    success_rates.unstack("controller").plot(kind="bar", ax=ax)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Success rate")
    ax.set_xlabel("Uncertainty tier")
    fig.tight_layout()
    fig.savefig(success_path, dpi=160)
    plt.close(fig)

    percentiles = plot_settings.get("envelope_percentiles", [5.0, 95.0])
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    for (tier, controller), frame in rollouts.groupby(
        ["tier", "controller"], sort=True
    ):
        grouped = frame.groupby("time_s")["alpha_error_rad"]
        median = grouped.median()
        low = grouped.quantile(float(percentiles[0]) / 100.0)
        high = grouped.quantile(float(percentiles[1]) / 100.0)
        ax.plot(median.index, median.to_numpy(), label=f"{tier}/{controller}")
        ax.fill_between(median.index, low.to_numpy(), high.to_numpy(), alpha=0.16)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Alpha error (rad)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(envelope_path, dpi=160)
    plt.close(fig)

    failure_counts = pd.crosstab(
        [summary["tier"], summary["controller"]], summary["failure_label"]
    )
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    failure_counts.sort_index().plot(kind="bar", stacked=True, ax=ax)
    ax.set_xlabel("Uncertainty tier / controller")
    ax.set_ylabel("Scenario count")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(failure_path, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for (tier, controller), frame in summary.groupby(["tier", "controller"], sort=True):
        ax.hist(
            frame["rms_alpha_error_rad"],
            alpha=0.45,
            bins=10,
            label=f"{tier}/{controller}",
        )
    ax.set_xlabel("RMS alpha error (rad)")
    ax.set_ylabel("Count")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(histogram_path, dpi=160)
    plt.close(fig)

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
        "Worst replay: "
        f"{worst_row['tier']}/{worst_row['controller']} "
        f"scenario {int(worst_row['scenario_id'])}"
    )
    axes[0].legend(loc="best")
    axes[1].legend(loc="best")
    fig.tight_layout()
    fig.savefig(worst_path, dpi=160)
    plt.close(fig)

    return {
        "success_rates_figure": success_path,
        "alpha_error_envelopes_figure": envelope_path,
        "failure_mode_figure": failure_path,
        "rms_error_histogram_figure": histogram_path,
        "worst_case_replay_figure": worst_path,
    }


def _rollout_one_controller(
    *,
    tier_name: str,
    controller_name: str,
    scenario: UncertaintyScenario,
    controller: PIDController | GainScheduledLQRController | None,
    reference_profile: pd.DataFrame,
    nmpc_reference: pd.DataFrame,
    plant_config: Phase1Config,
    phase4_config: Any,
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
    if controller is not None:
        controller.reset()
    rows: list[dict[str, Any]] = []
    last_nmpc_raw = 0.0
    solver_failure_seen = False
    for _idx, row in reference_profile.iterrows():
        measured_state = noisy_measurement(state, scenario, rng)
        schedule = _schedule_from_row(row, scenario)
        reference_state = np.array(
            [row["alpha_ref_rad"], row["q_ref_radps"], row["theta_ref_rad"]],
            dtype=float,
        )
        solver_status = "not_applicable"
        solve_time = 0.0
        if controller_name == "nominal_nmpc":
            should_solve = _is_nmpc_update_time(
                float(row["time_s"]), phase4_config.control_dt_s
            )
            if should_solve:
                nmpc_idx = min(
                    int(round(float(row["time_s"]) / phase4_config.control_dt_s)),
                    len(nmpc_reference) - 1,
                )
                horizon = nmpc_reference.iloc[
                    nmpc_idx : nmpc_idx + phase4_config.nmpc.horizon_steps + 1
                ]
                try:
                    last_nmpc_raw, step_log = solve_nmpc_step(
                        state=measured_state,
                        previous_flap_rad=actuator.previous_applied_rad,
                        horizon=horizon,
                        vehicle=plant_config.vehicle,
                        aero=plant_config.aero,
                        config=phase4_config.nmpc,
                    )
                    solver_status = str(step_log["solver_status"])
                    solve_time = float(step_log["solve_time_s"])
                    if solver_status != "Solve_Succeeded":
                        solver_failure_seen = True
                except RuntimeError:
                    solver_status = "RuntimeError"
                    solver_failure_seen = True
            raw_command = last_nmpc_raw
        elif controller is not None:
            raw_command = controller.command(
                state=measured_state,
                reference_state=reference_state,
                dt=dt,
                schedule=schedule,
            )
        else:
            raise ValueError(f"Unknown controller: {controller_name}")

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
        rows.append(
            {
                "scenario_id": scenario.scenario_id,
                "tier": tier_name,
                "seed": scenario.seed,
                "controller": controller_name,
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


def _schedule_from_row(
    row: pd.Series, scenario: UncertaintyScenario
) -> dict[str, float]:
    density = float(row["density_kgm3"]) * scenario.density_scale
    dynamic_pressure = float(row["dynamic_pressure_pa"]) * scenario.density_scale
    return {
        "altitude_m": float(row["altitude_m"]),
        "velocity_mps": float(row["velocity_mps"]),
        "mach": float(row["mach"]),
        "density_kgm3": density,
        "dynamic_pressure_pa": dynamic_pressure,
    }


def _has_corridor_violation(
    *, state: np.ndarray, row: pd.Series, tolerance: float
) -> bool:
    return bool(
        state[0] < row["alpha_min_rad"] - tolerance
        or state[0] > row["alpha_max_rad"] + tolerance
        or state[1] < row["q_min_radps"] - tolerance
        or state[1] > row["q_max_radps"] + tolerance
    )


def _uncertain_moment_for_log(
    *,
    state: np.ndarray,
    delta_flap_rad: float,
    row: pd.Series,
    vehicle: VehicleParams,
    aero: AeroParams,
    scenario: UncertaintyScenario,
) -> tuple[float, float, float]:
    schedule = _schedule_from_row(row, scenario)
    moment_nm, cm, effectiveness = scheduled_pitching_moment(
        state=state,
        delta_flap_rad=delta_flap_rad,
        schedule=schedule,
        vehicle=vehicle,
        aero=aero,
    )
    return moment_nm + scenario.external_disturbance_moment_nm, cm, effectiveness


def _is_nmpc_update_time(time_s: float, control_dt: float) -> bool:
    remainder = np.mod(time_s, control_dt)
    return bool(np.isclose(remainder, 0.0) or np.isclose(remainder, control_dt))


def _controller_seed_offset(controller_name: str) -> int:
    offsets = {"pid": 101, "gain_scheduled_lqr": 202, "nominal_nmpc": 303}
    return offsets.get(controller_name, 0)
