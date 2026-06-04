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
from reentry_mpc.phase2 import build_reference_profile, load_phase2_config
from reentry_mpc.phase4 import _downsample_reference_profile, load_phase4_config
from reentry_mpc.phase5 import (
    _controller_seed_offset,
    _has_corridor_violation,
    _is_nmpc_update_time,
    _schedule_from_row,
    _uncertain_moment_for_log,
    load_phase5_config,
    summarize_monte_carlo_rollout,
)
from reentry_mpc.scenario_mpc import (
    ScenarioMpcConfig,
    load_scenario_mpc_config,
    solve_scenario_mpc_step,
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
class Phase7Config:
    seed: int
    phase5_config: Path
    phase1_config: Path
    phase2_config: Path
    phase4_config: Path
    phase6_summary: Path
    controller_name: str
    scenario_mpc: ScenarioMpcConfig
    plot_settings: dict[str, Any]


def load_phase7_config(path: str | Path) -> Phase7Config:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)
    return Phase7Config(
        seed=int(raw["seed"]),
        phase5_config=Path(raw["phase5_config"]),
        phase1_config=Path(raw["phase1_config"]),
        phase2_config=Path(raw["phase2_config"]),
        phase4_config=Path(raw["phase4_config"]),
        phase6_summary=Path(raw["phase6_summary"]),
        controller_name=str(raw.get("controller_name", "scenario_nmpc")),
        scenario_mpc=load_scenario_mpc_config(raw["scenario_mpc"]),
        plot_settings=dict(raw.get("plot_settings", {})),
    )


def run_phase7_scenario_mpc(
    config_path: str | Path = "configs/phase7_scenario_mpc.yaml",
    output_dir: str | Path = "outputs/phase7_scenario_mpc",
    *,
    progress: bool = False,
) -> dict[str, Path | pd.DataFrame]:
    config = load_phase7_config(config_path)
    phase5_config = load_phase5_config(config.phase5_config)
    plant_config = load_phase1_config(config.phase1_config)
    phase2_config = load_phase2_config(config.phase2_config)
    phase4_config = load_phase4_config(config.phase4_config)
    reference_profile = build_reference_profile(phase2_config)
    nmpc_reference = _downsample_reference_profile(
        reference_profile, config.scenario_mpc.nmpc.dt
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    rollouts: list[pd.DataFrame] = []
    for tier_idx, tier in enumerate(phase5_config.tiers):
        scenario_count = tier.scenario_count
        if config.scenario_mpc.max_scenarios_per_tier is not None:
            scenario_count = min(
                scenario_count, config.scenario_mpc.max_scenarios_per_tier
            )
        for scenario_id in range(scenario_count):
            scenario_seed = phase5_config.seed + tier_idx * 10_000 + scenario_id
            scenario = sample_scenario(
                scenario_id=scenario_id,
                seed=scenario_seed,
                ranges=tier.uncertainty_ranges,
            )
            if progress:
                print(
                    "phase7_rollout "
                    f"tier={tier.name} scenario={scenario_id:03d} "
                    f"controller={config.controller_name}",
                    flush=True,
                )
            rollout = _rollout_scenario_nmpc(
                tier_name=tier.name,
                controller_name=config.controller_name,
                scenario=scenario,
                reference_profile=reference_profile,
                nmpc_reference=nmpc_reference,
                plant_config=plant_config,
                phase4_control_dt=phase4_config.control_dt_s,
                scenario_mpc_config=config.scenario_mpc,
                thresholds=phase5_config.failure_thresholds,
            )
            metrics = summarize_monte_carlo_rollout(
                rollout=rollout,
                tier_name=tier.name,
                controller_name=config.controller_name,
                scenario=scenario,
                thresholds=phase5_config.failure_thresholds,
            )
            run_dir = output_path / tier.name / f"scenario_{scenario_id:03d}"
            run_dir.mkdir(parents=True, exist_ok=True)
            trajectory_path = run_dir / "trajectory.csv"
            metrics_path = run_dir / "metrics.json"
            rollout.to_csv(trajectory_path, index=False)
            metrics_payload = {
                **metrics,
                "trajectory_csv": str(trajectory_path),
                "uncertainty_parameters": scenario.to_nested_dict(),
                "design_scenarios": [
                    design.__dict__ for design in config.scenario_mpc.design_scenarios
                ],
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
                    "design_scenario_count": len(config.scenario_mpc.design_scenarios),
                }
            )
            rollouts.append(rollout)

    summary = pd.DataFrame(summary_rows)
    combined_rollouts = pd.concat(rollouts, ignore_index=True)
    baseline_summary = _filter_baseline_to_phase7_subset(
        _load_baseline(config.phase6_summary), summary
    )
    comparison = pd.concat([baseline_summary, summary], ignore_index=True, sort=False)

    summary_path = output_path / "phase7_summary.csv"
    rollouts_path = output_path / "phase7_rollouts.csv"
    comparison_path = output_path / "phase7_vs_prior_summary.csv"
    summary.to_csv(summary_path, index=False)
    combined_rollouts.to_csv(rollouts_path, index=False)
    comparison.to_csv(comparison_path, index=False)
    figure_paths = write_phase7_figures(
        summary=summary,
        rollouts=combined_rollouts,
        comparison=comparison,
        output_dir=output_path,
        plot_settings=config.plot_settings,
    )
    return {
        "summary_csv": summary_path,
        "rollouts_csv": rollouts_path,
        "comparison_csv": comparison_path,
        "summary": summary,
        "rollouts": combined_rollouts,
        "comparison": comparison,
        **figure_paths,
    }


def _rollout_scenario_nmpc(
    *,
    tier_name: str,
    controller_name: str,
    scenario: UncertaintyScenario,
    reference_profile: pd.DataFrame,
    nmpc_reference: pd.DataFrame,
    plant_config: Any,
    phase4_control_dt: float,
    scenario_mpc_config: ScenarioMpcConfig,
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
    for _idx, row in reference_profile.iterrows():
        measured_state = noisy_measurement(state, scenario, rng)
        solver_status = "not_applicable"
        solve_time = 0.0
        objective_value = np.nan
        if _is_nmpc_update_time(float(row["time_s"]), phase4_control_dt):
            nmpc_idx = min(
                int(round(float(row["time_s"]) / scenario_mpc_config.nmpc.dt)),
                len(nmpc_reference) - 1,
            )
            horizon = nmpc_reference.iloc[
                nmpc_idx : nmpc_idx + scenario_mpc_config.nmpc.horizon_steps + 1
            ]
            try:
                last_raw, step_log = solve_scenario_mpc_step(
                    state=measured_state,
                    previous_flap_rad=actuator.previous_applied_rad,
                    horizon=horizon,
                    vehicle=plant_config.vehicle,
                    aero=plant_config.aero,
                    config=scenario_mpc_config,
                )
                solver_status = str(step_log["solver_status"])
                solve_time = float(step_log["solve_time_s"])
                objective_value = float(step_log["objective_value"])
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


def write_phase7_figures(
    *,
    summary: pd.DataFrame,
    rollouts: pd.DataFrame,
    comparison: pd.DataFrame,
    output_dir: Path,
    plot_settings: dict[str, Any],
) -> dict[str, Path]:
    success_path = output_dir / "scenario_nmpc_success_rates.png"
    comparison_path = output_dir / "phase7_vs_prior_success_rates.png"
    envelope_path = output_dir / "scenario_nmpc_alpha_error_envelopes.png"
    failure_path = output_dir / "scenario_nmpc_failure_modes.png"
    solve_time_path = output_dir / "scenario_nmpc_solve_time.png"

    _plot_success(summary, success_path)
    _plot_success(comparison, comparison_path)
    _plot_alpha_envelopes(rollouts, envelope_path, plot_settings)
    _plot_failures(summary, failure_path)
    _plot_solve_time(rollouts, solve_time_path)
    return {
        "success_rates_figure": success_path,
        "comparison_success_rates_figure": comparison_path,
        "alpha_error_envelopes_figure": envelope_path,
        "failure_mode_figure": failure_path,
        "solve_time_figure": solve_time_path,
    }


def _plot_success(summary: pd.DataFrame, path: Path) -> None:
    rates = (
        summary.assign(success=summary["failure_label"].eq("success"))
        .groupby(["tier", "controller"])["success"]
        .mean()
        .sort_index()
    )
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    rates.unstack("controller").plot(kind="bar", ax=ax)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Success rate")
    ax.set_xlabel("Uncertainty tier")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_alpha_envelopes(
    rollouts: pd.DataFrame, path: Path, plot_settings: dict[str, Any]
) -> None:
    percentiles = plot_settings.get("envelope_percentiles", [5.0, 95.0])
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    for tier, frame in rollouts.groupby("tier", sort=True):
        grouped = frame.groupby("time_s")["alpha_error_rad"]
        median = grouped.median()
        low = grouped.quantile(float(percentiles[0]) / 100.0)
        high = grouped.quantile(float(percentiles[1]) / 100.0)
        ax.plot(median.index, median.to_numpy(), label=f"{tier}/scenario_nmpc")
        ax.fill_between(median.index, low.to_numpy(), high.to_numpy(), alpha=0.16)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Alpha error (rad)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_failures(summary: pd.DataFrame, path: Path) -> None:
    counts = pd.crosstab(
        [summary["tier"], summary["controller"]], summary["failure_label"]
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    counts.sort_index().plot(kind="bar", stacked=True, ax=ax)
    ax.set_xlabel("Uncertainty tier / controller")
    ax.set_ylabel("Scenario count")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_solve_time(rollouts: pd.DataFrame, path: Path) -> None:
    solve_rows = rollouts[rollouts["solve_time_s"] > 0.0]
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for tier, frame in solve_rows.groupby("tier", sort=True):
        ax.plot(frame["time_s"], frame["solve_time_s"], ".", alpha=0.45, label=tier)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Solve time (s)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _load_baseline(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _filter_baseline_to_phase7_subset(
    baseline: pd.DataFrame, phase7_summary: pd.DataFrame
) -> pd.DataFrame:
    if baseline.empty:
        return baseline
    keys = phase7_summary[["tier", "scenario_id"]].drop_duplicates()
    return baseline.merge(keys, on=["tier", "scenario_id"], how="inner")


def _is_successful_solver_status(status: str) -> bool:
    return status in {"Solve_Succeeded", "Solved_To_Acceptable_Level"}
