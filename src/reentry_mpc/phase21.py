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
    _maybe_truncate_reference,
    summarize_controlled_recovery,
)
from reentry_mpc.uncertainty import (
    actuator_step,
    initialize_actuator,
    perturb_aero,
    sample_scenario,
    uncertain_rk4_step,
)


@dataclass(frozen=True)
class Phase21Config:
    seed: int
    phase5_config: Path
    phase1_config: Path
    phase2_config: Path
    phase20_output_dir: Path
    baseline_controller: str
    max_cases: int | None
    max_time_s: float | None
    recovery_thresholds: ControlledRecoveryThresholds
    plot_settings: dict[str, Any]


def load_phase21_config(path: str | Path) -> Phase21Config:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)
    thresholds = raw["failure_thresholds"]
    max_cases = raw.get("max_cases")
    max_time = raw.get("max_time_s")
    return Phase21Config(
        seed=int(raw["seed"]),
        phase5_config=Path(raw["phase5_config"]),
        phase1_config=Path(raw["phase1_config"]),
        phase2_config=Path(raw["phase2_config"]),
        phase20_output_dir=Path(raw["phase20_output_dir"]),
        baseline_controller=str(raw["baseline_controller"]),
        max_cases=None if max_cases is None else int(max_cases),
        max_time_s=None if max_time is None else float(max_time),
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


def run_phase21_missed_case_autopsy(
    config_path: str | Path = "configs/phase21_missed_case_autopsy.yaml",
    output_dir: str | Path = "outputs/phase21_missed_case_autopsy",
    *,
    progress: bool = False,
) -> dict[str, Path | pd.DataFrame]:
    config = load_phase21_config(config_path)
    phase5_config = load_phase5_config(config.phase5_config)
    plant = load_phase1_config(config.phase1_config)
    reference = _maybe_truncate_reference(
        build_reference_profile(load_phase2_config(config.phase2_config)),
        config.max_time_s,
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    missed_cases = select_missed_oracle_feasible_cases(config)
    if config.max_cases is not None:
        missed_cases = missed_cases.head(config.max_cases).copy()
    summary_rows: list[dict[str, Any]] = []
    rollout_frames: list[pd.DataFrame] = []
    for case in missed_cases.itertuples(index=False):
        tier_idx, tier = _tier_lookup(phase5_config, str(case.tier))
        scenario = sample_scenario(
            scenario_id=int(case.scenario_id),
            seed=phase5_config.seed + tier_idx * 10_000 + int(case.scenario_id),
            ranges=tier.uncertainty_ranges,
        )
        if progress:
            print(
                "phase21_replay "
                f"tier={case.tier} scenario={int(case.scenario_id):03d}",
                flush=True,
            )
        replay = rollout_oracle_command_replay(
            tier_name=str(case.tier),
            scenario=scenario,
            reference_profile=reference,
            plant_config=plant,
            oracle_trajectory_path=(
                config.phase20_output_dir
                / "oracle"
                / str(case.tier)
                / f"scenario_{int(case.scenario_id):03d}"
                / "oracle_trajectory.csv"
            ),
            thresholds=phase5_config.failure_thresholds,
        )
        replay_metrics = _summarize_replay(
            replay=replay,
            tier_name=str(case.tier),
            scenario=scenario,
            phase5_thresholds=phase5_config.failure_thresholds,
            recovery_thresholds=config.recovery_thresholds,
        )
        baseline_metrics = _baseline_metrics_for_case(config, case)
        transfer = build_transfer_row(
            baseline_metrics=baseline_metrics,
            replay_metrics=replay_metrics,
        )
        run_dir = output_path / str(case.tier) / f"scenario_{int(case.scenario_id):03d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        trajectory_path = run_dir / "oracle_replay_trajectory.csv"
        metrics_path = run_dir / "oracle_replay_metrics.json"
        replay.to_csv(trajectory_path, index=False)
        payload = {
            **replay_metrics,
            **transfer,
            "trajectory_csv": str(trajectory_path),
            "uncertainty_parameters": scenario.to_nested_dict(),
        }
        metrics_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        summary_rows.append(
            {
                "tier": str(case.tier),
                "scenario_id": int(case.scenario_id),
                **replay_metrics,
                **transfer,
                "trajectory_csv": str(trajectory_path),
                "metrics_json": str(metrics_path),
            }
        )
        rollout_frames.append(replay)
    summary = pd.DataFrame(summary_rows)
    rollouts = pd.concat(rollout_frames, ignore_index=True)
    comparison = summarize_phase21_transfer(summary)
    summary_path = output_path / "phase21_missed_case_summary.csv"
    rollouts_path = output_path / "phase21_oracle_replay_rollouts.csv"
    comparison_path = output_path / "phase21_transfer_comparison.csv"
    missed_path = output_path / "phase21_selected_missed_cases.csv"
    summary.to_csv(summary_path, index=False)
    rollouts.to_csv(rollouts_path, index=False)
    comparison.to_csv(comparison_path, index=False)
    missed_cases.to_csv(missed_path, index=False)
    figure_paths = write_phase21_figures(
        config=config,
        summary=summary,
        replay_rollouts=rollouts,
        missed_cases=missed_cases,
        output_dir=output_path,
    )
    return {
        "summary_csv": summary_path,
        "rollouts_csv": rollouts_path,
        "comparison_csv": comparison_path,
        "missed_cases_csv": missed_path,
        "summary": summary,
        "rollouts": rollouts,
        "comparison": comparison,
        **figure_paths,
    }


def select_missed_oracle_feasible_cases(config: Phase21Config) -> pd.DataFrame:
    diagnostics = pd.read_csv(
        config.phase20_output_dir / "phase20_failure_diagnostics.csv"
    )
    missed = diagnostics[
        diagnostics["controller"].eq(config.baseline_controller)
        & diagnostics["missed_oracle_feasible"].astype(bool)
    ].copy()
    return missed.sort_values(["tier", "scenario_id"]).reset_index(drop=True)


def rollout_oracle_command_replay(
    *,
    tier_name: str,
    scenario: Any,
    reference_profile: pd.DataFrame,
    plant_config: Any,
    oracle_trajectory_path: Path,
    thresholds: dict[str, float],
) -> pd.DataFrame:
    oracle = pd.read_csv(oracle_trajectory_path)
    aero = perturb_aero(plant_config.aero, scenario)
    first = reference_profile.iloc[0]
    state = np.array(
        [
            first["alpha_ref_rad"] + scenario.initial_error.alpha_rad,
            first["q_ref_radps"] + scenario.initial_error.q_radps,
            first["theta_ref_rad"] + scenario.initial_error.theta_rad,
        ],
        dtype=float,
    )
    dt = float(
        reference_profile["time_s"].iloc[1] - reference_profile["time_s"].iloc[0]
    )
    actuator = initialize_actuator(scenario, dt)
    rows: list[dict[str, Any]] = []
    for _idx, row in reference_profile.iterrows():
        raw_command = float(
            np.interp(
                float(row["time_s"]),
                oracle["time_s"].to_numpy(dtype=float),
                oracle["delta_flap_raw_rad"].to_numpy(dtype=float),
            )
        )
        applied, actuator_log = actuator_step(
            raw_command=raw_command,
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
            aero=aero,
            scenario=scenario,
        )
        rows.append(
            {
                "scenario_id": scenario.scenario_id,
                "tier": tier_name,
                "seed": scenario.seed,
                "controller": "oracle_command_replay",
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
                "alpha_error_rad": float(state[0] - row["alpha_ref_rad"]),
                "q_error_rad": float(state[1] - row["q_ref_radps"]),
                "theta_error_rad": float(state[2] - row["theta_ref_rad"]),
                "solver_status": "not_applicable",
                "solve_time_s": 0.0,
                "solver_failure": False,
                "learned_delta_flap_raw_rad": raw_command,
                "safety_delta_flap_raw_rad": raw_command,
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
            aero=aero,
            scenario=scenario,
            dt=dt,
        )
    return pd.DataFrame(rows)


def build_transfer_row(
    *, baseline_metrics: dict[str, Any], replay_metrics: dict[str, Any]
) -> dict[str, Any]:
    return {
        "baseline_failure_label": baseline_metrics["failure_label"],
        "baseline_max_alpha_miss_rad": float(
            baseline_metrics["max_alpha_corridor_miss_rad"]
        ),
        "baseline_first_alpha_violation_side": baseline_metrics[
            "first_alpha_violation_side"
        ],
        "baseline_first_alpha_violation_time_s": baseline_metrics[
            "first_alpha_violation_time_s"
        ],
        "replay_failure_label": replay_metrics["failure_label"],
        "replay_max_alpha_miss_rad": float(
            replay_metrics["max_alpha_corridor_miss_rad"]
        ),
        "replay_first_alpha_violation_side": replay_metrics[
            "first_alpha_violation_side"
        ],
        "replay_first_alpha_violation_time_s": replay_metrics[
            "first_alpha_violation_time_s"
        ],
        "replay_minus_baseline_alpha_miss_rad": float(
            replay_metrics["max_alpha_corridor_miss_rad"]
            - baseline_metrics["max_alpha_corridor_miss_rad"]
        ),
        "strict_success_recovered": bool(
            baseline_metrics["failure_label"] != "success"
            and replay_metrics["failure_label"] == "success"
        ),
    }


def summarize_phase21_transfer(summary: pd.DataFrame) -> pd.DataFrame:
    return (
        summary.groupby("tier", as_index=False)
        .agg(
            case_count=("scenario_id", "size"),
            recovered_strict_success_count=("strict_success_recovered", "sum"),
            recovered_strict_success_rate=("strict_success_recovered", "mean"),
            mean_replay_minus_baseline_alpha_miss_rad=(
                "replay_minus_baseline_alpha_miss_rad",
                "mean",
            ),
            max_replay_alpha_miss_rad=("replay_max_alpha_miss_rad", "max"),
        )
        .sort_values("tier")
    )


def write_phase21_figures(
    *,
    config: Phase21Config,
    summary: pd.DataFrame,
    replay_rollouts: pd.DataFrame,
    missed_cases: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    paths = {
        "transfer_png": output_dir / "oracle_replay_transfer_gap.png",
        "alpha_png": output_dir / "missed_case_alpha_replays.png",
        "command_png": output_dir / "missed_case_command_replays.png",
    }
    _plot_transfer(summary, paths["transfer_png"])
    _plot_case_replays(
        config=config,
        replay_rollouts=replay_rollouts,
        missed_cases=missed_cases,
        path=paths["alpha_png"],
        value="alpha",
    )
    _plot_case_replays(
        config=config,
        replay_rollouts=replay_rollouts,
        missed_cases=missed_cases,
        path=paths["command_png"],
        value="command",
    )
    return paths


def _summarize_replay(
    *,
    replay: pd.DataFrame,
    tier_name: str,
    scenario: Any,
    phase5_thresholds: dict[str, float],
    recovery_thresholds: ControlledRecoveryThresholds,
) -> dict[str, Any]:
    metrics = summarize_monte_carlo_rollout(
        rollout=replay,
        tier_name=tier_name,
        controller_name="oracle_command_replay",
        scenario=scenario,
        thresholds=phase5_thresholds,
    )
    diagnostics = summarize_corridor_diagnostics(
        rollout=replay,
        tolerance=phase5_thresholds["corridor_tolerance_rad"],
        control_dt_s=float(replay["time_s"].iloc[1] - replay["time_s"].iloc[0]),
    )
    recovery = summarize_controlled_recovery(
        rollout=replay, thresholds=recovery_thresholds
    )
    return {**metrics, **diagnostics, **recovery}


def _baseline_metrics_for_case(config: Phase21Config, case: Any) -> dict[str, Any]:
    diagnostics = pd.read_csv(
        config.phase20_output_dir / "phase20_failure_diagnostics.csv"
    )
    row = diagnostics[
        diagnostics["controller"].eq(config.baseline_controller)
        & diagnostics["tier"].eq(str(case.tier))
        & diagnostics["scenario_id"].eq(int(case.scenario_id))
    ].iloc[0]
    return row.to_dict()


def _tier_lookup(phase5_config: Any, tier_name: str) -> tuple[int, Any]:
    for idx, tier in enumerate(phase5_config.tiers):
        if tier.name == tier_name:
            return idx, tier
    raise ValueError(f"Unknown tier: {tier_name}")


def _plot_transfer(summary: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    colors = np.where(summary["strict_success_recovered"], "#2ca02c", "#d62728")
    ax.bar(
        summary["tier"] + " " + summary["scenario_id"].astype(str),
        summary["replay_minus_baseline_alpha_miss_rad"],
        color=colors,
    )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylabel("oracle replay minus baseline alpha miss [rad]")
    ax.set_xlabel("missed oracle-feasible case")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_case_replays(
    *,
    config: Phase21Config,
    replay_rollouts: pd.DataFrame,
    missed_cases: pd.DataFrame,
    path: Path,
    value: str,
) -> None:
    max_cases = int(config.plot_settings.get("max_case_plots", 4))
    cases = missed_cases.head(max_cases)
    fig, axes = plt.subplots(len(cases), 1, figsize=(9.0, 2.8 * len(cases)))
    if len(cases) == 1:
        axes = [axes]
    for ax, case in zip(axes, cases.itertuples(index=False), strict=False):
        baseline = pd.read_csv(
            config.phase20_output_dir
            / config.baseline_controller
            / str(case.tier)
            / f"scenario_{int(case.scenario_id):03d}"
            / "trajectory.csv"
        )
        replay = replay_rollouts[
            replay_rollouts["tier"].eq(str(case.tier))
            & replay_rollouts["scenario_id"].eq(int(case.scenario_id))
        ]
        if value == "alpha":
            ax.fill_between(
                baseline["time_s"],
                baseline["alpha_min_rad"],
                baseline["alpha_max_rad"],
                color="gray",
                alpha=0.14,
                label="alpha corridor",
            )
            ax.plot(baseline["time_s"], baseline["alpha_rad"], label="ridge baseline")
            ax.plot(replay["time_s"], replay["alpha_rad"], label="oracle replay")
            ax.set_ylabel("alpha [rad]")
        else:
            ax.plot(
                baseline["time_s"],
                baseline["safety_delta_flap_raw_rad"],
                label="ridge raw command",
            )
            ax.plot(
                replay["time_s"],
                replay["safety_delta_flap_raw_rad"],
                label="oracle raw command replay",
            )
            ax.set_ylabel("raw flap [rad]")
        ax.set_title(f"{case.tier} scenario {int(case.scenario_id)}")
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("time [s]")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
