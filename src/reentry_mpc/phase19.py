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
    UncertaintyScenario,
    actuator_step,
    initialize_actuator,
    noisy_measurement,
    perturb_aero,
    sample_scenario,
    uncertain_rk4_step,
)


@dataclass(frozen=True)
class OraclePolicyConfig:
    ridge_lambda: float
    use_only_feasible_or_near_feasible: bool
    feature_columns: list[str]
    safety_blend_gain: float
    safety_margin_rad: float
    command_clip_rad: float


@dataclass(frozen=True)
class Phase19Config:
    seed: int
    phase5_config: Path
    phase1_config: Path
    phase2_config: Path
    oracle_summary: Path
    oracle_trajectories: Path
    scenario_count_per_tier: int
    max_time_s: float | None
    policy: OraclePolicyConfig
    recovery_thresholds: ControlledRecoveryThresholds
    plot_settings: dict[str, Any]


@dataclass(frozen=True)
class FittedOraclePolicy:
    feature_columns: list[str]
    coefficients: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    target_mean: float


def load_phase19_config(path: str | Path) -> Phase19Config:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)
    thresholds = raw["failure_thresholds"]
    policy = raw["policy"]
    max_time = raw.get("max_time_s")
    return Phase19Config(
        seed=int(raw["seed"]),
        phase5_config=Path(raw["phase5_config"]),
        phase1_config=Path(raw["phase1_config"]),
        phase2_config=Path(raw["phase2_config"]),
        oracle_summary=Path(raw["oracle_summary"]),
        oracle_trajectories=Path(raw["oracle_trajectories"]),
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


def run_phase19_oracle_imitation(
    config_path: str | Path = "configs/phase19_oracle_imitation.yaml",
    output_dir: str | Path = "outputs/phase19_oracle_imitation",
    *,
    progress: bool = False,
) -> dict[str, Path | pd.DataFrame]:
    config = load_phase19_config(config_path)
    phase5_config = load_phase5_config(config.phase5_config)
    plant = load_phase1_config(config.phase1_config)
    reference = _maybe_truncate_reference(
        build_reference_profile(load_phase2_config(config.phase2_config)),
        config.max_time_s,
    )
    oracle_summary = pd.read_csv(config.oracle_summary)
    oracle_trajectories = pd.read_csv(config.oracle_trajectories)
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
            if progress:
                print(
                    "phase19_rollout " f"tier={tier.name} scenario={scenario_id:03d}",
                    flush=True,
                )
            rollout = rollout_oracle_imitation_policy(
                tier_name=tier.name,
                scenario=scenario,
                reference_profile=reference,
                plant_config=plant,
                policy=policy,
                policy_config=config.policy,
                thresholds=phase5_config.failure_thresholds,
            )
            metrics = summarize_monte_carlo_rollout(
                rollout=rollout,
                tier_name=tier.name,
                controller_name="oracle_imitation_policy",
                scenario=scenario,
                thresholds=phase5_config.failure_thresholds,
            )
            diagnostics = summarize_corridor_diagnostics(
                rollout=rollout,
                tolerance=phase5_config.failure_thresholds["corridor_tolerance_rad"],
                control_dt_s=_reference_dt(reference),
            )
            recovery = summarize_controlled_recovery(
                rollout=rollout, thresholds=config.recovery_thresholds
            )
            oracle_row = oracle_summary[
                oracle_summary["tier"].eq(tier.name)
                & oracle_summary["scenario_id"].eq(scenario_id)
            ]
            oracle_metrics = (
                {}
                if oracle_row.empty
                else {
                    "oracle_feasible": bool(oracle_row["oracle_feasible"].iloc[0]),
                    "oracle_near_feasible": bool(
                        oracle_row["oracle_near_feasible"].iloc[0]
                    ),
                    "oracle_max_alpha_miss_rad": float(
                        oracle_row["oracle_max_alpha_miss_rad"].iloc[0]
                    ),
                }
            )
            run_dir = output_path / tier.name / f"scenario_{scenario_id:03d}"
            run_dir.mkdir(parents=True, exist_ok=True)
            trajectory_path = run_dir / "trajectory.csv"
            metrics_path = run_dir / "metrics.json"
            rollout.to_csv(trajectory_path, index=False)
            payload = {
                **metrics,
                **diagnostics,
                **recovery,
                **oracle_metrics,
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
                    "trajectory_csv": str(trajectory_path),
                    "metrics_json": str(metrics_path),
                }
            )
            rollout_frames.append(rollout)
    summary = pd.DataFrame(summary_rows)
    rollouts = pd.concat(rollout_frames, ignore_index=True)
    comparison = summarize_oracle_imitation(summary)
    summary_path = output_path / "oracle_imitation_summary.csv"
    rollouts_path = output_path / "oracle_imitation_rollouts.csv"
    comparison_path = output_path / "oracle_imitation_comparison.csv"
    policy_path = output_path / "oracle_policy.json"
    training_path = output_path / "oracle_policy_training_data.csv"
    summary.to_csv(summary_path, index=False)
    rollouts.to_csv(rollouts_path, index=False)
    comparison.to_csv(comparison_path, index=False)
    training_frame.to_csv(training_path, index=False)
    policy_path.write_text(
        json.dumps(_policy_to_json(policy), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    figure_paths = write_phase19_figures(
        summary=summary,
        rollouts=rollouts,
        output_dir=output_path,
    )
    return {
        "summary_csv": summary_path,
        "rollouts_csv": rollouts_path,
        "comparison_csv": comparison_path,
        "policy_json": policy_path,
        "training_csv": training_path,
        "summary": summary,
        "rollouts": rollouts,
        "comparison": comparison,
        **figure_paths,
    }


def fit_oracle_policy(
    *,
    oracle_summary: pd.DataFrame,
    oracle_trajectories: pd.DataFrame,
    config: OraclePolicyConfig,
) -> tuple[FittedOraclePolicy, pd.DataFrame]:
    training = build_policy_training_frame(
        oracle_summary=oracle_summary,
        oracle_trajectories=oracle_trajectories,
        use_only_feasible_or_near_feasible=config.use_only_feasible_or_near_feasible,
    )
    features = training[config.feature_columns].to_numpy(dtype=float)
    target = training["delta_flap_raw_rad"].to_numpy(dtype=float)
    feature_mean = features.mean(axis=0)
    feature_scale = features.std(axis=0)
    feature_scale = np.where(feature_scale < 1.0e-8, 1.0, feature_scale)
    normalized = (features - feature_mean) / feature_scale
    centered_target = target - float(target.mean())
    design = np.column_stack([np.ones(len(normalized)), normalized])
    penalty = config.ridge_lambda * np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        design.T @ design + penalty, design.T @ centered_target
    )
    policy = FittedOraclePolicy(
        feature_columns=config.feature_columns,
        coefficients=coefficients,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        target_mean=float(target.mean()),
    )
    return policy, training


def build_policy_training_frame(
    *,
    oracle_summary: pd.DataFrame,
    oracle_trajectories: pd.DataFrame,
    use_only_feasible_or_near_feasible: bool,
) -> pd.DataFrame:
    summary = oracle_summary[["tier", "scenario_id", "oracle_near_feasible"]]
    frame = oracle_trajectories.merge(summary, on=["tier", "scenario_id"], how="left")
    if use_only_feasible_or_near_feasible:
        frame = frame[frame["oracle_near_feasible"].fillna(False).astype(bool)].copy()
    return add_policy_features(frame)


def rollout_oracle_imitation_policy(
    *,
    tier_name: str,
    scenario: UncertaintyScenario,
    reference_profile: pd.DataFrame,
    plant_config: Any,
    policy: FittedOraclePolicy,
    policy_config: OraclePolicyConfig,
    thresholds: dict[str, float],
    controller_name: str = "oracle_imitation_policy",
) -> pd.DataFrame:
    rng = np.random.default_rng(scenario.seed + 19_000)
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
    dt = _reference_dt(reference_profile)
    actuator = initialize_actuator(scenario, dt)
    rows: list[dict[str, Any]] = []
    actuator_tau = scenario.actuator_lag_s + 0.5 * scenario.actuator_delay_s
    prev_row: pd.Series | None = None
    for _idx, row in reference_profile.iterrows():
        measured = noisy_measurement(state, scenario, rng)
        feature_row = _feature_dict_from_state(
            state=measured,
            row=row,
            actuator_tau_s=actuator_tau,
            prev_row=prev_row,
            dt=dt,
        )
        prev_row = row
        learned_command = predict_oracle_policy(policy, feature_row)
        safety_command = _safety_blend_command(
            learned_command=learned_command,
            state=measured,
            row=row,
            config=policy_config,
        )
        raw_command = float(
            np.clip(
                safety_command,
                -policy_config.command_clip_rad,
                policy_config.command_clip_rad,
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
                "controller": controller_name,
                "time_s": float(row["time_s"]),
                "alpha_rad": float(state[0]),
                "q_radps": float(state[1]),
                "theta_rad": float(state[2]),
                "measured_alpha_rad": float(measured[0]),
                "measured_q_radps": float(measured[1]),
                "measured_theta_rad": float(measured[2]),
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
                "learned_delta_flap_raw_rad": float(learned_command),
                "safety_delta_flap_raw_rad": float(raw_command),
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


def add_policy_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = (
        frame.copy()
        .sort_values(["tier", "scenario_id", "time_s"])
        .reset_index(drop=True)
    )
    center = 0.5 * (result["alpha_min_rad"] + result["alpha_max_rad"])
    result["alpha_error_to_center_rad"] = result["alpha_rad"] - center
    result["alpha_margin_low_rad"] = result["alpha_rad"] - result["alpha_min_rad"]
    result["alpha_margin_high_rad"] = result["alpha_max_rad"] - result["alpha_rad"]
    result["dynamic_pressure_scaled"] = (
        result.get("dynamic_pressure_pa", 0.0) / 10_000.0
    )
    result["mach_scaled"] = result.get("mach", 0.0) / 20.0
    if "actuator_tau_prediction_s" in result:
        result["actuator_tau_s"] = result["actuator_tau_prediction_s"]
    else:
        result["actuator_tau_s"] = 0.5
    # Corridor rates: finite-difference within each scenario trajectory.
    # First step of each scenario gets zero rate (no look-back across scenarios).
    dt_col = result["time_s"].diff()
    same_scenario = result["tier"].eq(result["tier"].shift()) & result[
        "scenario_id"
    ].eq(result["scenario_id"].shift())
    alpha_min_diff = result["alpha_min_rad"].diff()
    alpha_max_diff = result["alpha_max_rad"].diff()
    result["alpha_min_rate_radps"] = np.where(
        same_scenario & (dt_col > 0), alpha_min_diff / dt_col, 0.0
    )
    result["alpha_max_rate_radps"] = np.where(
        same_scenario & (dt_col > 0), alpha_max_diff / dt_col, 0.0
    )
    return result


def predict_oracle_policy(
    policy: FittedOraclePolicy, features: dict[str, float]
) -> float:
    row = np.array([features[column] for column in policy.feature_columns], dtype=float)
    normalized = (row - policy.feature_mean) / policy.feature_scale
    design = np.concatenate([[1.0], normalized])
    return float(policy.target_mean + design @ policy.coefficients)


def summarize_oracle_imitation(summary: pd.DataFrame) -> pd.DataFrame:
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
            oracle_feasible_rate=("oracle_feasible", "mean"),
            mean_rms_alpha_error_rad=("rms_alpha_error_rad", "mean"),
            median_alpha_miss_rad=("max_alpha_corridor_miss_rad", "median"),
        )
    )


def write_phase19_figures(
    *, summary: pd.DataFrame, rollouts: pd.DataFrame, output_dir: Path
) -> dict[str, Path]:
    paths = {
        "success_png": output_dir / "oracle_imitation_success.png",
        "miss_png": output_dir / "oracle_imitation_alpha_miss.png",
        "envelope_png": output_dir / "oracle_imitation_alpha_envelopes.png",
    }
    _plot_success(summary, paths["success_png"])
    _plot_miss(summary, paths["miss_png"])
    _plot_envelopes(rollouts, paths["envelope_png"])
    return paths


def _feature_dict_from_state(
    *,
    state: np.ndarray,
    row: pd.Series,
    actuator_tau_s: float,
    prev_row: pd.Series | None = None,
    dt: float = 1.0,
) -> dict[str, float]:
    center = 0.5 * (float(row["alpha_min_rad"]) + float(row["alpha_max_rad"]))
    if prev_row is not None and dt > 0:
        alpha_min_rate = (
            float(row["alpha_min_rad"]) - float(prev_row["alpha_min_rad"])
        ) / dt
        alpha_max_rate = (
            float(row["alpha_max_rad"]) - float(prev_row["alpha_max_rad"])
        ) / dt
    else:
        alpha_min_rate = 0.0
        alpha_max_rate = 0.0
    return {
        "alpha_error_to_center_rad": float(state[0] - center),
        "q_radps": float(state[1]),
        "alpha_margin_low_rad": float(state[0] - row["alpha_min_rad"]),
        "alpha_margin_high_rad": float(row["alpha_max_rad"] - state[0]),
        "alpha_min_rate_radps": alpha_min_rate,
        "alpha_max_rate_radps": alpha_max_rate,
        "mach_scaled": float(row["mach"]) / 20.0,
        "dynamic_pressure_scaled": float(row["dynamic_pressure_pa"]) / 10_000.0,
        "actuator_tau_s": float(actuator_tau_s),
    }


def _safety_blend_command(
    *,
    learned_command: float,
    state: np.ndarray,
    row: pd.Series,
    config: OraclePolicyConfig,
) -> float:
    high_margin = float(row["alpha_max_rad"] - state[0])
    low_margin = float(state[0] - row["alpha_min_rad"])
    command = float(learned_command)
    if high_margin < config.safety_margin_rad:
        command += config.safety_blend_gain * (config.safety_margin_rad - high_margin)
    if low_margin < config.safety_margin_rad:
        command -= config.safety_blend_gain * (config.safety_margin_rad - low_margin)
    return command


def _reference_dt(reference_profile: pd.DataFrame) -> float:
    return float(
        reference_profile["time_s"].iloc[1] - reference_profile["time_s"].iloc[0]
    )


def _policy_to_json(policy: FittedOraclePolicy) -> dict[str, Any]:
    return {
        "feature_columns": policy.feature_columns,
        "coefficients": policy.coefficients.tolist(),
        "feature_mean": policy.feature_mean.tolist(),
        "feature_scale": policy.feature_scale.tolist(),
        "target_mean": policy.target_mean,
    }


def _plot_success(summary: pd.DataFrame, path: Path) -> None:
    grouped = (
        summary.assign(strict=summary["failure_label"].eq("success"))
        .groupby("tier")
        .agg(
            strict=("strict", "mean"),
            controlled=("controlled_recovery", "mean"),
            oracle=("oracle_feasible", "mean"),
        )
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = np.arange(len(grouped))
    ax.bar(x - 0.24, grouped["strict"], width=0.24, label="online strict")
    ax.bar(x, grouped["controlled"], width=0.24, label="online controlled")
    ax.bar(x + 0.24, grouped["oracle"], width=0.24, label="oracle feasible")
    ax.set_xticks(x, grouped["tier"])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("rate")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_miss(summary: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for tier, frame in summary.groupby("tier"):
        ax.scatter(
            frame["oracle_max_alpha_miss_rad"],
            frame["max_alpha_corridor_miss_rad"],
            label=tier,
            alpha=0.75,
        )
    ax.set_xlabel("oracle max alpha miss [rad]")
    ax.set_ylabel("online max alpha miss [rad]")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_envelopes(rollouts: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5), sharey=True)
    for ax, (tier, frame) in zip(axes, rollouts.groupby("tier"), strict=False):
        pivot = frame.pivot_table(
            index="time_s", columns="scenario_id", values="alpha_error_rad"
        )
        median = pivot.median(axis=1)
        low = pivot.quantile(0.05, axis=1)
        high = pivot.quantile(0.95, axis=1)
        ax.plot(median.index, median, label="oracle imitation")
        ax.fill_between(median.index, low, high, alpha=0.16)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title(tier)
        ax.set_xlabel("time [s]")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("alpha error [rad]")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
