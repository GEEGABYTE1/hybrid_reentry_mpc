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
from reentry_mpc.nmpc import NmpcSolverOptions
from reentry_mpc.phase2 import build_reference_profile, load_phase2_config
from reentry_mpc.phase5 import load_phase5_config, summarize_monte_carlo_rollout
from reentry_mpc.phase16 import summarize_corridor_diagnostics
from reentry_mpc.phase17 import (
    ControlledRecoveryThresholds,
    _maybe_truncate_reference,
    summarize_controlled_recovery,
)
from reentry_mpc.phase18 import (
    OracleClassification,
    SlackOracleSettings,
    _downsample_for_oracle,
    solve_slack_oracle,
    summarize_feasibility_ceiling,
)
from reentry_mpc.phase19 import (
    OraclePolicyConfig,
    fit_oracle_policy,
    rollout_oracle_imitation_policy,
)
from reentry_mpc.uncertainty import sample_scenario


@dataclass(frozen=True)
class PolicyVariant:
    name: str
    use_only_feasible_or_near_feasible: bool
    train_on_strict_feasible_only: bool
    safety_blend_gain: float
    safety_margin_rad: float
    command_clip_rad: float


@dataclass(frozen=True)
class Phase20Config:
    seed: int
    phase5_config: Path
    phase1_config: Path
    phase2_config: Path
    scenario_count_per_tier: int
    max_time_s: float | None
    oracle: SlackOracleSettings
    solver: NmpcSolverOptions
    classification: OracleClassification
    ridge_lambda: float
    feature_columns: list[str]
    variants: list[PolicyVariant]
    recovery_thresholds: ControlledRecoveryThresholds
    plot_settings: dict[str, Any]


def load_phase20_config(path: str | Path) -> Phase20Config:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)
    oracle = raw["oracle"]
    solver = raw["solver"]
    classification = raw["classification"]
    policy = raw["policy"]
    thresholds = raw["failure_thresholds"]
    max_time = raw.get("max_time_s")
    return Phase20Config(
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
            max_iter=int(solver["max_iter"]),
            acceptable_tol=float(solver["acceptable_tol"]),
            print_level=int(solver["print_level"]),
        ),
        classification=OracleClassification(
            feasible_alpha_miss_rad=float(classification["feasible_alpha_miss_rad"]),
            near_feasible_alpha_miss_rad=float(
                classification["near_feasible_alpha_miss_rad"]
            ),
        ),
        ridge_lambda=float(policy["ridge_lambda"]),
        feature_columns=[str(value) for value in policy["feature_columns"]],
        variants=[
            PolicyVariant(
                name=str(variant["name"]),
                use_only_feasible_or_near_feasible=bool(
                    variant["use_only_feasible_or_near_feasible"]
                ),
                train_on_strict_feasible_only=bool(
                    variant.get("train_on_strict_feasible_only", False)
                ),
                safety_blend_gain=float(variant["safety_blend_gain"]),
                safety_margin_rad=float(variant["safety_margin_rad"]),
                command_clip_rad=float(variant["command_clip_rad"]),
            )
            for variant in policy["variants"]
        ],
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


def run_phase20_full_oracle_imitation(
    config_path: str | Path = "configs/phase20_full_oracle_imitation.yaml",
    output_dir: str | Path = "outputs/phase20_full_oracle_imitation",
    *,
    progress: bool = False,
) -> dict[str, Path | pd.DataFrame]:
    config = load_phase20_config(config_path)
    phase5_config = load_phase5_config(config.phase5_config)
    plant = load_phase1_config(config.phase1_config)
    reference = _maybe_truncate_reference(
        build_reference_profile(load_phase2_config(config.phase2_config)),
        config.max_time_s,
    )
    oracle_reference = _downsample_for_oracle(reference, config.oracle.dt_s)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    oracle_summary, oracle_trajectories = _run_oracle_set(
        config=config,
        phase5_config=phase5_config,
        plant=plant,
        oracle_reference=oracle_reference,
        output_path=output_path,
        progress=progress,
    )
    ceiling = summarize_feasibility_ceiling(oracle_summary)
    online_summary, online_rollouts = _run_policy_variants(
        config=config,
        phase5_config=phase5_config,
        plant=plant,
        reference=reference,
        oracle_summary=oracle_summary,
        oracle_trajectories=oracle_trajectories,
        output_path=output_path,
        progress=progress,
    )
    comparison = summarize_phase20_comparison(online_summary)
    failure_diagnostics = build_failure_diagnostics(online_summary)
    ceiling_gap = build_ceiling_gap(comparison, ceiling)
    paths = _write_phase20_tables(
        output_path=output_path,
        oracle_summary=oracle_summary,
        oracle_trajectories=oracle_trajectories,
        ceiling=ceiling,
        online_summary=online_summary,
        online_rollouts=online_rollouts,
        comparison=comparison,
        failure_diagnostics=failure_diagnostics,
        ceiling_gap=ceiling_gap,
    )
    figure_paths = write_phase20_figures(
        comparison=comparison,
        ceiling_gap=ceiling_gap,
        diagnostics=failure_diagnostics,
        rollouts=online_rollouts,
        output_dir=output_path,
    )
    return {
        **paths,
        **figure_paths,
        "oracle_summary": oracle_summary,
        "oracle_trajectories": oracle_trajectories,
        "summary": online_summary,
        "rollouts": online_rollouts,
        "comparison": comparison,
        "failure_diagnostics": failure_diagnostics,
        "ceiling_gap": ceiling_gap,
    }


def summarize_phase20_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    return (
        summary.assign(
            strict_success=summary["failure_label"].eq("success"),
            controlled=summary["controlled_recovery"].astype(bool),
            missed_oracle_feasible=lambda frame: frame["oracle_feasible"].astype(bool)
            & ~frame["failure_label"].eq("success"),
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
            oracle_near_feasible_count=("oracle_near_feasible", "sum"),
            oracle_near_feasible_rate=("oracle_near_feasible", "mean"),
            missed_oracle_feasible_count=("missed_oracle_feasible", "sum"),
            mean_online_minus_oracle_alpha_miss_rad=(
                "online_minus_oracle_alpha_miss_rad",
                "mean",
            ),
            median_alpha_miss_rad=("max_alpha_corridor_miss_rad", "median"),
            mean_raw_applied_flap_gap_rad=("raw_applied_flap_gap_mean_rad", "mean"),
        )
        .sort_values(["tier", "controller"])
    )


def build_ceiling_gap(comparison: pd.DataFrame, ceiling: pd.DataFrame) -> pd.DataFrame:
    ceiling_cols = ceiling.rename(
        columns={
            "feasible_rate": "oracle_strict_feasible_rate",
            "near_feasible_rate": "oracle_near_feasible_rate",
            "feasible_count": "oracle_strict_feasible_count",
            "near_feasible_count": "oracle_near_feasible_count",
        }
    )[
        [
            "tier",
            "oracle_strict_feasible_count",
            "oracle_strict_feasible_rate",
            "oracle_near_feasible_count",
            "oracle_near_feasible_rate",
        ]
    ]
    comparison_base = comparison.drop(
        columns=["oracle_near_feasible_count", "oracle_near_feasible_rate"],
        errors="ignore",
    )
    result = comparison_base.merge(ceiling_cols, on="tier", how="left")
    result["strict_success_gap_vs_oracle_rate"] = (
        result["strict_success_rate"] - result["oracle_strict_feasible_rate"]
    )
    result["missed_feasible_scenarios_count"] = result["missed_oracle_feasible_count"]
    result["online_success_beyond_oracle_count"] = (
        result["strict_success_count"]
        - (
            result["oracle_strict_feasible_count"]
            - result["missed_oracle_feasible_count"]
        )
    ).clip(lower=0)
    return result


def build_failure_diagnostics(summary: pd.DataFrame) -> pd.DataFrame:
    result = summary.copy()
    result["strict_success"] = result["failure_label"].eq("success")
    result["online_minus_oracle_alpha_miss_rad"] = (
        result["max_alpha_corridor_miss_rad"] - result["oracle_max_alpha_miss_rad"]
    )
    result["oracle_feasibility_class"] = np.select(
        [
            result["oracle_feasible"].astype(bool),
            result["oracle_near_feasible"].astype(bool),
        ],
        ["strict_feasible", "near_feasible"],
        default="oracle_infeasible",
    )
    result["missed_oracle_feasible"] = (
        result["oracle_feasible"].astype(bool) & ~result["strict_success"]
    )
    return result


def write_phase20_figures(
    *,
    comparison: pd.DataFrame,
    ceiling_gap: pd.DataFrame,
    diagnostics: pd.DataFrame,
    rollouts: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    paths = {
        "success_png": output_dir / "full_oracle_vs_imitation_success.png",
        "gap_png": output_dir / "oracle_gap_by_scenario.png",
        "first_violation_png": output_dir / "first_violation_time_by_feasibility.png",
        "flap_lag_png": output_dir / "learned_vs_applied_flap_lag.png",
        "envelope_png": output_dir / "alpha_envelopes_full_oracle_imitation.png",
    }
    _plot_success(comparison, ceiling_gap, paths["success_png"])
    _plot_oracle_gap(diagnostics, paths["gap_png"])
    _plot_first_violation(diagnostics, paths["first_violation_png"])
    _plot_flap_lag(diagnostics, paths["flap_lag_png"])
    _plot_envelopes(rollouts, paths["envelope_png"])
    return paths


def _run_oracle_set(
    *,
    config: Phase20Config,
    phase5_config: Any,
    plant: Any,
    oracle_reference: pd.DataFrame,
    output_path: Path,
    progress: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
                    f"phase20_oracle tier={tier.name} scenario={scenario_id:03d}",
                    flush=True,
                )
            trajectory, metrics = solve_slack_oracle(
                tier_name=tier.name,
                scenario=scenario,
                reference=oracle_reference,
                vehicle=plant.vehicle,
                aero=plant.aero,
                settings=config.oracle,
                solver=config.solver,
                classification=config.classification,
                tolerance=phase5_config.failure_thresholds["corridor_tolerance_rad"],
            )
            run_dir = output_path / "oracle" / tier.name / f"scenario_{scenario_id:03d}"
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
                    "oracle_trajectory_csv": str(trajectory_path),
                    "oracle_metrics_json": str(metrics_path),
                }
            )
            trajectory_frames.append(trajectory)
    return pd.DataFrame(summary_rows), pd.concat(trajectory_frames, ignore_index=True)


def _run_policy_variants(
    *,
    config: Phase20Config,
    phase5_config: Any,
    plant: Any,
    reference: pd.DataFrame,
    oracle_summary: pd.DataFrame,
    oracle_trajectories: pd.DataFrame,
    output_path: Path,
    progress: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    rollout_frames: list[pd.DataFrame] = []
    for variant in config.variants:
        training_summary = _filter_training_summary(oracle_summary, variant)
        policy_config = OraclePolicyConfig(
            ridge_lambda=config.ridge_lambda,
            use_only_feasible_or_near_feasible=variant.use_only_feasible_or_near_feasible,
            feature_columns=config.feature_columns,
            safety_blend_gain=variant.safety_blend_gain,
            safety_margin_rad=variant.safety_margin_rad,
            command_clip_rad=variant.command_clip_rad,
        )
        policy, training_frame = fit_oracle_policy(
            oracle_summary=training_summary,
            oracle_trajectories=oracle_trajectories,
            config=policy_config,
        )
        variant_dir = output_path / variant.name
        variant_dir.mkdir(parents=True, exist_ok=True)
        training_frame.to_csv(
            variant_dir / "oracle_policy_training_data.csv", index=False
        )
        (variant_dir / "oracle_policy.json").write_text(
            json.dumps(
                {
                    "feature_columns": policy.feature_columns,
                    "coefficients": policy.coefficients.tolist(),
                    "feature_mean": policy.feature_mean.tolist(),
                    "feature_scale": policy.feature_scale.tolist(),
                    "target_mean": policy.target_mean,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
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
                        "phase20_policy "
                        f"variant={variant.name} "
                        f"tier={tier.name} scenario={scenario_id:03d}",
                        flush=True,
                    )
                rollout = rollout_oracle_imitation_policy(
                    tier_name=tier.name,
                    scenario=scenario,
                    reference_profile=reference,
                    plant_config=plant,
                    policy=policy,
                    policy_config=policy_config,
                    thresholds=phase5_config.failure_thresholds,
                    controller_name=variant.name,
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
                    control_dt_s=_reference_dt(reference),
                )
                recovery = summarize_controlled_recovery(
                    rollout=rollout, thresholds=config.recovery_thresholds
                )
                oracle_metrics = _oracle_metrics_for(
                    oracle_summary, tier.name, scenario_id
                )
                flap_gap = _flap_gap_metrics(rollout)
                run_dir = variant_dir / tier.name / f"scenario_{scenario_id:03d}"
                run_dir.mkdir(parents=True, exist_ok=True)
                trajectory_path = run_dir / "trajectory.csv"
                metrics_path = run_dir / "metrics.json"
                rollout.to_csv(trajectory_path, index=False)
                row = {
                    "tier": tier.name,
                    **scenario.to_flat_dict(),
                    **metrics,
                    **diagnostics,
                    **recovery,
                    **oracle_metrics,
                    **flap_gap,
                    "online_minus_oracle_alpha_miss_rad": float(
                        diagnostics["max_alpha_corridor_miss_rad"]
                        - oracle_metrics["oracle_max_alpha_miss_rad"]
                    ),
                    "trajectory_csv": str(trajectory_path),
                    "metrics_json": str(metrics_path),
                }
                metrics_path.write_text(
                    json.dumps(
                        {
                            **row,
                            "uncertainty_parameters": scenario.to_nested_dict(),
                        },
                        indent=2,
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )
                summary_rows.append(row)
                rollout_frames.append(rollout)
    return pd.DataFrame(summary_rows), pd.concat(rollout_frames, ignore_index=True)


def _filter_training_summary(
    oracle_summary: pd.DataFrame, variant: PolicyVariant
) -> pd.DataFrame:
    summary = oracle_summary.copy()
    if variant.train_on_strict_feasible_only:
        # Collapse oracle_near_feasible to oracle_feasible so that
        # use_only_feasible_or_near_feasible (required=True for this variant)
        # selects only strict-feasible trajectories via build_policy_training_frame.
        summary["oracle_near_feasible"] = summary["oracle_feasible"].astype(bool)
    return summary


def _oracle_metrics_for(
    oracle_summary: pd.DataFrame, tier_name: str, scenario_id: int
) -> dict[str, Any]:
    row = oracle_summary[
        oracle_summary["tier"].eq(tier_name)
        & oracle_summary["scenario_id"].eq(scenario_id)
    ].iloc[0]
    return {
        "oracle_feasible": bool(row["oracle_feasible"]),
        "oracle_near_feasible": bool(row["oracle_near_feasible"]),
        "oracle_max_alpha_miss_rad": float(row["oracle_max_alpha_miss_rad"]),
        "oracle_max_q_miss_radps": float(row["oracle_max_q_miss_radps"]),
    }


def _flap_gap_metrics(rollout: pd.DataFrame) -> dict[str, float]:
    gap = rollout["learned_delta_flap_raw_rad"].to_numpy(dtype=float) - rollout[
        "delta_flap_rad"
    ].to_numpy(dtype=float)
    safety_gap = rollout["safety_delta_flap_raw_rad"].to_numpy(dtype=float) - rollout[
        "delta_flap_rad"
    ].to_numpy(dtype=float)
    return {
        "raw_applied_flap_gap_mean_rad": float(np.mean(np.abs(gap))),
        "raw_applied_flap_gap_max_rad": float(np.max(np.abs(gap))),
        "safety_applied_flap_gap_mean_rad": float(np.mean(np.abs(safety_gap))),
        "safety_applied_flap_gap_max_rad": float(np.max(np.abs(safety_gap))),
    }


def _write_phase20_tables(
    *,
    output_path: Path,
    oracle_summary: pd.DataFrame,
    oracle_trajectories: pd.DataFrame,
    ceiling: pd.DataFrame,
    online_summary: pd.DataFrame,
    online_rollouts: pd.DataFrame,
    comparison: pd.DataFrame,
    failure_diagnostics: pd.DataFrame,
    ceiling_gap: pd.DataFrame,
) -> dict[str, Path]:
    paths = {
        "oracle_summary_csv": output_path / "phase20_oracle_summary.csv",
        "oracle_trajectories_csv": output_path / "phase20_oracle_trajectories.csv",
        "oracle_ceiling_csv": output_path / "phase20_oracle_ceiling.csv",
        "summary_csv": output_path / "phase20_summary.csv",
        "rollouts_csv": output_path / "phase20_rollouts.csv",
        "comparison_csv": output_path / "phase20_comparison.csv",
        "failure_diagnostics_csv": output_path / "phase20_failure_diagnostics.csv",
        "ceiling_gap_csv": output_path / "phase20_ceiling_gap.csv",
    }
    oracle_summary.to_csv(paths["oracle_summary_csv"], index=False)
    oracle_trajectories.to_csv(paths["oracle_trajectories_csv"], index=False)
    ceiling.to_csv(paths["oracle_ceiling_csv"], index=False)
    online_summary.to_csv(paths["summary_csv"], index=False)
    online_rollouts.to_csv(paths["rollouts_csv"], index=False)
    comparison.to_csv(paths["comparison_csv"], index=False)
    failure_diagnostics.to_csv(paths["failure_diagnostics_csv"], index=False)
    ceiling_gap.to_csv(paths["ceiling_gap_csv"], index=False)
    return paths


def _reference_dt(reference: pd.DataFrame) -> float:
    return float(reference["time_s"].iloc[1] - reference["time_s"].iloc[0])


def _plot_success(
    comparison: pd.DataFrame, ceiling_gap: pd.DataFrame, path: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4), sharey=True)
    for ax, (tier, frame) in zip(axes, comparison.groupby("tier"), strict=False):
        labels = frame["controller"].to_list()
        x = np.arange(len(frame))
        ax.bar(x - 0.2, frame["strict_success_rate"], width=0.2, label="online strict")
        ax.bar(
            x,
            frame["controlled_recovery_rate"],
            width=0.2,
            label="controlled recovery",
        )
        oracle_rate = ceiling_gap[ceiling_gap["tier"].eq(tier)][
            "oracle_strict_feasible_rate"
        ].iloc[0]
        ax.bar(
            x + 0.2,
            np.full(len(frame), oracle_rate),
            width=0.2,
            label="oracle ceiling",
        )
        ax.set_xticks(x, labels, rotation=25, ha="right")
        ax.set_title(tier)
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("rate")
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_oracle_gap(diagnostics: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4), sharey=True)
    for ax, (tier, frame) in zip(axes, diagnostics.groupby("tier"), strict=False):
        for controller, ctrl_frame in frame.groupby("controller"):
            ax.scatter(
                ctrl_frame["scenario_id"],
                ctrl_frame["online_minus_oracle_alpha_miss_rad"],
                label=controller,
                alpha=0.7,
                s=22,
            )
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title(tier)
        ax.set_xlabel("scenario id")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("online minus oracle alpha miss [rad]")
    axes[-1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_first_violation(diagnostics: pd.DataFrame, path: Path) -> None:
    data = diagnostics[~diagnostics["strict_success"]].copy()
    data["first_alpha_violation_time_s"] = pd.to_numeric(
        data["first_alpha_violation_time_s"], errors="coerce"
    )
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    for label, frame in data.groupby("oracle_feasibility_class"):
        ax.scatter(
            frame["max_alpha_corridor_miss_rad"],
            frame["first_alpha_violation_time_s"],
            label=label,
            alpha=0.72,
        )
    ax.set_xlabel("online max alpha miss [rad]")
    ax.set_ylabel("first alpha violation time [s]")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_flap_lag(diagnostics: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    grouped = diagnostics.groupby(["tier", "controller"], as_index=False).agg(
        mean_gap=("raw_applied_flap_gap_mean_rad", "mean"),
        max_gap=("raw_applied_flap_gap_max_rad", "mean"),
    )
    x = np.arange(len(grouped))
    labels = grouped["tier"] + "\n" + grouped["controller"]
    ax.bar(x - 0.16, grouped["mean_gap"], width=0.32, label="mean abs gap")
    ax.bar(x + 0.16, grouped["max_gap"], width=0.32, label="mean max gap")
    ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.set_ylabel("raw learned minus applied flap [rad]")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_envelopes(rollouts: pd.DataFrame, path: Path) -> None:
    controllers = list(rollouts["controller"].drop_duplicates())
    tiers = list(rollouts["tier"].drop_duplicates())
    fig, axes = plt.subplots(
        len(tiers),
        len(controllers),
        figsize=(4.2 * len(controllers), 3.6 * len(tiers)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for row_idx, tier in enumerate(tiers):
        for col_idx, controller in enumerate(controllers):
            ax = axes[row_idx][col_idx]
            frame = rollouts[
                rollouts["tier"].eq(tier) & rollouts["controller"].eq(controller)
            ]
            pivot = frame.pivot_table(
                index="time_s", columns="scenario_id", values="alpha_error_rad"
            )
            median = pivot.median(axis=1)
            low = pivot.quantile(0.05, axis=1)
            high = pivot.quantile(0.95, axis=1)
            ax.plot(median.index, median, linewidth=1.5)
            ax.fill_between(median.index, low, high, alpha=0.18)
            ax.axhline(0.0, color="black", linewidth=0.7)
            ax.set_title(f"{tier}: {controller}", fontsize=9)
            ax.grid(True, alpha=0.22)
    for ax in axes[-1]:
        ax.set_xlabel("time [s]")
    for ax in axes[:, 0]:
        ax.set_ylabel("alpha error [rad]")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
