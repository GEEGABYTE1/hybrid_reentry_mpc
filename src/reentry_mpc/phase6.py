from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from reentry_mpc.artifacts import plt
from reentry_mpc.longitudinal import load_phase1_config
from reentry_mpc.phase2 import build_reference_profile, load_phase2_config
from reentry_mpc.phase4 import _downsample_reference_profile, load_phase4_config
from reentry_mpc.phase5 import (
    _rollout_one_controller,
    load_phase5_config,
    summarize_monte_carlo_rollout,
)
from reentry_mpc.uncertainty import sample_scenario


@dataclass(frozen=True)
class TighteningMargins:
    alpha_margin_rad: float
    q_margin_radps: float


@dataclass(frozen=True)
class Phase6Config:
    seed: int
    phase5_config: Path
    phase1_config: Path
    phase2_config: Path
    phase4_config: Path
    phase5_summary: Path
    controller_name: str
    tightening: TighteningMargins
    plot_settings: dict[str, Any]


def load_phase6_config(path: str | Path) -> Phase6Config:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)
    tightening = raw["tightening"]
    return Phase6Config(
        seed=int(raw["seed"]),
        phase5_config=Path(raw["phase5_config"]),
        phase1_config=Path(raw["phase1_config"]),
        phase2_config=Path(raw["phase2_config"]),
        phase4_config=Path(raw["phase4_config"]),
        phase5_summary=Path(raw["phase5_summary"]),
        controller_name=str(raw.get("controller_name", "tightened_nmpc")),
        tightening=TighteningMargins(
            alpha_margin_rad=float(tightening["alpha_margin_rad"]),
            q_margin_radps=float(tightening["q_margin_radps"]),
        ),
        plot_settings=dict(raw.get("plot_settings", {})),
    )


def tighten_reference_profile(
    reference_profile: pd.DataFrame, margins: TighteningMargins
) -> pd.DataFrame:
    # Return a planning profile with a tightened state corridor.
    tightened = reference_profile.copy()
    tightened["alpha_min_rad"] = tightened["alpha_min_rad"] + margins.alpha_margin_rad
    tightened["alpha_max_rad"] = tightened["alpha_max_rad"] - margins.alpha_margin_rad
    tightened["q_min_radps"] = tightened["q_min_radps"] + margins.q_margin_radps
    tightened["q_max_radps"] = tightened["q_max_radps"] - margins.q_margin_radps
    if (tightened["alpha_min_rad"] >= tightened["alpha_max_rad"]).any():
        raise ValueError("Alpha tightening margin closes or inverts the corridor.")
    if (tightened["q_min_radps"] >= tightened["q_max_radps"]).any():
        raise ValueError("Pitch-rate tightening margin closes or inverts the corridor.")
    return tightened


def run_phase6_robust_mpc(
    config_path: str | Path = "configs/phase6_robust_mpc.yaml",
    output_dir: str | Path = "outputs/phase6_robust_mpc",
    *,
    progress: bool = False,
) -> dict[str, Path | pd.DataFrame]:
    config = load_phase6_config(config_path)
    phase5_config = load_phase5_config(config.phase5_config)
    plant_config = load_phase1_config(config.phase1_config)
    phase2_config = load_phase2_config(config.phase2_config)
    phase4_config = load_phase4_config(config.phase4_config)
    reference_profile = build_reference_profile(phase2_config)
    tightened_reference = tighten_reference_profile(
        reference_profile, config.tightening
    )
    tightened_nmpc_reference = _downsample_reference_profile(
        tightened_reference, phase4_config.control_dt_s
    )

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
            if progress:
                print(
                    "phase6_rollout "
                    f"tier={tier.name} scenario={scenario_id:03d} "
                    f"controller={config.controller_name}",
                    flush=True,
                )
            rollout = _rollout_one_controller(
                tier_name=tier.name,
                controller_name="nominal_nmpc",
                scenario=scenario,
                controller=None,
                reference_profile=reference_profile,
                nmpc_reference=tightened_nmpc_reference,
                plant_config=plant_config,
                phase4_config=phase4_config,
                thresholds=phase5_config.failure_thresholds,
            )
            rollout["controller"] = config.controller_name
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
                "tightening": {
                    "alpha_margin_rad": config.tightening.alpha_margin_rad,
                    "q_margin_radps": config.tightening.q_margin_radps,
                },
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
                    "alpha_tightening_margin_rad": config.tightening.alpha_margin_rad,
                    "q_tightening_margin_radps": config.tightening.q_margin_radps,
                }
            )
            rollouts.append(rollout)

    summary = pd.DataFrame(summary_rows)
    combined_rollouts = pd.concat(rollouts, ignore_index=True)
    baseline_summary = _load_phase5_baseline(config.phase5_summary)
    comparison = pd.concat([baseline_summary, summary], ignore_index=True, sort=False)

    summary_path = output_path / "phase6_summary.csv"
    rollouts_path = output_path / "phase6_rollouts.csv"
    comparison_path = output_path / "phase6_vs_phase5_summary.csv"
    summary.to_csv(summary_path, index=False)
    combined_rollouts.to_csv(rollouts_path, index=False)
    comparison.to_csv(comparison_path, index=False)
    figure_paths = write_phase6_figures(
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


def write_phase6_figures(
    *,
    summary: pd.DataFrame,
    rollouts: pd.DataFrame,
    comparison: pd.DataFrame,
    output_dir: Path,
    plot_settings: dict[str, Any],
) -> dict[str, Path]:
    
    success_path = output_dir / "tightened_nmpc_success_rates.png"
    comparison_path = output_dir / "phase6_vs_phase5_success_rates.png"
    envelope_path = output_dir / "tightened_nmpc_alpha_error_envelopes.png"
    failure_path = output_dir / "tightened_nmpc_failure_modes.png"
    worst_path = output_dir / "tightened_nmpc_worst_case_replay.png"

    success_rates = (
        summary.assign(success=summary["failure_label"].eq("success"))
        .groupby(["tier", "controller"])["success"]
        .mean()
        .sort_index()
    )
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    success_rates.unstack("controller").plot(kind="bar", ax=ax)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Success rate")
    ax.set_xlabel("Uncertainty tier")
    fig.tight_layout()
    fig.savefig(success_path, dpi=160)
    plt.close(fig)

    compared = comparison[
        comparison["controller"].isin(
            ["pid", "gain_scheduled_lqr", "nominal_nmpc", "tightened_nmpc"]
        )
    ]
    compared_rates = (
        compared.assign(success=compared["failure_label"].eq("success"))
        .groupby(["tier", "controller"])["success"]
        .mean()
        .sort_index()
    )
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    compared_rates.unstack("controller").plot(kind="bar", ax=ax)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Success rate")
    ax.set_xlabel("Uncertainty tier")
    fig.tight_layout()
    fig.savefig(comparison_path, dpi=160)
    plt.close(fig)

    percentiles = plot_settings.get("envelope_percentiles", [5.0, 95.0])
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    for tier, frame in rollouts.groupby("tier", sort=True):
        grouped = frame.groupby("time_s")["alpha_error_rad"]
        median = grouped.median()
        low = grouped.quantile(float(percentiles[0]) / 100.0)
        high = grouped.quantile(float(percentiles[1]) / 100.0)
        ax.plot(median.index, median.to_numpy(), label=f"{tier}/tightened_nmpc")
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
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    failure_counts.sort_index().plot(kind="bar", stacked=True, ax=ax)
    ax.set_xlabel("Uncertainty tier / controller")
    ax.set_ylabel("Scenario count")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(failure_path, dpi=160)
    plt.close(fig)

    worst_row = (
        summary.assign(non_success=summary["failure_label"].ne("success"))
        .sort_values(["non_success", "max_alpha_error_rad"], ascending=[False, False])
        .iloc[0]
    )
    worst = rollouts[
        (rollouts["tier"] == worst_row["tier"])
        & (rollouts["scenario_id"] == int(worst_row["scenario_id"]))
    ]
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.4), sharex=True)
    axes[0].fill_between(
        worst["time_s"],
        worst["alpha_min_rad"],
        worst["alpha_max_rad"],
        color="gray",
        alpha=0.22,
        label="evaluation corridor",
    )
    axes[0].plot(worst["time_s"], worst["alpha_rad"], label="alpha")
    axes[0].plot(worst["time_s"], worst["alpha_ref_rad"], linestyle="--", label="ref")
    axes[1].plot(worst["time_s"], worst["delta_flap_rad"], label="applied flap")
    axes[0].set_ylabel("Alpha (rad)")
    axes[1].set_ylabel("Flap (rad)")
    axes[1].set_xlabel("Time (s)")
    axes[0].set_title(
        "Worst tightened NMPC replay: "
        f"{worst_row['tier']} scenario {int(worst_row['scenario_id'])}"
    )
    axes[0].legend(loc="best")
    axes[1].legend(loc="best")
    fig.tight_layout()
    fig.savefig(worst_path, dpi=160)
    plt.close(fig)

    return {
        "success_rates_figure": success_path,
        "comparison_success_rates_figure": comparison_path,
        "alpha_error_envelopes_figure": envelope_path,
        "failure_mode_figure": failure_path,
        "worst_case_replay_figure": worst_path,
    }


def _load_phase5_baseline(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)
