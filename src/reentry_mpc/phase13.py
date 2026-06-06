from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from reentry_mpc.artifacts import plt


@dataclass(frozen=True)
class Phase13Config:
    phase12_summary: Path
    phase12_rollouts: Path
    alpha_slack_bins_rad: list[float]
    plot_settings: dict[str, Any]


def load_phase13_config(path: str | Path) -> Phase13Config:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)
    return Phase13Config(
        phase12_summary=Path(raw["phase12_summary"]),
        phase12_rollouts=Path(raw["phase12_rollouts"]),
        alpha_slack_bins_rad=[float(value) for value in raw["alpha_slack_bins_rad"]],
        plot_settings=dict(raw.get("plot_settings", {})),
    )


def run_phase13_feasibility_diagnostics(
    config_path: str | Path = "configs/phase13_feasibility_diagnostics.yaml",
    output_dir: str | Path = "outputs/phase13_feasibility_diagnostics",
) -> dict[str, Path | pd.DataFrame]:
    config = load_phase13_config(config_path)
    summary = pd.read_csv(config.phase12_summary)
    rollouts = pd.read_csv(config.phase12_rollouts)
    diagnostics = compute_feasibility_diagnostics(summary=summary, rollouts=rollouts)
    slack_summary = compute_slack_summary(
        diagnostics=diagnostics,
        bins=config.alpha_slack_bins_rad,
    )
    failure_timing = compute_failure_timing(diagnostics=diagnostics)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    diagnostics_path = output_path / "feasibility_diagnostics.csv"
    slack_path = output_path / "corridor_slack_summary.csv"
    timing_path = output_path / "failure_timing.csv"
    diagnostics.to_csv(diagnostics_path, index=False)
    slack_summary.to_csv(slack_path, index=False)
    failure_timing.to_csv(timing_path, index=False)
    figure_paths = write_phase13_figures(
        diagnostics=diagnostics,
        failure_timing=failure_timing,
        output_dir=output_path,
        controller_order=list(
            config.plot_settings.get(
                "controller_order",
                sorted(diagnostics["controller"].unique()),
            )
        ),
    )
    return {
        "diagnostics_csv": diagnostics_path,
        "slack_summary_csv": slack_path,
        "failure_timing_csv": timing_path,
        "diagnostics": diagnostics,
        "slack_summary": slack_summary,
        "failure_timing": failure_timing,
        **figure_paths,
    }


def compute_feasibility_diagnostics(
    *, summary: pd.DataFrame, rollouts: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["tier", "controller", "scenario_id"]
    summary_keyed = summary.set_index(group_cols)
    for key, rollout in rollouts.groupby(group_cols, sort=True):
        tier, controller, scenario_id = key
        metrics = summary_keyed.loc[key].to_dict()
        alpha_low = rollout["alpha_min_rad"] - rollout["alpha_rad"]
        alpha_high = rollout["alpha_rad"] - rollout["alpha_max_rad"]
        q_low = rollout["q_min_radps"] - rollout["q_radps"]
        q_high = rollout["q_radps"] - rollout["q_max_radps"]
        alpha_violation = np.maximum.reduce(
            [alpha_low.to_numpy(), alpha_high.to_numpy(), np.zeros(len(rollout))]
        )
        q_violation = np.maximum.reduce(
            [q_low.to_numpy(), q_high.to_numpy(), np.zeros(len(rollout))]
        )
        alpha_violation_mask = alpha_violation > 0.0
        q_violation_mask = q_violation > 0.0
        dt = _median_dt(rollout["time_s"])
        first_alpha_time = _first_time(rollout, alpha_violation_mask)
        first_q_time = _first_time(rollout, q_violation_mask)
        rows.append(
            {
                "tier": tier,
                "controller": controller,
                "scenario_id": int(scenario_id),
                "failure_label": metrics["failure_label"],
                "success": metrics["failure_label"] == "success",
                "needed_alpha_corridor_expansion_rad": float(alpha_violation.max()),
                "needed_q_corridor_expansion_radps": float(q_violation.max()),
                "alpha_violation_duration_s": float(alpha_violation_mask.sum() * dt),
                "q_violation_duration_s": float(q_violation_mask.sum() * dt),
                "first_alpha_violation_time_s": first_alpha_time,
                "first_q_violation_time_s": first_q_time,
                "flap_saturation_fraction": float(rollout["flap_saturated"].mean()),
                "flap_rate_saturation_fraction": float(
                    rollout["flap_rate_saturated"].mean()
                ),
                "initial_alpha_error_rad": float(
                    rollout["initial_alpha_error_rad"].iloc[0]
                ),
                "initial_q_error_radps": float(
                    rollout["initial_q_error_radps"].iloc[0]
                ),
                "actuator_lag_s": float(rollout["actuator_lag_s"].iloc[0]),
                "actuator_delay_s": float(rollout["actuator_delay_s"].iloc[0]),
                "density_scale": float(rollout["density_scale"].iloc[0]),
                "cm_delta_scale": float(rollout["cm_delta_scale"].iloc[0]),
                "external_disturbance_moment_nm": float(
                    rollout["external_disturbance_moment_nm"].iloc[0]
                ),
                "rms_alpha_error_rad": float(metrics["rms_alpha_error_rad"]),
                "max_alpha_error_rad": float(metrics["max_alpha_error_rad"]),
            }
        )
    return pd.DataFrame(rows)


def compute_slack_summary(
    *, diagnostics: pd.DataFrame, bins: list[float]
) -> pd.DataFrame:
    data = diagnostics.copy()
    data["alpha_slack_bin_rad"] = pd.cut(
        data["needed_alpha_corridor_expansion_rad"],
        bins=bins,
        include_lowest=True,
        right=True,
    ).astype(str)
    return (
        data.groupby(["tier", "controller", "alpha_slack_bin_rad"])
        .size()
        .reset_index(name="rollout_count")
    )


def compute_failure_timing(*, diagnostics: pd.DataFrame) -> pd.DataFrame:
    failed = diagnostics[~diagnostics["success"]].copy()
    return (
        failed.groupby(["tier", "controller", "failure_label"])
        .agg(
            rollout_count=("scenario_id", "size"),
            median_first_alpha_violation_time_s=(
                "first_alpha_violation_time_s",
                "median",
            ),
            median_alpha_violation_duration_s=(
                "alpha_violation_duration_s",
                "median",
            ),
            median_needed_alpha_expansion_rad=(
                "needed_alpha_corridor_expansion_rad",
                "median",
            ),
            median_flap_saturation_fraction=(
                "flap_saturation_fraction",
                "median",
            ),
            median_flap_rate_saturation_fraction=(
                "flap_rate_saturation_fraction",
                "median",
            ),
        )
        .reset_index()
    )


def write_phase13_figures(
    *,
    diagnostics: pd.DataFrame,
    failure_timing: pd.DataFrame,
    output_dir: Path,
    controller_order: list[str],
) -> dict[str, Path]:
    paths = {
        "slack_histogram_png": output_dir / "alpha_corridor_slack_histogram.png",
        "first_violation_png": output_dir / "first_violation_time.png",
        "slack_vs_initial_png": output_dir / "slack_vs_initial_error.png",
        "failure_heatmap_png": output_dir / "failure_diagnostics_heatmap.png",
    }
    _plot_slack_histogram(diagnostics, paths["slack_histogram_png"], controller_order)
    _plot_first_violation(diagnostics, paths["first_violation_png"], controller_order)
    _plot_slack_vs_initial(diagnostics, paths["slack_vs_initial_png"])
    _plot_failure_heatmap(failure_timing, paths["failure_heatmap_png"])
    return paths


def _plot_slack_histogram(
    diagnostics: pd.DataFrame, path: Path, controller_order: list[str]
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, (tier, tier_data) in zip(axes, diagnostics.groupby("tier"), strict=False):
        for controller in controller_order:
            subset = tier_data[tier_data["controller"] == controller]
            if subset.empty:
                continue
            ax.hist(
                subset["needed_alpha_corridor_expansion_rad"],
                bins=18,
                alpha=0.45,
                label=controller,
            )
        ax.set_title(tier)
        ax.set_xlabel("needed alpha corridor expansion [rad]")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("rollout count")
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_first_violation(
    diagnostics: pd.DataFrame, path: Path, controller_order: list[str]
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, (tier, tier_data) in zip(axes, diagnostics.groupby("tier"), strict=False):
        positions = []
        labels = []
        values = []
        for idx, controller in enumerate(controller_order):
            subset = tier_data[
                (tier_data["controller"] == controller)
                & tier_data["first_alpha_violation_time_s"].notna()
            ]
            if subset.empty:
                continue
            positions.append(idx)
            labels.append(controller)
            values.append(subset["first_alpha_violation_time_s"].to_numpy())
        if values:
            ax.boxplot(values, positions=positions, widths=0.55)
            ax.set_xticks(positions, labels, rotation=20, ha="right")
        ax.set_title(tier)
        ax.set_ylabel("first alpha violation time [s]")
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_slack_vs_initial(diagnostics: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, (tier, tier_data) in zip(axes, diagnostics.groupby("tier"), strict=False):
        for controller, subset in tier_data.groupby("controller"):
            ax.scatter(
                subset["initial_alpha_error_rad"],
                subset["needed_alpha_corridor_expansion_rad"],
                s=28,
                alpha=0.65,
                label=controller,
            )
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title(tier)
        ax.set_xlabel("initial alpha error [rad]")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("needed alpha expansion [rad]")
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_failure_heatmap(failure_timing: pd.DataFrame, path: Path) -> None:
    if failure_timing.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No failures", ha="center", va="center")
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return
    pivot = failure_timing.pivot_table(
        index=["tier", "controller"],
        columns="failure_label",
        values="median_needed_alpha_expansion_rad",
        fill_value=0.0,
    )
    fig, ax = plt.subplots(figsize=(10, 4.8))
    image = ax.imshow(pivot.to_numpy(), aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=20, ha="right")
    ax.set_yticks(
        np.arange(len(pivot.index)),
        [f"{tier}\n{controller}" for tier, controller in pivot.index],
    )
    ax.set_title("median needed alpha expansion by failure label")
    fig.colorbar(image, ax=ax, label="rad")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _first_time(rollout: pd.DataFrame, mask: np.ndarray) -> float:
    if not mask.any():
        return float("nan")
    return float(rollout.loc[mask, "time_s"].iloc[0])


def _median_dt(time_s: pd.Series) -> float:
    values = time_s.to_numpy(dtype=float)
    if len(values) < 2:
        return 0.0
    return float(np.median(np.diff(values)))
