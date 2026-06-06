from __future__ import annotations

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
from reentry_mpc.phase3 import build_controllers, load_phase3_config
from reentry_mpc.phase6 import TighteningMargins, tighten_reference_profile
from reentry_mpc.phase10 import load_residual_model

CONTROLLERS = [
    "pid",
    "gain_scheduled_lqr",
    "nominal_nmpc",
    "residual_corrected_nmpc",
    "residual_corrected_tightened_nmpc",
]


@dataclass(frozen=True)
class Phase14Config:
    seed: int
    phase1_config: Path
    phase2_config: Path
    phase3_config: Path
    phase5_summary: Path
    phase12_summary: Path
    residual_model_checkpoint: Path
    horizon_lengths: list[int]
    control_frequencies_hz: list[float]
    sample_count: int
    warm_start_modes: list[bool]
    tightening: TighteningMargins
    weights: NmpcWeights
    solver: NmpcSolverOptions
    budgets_ms: dict[str, float]


def load_phase14_config(path: str | Path) -> Phase14Config:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)
    return Phase14Config(
        seed=int(raw["seed"]),
        phase1_config=Path(raw["phase1_config"]),
        phase2_config=Path(raw["phase2_config"]),
        phase3_config=Path(raw["phase3_config"]),
        phase5_summary=Path(raw["phase5_summary"]),
        phase12_summary=Path(raw["phase12_summary"]),
        residual_model_checkpoint=Path(raw["residual_model_checkpoint"]),
        horizon_lengths=[int(value) for value in raw["horizon_lengths"]],
        control_frequencies_hz=[
            float(value) for value in raw["control_frequencies_hz"]
        ],
        sample_count=int(raw["sample_count"]),
        warm_start_modes=[bool(value) for value in raw["warm_start_modes"]],
        tightening=TighteningMargins(
            alpha_margin_rad=float(raw["tightening"]["alpha_margin_rad"]),
            q_margin_radps=float(raw["tightening"]["q_margin_radps"]),
        ),
        weights=NmpcWeights(**{k: float(v) for k, v in raw["weights"].items()}),
        solver=NmpcSolverOptions(
            max_iter=int(raw["solver"]["max_iter"]),
            acceptable_tol=float(raw["solver"]["acceptable_tol"]),
            print_level=int(raw["solver"]["print_level"]),
        ),
        budgets_ms={str(key): float(value) for key, value in raw["budgets_ms"].items()},
    )


def run_phase14_realtime_timing(
    config_path: str | Path = "configs/phase14_realtime_timing.yaml",
    output_dir: str | Path = "outputs/phase14_realtime_timing",
) -> dict[str, Path | pd.DataFrame]:
    config = load_phase14_config(config_path)
    plant_config = load_phase1_config(config.phase1_config)
    phase2_config = load_phase2_config(config.phase2_config)
    phase3_config = load_phase3_config(config.phase3_config)
    reference = build_reference_profile(phase2_config)
    tightened_reference = tighten_reference_profile(reference, config.tightening)
    residual_model = load_residual_model(config.residual_model_checkpoint)
    controllers = build_controllers(
        config=phase3_config,
        plant_config=plant_config,
        dt=_reference_dt(reference),
    )
    sample_times = _sample_times(reference, config.sample_count)
    rows: list[dict[str, Any]] = []
    for warm_start in config.warm_start_modes:
        for frequency_hz in config.control_frequencies_hz:
            control_dt = 1.0 / frequency_hz
            for horizon_steps in config.horizon_lengths:
                nmpc_config = NmpcConfig(
                    horizon_steps=horizon_steps,
                    dt=control_dt,
                    weights=config.weights,
                    solver=config.solver,
                )
                for controller_name in CONTROLLERS:
                    for sample_idx, sample_time in enumerate(sample_times):
                        rows.append(
                            _benchmark_one_call(
                                controller_name=controller_name,
                                sample_idx=sample_idx,
                                sample_time=sample_time,
                                reference=reference,
                                tightened_reference=tightened_reference,
                                plant_config=plant_config,
                                controllers=controllers,
                                residual_model=residual_model,
                                nmpc_config=nmpc_config,
                                frequency_hz=frequency_hz,
                                warm_start=warm_start,
                            )
                        )
    raw_timings = pd.DataFrame(rows)
    summary = summarize_realtime_timings(
        raw_timings=raw_timings,
        budgets_ms=config.budgets_ms,
        phase5_summary=_load_optional(config.phase5_summary),
        phase12_summary=_load_optional(config.phase12_summary),
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    raw_path = output_path / "onboard_timing_raw.csv"
    summary_path = output_path / "onboard_feasibility_table.csv"
    summary_md_path = output_path / "onboard_feasibility_summary.md"
    raw_timings.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    summary_md_path.write_text(
        _summary_markdown(summary=summary, budgets_ms=config.budgets_ms),
        encoding="utf-8",
    )
    figure_paths = write_phase14_figures(summary=summary, output_dir=output_path)
    return {
        "raw_csv": raw_path,
        "summary_csv": summary_path,
        "summary_md": summary_md_path,
        "raw_timings": raw_timings,
        "summary": summary,
        **figure_paths,
    }


def summarize_realtime_timings(
    *,
    raw_timings: pd.DataFrame,
    budgets_ms: dict[str, float],
    phase5_summary: pd.DataFrame,
    phase12_summary: pd.DataFrame,
) -> pd.DataFrame:
    grouped = (
        raw_timings.groupby(
            [
                "controller",
                "horizon_steps",
                "control_frequency_hz",
                "warm_start",
            ]
        )
        .agg(
            sample_count=("sample_idx", "size"),
            mean_solve_time_ms=("solve_time_ms", "mean"),
            median_solve_time_ms=("solve_time_ms", "median"),
            p95_solve_time_ms=("solve_time_ms", lambda s: float(s.quantile(0.95))),
            max_solve_time_ms=("solve_time_ms", "max"),
            mean_nn_inference_time_ms=("nn_inference_time_ms", "mean"),
            mean_total_loop_time_ms=("total_loop_time_ms", "mean"),
            median_total_loop_time_ms=("total_loop_time_ms", "median"),
            p95_total_loop_time_ms=(
                "total_loop_time_ms",
                lambda s: float(s.quantile(0.95)),
            ),
            max_total_loop_time_ms=("total_loop_time_ms", "max"),
            solver_failure_rate=("solver_failed", "mean"),
        )
        .reset_index()
    )
    grouped["budget_10hz_ms"] = budgets_ms["10 Hz"]
    grouped["budget_20hz_ms"] = budgets_ms["20 Hz"]
    grouped["budget_50hz_ms"] = budgets_ms["50 Hz"]
    grouped["meets_10hz_budget_p95"] = (
        grouped["p95_total_loop_time_ms"] <= budgets_ms["10 Hz"]
    )
    grouped["meets_20hz_budget_p95"] = (
        grouped["p95_total_loop_time_ms"] <= budgets_ms["20 Hz"]
    )
    grouped["meets_50hz_budget_p95"] = (
        grouped["p95_total_loop_time_ms"] <= budgets_ms["50 Hz"]
    )
    success = _controller_success_rates(phase5_summary, phase12_summary)
    return grouped.merge(success, on="controller", how="left")


def write_phase14_figures(
    *, summary: pd.DataFrame, output_dir: Path
) -> dict[str, Path]:
    paths = {
        "histogram_png": output_dir / "solve_time_histogram.png",
        "vs_horizon_png": output_dir / "solve_time_vs_horizon.png",
        "pareto_png": output_dir / "performance_vs_timing_pareto.png",
    }
    _plot_solve_time_histogram(summary, paths["histogram_png"])
    _plot_solve_time_vs_horizon(summary, paths["vs_horizon_png"])
    _plot_pareto(summary, paths["pareto_png"])
    return paths


def _benchmark_one_call(
    *,
    controller_name: str,
    sample_idx: int,
    sample_time: float,
    reference: pd.DataFrame,
    tightened_reference: pd.DataFrame,
    plant_config: Any,
    controllers: dict[str, Any],
    residual_model: Any,
    nmpc_config: NmpcConfig,
    frequency_hz: float,
    warm_start: bool,
) -> dict[str, Any]:
    del warm_start
    state, reference_state = _state_at_time(reference, sample_time)
    previous_flap = 0.0
    loop_start = time.perf_counter()
    solve_time = 0.0
    nn_time = 0.0
    solver_failed = False
    status = "not_applicable"
    if controller_name in {"pid", "gain_scheduled_lqr"}:
        row = _interpolated_horizon(reference, sample_time, 1, nmpc_config.dt).iloc[0]
        schedule = {
            "altitude_m": float(row["altitude_m"]),
            "velocity_mps": float(row["velocity_mps"]),
            "mach": float(row["mach"]),
            "density_kgm3": float(row["density_kgm3"]),
            "dynamic_pressure_pa": float(row["dynamic_pressure_pa"]),
        }
        start = time.perf_counter()
        _ = controllers[controller_name].command(
            state=state,
            reference_state=reference_state,
            dt=nmpc_config.dt,
            schedule=schedule,
        )
        solve_time = time.perf_counter() - start
    else:
        planning_reference = (
            tightened_reference
            if controller_name == "residual_corrected_tightened_nmpc"
            else reference
        )
        horizon = _interpolated_horizon(
            planning_reference,
            sample_time,
            nmpc_config.horizon_steps + 1,
            nmpc_config.dt,
        )
        residual_biases = np.zeros(nmpc_config.horizon_steps, dtype=float)
        if controller_name in {
            "residual_corrected_nmpc",
            "residual_corrected_tightened_nmpc",
        }:
            residual_biases, nn_time = build_horizon_residual_biases(
                loaded_model=residual_model,
                state=state,
                previous_flap_rad=previous_flap,
                horizon=horizon,
                horizon_steps=nmpc_config.horizon_steps,
            )
        try:
            _, step_log = solve_horizon_biased_nmpc_step(
                state=state,
                previous_flap_rad=previous_flap,
                horizon=horizon,
                vehicle=plant_config.vehicle,
                aero=plant_config.aero,
                config=nmpc_config,
                residual_q_dot_biases=residual_biases,
            )
            solve_time = float(step_log["solve_time_s"])
            status = str(step_log["solver_status"])
            solver_failed = status != "Solve_Succeeded"
        except RuntimeError:
            status = "RuntimeError"
            solver_failed = True
    total_loop = time.perf_counter() - loop_start
    return {
        "controller": controller_name,
        "sample_idx": sample_idx,
        "sample_time_s": sample_time,
        "horizon_steps": nmpc_config.horizon_steps,
        "control_frequency_hz": frequency_hz,
        "control_dt_s": nmpc_config.dt,
        "warm_start": False,
        "warm_start_implemented": False,
        "solver_status": status,
        "solver_failed": solver_failed,
        "solve_time_ms": solve_time * 1000.0,
        "nn_inference_time_ms": nn_time * 1000.0,
        "total_loop_time_ms": total_loop * 1000.0,
    }


def _plot_solve_time_histogram(summary: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for controller, data in summary.groupby("controller"):
        ax.hist(data["p95_total_loop_time_ms"], bins=18, alpha=0.45, label=controller)
    for budget in [100.0, 50.0, 20.0]:
        ax.axvline(budget, color="black", linestyle="--", linewidth=0.9)
    ax.set_xlabel("p95 total control-loop time [ms]")
    ax.set_ylabel("configuration count")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_solve_time_vs_horizon(summary: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, (freq, data) in zip(
        axes, summary.groupby("control_frequency_hz"), strict=False
    ):
        for controller, subset in data.groupby("controller"):
            profile = subset.groupby("horizon_steps")["p95_total_loop_time_ms"].median()
            ax.plot(profile.index, profile, marker="o", label=controller)
        ax.axhline(1000.0 / freq, color="black", linestyle="--", linewidth=1.0)
        ax.set_title(f"{freq:g} Hz budget")
        ax.set_xlabel("horizon length N")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("p95 total loop time [ms]")
    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        axes[-1].legend(handles, labels, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_pareto(summary: pd.DataFrame, path: Path) -> None:
    best = (
        summary.groupby("controller")
        .agg(
            best_success_rate=("success_rate", "max"),
            min_p95_loop_ms=("p95_total_loop_time_ms", "min"),
        )
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(best["min_p95_loop_ms"], best["best_success_rate"], s=80)
    for row in best.itertuples():
        ax.annotate(row.controller, (row.min_p95_loop_ms, row.best_success_rate))
    for budget, label in [(100.0, "10 Hz"), (50.0, "20 Hz"), (20.0, "50 Hz")]:
        ax.axvline(budget, color="black", linestyle="--", linewidth=0.9)
        ax.text(budget, 0.0, label, rotation=90, va="bottom", ha="right")
    ax.set_xlabel("best p95 total loop time [ms]")
    ax.set_ylabel("best paired benchmark success rate")
    ax.set_ylim(-0.02, 1.0)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _summary_markdown(summary: pd.DataFrame, budgets_ms: dict[str, float]) -> str:
    fastest = summary.sort_values("p95_total_loop_time_ms").head(8)
    budget_counts = {
        label: int((summary["p95_total_loop_time_ms"] <= budget).sum())
        for label, budget in budgets_ms.items()
    }
    lines = [
        "# Phase 14 Real-Time Feasibility Summary",
        "",
        "Budgets:",
        *(f"- {label}: {budget:g} ms" for label, budget in budgets_ms.items()),
        "",
        "Configurations meeting p95 total-loop budgets:",
        *(f"- {label}: {count}" for label, count in budget_counts.items()),
        "",
        "Fastest configurations by p95 total-loop time:",
        "",
        _markdown_table(
            fastest[
                [
                    "controller",
                    "horizon_steps",
                    "control_frequency_hz",
                    "p95_total_loop_time_ms",
                    "solver_failure_rate",
                    "success_rate",
                ]
            ]
        ),
        "",
        "Warm-start note: warm starts are not implemented in this benchmark yet; "
        "all rows use `warm_start=false` and `warm_start_implemented=false`.",
    ]
    return "\n".join(lines)


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.itertuples(index=False):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _controller_success_rates(
    phase5_summary: pd.DataFrame, phase12_summary: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not phase5_summary.empty:
        phase5 = phase5_summary[
            phase5_summary["controller"].isin(["pid", "gain_scheduled_lqr"])
        ]
        for controller, data in phase5.groupby("controller"):
            rows.append(
                {
                    "controller": controller,
                    "success_rate": float(data["failure_label"].eq("success").mean()),
                    "performance_source": "phase5_monte_carlo",
                }
            )
    if not phase12_summary.empty:
        phase12 = phase12_summary[
            phase12_summary["controller"].isin(
                [
                    "nominal_nmpc",
                    "residual_corrected_nmpc",
                    "residual_corrected_tightened_nmpc",
                ]
            )
        ]
        for controller, data in phase12.groupby("controller"):
            rows.append(
                {
                    "controller": controller,
                    "success_rate": float(data["failure_label"].eq("success").mean()),
                    "performance_source": "phase12_learning_augmented_mpc",
                }
            )
    return pd.DataFrame(rows)


def _interpolated_horizon(
    reference: pd.DataFrame, start_time: float, rows: int, dt: float
) -> pd.DataFrame:
    times = start_time + np.arange(rows, dtype=float) * dt
    source_time = reference["time_s"].to_numpy(dtype=float)
    data: dict[str, np.ndarray] = {"time_s": times}
    for column in reference.columns:
        if column == "time_s":
            continue
        if pd.api.types.is_numeric_dtype(reference[column]):
            data[column] = np.interp(
                times,
                source_time,
                reference[column].to_numpy(dtype=float),
                left=float(reference[column].iloc[0]),
                right=float(reference[column].iloc[-1]),
            )
    return pd.DataFrame(data)


def _state_at_time(
    reference: pd.DataFrame, time_s: float
) -> tuple[np.ndarray, np.ndarray]:
    horizon = _interpolated_horizon(reference, time_s, 1, _reference_dt(reference))
    row = horizon.iloc[0]
    reference_state = np.array(
        [row["alpha_ref_rad"], row["q_ref_radps"], row["theta_ref_rad"]],
        dtype=float,
    )
    perturbation = np.array([-0.015, 0.015, 0.005], dtype=float)
    return reference_state + perturbation, reference_state


def _sample_times(reference: pd.DataFrame, sample_count: int) -> np.ndarray:
    start = float(reference["time_s"].iloc[2])
    stop = float(reference["time_s"].iloc[-3])
    return np.linspace(start, stop, sample_count)


def _reference_dt(reference: pd.DataFrame) -> float:
    return float(np.median(np.diff(reference["time_s"].to_numpy(dtype=float))))


def _load_optional(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)
