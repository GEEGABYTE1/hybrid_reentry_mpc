from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from reentry_mpc.artifacts import plt
from reentry_mpc.baseline_controllers import GainScheduledLQRController
from reentry_mpc.learning_augmented_mpc import (
    build_horizon_residual_biases,
    solve_horizon_biased_nmpc_step,
)
from reentry_mpc.longitudinal import AeroParams, load_phase1_config
from reentry_mpc.nmpc import NmpcConfig, NmpcSolverOptions, NmpcWeights
from reentry_mpc.phase2 import build_reference_profile, load_phase2_config
from reentry_mpc.phase3 import build_controllers, load_phase3_config
from reentry_mpc.phase5 import (
    _has_corridor_violation,
    load_phase5_config,
    summarize_monte_carlo_rollout,
)
from reentry_mpc.phase6 import TighteningMargins, tighten_reference_profile
from reentry_mpc.phase10 import load_residual_model
from reentry_mpc.uncertainty import (
    InitialStateError,
    SensorNoiseStd,
    UncertaintyScenario,
    initialize_actuator,
    perturb_aero,
)


@dataclass(frozen=True)
class FaultCase:
    name: str
    trigger_time_s: float
    parameters: dict[str, float]


@dataclass(frozen=True)
class FallbackConfig:
    residual_error_threshold_rad: float
    repeated_solver_failure_limit: int
    hold_previous_on_fault: bool


@dataclass(frozen=True)
class Phase15Config:
    seed: int
    phase1_config: Path
    phase2_config: Path
    phase3_config: Path
    phase5_config: Path
    residual_model_checkpoint: Path
    controllers: list[str]
    initial_error: np.ndarray
    tightening: TighteningMargins
    fallback: FallbackConfig
    faults: list[FaultCase]
    nmpc: NmpcConfig


def load_phase15_config(path: str | Path) -> Phase15Config:
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
    return Phase15Config(
        seed=int(raw["seed"]),
        phase1_config=Path(raw["phase1_config"]),
        phase2_config=Path(raw["phase2_config"]),
        phase3_config=Path(raw["phase3_config"]),
        phase5_config=Path(raw["phase5_config"]),
        residual_model_checkpoint=Path(raw["residual_model_checkpoint"]),
        controllers=[str(value) for value in raw["controllers"]],
        initial_error=np.array(
            [
                float(raw["initial_error"]["alpha_rad"]),
                float(raw["initial_error"]["q_radps"]),
                float(raw["initial_error"]["theta_rad"]),
            ],
            dtype=float,
        ),
        tightening=TighteningMargins(
            alpha_margin_rad=float(raw["tightening"]["alpha_margin_rad"]),
            q_margin_radps=float(raw["tightening"]["q_margin_radps"]),
        ),
        fallback=FallbackConfig(
            residual_error_threshold_rad=float(
                raw["fallback"]["residual_error_threshold_rad"]
            ),
            repeated_solver_failure_limit=int(
                raw["fallback"]["repeated_solver_failure_limit"]
            ),
            hold_previous_on_fault=bool(raw["fallback"]["hold_previous_on_fault"]),
        ),
        faults=[
            FaultCase(
                name=str(item["name"]),
                trigger_time_s=float(item["trigger_time_s"]),
                parameters={
                    key: float(value)
                    for key, value in item.items()
                    if key not in {"name", "trigger_time_s"}
                },
            )
            for item in raw["faults"]
        ],
        nmpc=NmpcConfig(
            horizon_steps=int(raw["nmpc"]["horizon_steps"]),
            dt=float(raw["nmpc"]["control_dt_s"]),
            weights=weights,
            solver=solver,
        ),
    )


def run_phase15_fault_injection(
    config_path: str | Path = "configs/phase15_fault_injection.yaml",
    output_dir: str | Path = "outputs/phase15_fault_injection",
) -> dict[str, Path | pd.DataFrame]:
    config = load_phase15_config(config_path)
    plant = load_phase1_config(config.phase1_config)
    phase2 = load_phase2_config(config.phase2_config)
    phase3 = load_phase3_config(config.phase3_config)
    phase5 = load_phase5_config(config.phase5_config)
    reference = build_reference_profile(phase2)
    tightened_reference = tighten_reference_profile(reference, config.tightening)
    residual_model = load_residual_model(config.residual_model_checkpoint)
    controllers = build_controllers(
        config=phase3,
        plant_config=plant,
        dt=float(np.median(np.diff(reference["time_s"]))),
    )
    lqr = controllers["gain_scheduled_lqr"]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    rollouts: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    for fault in config.faults:
        for controller_name in config.controllers:
            rollout = rollout_fault_case(
                controller_name=controller_name,
                fault=fault,
                reference=reference,
                tightened_reference=tightened_reference,
                plant=plant,
                residual_model=residual_model,
                lqr=lqr,
                config=config,
                thresholds=phase5.failure_thresholds,
            )
            scenario = _summary_scenario(config, fault)
            metrics = summarize_monte_carlo_rollout(
                rollout=rollout,
                tier_name="fault_injection",
                controller_name=controller_name,
                scenario=scenario,
                thresholds=phase5.failure_thresholds,
            )
            run_dir = output_path / fault.name / controller_name
            run_dir.mkdir(parents=True, exist_ok=True)
            trajectory_path = run_dir / "trajectory.csv"
            metrics_path = run_dir / "metrics.json"
            rollout.to_csv(trajectory_path, index=False)
            metrics_payload = {
                **metrics,
                "fault": fault.name,
                "fault_parameters": fault.parameters,
                "trajectory_csv": str(trajectory_path),
                "fallback_counts": _fallback_counts(rollout),
            }
            metrics_path.write_text(
                json.dumps(metrics_payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            summary_rows.append(
                {
                    "fault": fault.name,
                    **metrics,
                    **_fallback_counts(rollout),
                    "trajectory_csv": str(trajectory_path),
                    "metrics_json": str(metrics_path),
                }
            )
            rollouts.append(rollout)
    combined = pd.concat(rollouts, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    summary_path = output_path / "fault_metrics_table.csv"
    rollouts_path = output_path / "fault_rollouts.csv"
    limitations_path = output_path / "fault_limitations.md"
    summary.to_csv(summary_path, index=False)
    combined.to_csv(rollouts_path, index=False)
    limitations_path.write_text(_fault_limitations_markdown(summary), encoding="utf-8")
    figure_paths = write_phase15_figures(
        summary=summary, rollouts=combined, output_dir=output_path
    )
    return {
        "summary_csv": summary_path,
        "rollouts_csv": rollouts_path,
        "limitations_md": limitations_path,
        "summary": summary,
        "rollouts": combined,
        **figure_paths,
    }


def rollout_fault_case(
    *,
    controller_name: str,
    fault: FaultCase,
    reference: pd.DataFrame,
    tightened_reference: pd.DataFrame,
    plant: Any,
    residual_model: Any,
    lqr: GainScheduledLQRController,
    config: Phase15Config,
    thresholds: dict[str, float],
) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed)
    scenario = _summary_scenario(config, fault)
    perturbed_aero = perturb_aero(plant.aero, scenario)
    first = reference.iloc[0]
    state = (
        np.array(
            [first["alpha_ref_rad"], first["q_ref_radps"], first["theta_ref_rad"]],
            dtype=float,
        )
        + config.initial_error
    )
    dt = float(np.median(np.diff(reference["time_s"])))
    actuator = initialize_actuator(scenario, dt)
    fault_delay_buffer: list[float] = []
    last_raw = 0.0
    last_feasible_raw = 0.0
    repeated_solver_failures = 0
    safe_mode_active = False
    stuck_flap_value: float | None = None
    rows: list[dict[str, Any]] = []
    for _idx, row in reference.iterrows():
        time_s = float(row["time_s"])
        fault_active = time_s >= fault.trigger_time_s
        measured_state = state.copy()
        measured_state += rng.normal(0.0, [0.0005, 0.0002, 0.0005])
        if fault_active and fault.name == "biased_alpha_measurement":
            measured_state[0] += fault.parameters["alpha_bias_rad"]
        reference_state = np.array(
            [row["alpha_ref_rad"], row["q_ref_radps"], row["theta_ref_rad"]],
            dtype=float,
        )
        residual_error = float(abs(measured_state[0] - row["alpha_ref_rad"]))
        use_fallbacks = controller_name.endswith("with_fallback")
        use_tightened = bool(
            use_fallbacks
            and residual_error > config.fallback.residual_error_threshold_rad
        )
        solver_status = "held"
        fallback_action = "none"
        raw_command = last_raw
        if _is_update_time(time_s, config.nmpc.dt):
            if safe_mode_active:
                raw_command = _lqr_command(
                    lqr, measured_state, reference_state, row, dt
                )
                fallback_action = "lqr_safe_mode"
                solver_status = "safe_mode"
            else:
                planning = tightened_reference if use_tightened else reference
                nmpc_idx = min(
                    int(round(time_s / config.nmpc.dt)),
                    len(planning) - 1,
                )
                horizon = planning.iloc[
                    nmpc_idx : nmpc_idx + config.nmpc.horizon_steps + 1
                ]
                residual_biases, _ = build_horizon_residual_biases(
                    loaded_model=residual_model,
                    state=measured_state,
                    previous_flap_rad=actuator.previous_applied_rad,
                    horizon=horizon,
                    horizon_steps=config.nmpc.horizon_steps,
                )
                try:
                    raw_command, step_log = solve_horizon_biased_nmpc_step(
                        state=measured_state,
                        previous_flap_rad=actuator.previous_applied_rad,
                        horizon=horizon,
                        vehicle=plant.vehicle,
                        aero=plant.aero,
                        config=config.nmpc,
                        residual_q_dot_biases=residual_biases,
                    )
                    solver_status = str(step_log["solver_status"])
                    if solver_status == "Solve_Succeeded":
                        repeated_solver_failures = 0
                        last_feasible_raw = raw_command
                        fallback_action = (
                            "constraint_tightening" if use_tightened else "none"
                        )
                    else:
                        repeated_solver_failures += 1
                        if use_fallbacks:
                            raw_command = last_feasible_raw
                            fallback_action = "previous_feasible_control"
                except RuntimeError:
                    solver_status = "RuntimeError"
                    repeated_solver_failures += 1
                    if use_fallbacks:
                        raw_command = last_feasible_raw
                        fallback_action = "previous_feasible_control"
                if (
                    use_fallbacks
                    and repeated_solver_failures
                    >= config.fallback.repeated_solver_failure_limit
                ):
                    safe_mode_active = True
        last_raw = float(raw_command)
        applied_control, actuator_log, fault_delay_buffer = _faulted_actuator_step(
            raw_command=float(raw_command),
            actuator=actuator,
            row=row,
            dt=dt,
            fault=fault,
            fault_active=fault_active,
            delay_buffer=fault_delay_buffer,
        )
        if fault_active and fault.name == "stuck_flap":
            if stuck_flap_value is None:
                stuck_flap_value = fault.parameters.get(
                    "stuck_flap_rad", applied_control
                )
            applied_control = float(stuck_flap_value)
            actuator.previous_applied_rad = applied_control
            actuator_log["delta_flap_rad"] = applied_control
            actuator_log["fault_stuck_flap_override"] = True
        else:
            actuator_log["fault_stuck_flap_override"] = False
        alpha_error = float(state[0] - row["alpha_ref_rad"])
        q_error = float(state[1] - row["q_ref_radps"])
        rows.append(
            {
                "scenario_id": 0,
                "tier": "fault_injection",
                "seed": config.seed,
                "fault": fault.name,
                "controller": controller_name,
                "time_s": time_s,
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
                "theta_error_rad": float(state[2] - row["theta_ref_rad"]),
                "solver_status": solver_status,
                "solver_failure": repeated_solver_failures > 0,
                "fault_active": fault_active,
                "fallback_action": fallback_action,
                "safe_mode_active": safe_mode_active,
                "constraint_tightening_active": use_tightened,
                "residual_error_rad": residual_error,
                **actuator_log,
                "altitude_m": float(row["altitude_m"]),
                "velocity_mps": float(row["velocity_mps"]),
                "mach": float(row["mach"]),
                "density_kgm3": float(row["density_kgm3"])
                * _fault_density_scale(fault, fault_active),
                "dynamic_pressure_pa": float(row["dynamic_pressure_pa"])
                * _fault_density_scale(fault, fault_active),
                **scenario.to_flat_dict(),
                "corridor_violation": _has_corridor_violation(
                    state=state,
                    row=row,
                    tolerance=thresholds["corridor_tolerance_rad"],
                ),
            }
        )
        state = _faulted_rk4_step(
            state=state,
            delta_flap_rad=applied_control,
            row=row,
            aero=perturbed_aero,
            plant=plant,
            scenario=scenario,
            fault=fault,
            fault_active=fault_active,
            dt=dt,
        )
    return pd.DataFrame(rows)


def write_phase15_figures(
    *, summary: pd.DataFrame, rollouts: pd.DataFrame, output_dir: Path
) -> dict[str, Path]:
    paths = {
        "success_png": output_dir / "fault_success_rates.png",
        "fallback_png": output_dir / "fallback_comparison.png",
        "case_studies_png": output_dir / "fault_case_studies.png",
    }
    _plot_fault_success(summary, paths["success_png"])
    _plot_fallback_comparison(summary, paths["fallback_png"])
    _plot_case_studies(rollouts, paths["case_studies_png"])
    return paths


def _faulted_actuator_step(
    *,
    raw_command: float,
    actuator: Any,
    row: pd.Series,
    dt: float,
    fault: FaultCase,
    fault_active: bool,
    delay_buffer: list[float],
) -> tuple[float, dict[str, float | bool], list[float]]:
    command = raw_command
    if fault_active and fault.name == "delayed_actuator":
        extra_steps = max(1, int(round(fault.parameters["extra_delay_s"] / dt)))
        if not delay_buffer:
            delay_buffer = [actuator.previous_applied_rad] * extra_steps
        delay_buffer.append(float(raw_command))
        command = delay_buffer.pop(0)
    from reentry_mpc.uncertainty import actuator_step  # noqa: PLC0415

    scenario = _zero_scenario()
    applied, log = actuator_step(
        raw_command=command,
        actuator=actuator,
        scenario=scenario,
        row=row,
        dt=dt,
    )
    log["fault_extra_actuator_delay_active"] = bool(
        fault_active and fault.name == "delayed_actuator"
    )
    return applied, log, delay_buffer


def _faulted_rk4_step(
    *,
    state: np.ndarray,
    delta_flap_rad: float,
    row: pd.Series,
    aero: AeroParams,
    plant: Any,
    scenario: UncertaintyScenario,
    fault: FaultCase,
    fault_active: bool,
    dt: float,
) -> np.ndarray:
    def f(local_state: np.ndarray) -> np.ndarray:
        return _faulted_derivatives(
            state=local_state,
            delta_flap_rad=delta_flap_rad,
            row=row,
            aero=aero,
            plant=plant,
            scenario=scenario,
            fault=fault,
            fault_active=fault_active,
        )

    k1 = f(state)
    k2 = f(state + 0.5 * dt * k1)
    k3 = f(state + 0.5 * dt * k2)
    k4 = f(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _faulted_derivatives(
    *,
    state: np.ndarray,
    delta_flap_rad: float,
    row: pd.Series,
    aero: AeroParams,
    plant: Any,
    scenario: UncertaintyScenario,
    fault: FaultCase,
    fault_active: bool,
) -> np.ndarray:
    dynamic_pressure = float(row["dynamic_pressure_pa"]) * _fault_density_scale(
        fault, fault_active
    )
    effectiveness_scale = (
        fault.parameters["cm_delta_scale"]
        if fault_active and fault.name == "reduced_flap_effectiveness"
        else 1.0
    )
    effectiveness = _flap_effectiveness(
        float(row["mach"]), float(row["altitude_m"]), aero
    )
    alpha_slope = aero.cm_alpha + aero.cm_alpha_mach_slope * (float(row["mach"]) - 15.0)
    q_hat = (
        state[1]
        * plant.vehicle.reference_length_m
        / max(2.0 * float(row["velocity_mps"]), 1.0)
    )
    cm = (
        aero.cm0
        + alpha_slope * state[0]
        + aero.cm_q * q_hat
        + aero.cm_delta * effectiveness * effectiveness_scale * delta_flap_rad
    )
    disturbance = scenario.external_disturbance_moment_nm
    if fault_active and fault.name == "large_unmodeled_disturbance":
        disturbance += fault.parameters["disturbance_moment_nm"]
    moment = (
        dynamic_pressure
        * plant.vehicle.reference_area_m2
        * plant.vehicle.reference_length_m
        * cm
        + disturbance
    )
    q_dot = moment / plant.vehicle.pitch_inertia_kgm2
    return np.array([state[1] - 0.22 * state[0], q_dot, state[1]], dtype=float)


def _plot_fault_success(summary: pd.DataFrame, path: Path) -> None:
    data = summary.assign(success=summary["failure_label"].eq("success"))
    pivot = data.pivot_table(
        index="fault", columns="controller", values="success", aggfunc="mean"
    ).fillna(0.0)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("success rate")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_fallback_comparison(summary: pd.DataFrame, path: Path) -> None:
    columns = [
        "previous_feasible_control_count",
        "constraint_tightening_count",
        "lqr_safe_mode_count",
    ]
    data = summary.groupby("controller")[columns].sum()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    data.plot(kind="bar", stacked=True, ax=ax)
    ax.set_ylabel("fallback activations")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_case_studies(rollouts: pd.DataFrame, path: Path) -> None:
    selected_faults = [
        "stuck_flap",
        "biased_alpha_measurement",
        "large_unmodeled_disturbance",
    ]
    fig, axes = plt.subplots(len(selected_faults), 1, figsize=(10, 8), sharex=True)
    for ax, fault in zip(axes, selected_faults, strict=False):
        subset = rollouts[rollouts["fault"] == fault]
        for controller, data in subset.groupby("controller"):
            ax.plot(data["time_s"], data["alpha_rad"], label=controller)
        ref = subset.groupby("time_s", as_index=False).first()
        ax.fill_between(
            ref["time_s"],
            ref["alpha_min_rad"],
            ref["alpha_max_rad"],
            color="gray",
            alpha=0.18,
        )
        ax.axvline(
            float(subset["time_s"][subset["fault_active"]].min()),
            color="red",
            linestyle="--",
        )
        ax.set_title(fault)
        ax.set_ylabel("alpha [rad]")
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("time [s]")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _fallback_counts(rollout: pd.DataFrame) -> dict[str, int]:
    return {
        "previous_feasible_control_count": int(
            (rollout["fallback_action"] == "previous_feasible_control").sum()
        ),
        "constraint_tightening_count": int(
            (rollout["fallback_action"] == "constraint_tightening").sum()
        ),
        "lqr_safe_mode_count": int(
            (rollout["fallback_action"] == "lqr_safe_mode").sum()
        ),
        "fault_active_count": int(rollout["fault_active"].sum()),
    }


def _fault_limitations_markdown(summary: pd.DataFrame) -> str:
    failures = summary[summary["failure_label"] != "success"]
    return "\n".join(
        [
            "# Fault Injection Limitations",
            "",
            "- Fault cases are deterministic stress probes, not statistically "
            "calibrated failure probabilities.",
            "- The stuck-flap and actuator-delay models are simplified "
            "actuator-level approximations.",
            "- Safe-mode LQR fallback is implemented after repeated solver "
            "failures, but most current failures are corridor/dynamics "
            "failures rather than solver failures.",
            "- Constraint tightening is triggered from measured alpha residual "
            "error; it is not a formal fault detector.",
            "",
            f"Failed fault/controller rows: {len(failures)} of {len(summary)}.",
        ]
    )


def _summary_scenario(config: Phase15Config, fault: FaultCase) -> UncertaintyScenario:
    del fault
    return UncertaintyScenario(
        scenario_id=0,
        seed=config.seed,
        density_scale=1.0,
        cm_alpha_scale=1.0,
        cm_delta_scale=1.0,
        cm_q_scale=1.0,
        actuator_lag_s=0.0,
        actuator_delay_s=0.0,
        sensor_noise_std=SensorNoiseStd(0.0005, 0.0002, 0.0005),
        initial_error=InitialStateError(
            alpha_rad=float(config.initial_error[0]),
            q_radps=float(config.initial_error[1]),
            theta_rad=float(config.initial_error[2]),
        ),
        external_disturbance_moment_nm=0.0,
    )


def _zero_scenario() -> UncertaintyScenario:
    return UncertaintyScenario(
        scenario_id=0,
        seed=0,
        density_scale=1.0,
        cm_alpha_scale=1.0,
        cm_delta_scale=1.0,
        cm_q_scale=1.0,
        actuator_lag_s=0.0,
        actuator_delay_s=0.0,
        sensor_noise_std=SensorNoiseStd(0.0, 0.0, 0.0),
        initial_error=InitialStateError(0.0, 0.0, 0.0),
        external_disturbance_moment_nm=0.0,
    )


def _lqr_command(
    lqr: GainScheduledLQRController,
    measured_state: np.ndarray,
    reference_state: np.ndarray,
    row: pd.Series,
    dt: float,
) -> float:
    return lqr.command(
        state=measured_state,
        reference_state=reference_state,
        dt=dt,
        schedule={
            "altitude_m": float(row["altitude_m"]),
            "velocity_mps": float(row["velocity_mps"]),
            "mach": float(row["mach"]),
            "density_kgm3": float(row["density_kgm3"]),
            "dynamic_pressure_pa": float(row["dynamic_pressure_pa"]),
        },
    )


def _fault_density_scale(fault: FaultCase, fault_active: bool) -> float:
    if fault_active and fault.name == "sudden_density_jump":
        return fault.parameters["density_scale"]
    return 1.0


def _is_update_time(time_s: float, control_dt: float) -> bool:
    return bool(np.isclose(np.mod(time_s, control_dt), 0.0))


def _flap_effectiveness(mach: float, altitude_m: float, aero: AeroParams) -> float:
    from reentry_mpc.longitudinal import flap_effectiveness  # noqa: PLC0415

    return flap_effectiveness(mach, altitude_m, aero)
