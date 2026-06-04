from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import casadi as ca
import numpy as np
import pandas as pd

from reentry_mpc.longitudinal import AeroParams, VehicleParams, flap_effectiveness
from reentry_mpc.nmpc import (
    NmpcConfig,
    NmpcSolverOptions,
    NmpcWeights,
    _add_control_constraints,
    _build_step_log,
    _pad_horizon,
)


@dataclass(frozen=True)
class DesignScenario:
    name: str
    density_scale: float
    cm_alpha_scale: float
    cm_delta_scale: float
    cm_q_scale: float
    external_disturbance_moment_nm: float


@dataclass(frozen=True)
class ScenarioMpcConfig:
    nmpc: NmpcConfig
    design_scenarios: list[DesignScenario]
    max_scenarios_per_tier: int | None


def load_scenario_mpc_config(raw: dict[str, Any]) -> ScenarioMpcConfig:
    weights = NmpcWeights(
        **{key: float(value) for key, value in raw["weights"].items()}
    )
    solver = NmpcSolverOptions(
        max_iter=int(raw["solver"]["max_iter"]),
        acceptable_tol=float(raw["solver"]["acceptable_tol"]),
        print_level=int(raw["solver"]["print_level"]),
    )
    return ScenarioMpcConfig(
        nmpc=NmpcConfig(
            horizon_steps=int(raw["horizon_steps"]),
            dt=float(raw["control_dt_s"]),
            weights=weights,
            solver=solver,
        ),
        design_scenarios=[
            DesignScenario(
                name=str(item["name"]),
                density_scale=float(item["density_scale"]),
                cm_alpha_scale=float(item["cm_alpha_scale"]),
                cm_delta_scale=float(item["cm_delta_scale"]),
                cm_q_scale=float(item["cm_q_scale"]),
                external_disturbance_moment_nm=float(
                    item["external_disturbance_moment_nm"]
                ),
            )
            for item in raw["design_scenarios"]
        ],
        max_scenarios_per_tier=(
            None
            if raw.get("max_scenarios_per_tier") is None
            else int(raw["max_scenarios_per_tier"])
        ),
    )


def solve_scenario_mpc_step(
    *,
    state: np.ndarray,
    previous_flap_rad: float,
    horizon: pd.DataFrame,
    vehicle: VehicleParams,
    aero: AeroParams,
    config: ScenarioMpcConfig,
) -> tuple[float, dict[str, Any]]:
    
    horizon = _pad_horizon(horizon, config.nmpc.horizon_steps + 1)
    opti = ca.Opti()
    u_var = opti.variable(1, config.nmpc.horizon_steps)
    x_vars = [
        opti.variable(3, config.nmpc.horizon_steps + 1) for _ in config.design_scenarios
    ]
    alpha_slacks = [
        opti.variable(2, config.nmpc.horizon_steps + 1) for _ in config.design_scenarios
    ]
    q_slacks = [
        opti.variable(2, config.nmpc.horizon_steps + 1) for _ in config.design_scenarios
    ]

    objective = 0
    for scenario_idx, design in enumerate(config.design_scenarios):
        x_var = x_vars[scenario_idx]
        alpha_slack = alpha_slacks[scenario_idx]
        q_slack = q_slacks[scenario_idx]
        opti.subject_to(x_var[:, 0] == state)
        opti.subject_to(ca.vec(alpha_slack) >= 0)
        opti.subject_to(ca.vec(q_slack) >= 0)
        for k_idx in range(config.nmpc.horizon_steps):
            row = horizon.iloc[k_idx]
            ref = ca.vertcat(
                row["alpha_ref_rad"], row["q_ref_radps"], row["theta_ref_rad"]
            )
            error = x_var[:, k_idx] - ref
            objective += (
                config.nmpc.weights.alpha * error[0] ** 2
                + config.nmpc.weights.q * error[1] ** 2
                + config.nmpc.weights.theta * error[2] ** 2
            ) / len(config.design_scenarios)
            next_state = _scenario_rk4_symbolic(
                x_var[:, k_idx],
                u_var[0, k_idx],
                row,
                vehicle,
                aero,
                design,
                config.nmpc.dt,
            )
            opti.subject_to(x_var[:, k_idx + 1] == next_state)
            objective += config.nmpc.weights.state_slack * (
                alpha_slack[0, k_idx] ** 2
                + alpha_slack[1, k_idx] ** 2
                + q_slack[0, k_idx] ** 2
                + q_slack[1, k_idx] ** 2
            )
            _add_state_constraints(
                opti,
                x_var[:, k_idx],
                row,
                alpha_slack[:, k_idx],
                q_slack[:, k_idx],
            )

        terminal_row = horizon.iloc[config.nmpc.horizon_steps]
        terminal_ref = ca.vertcat(
            terminal_row["alpha_ref_rad"],
            terminal_row["q_ref_radps"],
            terminal_row["theta_ref_rad"],
        )
        terminal_error = x_var[:, config.nmpc.horizon_steps] - terminal_ref
        objective += (
            config.nmpc.weights.terminal_alpha * terminal_error[0] ** 2
            + config.nmpc.weights.q * terminal_error[1] ** 2
            + config.nmpc.weights.theta * terminal_error[2] ** 2
        ) / len(config.design_scenarios)
        _add_state_constraints(
            opti,
            x_var[:, config.nmpc.horizon_steps],
            terminal_row,
            alpha_slack[:, config.nmpc.horizon_steps],
            q_slack[:, config.nmpc.horizon_steps],
        )

    for k_idx in range(config.nmpc.horizon_steps):
        row = horizon.iloc[k_idx]
        previous_u = previous_flap_rad if k_idx == 0 else u_var[0, k_idx - 1]
        delta_u = u_var[0, k_idx] - previous_u
        objective += (
            config.nmpc.weights.control * u_var[0, k_idx] ** 2
            + config.nmpc.weights.flap_rate * delta_u**2
        )
        _add_control_constraints(opti, u_var[0, k_idx], delta_u, row, config.nmpc.dt)

    opti.minimize(objective)
    for x_var in x_vars:
        opti.set_initial(
            x_var, _initial_state_guess(state, horizon, config.nmpc.horizon_steps)
        )
    opti.set_initial(u_var, previous_flap_rad)
    opti.solver(
        "ipopt",
        {"print_time": False, "error_on_fail": False},
        {
            "max_iter": config.nmpc.solver.max_iter,
            "acceptable_tol": config.nmpc.solver.acceptable_tol,
            "print_level": config.nmpc.solver.print_level,
            "sb": "yes",
        },
    )

    start = time.perf_counter()
    solution = opti.solve()
    solve_time = time.perf_counter() - start
    status = opti.stats().get("return_status", "unknown")
    first_control = float(solution.value(u_var[0, 0]))
    objective_value = float(solution.value(objective))
    predicted_control = np.array(solution.value(u_var), dtype=float).reshape(-1)
    predicted_states = [
        np.array(solution.value(x_var), dtype=float) for x_var in x_vars
    ]
    nominal_state = predicted_states[0]
    log = _build_step_log(
        status=status,
        solve_time=solve_time,
        objective_value=objective_value,
        first_control=first_control,
        predicted_state=nominal_state,
        predicted_control=predicted_control,
        previous_flap_rad=previous_flap_rad,
        horizon=horizon,
        dt=config.nmpc.dt,
    )
    log.update(_scenario_log(predicted_states, horizon, config.design_scenarios))
    return first_control, log


def _scenario_dynamics_symbolic(
    state: ca.MX,
    delta_flap_rad: ca.MX,
    row: pd.Series,
    vehicle: VehicleParams,
    aero: AeroParams,
    design: DesignScenario,
) -> ca.MX:
    effectiveness = flap_effectiveness(
        float(row["mach"]), float(row["altitude_m"]), aero
    )
    alpha_slope = aero.cm_alpha * design.cm_alpha_scale + aero.cm_alpha_mach_slope * (
        float(row["mach"]) - 15.0
    )
    q_hat = (
        state[1]
        * vehicle.reference_length_m
        / max(2.0 * float(row["velocity_mps"]), 1.0)
    )
    cm = (
        aero.cm0
        + alpha_slope * state[0]
        + aero.cm_q * design.cm_q_scale * q_hat
        + aero.cm_delta * design.cm_delta_scale * effectiveness * delta_flap_rad
    )
    moment = (
        float(row["dynamic_pressure_pa"])
        * design.density_scale
        * vehicle.reference_area_m2
        * vehicle.reference_length_m
        * cm
        + design.external_disturbance_moment_nm
    )
    q_dot = moment / vehicle.pitch_inertia_kgm2
    alpha_dot = state[1] - 0.22 * state[0]
    theta_dot = state[1]
    return ca.vertcat(alpha_dot, q_dot, theta_dot)


def _scenario_rk4_symbolic(
    state: ca.MX,
    delta_flap_rad: ca.MX,
    row: pd.Series,
    vehicle: VehicleParams,
    aero: AeroParams,
    design: DesignScenario,
    dt: float,
) -> ca.MX:
    k1 = _scenario_dynamics_symbolic(state, delta_flap_rad, row, vehicle, aero, design)
    k2 = _scenario_dynamics_symbolic(
        state + 0.5 * dt * k1, delta_flap_rad, row, vehicle, aero, design
    )
    k3 = _scenario_dynamics_symbolic(
        state + 0.5 * dt * k2, delta_flap_rad, row, vehicle, aero, design
    )
    k4 = _scenario_dynamics_symbolic(
        state + dt * k3, delta_flap_rad, row, vehicle, aero, design
    )
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _add_state_constraints(
    opti: ca.Opti,
    state: ca.MX,
    row: pd.Series,
    alpha_slack: ca.MX,
    q_slack: ca.MX,
) -> None:
    opti.subject_to(state[0] + alpha_slack[0] >= float(row["alpha_min_rad"]))
    opti.subject_to(state[0] - alpha_slack[1] <= float(row["alpha_max_rad"]))
    opti.subject_to(state[1] + q_slack[0] >= float(row["q_min_radps"]))
    opti.subject_to(state[1] - q_slack[1] <= float(row["q_max_radps"]))


def _initial_state_guess(
    state: np.ndarray, horizon: pd.DataFrame, horizon_steps: int
) -> np.ndarray:
    guess = np.zeros((3, horizon_steps + 1))
    guess[:, 0] = state
    for idx in range(1, horizon_steps + 1):
        row = horizon.iloc[idx]
        guess[:, idx] = [
            row["alpha_ref_rad"],
            row["q_ref_radps"],
            row["theta_ref_rad"],
        ]
    return guess


def _scenario_log(
    predicted_states: list[np.ndarray],
    horizon: pd.DataFrame,
    design_scenarios: list[DesignScenario],
) -> dict[str, Any]:
    alpha_min = horizon["alpha_min_rad"].iloc[: predicted_states[0].shape[1]].to_numpy()
    alpha_max = horizon["alpha_max_rad"].iloc[: predicted_states[0].shape[1]].to_numpy()
    max_alpha_violation = 0.0
    worst_name = design_scenarios[0].name
    for state, design in zip(predicted_states, design_scenarios, strict=True):
        alpha_values = state[0, :]
        lower = np.maximum(alpha_min - alpha_values, 0.0)
        upper = np.maximum(alpha_values - alpha_max, 0.0)
        scenario_violation = float(max(lower.max(), upper.max()))
        if scenario_violation >= max_alpha_violation:
            max_alpha_violation = scenario_violation
            worst_name = design.name
    return {
        "design_scenario_count": len(design_scenarios),
        "max_design_scenario_alpha_violation_rad": max_alpha_violation,
        "worst_design_scenario": worst_name,
    }
