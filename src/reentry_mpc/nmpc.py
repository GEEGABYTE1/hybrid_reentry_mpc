# nominal nonlinear mpc utilities using casdi
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import casadi as ca
import numpy as np
import pandas as pd

from reentry_mpc.longitudinal import AeroParams, VehicleParams, flap_effectiveness


@dataclass(frozen=True)
class NmpcWeights:
    alpha: float
    q: float
    theta: float
    control: float
    flap_rate: float
    terminal_alpha: float
    state_slack: float


@dataclass(frozen=True)
class NmpcSolverOptions:

    max_iter: int
    acceptable_tol: float
    print_level: int


@dataclass(frozen=True)
class NmpcConfig:
    horizon_steps: int
    dt: float
    weights: NmpcWeights
    solver: NmpcSolverOptions


def solve_nmpc_step(
    *,
    state: np.ndarray,
    previous_flap_rad: float,
    horizon: pd.DataFrame,
    vehicle: VehicleParams,
    aero: AeroParams,
    config: NmpcConfig,
) -> tuple[float, dict[str, Any]]:
    # solving one finite-horizon NMPC problem and return the first control
    horizon = _pad_horizon(horizon, config.horizon_steps + 1)
    opti = ca.Opti()
    x_var = opti.variable(3, config.horizon_steps + 1)
    u_var = opti.variable(1, config.horizon_steps)
    alpha_slack = opti.variable(2, config.horizon_steps + 1)
    q_slack = opti.variable(2, config.horizon_steps + 1)
    opti.subject_to(x_var[:, 0] == state)
    opti.subject_to(ca.vec(alpha_slack) >= 0)
    opti.subject_to(ca.vec(q_slack) >= 0)

    objective = 0
    for k_idx in range(config.horizon_steps):
        row = horizon.iloc[k_idx]
        ref = ca.vertcat(row["alpha_ref_rad"], row["q_ref_radps"], row["theta_ref_rad"])
        error = x_var[:, k_idx] - ref
        objective += (
            config.weights.alpha * error[0] ** 2
            + config.weights.q * error[1] ** 2
            + config.weights.theta * error[2] ** 2
            + config.weights.control * u_var[0, k_idx] ** 2
        )
        previous_u = previous_flap_rad if k_idx == 0 else u_var[0, k_idx - 1]
        delta_u = u_var[0, k_idx] - previous_u
        objective += config.weights.flap_rate * delta_u**2

        next_state = _rk4_symbolic(
            x_var[:, k_idx],
            u_var[0, k_idx],
            row,
            vehicle,
            aero,
            config.dt,
        )
        opti.subject_to(x_var[:, k_idx + 1] == next_state)
        objective += config.weights.state_slack * (
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
        _add_control_constraints(opti, u_var[0, k_idx], delta_u, row, config.dt)

    terminal_row = horizon.iloc[config.horizon_steps]
    terminal_ref = ca.vertcat(
        terminal_row["alpha_ref_rad"],
        terminal_row["q_ref_radps"],
        terminal_row["theta_ref_rad"],
    )
    terminal_error = x_var[:, config.horizon_steps] - terminal_ref
    objective += (
        config.weights.terminal_alpha * terminal_error[0] ** 2
        + config.weights.q * terminal_error[1] ** 2
        + config.weights.theta * terminal_error[2] ** 2
    )
    objective += config.weights.state_slack * (
        alpha_slack[0, config.horizon_steps] ** 2
        + alpha_slack[1, config.horizon_steps] ** 2
        + q_slack[0, config.horizon_steps] ** 2
        + q_slack[1, config.horizon_steps] ** 2
    )
    _add_state_constraints(
        opti,
        x_var[:, config.horizon_steps],
        terminal_row,
        alpha_slack[:, config.horizon_steps],
        q_slack[:, config.horizon_steps],
    )

    opti.minimize(objective)
    opti.set_initial(x_var, _initial_state_guess(state, horizon, config.horizon_steps))
    opti.set_initial(u_var, previous_flap_rad)
    opti.solver(
        "ipopt",
        {"print_time": False, "error_on_fail": False},
        {
            "max_iter": config.solver.max_iter,
            "acceptable_tol": config.solver.acceptable_tol,
            "print_level": config.solver.print_level,
            "sb": "yes",
        },
    )

    start = time.perf_counter()
    solution = opti.solve()
    solve_time = time.perf_counter() - start
    status = opti.stats().get("return_status", "unknown")
    first_control = float(solution.value(u_var[0, 0]))
    objective_value = float(solution.value(objective))
    predicted_state = np.array(solution.value(x_var), dtype=float)
    predicted_control = np.array(solution.value(u_var), dtype=float).reshape(-1)
    log = _build_step_log(
        status=status,
        solve_time=solve_time,
        objective_value=objective_value,
        first_control=first_control,
        predicted_state=predicted_state,
        predicted_control=predicted_control,
        previous_flap_rad=previous_flap_rad,
        horizon=horizon,
        dt=config.dt,
    )
    return first_control, log


def apply_flap_limits(
    *, raw_command: float, previous_flap_rad: float, row: pd.Series, dt: float
) -> tuple[float, bool, bool]:
    min_step = float(row["flap_rate_min_radps"]) * dt
    max_step = float(row["flap_rate_max_radps"]) * dt
    requested_step = raw_command - previous_flap_rad
    limited_step = float(np.clip(requested_step, min_step, max_step))
    rate_limited = previous_flap_rad + limited_step
    limited = float(np.clip(rate_limited, row["flap_min_rad"], row["flap_max_rad"]))
    return (
        limited,
        bool(not np.isclose(limited, rate_limited)),
        bool(not np.isclose(limited_step, requested_step)),
    )


def nmpc_derivatives_numeric(
    *,
    state: np.ndarray,
    delta_flap_rad: float,
    row: pd.Series,
    vehicle: VehicleParams,
    aero: AeroParams,
) -> np.ndarray:
    """Numeric dynamics matching the symbolic NMPC dynamics."""

    effectiveness = flap_effectiveness(
        float(row["mach"]), float(row["altitude_m"]), aero
    )
    alpha_slope = aero.cm_alpha + aero.cm_alpha_mach_slope * (float(row["mach"]) - 15.0)
    q_hat = (
        state[1]
        * vehicle.reference_length_m
        / max(2.0 * float(row["velocity_mps"]), 1.0)
    )
    cm = (
        aero.cm0
        + alpha_slope * state[0]
        + aero.cm_q * q_hat
        + aero.cm_delta * effectiveness * delta_flap_rad
    )
    moment = (
        float(row["dynamic_pressure_pa"])
        * vehicle.reference_area_m2
        * vehicle.reference_length_m
        * cm
    )
    q_dot = moment / vehicle.pitch_inertia_kgm2
    alpha_dot = state[1] - 0.22 * state[0]
    theta_dot = state[1]
    return np.array([alpha_dot, q_dot, theta_dot], dtype=float)


def rk4_step_numeric(
    *,
    state: np.ndarray,
    delta_flap_rad: float,
    row: pd.Series,
    vehicle: VehicleParams,
    aero: AeroParams,
    dt: float,
) -> np.ndarray:
    # rk4
    k1 = nmpc_derivatives_numeric(
        state=state, delta_flap_rad=delta_flap_rad, row=row, vehicle=vehicle, aero=aero
    )
    k2 = nmpc_derivatives_numeric(
        state=state + 0.5 * dt * k1,
        delta_flap_rad=delta_flap_rad,
        row=row,
        vehicle=vehicle,
        aero=aero,
    )
    k3 = nmpc_derivatives_numeric(
        state=state + 0.5 * dt * k2,
        delta_flap_rad=delta_flap_rad,
        row=row,
        vehicle=vehicle,
        aero=aero,
    )
    k4 = nmpc_derivatives_numeric(
        state=state + dt * k3,
        delta_flap_rad=delta_flap_rad,
        row=row,
        vehicle=vehicle,
        aero=aero,
    )
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _dynamics_symbolic(
    state: ca.MX,
    delta_flap_rad: ca.MX,
    row: pd.Series,
    vehicle: VehicleParams,
    aero: AeroParams,
) -> ca.MX:
    effectiveness = flap_effectiveness(
        float(row["mach"]), float(row["altitude_m"]), aero
    )
    alpha_slope = aero.cm_alpha + aero.cm_alpha_mach_slope * (float(row["mach"]) - 15.0)
    q_hat = (
        state[1]
        * vehicle.reference_length_m
        / max(2.0 * float(row["velocity_mps"]), 1.0)
    )
    cm = (
        aero.cm0
        + alpha_slope * state[0]
        + aero.cm_q * q_hat
        + aero.cm_delta * effectiveness * delta_flap_rad
    )
    moment = (
        float(row["dynamic_pressure_pa"])
        * vehicle.reference_area_m2
        * vehicle.reference_length_m
        * cm
    )
    q_dot = moment / vehicle.pitch_inertia_kgm2
    alpha_dot = state[1] - 0.22 * state[0]
    theta_dot = state[1]
    return ca.vertcat(alpha_dot, q_dot, theta_dot)


def _rk4_symbolic(
    state: ca.MX,
    delta_flap_rad: ca.MX,
    row: pd.Series,
    vehicle: VehicleParams,
    aero: AeroParams,
    dt: float,
) -> ca.MX:
    k1 = _dynamics_symbolic(state, delta_flap_rad, row, vehicle, aero)
    k2 = _dynamics_symbolic(state + 0.5 * dt * k1, delta_flap_rad, row, vehicle, aero)
    k3 = _dynamics_symbolic(state + 0.5 * dt * k2, delta_flap_rad, row, vehicle, aero)
    k4 = _dynamics_symbolic(state + dt * k3, delta_flap_rad, row, vehicle, aero)
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


def _add_control_constraints(
    opti: ca.Opti, control: ca.MX, delta_u: ca.MX, row: pd.Series, dt: float
) -> None:
    opti.subject_to(control >= float(row["flap_min_rad"]))
    opti.subject_to(control <= float(row["flap_max_rad"]))
    opti.subject_to(delta_u >= float(row["flap_rate_min_radps"]) * dt)
    opti.subject_to(delta_u <= float(row["flap_rate_max_radps"]) * dt)


def _pad_horizon(horizon: pd.DataFrame, required_rows: int) -> pd.DataFrame:
    if len(horizon) >= required_rows:
        return horizon.iloc[:required_rows].reset_index(drop=True)
    last_row = horizon.iloc[[-1]]
    padding = [last_row] * (required_rows - len(horizon))
    return pd.concat([horizon, *padding], ignore_index=True)


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


def _build_step_log(
    *,
    status: str,
    solve_time: float,
    objective_value: float,
    first_control: float,
    predicted_state: np.ndarray,
    predicted_control: np.ndarray,
    previous_flap_rad: float,
    horizon: pd.DataFrame,
    dt: float,
) -> dict[str, Any]:
    alpha_min = horizon["alpha_min_rad"].iloc[: predicted_state.shape[1]].to_numpy()
    alpha_max = horizon["alpha_max_rad"].iloc[: predicted_state.shape[1]].to_numpy()
    q_min = horizon["q_min_radps"].iloc[: predicted_state.shape[1]].to_numpy()
    q_max = horizon["q_max_radps"].iloc[: predicted_state.shape[1]].to_numpy()
    alpha_values = predicted_state[0, :]
    q_values = predicted_state[1, :]
    alpha_lower_violation = np.maximum(alpha_min - alpha_values, 0.0)
    alpha_upper_violation = np.maximum(alpha_values - alpha_max, 0.0)
    q_lower_violation = np.maximum(q_min - q_values, 0.0)
    q_upper_violation = np.maximum(q_values - q_max, 0.0)
    flap_min = horizon["flap_min_rad"].iloc[: len(predicted_control)].to_numpy()
    flap_max = horizon["flap_max_rad"].iloc[: len(predicted_control)].to_numpy()
    flap_lower_violation = np.maximum(flap_min - predicted_control, 0.0)
    flap_upper_violation = np.maximum(predicted_control - flap_max, 0.0)
    previous_controls = np.concatenate([[previous_flap_rad], predicted_control[:-1]])
    flap_rate = (predicted_control - previous_controls) / dt
    rate_min = horizon["flap_rate_min_radps"].iloc[: len(predicted_control)].to_numpy()
    rate_max = horizon["flap_rate_max_radps"].iloc[: len(predicted_control)].to_numpy()
    rate_lower_violation = np.maximum(rate_min - flap_rate, 0.0)
    rate_upper_violation = np.maximum(flap_rate - rate_max, 0.0)
    tolerance = 1.0e-4
    return {
        "solver_status": status,
        "solve_time_s": solve_time,
        "objective_value": objective_value,
        "first_control_action_rad": first_control,
        "max_alpha_violation_rad": float(
            max(alpha_lower_violation.max(), alpha_upper_violation.max())
        ),
        "max_q_violation_radps": float(
            max(q_lower_violation.max(), q_upper_violation.max())
        ),
        "max_flap_violation_rad": float(
            max(flap_lower_violation.max(), flap_upper_violation.max())
        ),
        "max_flap_rate_violation_rad": float(
            max(rate_lower_violation.max(), rate_upper_violation.max())
        ),
        "alpha_constraint_active": bool(
            np.any(
                np.isclose(alpha_values, alpha_min, atol=tolerance)
                | np.isclose(alpha_values, alpha_max, atol=tolerance)
            )
        ),
        "q_constraint_active": bool(
            np.any(
                np.isclose(q_values, q_min, atol=tolerance)
                | np.isclose(q_values, q_max, atol=tolerance)
            )
        ),
        "flap_saturated": bool(
            np.any(
                np.isclose(predicted_control, flap_min, atol=tolerance)
                | np.isclose(predicted_control, flap_max, atol=tolerance)
            )
        ),
        "flap_rate_saturated": bool(
            np.any(
                np.isclose(flap_rate, rate_min, atol=tolerance)
                | np.isclose(flap_rate, rate_max, atol=tolerance)
            )
        ),
    }
