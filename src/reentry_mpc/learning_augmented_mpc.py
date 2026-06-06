from __future__ import annotations

import time
from typing import Any

import casadi as ca
import numpy as np
import pandas as pd
import torch

from reentry_mpc.longitudinal import AeroParams, VehicleParams, flap_effectiveness
from reentry_mpc.nmpc import (
    NmpcConfig,
    _add_control_constraints,
    _add_state_constraints,
    _build_step_log,
    _initial_state_guess,
    _pad_horizon,
    nmpc_derivatives_numeric,
)
from reentry_mpc.phase9 import normalize_X
from reentry_mpc.phase10 import LoadedResidualModel


def build_horizon_residual_biases(
    *,
    loaded_model: LoadedResidualModel,
    state: np.ndarray,
    previous_flap_rad: float,
    horizon: pd.DataFrame,
    horizon_steps: int,
) -> tuple[np.ndarray, float]:
    #predict one residual q-dot bias for each horizon interval

    start = time.perf_counter()
    horizon = _pad_horizon(horizon, horizon_steps + 1)
    features: list[list[float]] = []
    for idx in range(horizon_steps):
        row = horizon.iloc[idx]
        if idx == 0:
            feature_state = state
        else:
            feature_state = np.array(
                [
                    row["alpha_ref_rad"],
                    row["q_ref_radps"],
                    row["theta_ref_rad"],
                ],
                dtype=float,
            )
        features.append(
            [
                float(feature_state[0]),
                float(feature_state[1]),
                float(feature_state[2]),
                float(previous_flap_rad),
                float(row["mach"]),
                float(row["altitude_m"]),
                float(row["velocity_mps"]),
                float(row["density_kgm3"]),
                float(row["dynamic_pressure_pa"]),
            ]
        )
    feature_array = np.asarray(features, dtype=np.float32)
    with torch.no_grad():
        x_norm = torch.tensor(
            normalize_X(feature_array, loaded_model.normalizer), dtype=torch.float32
        )
        y_norm = loaded_model.model(x_norm).detach().cpu().numpy()
    residual = (
        y_norm * loaded_model.normalizer["y_std"] + loaded_model.normalizer["y_mean"]
    )
    elapsed = time.perf_counter() - start
    return residual[:, 1].astype(float), elapsed


def biased_nmpc_derivatives_numeric(
    *,
    state: np.ndarray,
    delta_flap_rad: float,
    row: pd.Series,
    vehicle: VehicleParams,
    aero: AeroParams,
    residual_q_dot_bias: float,
) -> np.ndarray:
    nominal = nmpc_derivatives_numeric(
        state=state,
        delta_flap_rad=delta_flap_rad,
        row=row,
        vehicle=vehicle,
        aero=aero,
    )
    return nominal + np.array([0.0, residual_q_dot_bias, 0.0], dtype=float)


def solve_horizon_biased_nmpc_step(
    *,
    state: np.ndarray,
    previous_flap_rad: float,
    horizon: pd.DataFrame,
    vehicle: VehicleParams,
    aero: AeroParams,
    config: NmpcConfig,
    residual_q_dot_biases: np.ndarray,
) -> tuple[float, dict[str, Any]]:
    horizon = _pad_horizon(horizon, config.horizon_steps + 1)
    biases = _pad_biases(residual_q_dot_biases, config.horizon_steps)
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
        next_state = _biased_rk4_symbolic(
            state=x_var[:, k_idx],
            delta_flap_rad=u_var[0, k_idx],
            row=row,
            vehicle=vehicle,
            aero=aero,
            dt=config.dt,
            residual_q_dot_bias=float(biases[k_idx]),
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
    log.update(
        {
            "predicted_residual_q_dot_first": float(biases[0]),
            "predicted_residual_q_dot_mean": float(np.mean(biases)),
            "predicted_residual_q_dot_max_abs": float(np.max(np.abs(biases))),
        }
    )
    return first_control, log


def _biased_dynamics_symbolic(
    *,
    state: ca.MX,
    delta_flap_rad: ca.MX,
    row: pd.Series,
    vehicle: VehicleParams,
    aero: AeroParams,
    residual_q_dot_bias: float,
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
    q_dot = moment / vehicle.pitch_inertia_kgm2 + float(residual_q_dot_bias)
    alpha_dot = state[1] - 0.22 * state[0]
    theta_dot = state[1]
    return ca.vertcat(alpha_dot, q_dot, theta_dot)


def _biased_rk4_symbolic(
    *,
    state: ca.MX,
    delta_flap_rad: ca.MX,
    row: pd.Series,
    vehicle: VehicleParams,
    aero: AeroParams,
    dt: float,
    residual_q_dot_bias: float,
) -> ca.MX:
    k1 = _biased_dynamics_symbolic(
        state=state,
        delta_flap_rad=delta_flap_rad,
        row=row,
        vehicle=vehicle,
        aero=aero,
        residual_q_dot_bias=residual_q_dot_bias,
    )
    k2 = _biased_dynamics_symbolic(
        state=state + 0.5 * dt * k1,
        delta_flap_rad=delta_flap_rad,
        row=row,
        vehicle=vehicle,
        aero=aero,
        residual_q_dot_bias=residual_q_dot_bias,
    )
    k3 = _biased_dynamics_symbolic(
        state=state + 0.5 * dt * k2,
        delta_flap_rad=delta_flap_rad,
        row=row,
        vehicle=vehicle,
        aero=aero,
        residual_q_dot_bias=residual_q_dot_bias,
    )
    k4 = _biased_dynamics_symbolic(
        state=state + dt * k3,
        delta_flap_rad=delta_flap_rad,
        row=row,
        vehicle=vehicle,
        aero=aero,
        residual_q_dot_bias=residual_q_dot_bias,
    )
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _pad_biases(values: np.ndarray, horizon_steps: int) -> np.ndarray:
    biases = np.asarray(values, dtype=float).reshape(-1)
    if biases.size == 0:
        return np.zeros(horizon_steps, dtype=float)
    if biases.size >= horizon_steps:
        return biases[:horizon_steps]
    padding = np.full(horizon_steps - biases.size, biases[-1], dtype=float)
    return np.concatenate([biases, padding])
