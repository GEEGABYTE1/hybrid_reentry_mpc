
#linearization utilities

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from reentry_mpc.longitudinal import (
    AeroParams,
    VehicleParams,
    scheduled_pitching_moment,
)


@dataclass(frozen=True)
class LinearModel:
    a_discrete: np.ndarray
    b_discrete: np.ndarray
    a_continuous: np.ndarray
    b_continuous: np.ndarray


def derivatives_for_schedule(
    *,
    state: np.ndarray,
    delta_flap_rad: float,
    schedule: dict[str, float],
    vehicle: VehicleParams,
    aero: AeroParams,
) -> np.ndarray:
    
    moment_nm, _cm, _effectiveness = scheduled_pitching_moment(
        state=state,
        delta_flap_rad=delta_flap_rad,
        schedule=schedule,
        vehicle=vehicle,
        aero=aero,
    )
    q_dot = moment_nm / vehicle.pitch_inertia_kgm2
    alpha_dot = q_dot - 0.22 * state[0]
    theta_dot = state[1]
    return np.array([alpha_dot, q_dot, theta_dot], dtype=float)


def finite_difference_linearization(
    *,
    state: np.ndarray,
    delta_flap_rad: float,
    schedule: dict[str, float],
    vehicle: VehicleParams,
    aero: AeroParams,
    dt: float,
    state_eps: float = 1.0e-5,
    control_eps: float = 1.0e-5,
) -> LinearModel:
    
    #we are forwarding with Euler

    n_state = state.size
    a_continuous = np.zeros((n_state, n_state))
    for idx in range(n_state):
        perturb = np.zeros(n_state)
        perturb[idx] = state_eps
        f_plus = derivatives_for_schedule(
            state=state + perturb,
            delta_flap_rad=delta_flap_rad,
            schedule=schedule,
            vehicle=vehicle,
            aero=aero,
        )
        f_minus = derivatives_for_schedule(
            state=state - perturb,
            delta_flap_rad=delta_flap_rad,
            schedule=schedule,
            vehicle=vehicle,
            aero=aero,
        )
        a_continuous[:, idx] = (f_plus - f_minus) / (2.0 * state_eps)

    f_plus = derivatives_for_schedule(
        state=state,
        delta_flap_rad=delta_flap_rad + control_eps,
        schedule=schedule,
        vehicle=vehicle,
        aero=aero,
    )
    f_minus = derivatives_for_schedule(
        state=state,
        delta_flap_rad=delta_flap_rad - control_eps,
        schedule=schedule,
        vehicle=vehicle,
        aero=aero,
    )
    b_continuous = ((f_plus - f_minus) / (2.0 * control_eps)).reshape(n_state, 1)
    a_discrete = np.eye(n_state) + dt * a_continuous
    b_discrete = dt * b_continuous
    return LinearModel(
        a_discrete=a_discrete,
        b_discrete=b_discrete,
        a_continuous=a_continuous,
        b_continuous=b_continuous,
    )


def solve_discrete_lqr_gain(
    *,
    a_discrete: np.ndarray,
    b_discrete: np.ndarray,
    q_weight: np.ndarray,
    r_weight: np.ndarray,
    max_iterations: int = 500,
    tolerance: float = 1.0e-10,
) -> np.ndarray:
    

    p_matrix = q_weight.copy()
    for _iteration in range(max_iterations): #riccati iteration 
        gain_term = r_weight + b_discrete.T @ p_matrix @ b_discrete
        p_next = (
            a_discrete.T @ p_matrix @ a_discrete
            - a_discrete.T
            @ p_matrix
            @ b_discrete
            @ np.linalg.solve(gain_term, b_discrete.T @ p_matrix @ a_discrete)
            + q_weight
        )
        if np.max(np.abs(p_next - p_matrix)) < tolerance:
            p_matrix = p_next
            break
        p_matrix = p_next
    return np.linalg.solve(
        r_weight + b_discrete.T @ p_matrix @ b_discrete,
        b_discrete.T @ p_matrix @ a_discrete,
    )
