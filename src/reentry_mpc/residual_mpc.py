from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import casadi as ca
import numpy as np
import pandas as pd

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
from reentry_mpc.phase8 import FEATURE_NAMES, TARGET_NAMES


@dataclass(frozen=True)
class ResidualSurrogate:
    feature_mode: str
    feature_names: list[str]
    target_name: str
    x_mean: list[float]
    x_std: list[float]
    coefficients: list[float]
    ridge_lambda: float

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def fit_residual_surrogate(
    *,
    dataset_dir: str | Path,
    ridge_lambda: float,
    feature_mode: str,
) -> ResidualSurrogate:
    train = np.load(Path(dataset_dir) / "train.npz", allow_pickle=False)
    X = train["X"].astype(float)
    y_all = train["y"].astype(float)
    target_idx = TARGET_NAMES.index("residual_q_dot")
    y = y_all[:, target_idx]
    x_mean = X.mean(axis=0)
    x_std = np.maximum(X.std(axis=0), 1.0e-8)
    Xn = (X - x_mean) / x_std
    design = polynomial_design_numpy(Xn, feature_mode)
    penalty = float(ridge_lambda) * np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    coeffs = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return ResidualSurrogate(
        feature_mode=feature_mode,
        feature_names=list(FEATURE_NAMES),
        target_name="residual_q_dot",
        x_mean=x_mean.astype(float).tolist(),
        x_std=x_std.astype(float).tolist(),
        coefficients=coeffs.astype(float).tolist(),
        ridge_lambda=float(ridge_lambda),
    )


def save_residual_surrogate(surrogate: ResidualSurrogate, path: str | Path) -> Path:
    output = Path(path)
    output.write_text(
        json.dumps(surrogate.to_json_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output


def load_residual_surrogate(path: str | Path) -> ResidualSurrogate:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return ResidualSurrogate(
        feature_mode=str(payload["feature_mode"]),
        feature_names=[str(value) for value in payload["feature_names"]],
        target_name=str(payload["target_name"]),
        x_mean=[float(value) for value in payload["x_mean"]],
        x_std=[float(value) for value in payload["x_std"]],
        coefficients=[float(value) for value in payload["coefficients"]],
        ridge_lambda=float(payload["ridge_lambda"]),
    )


def polynomial_design_numpy(Xn: np.ndarray, feature_mode: str) -> np.ndarray:
    if feature_mode not in {"linear", "quadratic"}:
        raise ValueError(f"Unsupported residual surrogate feature mode: {feature_mode}")
    parts = [np.ones((Xn.shape[0], 1)), Xn]
    if feature_mode == "quadratic":
        parts.append(Xn**2)
    return np.concatenate(parts, axis=1)


def predict_residual_qdot_numpy(
    *, features: np.ndarray, surrogate: ResidualSurrogate
) -> np.ndarray:
    Xn = (features - np.asarray(surrogate.x_mean)) / np.asarray(surrogate.x_std)
    design = polynomial_design_numpy(np.atleast_2d(Xn), surrogate.feature_mode)
    return design @ np.asarray(surrogate.coefficients)


def predict_residual_qdot_symbolic(
    *,
    state: ca.MX,
    delta_flap_rad: ca.MX,
    row: pd.Series,
    surrogate: ResidualSurrogate,
) -> ca.MX:
    raw_features = [
        state[0],
        state[1],
        state[2],
        delta_flap_rad,
        float(row["mach"]),
        float(row["altitude_m"]),
        float(row["velocity_mps"]),
        float(row["density_kgm3"]),
        float(row["dynamic_pressure_pa"]),
    ]
    normalized = [
        (value - surrogate.x_mean[idx]) / surrogate.x_std[idx]
        for idx, value in enumerate(raw_features)
    ]
    coeffs = surrogate.coefficients
    expression = coeffs[0]
    offset = 1
    for idx, value in enumerate(normalized):
        expression += coeffs[offset + idx] * value
    if surrogate.feature_mode == "quadratic":
        offset += len(normalized)
        for idx, value in enumerate(normalized):
            expression += coeffs[offset + idx] * value**2
    return expression


def residual_augmented_derivatives_numeric(
    *,
    state: np.ndarray,
    delta_flap_rad: float,
    row: pd.Series,
    vehicle: VehicleParams,
    aero: AeroParams,
    surrogate: ResidualSurrogate,
    residual_gain: float,
) -> np.ndarray:
    nominal = nmpc_derivatives_numeric(
        state=state,
        delta_flap_rad=delta_flap_rad,
        row=row,
        vehicle=vehicle,
        aero=aero,
    )
    features = np.array(
        [
            state[0],
            state[1],
            state[2],
            delta_flap_rad,
            float(row["mach"]),
            float(row["altitude_m"]),
            float(row["velocity_mps"]),
            float(row["density_kgm3"]),
            float(row["dynamic_pressure_pa"]),
        ],
        dtype=float,
    )
    residual_qdot = float(
        predict_residual_qdot_numpy(features=features, surrogate=surrogate)[0]
    )
    return nominal + np.array([0.0, residual_gain * residual_qdot, 0.0], dtype=float)


def solve_residual_mpc_step(
    *,
    state: np.ndarray,
    previous_flap_rad: float,
    horizon: pd.DataFrame,
    vehicle: VehicleParams,
    aero: AeroParams,
    config: NmpcConfig,
    surrogate: ResidualSurrogate,
    residual_gain: float,
) -> tuple[float, dict[str, Any]]:
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
    residual_values: list[ca.MX] = []
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
        next_state, residual_qdot = _residual_rk4_symbolic(
            state=x_var[:, k_idx],
            delta_flap_rad=u_var[0, k_idx],
            row=row,
            vehicle=vehicle,
            aero=aero,
            dt=config.dt,
            surrogate=surrogate,
            residual_gain=residual_gain,
        )
        residual_values.append(residual_qdot)
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
    residual_numeric = np.array(
        [float(solution.value(value)) for value in residual_values], dtype=float
    )
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
            "residual_gain": float(residual_gain),
            "predicted_residual_q_dot_first": float(residual_numeric[0]),
            "predicted_residual_q_dot_mean": float(residual_numeric.mean()),
            "predicted_residual_q_dot_max_abs": float(np.abs(residual_numeric).max()),
        }
    )
    return first_control, log


def _residual_dynamics_symbolic(
    *,
    state: ca.MX,
    delta_flap_rad: ca.MX,
    row: pd.Series,
    vehicle: VehicleParams,
    aero: AeroParams,
    surrogate: ResidualSurrogate,
    residual_gain: float,
) -> tuple[ca.MX, ca.MX]:
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
    q_dot_nominal = moment / vehicle.pitch_inertia_kgm2
    residual_qdot = predict_residual_qdot_symbolic(
        state=state,
        delta_flap_rad=delta_flap_rad,
        row=row,
        surrogate=surrogate,
    )
    alpha_dot = state[1] - 0.22 * state[0]
    q_dot = q_dot_nominal + float(residual_gain) * residual_qdot
    theta_dot = state[1]
    return ca.vertcat(alpha_dot, q_dot, theta_dot), residual_qdot


def _residual_rk4_symbolic(
    *,
    state: ca.MX,
    delta_flap_rad: ca.MX,
    row: pd.Series,
    vehicle: VehicleParams,
    aero: AeroParams,
    dt: float,
    surrogate: ResidualSurrogate,
    residual_gain: float,
) -> tuple[ca.MX, ca.MX]:
    k1, residual_qdot = _residual_dynamics_symbolic(
        state=state,
        delta_flap_rad=delta_flap_rad,
        row=row,
        vehicle=vehicle,
        aero=aero,
        surrogate=surrogate,
        residual_gain=residual_gain,
    )
    k2, _ = _residual_dynamics_symbolic(
        state=state + 0.5 * dt * k1,
        delta_flap_rad=delta_flap_rad,
        row=row,
        vehicle=vehicle,
        aero=aero,
        surrogate=surrogate,
        residual_gain=residual_gain,
    )
    k3, _ = _residual_dynamics_symbolic(
        state=state + 0.5 * dt * k2,
        delta_flap_rad=delta_flap_rad,
        row=row,
        vehicle=vehicle,
        aero=aero,
        surrogate=surrogate,
        residual_gain=residual_gain,
    )
    k4, _ = _residual_dynamics_symbolic(
        state=state + dt * k3,
        delta_flap_rad=delta_flap_rad,
        row=row,
        vehicle=vehicle,
        aero=aero,
        surrogate=surrogate,
        residual_gain=residual_gain,
    )
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4), residual_qdot
