from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd

from reentry_mpc.linearization import derivatives_for_schedule
from reentry_mpc.longitudinal import (
    AeroParams,
    VehicleParams,
    scheduled_pitching_moment,
)


class ControllerProtocol(Protocol):
    # Controller interface used by the shared rollout runner.

    def reset(self) -> None:
        pass

    def command(
        self,
        *,
        state: np.ndarray,
        reference_state: np.ndarray,
        dt: float,
        schedule: dict[str, float],
    ) -> float:
        """Return raw flap command."""


def rollout_controller(
    *,
    controller_name: str,
    controller: ControllerProtocol,
    reference_profile: pd.DataFrame,
    vehicle: VehicleParams,
    aero: AeroParams,
    initial_state: np.ndarray,
) -> pd.DataFrame:

    controller.reset()
    state = initial_state.copy()
    previous_flap = 0.0
    rows: list[dict[str, float | bool | str]] = []
    time_values = reference_profile["time_s"].to_numpy(dtype=float)
    dt_values = np.diff(time_values, append=time_values[-1])
    if len(dt_values) > 1:
        dt_values[-1] = dt_values[-2]

    for idx, row in reference_profile.iterrows():
        dt = float(dt_values[idx])
        schedule = _schedule_from_row(row)
        reference_state = np.array(
            [
                row["alpha_ref_rad"],
                row["q_ref_radps"],
                row["theta_ref_rad"],
            ],
            dtype=float,
        )
        raw_command = controller.command(
            state=state,
            reference_state=reference_state,
            dt=dt,
            schedule=schedule,
        )
        limited_command, flap_saturated, rate_saturated = _apply_actuator_limits(
            raw_command=float(raw_command),
            previous_flap=previous_flap,
            dt=dt,
            row=row,
        )
        moment_nm, cm, effectiveness = scheduled_pitching_moment(
            state=state,
            delta_flap_rad=limited_command,
            schedule=schedule,
            vehicle=vehicle,
            aero=aero,
        )
        alpha_error = float(state[0] - row["alpha_ref_rad"])
        q_error = float(state[1] - row["q_ref_radps"])
        theta_error = float(state[2] - row["theta_ref_rad"])
        corridor_violation = bool(
            state[0] < row["alpha_min_rad"]
            or state[0] > row["alpha_max_rad"]
            or state[1] < row["q_min_radps"]
            or state[1] > row["q_max_radps"]
        )
        rows.append(
            {
                "controller": controller_name,
                "time_s": float(row["time_s"]),
                "alpha_rad": float(state[0]),
                "q_radps": float(state[1]),
                "theta_rad": float(state[2]),
                "alpha_ref_rad": float(row["alpha_ref_rad"]),
                "q_ref_radps": float(row["q_ref_radps"]),
                "theta_ref_rad": float(row["theta_ref_rad"]),
                "alpha_min_rad": float(row["alpha_min_rad"]),
                "alpha_max_rad": float(row["alpha_max_rad"]),
                "q_min_radps": float(row["q_min_radps"]),
                "q_max_radps": float(row["q_max_radps"]),
                "delta_flap_raw_rad": float(raw_command),
                "delta_flap_rad": float(limited_command),
                "delta_flap_rate_radps": float((limited_command - previous_flap) / dt),
                "flap_min_rad": float(row["flap_min_rad"]),
                "flap_max_rad": float(row["flap_max_rad"]),
                "flap_rate_min_radps": float(row["flap_rate_min_radps"]),
                "flap_rate_max_radps": float(row["flap_rate_max_radps"]),
                "flap_saturated": flap_saturated,
                "flap_rate_saturated": rate_saturated,
                "corridor_violation": corridor_violation,
                "alpha_error_rad": alpha_error,
                "q_error_rad": q_error,
                "theta_error_rad": theta_error,
                "pitching_moment_nm": moment_nm,
                "cm": cm,
                "flap_effectiveness": effectiveness,
                **schedule,
            }
        )
        previous_flap = limited_command
        state = _rk4_step_fixed_schedule(
            state=state,
            delta_flap_rad=limited_command,
            dt=dt,
            schedule=schedule,
            vehicle=vehicle,
            aero=aero,
        )
    return pd.DataFrame(rows)


def _schedule_from_row(row: pd.Series) -> dict[str, float]:
    return {
        "altitude_m": float(row["altitude_m"]),
        "velocity_mps": float(row["velocity_mps"]),
        "mach": float(row["mach"]),
        "density_kgm3": float(row["density_kgm3"]),
        "dynamic_pressure_pa": float(row["dynamic_pressure_pa"]),
    }


def _apply_actuator_limits(
    *,
    raw_command: float,
    previous_flap: float,
    dt: float,
    row: pd.Series,
) -> tuple[float, bool, bool]:
    min_step = float(row["flap_rate_min_radps"]) * dt
    max_step = float(row["flap_rate_max_radps"]) * dt
    requested_step = raw_command - previous_flap
    limited_step = float(np.clip(requested_step, min_step, max_step))
    rate_limited_command = previous_flap + limited_step
    limited_command = float(
        np.clip(rate_limited_command, row["flap_min_rad"], row["flap_max_rad"])
    )
    flap_saturated = not np.isclose(limited_command, rate_limited_command)
    rate_saturated = not np.isclose(limited_step, requested_step)
    return limited_command, bool(flap_saturated), bool(rate_saturated)


def _rk4_step_fixed_schedule(
    *,
    state: np.ndarray,
    delta_flap_rad: float,
    dt: float,
    schedule: dict[str, float],
    vehicle: VehicleParams,
    aero: AeroParams,
) -> np.ndarray:
    k1 = derivatives_for_schedule(
        state=state,
        delta_flap_rad=delta_flap_rad,
        schedule=schedule,
        vehicle=vehicle,
        aero=aero,
    )
    k2 = derivatives_for_schedule(
        state=state + 0.5 * dt * k1,
        delta_flap_rad=delta_flap_rad,
        schedule=schedule,
        vehicle=vehicle,
        aero=aero,
    )
    k3 = derivatives_for_schedule(
        state=state + 0.5 * dt * k2,
        delta_flap_rad=delta_flap_rad,
        schedule=schedule,
        vehicle=vehicle,
        aero=aero,
    )
    k4 = derivatives_for_schedule(
        state=state + dt * k3,
        delta_flap_rad=delta_flap_rad,
        schedule=schedule,
        vehicle=vehicle,
        aero=aero,
    )
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
