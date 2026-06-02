# PID and gain-scheduled LQR baseline controllers.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from reentry_mpc.linearization import (
    finite_difference_linearization,
    solve_discrete_lqr_gain,
)
from reentry_mpc.longitudinal import AeroParams, VehicleParams


@dataclass
class PIDController:
    kp_alpha: float
    ki_alpha: float
    kd_alpha: float
    kp_q: float
    anti_windup_limit: float
    integral_alpha: float = 0.0
    previous_alpha_error: float | None = None

    def reset(self) -> None:
        self.integral_alpha = 0.0
        self.previous_alpha_error = None

    def command(
        self,
        *,
        state: np.ndarray,
        reference_state: np.ndarray,
        dt: float,
        schedule: dict[str, float],
    ) -> float:
        # Return raw flap command before actuator limits.

        del schedule
        alpha_error = state[0] - reference_state[0]
        q_error = state[1] - reference_state[1]
        self.integral_alpha = float(
            np.clip(
                self.integral_alpha + alpha_error * dt,
                -self.anti_windup_limit,
                self.anti_windup_limit,
            )
        )
        if self.previous_alpha_error is None:
            alpha_error_rate = 0.0
        else:
            alpha_error_rate = (alpha_error - self.previous_alpha_error) / dt
        self.previous_alpha_error = float(alpha_error)
        return float(
            self.kp_alpha * alpha_error
            + self.ki_alpha * self.integral_alpha
            + self.kd_alpha * alpha_error_rate
            + self.kp_q * q_error
        )


@dataclass(frozen=True)
class ScheduledGain:
    """One local LQR gain tied to a scheduling point."""

    label: str
    altitude_m: float
    velocity_mps: float
    dynamic_pressure_pa: float
    gain: np.ndarray


class GainScheduledLQRController:
    # Nearest-neighbor gain-scheduled LQR over dynamic pressure.

    def __init__(
        self,
        *,
        scheduled_gains: list[ScheduledGain],
        feedforward_flap_rad: float = 0.0,
    ) -> None:
        self.scheduled_gains = sorted(
            scheduled_gains, key=lambda item: item.dynamic_pressure_pa
        )
        self.feedforward_flap_rad = feedforward_flap_rad

    def reset(self) -> None:
        """Reset controller memory."""

    def select_gain(self, schedule: dict[str, float]) -> ScheduledGain:
        """Select nearest gain by dynamic pressure."""

        qbar = schedule["dynamic_pressure_pa"]
        return min(
            self.scheduled_gains,
            key=lambda item: abs(item.dynamic_pressure_pa - qbar),
        )

    def command(
        self,
        *,
        state: np.ndarray,
        reference_state: np.ndarray,
        dt: float,
        schedule: dict[str, float],
    ) -> float:
        # Return raw LQR flap command before actuator limits.

        del dt
        selected = self.select_gain(schedule)
        error = (state - reference_state).reshape(3, 1)
        return float(self.feedforward_flap_rad - (selected.gain @ error)[0, 0])


def build_lqr_controller(
    *,
    schedule_points: list[dict[str, float | str]],
    q_weights: dict[str, float],
    r_weight: float,
    dt: float,
    vehicle: VehicleParams,
    aero: AeroParams,
) -> GainScheduledLQRController:
    # Build local gains for the configured schedule points.

    q_matrix = np.diag([q_weights["alpha"], q_weights["q"], q_weights["theta"]]).astype(
        float
    )
    r_matrix = np.array([[float(r_weight)]])
    scheduled_gains: list[ScheduledGain] = []
    for point in schedule_points:
        schedule = {
            "altitude_m": float(point["altitude_m"]),
            "velocity_mps": float(point["velocity_mps"]),
            "mach": _mach_from_schedule_point(point),
            "density_kgm3": _density_from_schedule_point(point),
            "dynamic_pressure_pa": _qbar_from_schedule_point(point),
        }
        model = finite_difference_linearization(
            state=np.zeros(3),
            delta_flap_rad=0.0,
            schedule=schedule,
            vehicle=vehicle,
            aero=aero,
            dt=dt,
        )
        gain = solve_discrete_lqr_gain(
            a_discrete=model.a_discrete,
            b_discrete=model.b_discrete,
            q_weight=q_matrix,
            r_weight=r_matrix,
        )
        scheduled_gains.append(
            ScheduledGain(
                label=str(point["label"]),
                altitude_m=schedule["altitude_m"],
                velocity_mps=schedule["velocity_mps"],
                dynamic_pressure_pa=schedule["dynamic_pressure_pa"],
                gain=gain,
            )
        )
    return GainScheduledLQRController(scheduled_gains=scheduled_gains)


def _schedule_from_atmosphere(point: dict[str, float | str]) -> dict[str, float]:
    from reentry_mpc.atmosphere import (  # noqa: PLC0415
        dynamic_pressure_pa,
        mach_number,
        standard_atmosphere,
    )

    altitude = float(point["altitude_m"])
    velocity = float(point["velocity_mps"])
    atmosphere = standard_atmosphere(altitude)
    return {
        "altitude_m": altitude,
        "velocity_mps": velocity,
        "mach": mach_number(velocity, altitude),
        "density_kgm3": atmosphere.density_kgm3,
        "dynamic_pressure_pa": dynamic_pressure_pa(velocity, atmosphere.density_kgm3),
    }


def _mach_from_schedule_point(point: dict[str, float | str]) -> float:
    return _schedule_from_atmosphere(point)["mach"]


def _density_from_schedule_point(point: dict[str, float | str]) -> float:
    return _schedule_from_atmosphere(point)["density_kgm3"]


def _qbar_from_schedule_point(point: dict[str, float | str]) -> float:
    return _schedule_from_atmosphere(point)["dynamic_pressure_pa"]
