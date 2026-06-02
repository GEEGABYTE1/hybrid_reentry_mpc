# scenario samling and uncertain rollout utilities f
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from reentry_mpc.longitudinal import AeroParams, VehicleParams, flap_effectiveness


@dataclass(frozen=True)
class SensorNoiseStd:
    # per-state sensor noise std
    alpha_rad: float
    q_radps: float
    theta_rad: float


@dataclass(frozen=True)
class InitialStateError:
    # initial state offset from first ref state
    alpha_rad: float
    q_radps: float
    theta_rad: float


@dataclass(frozen=True)
class UncertaintyScenario:
    # one sampled monte carlo unc scenario
    scenario_id: int
    seed: int
    density_scale: float
    cm_alpha_scale: float
    cm_delta_scale: float
    cm_q_scale: float
    actuator_lag_s: float
    actuator_delay_s: float
    sensor_noise_std: SensorNoiseStd
    initial_error: InitialStateError
    external_disturbance_moment_nm: float

    def to_flat_dict(self) -> dict[str, float | int]:
        """Return a flat dictionary suitable for CSV/JSON artifacts."""

        return {
            "scenario_id": self.scenario_id,
            "seed": self.seed,
            "density_scale": self.density_scale,
            "cm_alpha_scale": self.cm_alpha_scale,
            "cm_delta_scale": self.cm_delta_scale,
            "cm_q_scale": self.cm_q_scale,
            "actuator_lag_s": self.actuator_lag_s,
            "actuator_delay_s": self.actuator_delay_s,
            "sensor_noise_alpha_rad": self.sensor_noise_std.alpha_rad,
            "sensor_noise_q_radps": self.sensor_noise_std.q_radps,
            "sensor_noise_theta_rad": self.sensor_noise_std.theta_rad,
            "initial_alpha_error_rad": self.initial_error.alpha_rad,
            "initial_q_error_radps": self.initial_error.q_radps,
            "initial_theta_error_rad": self.initial_error.theta_rad,
            "external_disturbance_moment_nm": self.external_disturbance_moment_nm,
        }

    def to_nested_dict(self) -> dict[str, Any]:

        return asdict(self)


@dataclass
class ActuatorState:
    delay_buffer: list[float]
    lagged_command_rad: float = 0.0
    previous_applied_rad: float = 0.0


def sample_scenario(
    *,
    scenario_id: int,
    seed: int,
    ranges: dict[str, Any],
) -> UncertaintyScenario:
    rng = np.random.default_rng(seed)
    sensor_ranges = ranges["sensor_noise_std"]
    center = ranges["initial_error_center"]
    half_width = ranges["initial_error_half_width"]
    return UncertaintyScenario(
        scenario_id=scenario_id,
        seed=seed,
        density_scale=_uniform(rng, ranges["density_scale"]),
        cm_alpha_scale=_uniform(rng, ranges["cm_alpha_scale"]),
        cm_delta_scale=_uniform(rng, ranges["cm_delta_scale"]),
        cm_q_scale=_uniform(rng, ranges["cm_q_scale"]),
        actuator_lag_s=_uniform(rng, ranges["actuator_lag_s"]),
        actuator_delay_s=float(rng.choice(ranges["actuator_delay_s_choices"])),
        sensor_noise_std=SensorNoiseStd(
            alpha_rad=_uniform(rng, sensor_ranges["alpha_rad"]),
            q_radps=_uniform(rng, sensor_ranges["q_radps"]),
            theta_rad=_uniform(rng, sensor_ranges["theta_rad"]),
        ),
        initial_error=InitialStateError(
            alpha_rad=float(center["alpha_rad"])
            + _uniform(rng, [-half_width["alpha_rad"], half_width["alpha_rad"]]),
            q_radps=float(center["q_radps"])
            + _uniform(rng, [-half_width["q_radps"], half_width["q_radps"]]),
            theta_rad=float(center["theta_rad"])
            + _uniform(rng, [-half_width["theta_rad"], half_width["theta_rad"]]),
        ),
        external_disturbance_moment_nm=_uniform(
            rng, ranges["external_disturbance_moment_nm"]
        ),
    )


def perturb_aero(aero: AeroParams, scenario: UncertaintyScenario) -> AeroParams:
    return AeroParams(
        cm0=aero.cm0,
        cm_alpha=aero.cm_alpha * scenario.cm_alpha_scale,
        cm_q=aero.cm_q * scenario.cm_q_scale,
        cm_delta=aero.cm_delta * scenario.cm_delta_scale,
        cm_alpha_mach_slope=aero.cm_alpha_mach_slope,
        min_effectiveness=aero.min_effectiveness,
        max_effectiveness=aero.max_effectiveness,
    )


def noisy_measurement(
    state: np.ndarray, scenario: UncertaintyScenario, rng: np.random.Generator
) -> np.ndarray:
    std = np.array(
        [
            scenario.sensor_noise_std.alpha_rad,
            scenario.sensor_noise_std.q_radps,
            scenario.sensor_noise_std.theta_rad,
        ],
        dtype=float,
    )
    return state + rng.normal(0.0, std)


def initialize_actuator(scenario: UncertaintyScenario, dt: float) -> ActuatorState:
    delay_steps = int(round(scenario.actuator_delay_s / dt))
    return ActuatorState(delay_buffer=[0.0] * delay_steps)


def actuator_step(
    *,
    raw_command: float,
    actuator: ActuatorState,
    scenario: UncertaintyScenario,
    row: pd.Series,
    dt: float,
) -> tuple[float, dict[str, float | bool]]:
    if actuator.delay_buffer:
        actuator.delay_buffer.append(float(raw_command))
        delayed_command = actuator.delay_buffer.pop(0)
    else:
        delayed_command = float(raw_command)

    if scenario.actuator_lag_s <= 0.0:
        lagged_command = delayed_command
    else:
        lag_gain = 1.0 - float(np.exp(-dt / scenario.actuator_lag_s))
        lagged_command = actuator.lagged_command_rad + lag_gain * (
            delayed_command - actuator.lagged_command_rad
        )
    actuator.lagged_command_rad = float(lagged_command)

    min_step = float(row["flap_rate_min_radps"]) * dt
    max_step = float(row["flap_rate_max_radps"]) * dt
    requested_step = lagged_command - actuator.previous_applied_rad
    limited_step = float(np.clip(requested_step, min_step, max_step))
    rate_limited_command = actuator.previous_applied_rad + limited_step
    applied_command = float(
        np.clip(rate_limited_command, row["flap_min_rad"], row["flap_max_rad"])
    )
    flap_saturated = not np.isclose(applied_command, rate_limited_command)
    rate_saturated = not np.isclose(limited_step, requested_step)
    flap_rate = (applied_command - actuator.previous_applied_rad) / dt
    actuator.previous_applied_rad = applied_command
    return applied_command, {
        "delta_flap_raw_rad": float(raw_command),
        "delta_flap_delayed_rad": float(delayed_command),
        "delta_flap_lagged_rad": float(lagged_command),
        "delta_flap_rad": float(applied_command),
        "delta_flap_rate_radps": float(flap_rate),
        "flap_saturated": bool(flap_saturated),
        "flap_rate_saturated": bool(rate_saturated),
    }


def uncertain_derivatives(
    *,
    state: np.ndarray,
    delta_flap_rad: float,
    row: pd.Series,
    vehicle: VehicleParams,
    aero: AeroParams,
    scenario: UncertaintyScenario,
) -> np.ndarray:
    dynamic_pressure = float(row["dynamic_pressure_pa"]) * scenario.density_scale
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
        dynamic_pressure * vehicle.reference_area_m2 * vehicle.reference_length_m * cm
        + scenario.external_disturbance_moment_nm
    )
    q_dot = moment / vehicle.pitch_inertia_kgm2
    alpha_dot = state[1] - 0.22 * state[0]
    theta_dot = state[1]
    return np.array([alpha_dot, q_dot, theta_dot], dtype=float)


def uncertain_rk4_step(
    *,
    state: np.ndarray,
    delta_flap_rad: float,
    row: pd.Series,
    vehicle: VehicleParams,
    aero: AeroParams,
    scenario: UncertaintyScenario,
    dt: float,
) -> np.ndarray:
    k1 = uncertain_derivatives(
        state=state,
        delta_flap_rad=delta_flap_rad,
        row=row,
        vehicle=vehicle,
        aero=aero,
        scenario=scenario,
    )
    k2 = uncertain_derivatives(
        state=state + 0.5 * dt * k1,
        delta_flap_rad=delta_flap_rad,
        row=row,
        vehicle=vehicle,
        aero=aero,
        scenario=scenario,
    )
    k3 = uncertain_derivatives(
        state=state + 0.5 * dt * k2,
        delta_flap_rad=delta_flap_rad,
        row=row,
        vehicle=vehicle,
        aero=aero,
        scenario=scenario,
    )
    k4 = uncertain_derivatives(
        state=state + dt * k3,
        delta_flap_rad=delta_flap_rad,
        row=row,
        vehicle=vehicle,
        aero=aero,
        scenario=scenario,
    )
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _uniform(rng: np.random.Generator, bounds: list[float]) -> float:
    return float(rng.uniform(float(bounds[0]), float(bounds[1])))
