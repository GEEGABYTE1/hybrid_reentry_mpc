# Reduced-order longitudinal reentry attitude dynamics.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from reentry_mpc.schedules import (
    ReferenceTrajectory,
    build_linear_reference_trajectory,
)


@dataclass(frozen=True)
class VehicleParams:
    reference_area_m2: float
    reference_length_m: float
    pitch_inertia_kgm2: float


@dataclass(frozen=True)
class AeroParams:
    cm0: float
    cm_alpha: float
    cm_q: float
    cm_delta: float
    cm_alpha_mach_slope: float
    min_effectiveness: float
    max_effectiveness: float


@dataclass(frozen=True)
class Phase1Config:
    seed: int
    dt: float
    duration_s: float
    initial_state: np.ndarray
    reference: ReferenceTrajectory
    vehicle: VehicleParams
    aero: AeroParams
    trim_flap_rad: float
    step_time_s: float
    step_delta_rad: float


def load_phase1_config(path: str | Path) -> Phase1Config:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)

    reference = build_linear_reference_trajectory(
        duration_s=float(raw["duration_s"]),
        dt=float(raw["dt"]),
        start_altitude_m=float(raw["reference_trajectory"]["start_altitude_m"]),
        end_altitude_m=float(raw["reference_trajectory"]["end_altitude_m"]),
        start_velocity_mps=float(raw["reference_trajectory"]["start_velocity_mps"]),
        end_velocity_mps=float(raw["reference_trajectory"]["end_velocity_mps"]),
    )
    return Phase1Config(
        seed=int(raw["seed"]),
        dt=float(raw["dt"]),
        duration_s=float(raw["duration_s"]),
        initial_state=np.array(
            [
                float(raw["initial_state"]["alpha_rad"]),
                float(raw["initial_state"]["q_radps"]),
                float(raw["initial_state"]["theta_rad"]),
            ],
            dtype=float,
        ),
        reference=reference,
        vehicle=VehicleParams(
            reference_area_m2=float(raw["vehicle"]["reference_area_m2"]),
            reference_length_m=float(raw["vehicle"]["reference_length_m"]),
            pitch_inertia_kgm2=float(raw["vehicle"]["pitch_inertia_kgm2"]),
        ),
        aero=AeroParams(
            cm0=float(raw["aero"]["cm0"]),
            cm_alpha=float(raw["aero"]["cm_alpha"]),
            cm_q=float(raw["aero"]["cm_q"]),
            cm_delta=float(raw["aero"]["cm_delta"]),
            cm_alpha_mach_slope=float(raw["aero"]["cm_alpha_mach_slope"]),
            min_effectiveness=float(raw["aero"]["min_effectiveness"]),
            max_effectiveness=float(raw["aero"]["max_effectiveness"]),
        ),
        trim_flap_rad=float(raw["control"]["trim_flap_rad"]),
        step_time_s=float(raw["control"]["step_time_s"]),
        step_delta_rad=float(raw["control"]["step_delta_rad"]),
    )


def flap_effectiveness(mach: float, altitude_m: float, aero: AeroParams) -> float:
    mach_factor = np.interp(mach, [8.0, 18.0, 28.0], [1.0, 0.72, 0.48])
    altitude_factor = np.interp(altitude_m, [35000.0, 80000.0], [1.0, 0.62])
    effectiveness = mach_factor * altitude_factor
    return float(np.clip(effectiveness, aero.min_effectiveness, aero.max_effectiveness))


def pitching_moment_coefficient(
    *,
    alpha_rad: float,
    q_radps: float,
    delta_flap_rad: float,
    mach: float,
    altitude_m: float,
    velocity_mps: float,
    vehicle: VehicleParams,
    aero: AeroParams,
) -> float:
    alpha_slope = aero.cm_alpha + aero.cm_alpha_mach_slope * (mach - 15.0)
    q_hat = q_radps * vehicle.reference_length_m / max(2.0 * velocity_mps, 1.0)
    effectiveness = flap_effectiveness(mach, altitude_m=altitude_m, aero=aero)
    return float(
        aero.cm0
        + alpha_slope * alpha_rad
        + aero.cm_q * q_hat
        + aero.cm_delta * effectiveness * delta_flap_rad
    )


def scheduled_pitching_moment(
    *,
    state: np.ndarray,
    delta_flap_rad: float,
    schedule: dict[str, float],
    vehicle: VehicleParams,
    aero: AeroParams,
) -> tuple[float, float, float]:
    alpha_rad, q_radps, _theta_rad = state
    effectiveness = flap_effectiveness(schedule["mach"], schedule["altitude_m"], aero)
    alpha_slope = aero.cm_alpha + aero.cm_alpha_mach_slope * (schedule["mach"] - 15.0)
    q_hat = (
        q_radps * vehicle.reference_length_m / max(2.0 * schedule["velocity_mps"], 1.0)
    )
    cm = (
        aero.cm0
        + alpha_slope * alpha_rad
        + aero.cm_q * q_hat
        + aero.cm_delta * effectiveness * delta_flap_rad
    )
    moment_nm = (
        schedule["dynamic_pressure_pa"]
        * vehicle.reference_area_m2
        * vehicle.reference_length_m
        * cm
    )
    return float(moment_nm), float(cm), effectiveness


def longitudinal_derivatives(
    *,
    time_s: float,
    state: np.ndarray,
    delta_flap_rad: float,
    config: Phase1Config,
) -> np.ndarray:
    schedule = config.reference.sample(time_s)
    moment_nm, _cm, _effectiveness = scheduled_pitching_moment(
        state=state,
        delta_flap_rad=delta_flap_rad,
        schedule=schedule,
        vehicle=config.vehicle,
        aero=config.aero,
    )
    q_dot = moment_nm / config.vehicle.pitch_inertia_kgm2
    alpha_dot = q_dot - 0.22 * state[0]
    theta_dot = state[1]
    return np.array([alpha_dot, q_dot, theta_dot], dtype=float)


def rk4_step(
    *,
    time_s: float,
    state: np.ndarray,
    delta_flap_rad: float,
    dt: float,
    config: Phase1Config,
) -> np.ndarray:
    k1 = longitudinal_derivatives(
        time_s=time_s, state=state, delta_flap_rad=delta_flap_rad, config=config
    )
    k2 = longitudinal_derivatives(
        time_s=time_s + 0.5 * dt,
        state=state + 0.5 * dt * k1,
        delta_flap_rad=delta_flap_rad,
        config=config,
    )
    k3 = longitudinal_derivatives(
        time_s=time_s + 0.5 * dt,
        state=state + 0.5 * dt * k2,
        delta_flap_rad=delta_flap_rad,
        config=config,
    )
    k4 = longitudinal_derivatives(
        time_s=time_s + dt,
        state=state + dt * k3,
        delta_flap_rad=delta_flap_rad,
        config=config,
    )
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def open_loop_control(time_s: float, config: Phase1Config) -> float:
    if time_s >= config.step_time_s:
        return config.trim_flap_rad + config.step_delta_rad
    return config.trim_flap_rad


def simulate_open_loop(config: Phase1Config) -> pd.DataFrame:
    state = config.initial_state.copy()
    rows: list[dict[str, float]] = []
    for time_s in config.reference.time_s:
        delta = open_loop_control(float(time_s), config)
        schedule = config.reference.sample(float(time_s))
        moment_nm, cm, effectiveness = scheduled_pitching_moment(
            state=state,
            delta_flap_rad=delta,
            schedule=schedule,
            vehicle=config.vehicle,
            aero=config.aero,
        )
        rows.append(
            {
                "time_s": float(time_s),
                "alpha_rad": float(state[0]),
                "q_radps": float(state[1]),
                "theta_rad": float(state[2]),
                "delta_flap_rad": float(delta),
                "pitching_moment_nm": moment_nm,
                "cm": cm,
                "flap_effectiveness": effectiveness,
                **schedule,
            }
        )
        state = rk4_step(
            time_s=float(time_s),
            state=state,
            delta_flap_rad=delta,
            dt=config.dt,
            config=config,
        )
    return pd.DataFrame(rows)
