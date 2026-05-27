# reference-trajectory scheduling utilities

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from reentry_mpc.atmosphere import (
    dynamic_pressure_pa,
    mach_number,
    standard_atmosphere,
)


@dataclass(frozen=True)
class ReferenceTrajectory:
    time_s: np.ndarray
    altitude_m: np.ndarray
    velocity_mps: np.ndarray

    def sample(self, time_s: float) -> dict[str, float]:
        time = float(np.clip(time_s, self.time_s[0], self.time_s[-1]))
        altitude = float(np.interp(time, self.time_s, self.altitude_m))
        velocity = float(np.interp(time, self.time_s, self.velocity_mps))
        atmosphere = standard_atmosphere(altitude)
        mach = mach_number(velocity, altitude)
        qbar = dynamic_pressure_pa(velocity, atmosphere.density_kgm3)
        return {
            "altitude_m": altitude,
            "velocity_mps": velocity,
            "mach": mach,
            "density_kgm3": atmosphere.density_kgm3,
            "dynamic_pressure_pa": qbar,
        }

    def to_frame(self) -> pd.DataFrame:
        rows = [
            {"time_s": float(time), **self.sample(float(time))} for time in self.time_s
        ]
        return pd.DataFrame(rows)


def build_linear_reference_trajectory(
    *,
    duration_s: float,
    dt: float,
    start_altitude_m: float,
    end_altitude_m: float,
    start_velocity_mps: float,
    end_velocity_mps: float,
) -> ReferenceTrajectory:
    steps = int(round(duration_s / dt)) + 1
    time = np.linspace(0.0, duration_s, steps)
    altitude = np.linspace(start_altitude_m, end_altitude_m, steps)
    velocity = np.linspace(start_velocity_mps, end_velocity_mps, steps)
    return ReferenceTrajectory(
        time_s=time,
        altitude_m=altitude,
        velocity_mps=velocity,
    )
