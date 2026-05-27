# standard-atmosphere approximations for reduced-order reentry simulation.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GAMMA_AIR = 1.4
GAS_CONSTANT_AIR = 287.05
SEA_LEVEL_DENSITY_KGM3 = 1.225
SEA_LEVEL_PRESSURE_PA = 101325.0
SEA_LEVEL_TEMPERATURE_K = 288.15
TROPOSPHERE_LAPSE_K_PER_M = -0.0065
TROPOPAUSE_ALTITUDE_M = 11000.0
TROPOPAUSE_TEMPERATURE_K = 216.65
GRAVITY_MPS2 = 9.80665


@dataclass(frozen=True)
class AtmosphereSample:
    altitude_m: float
    temperature_k: float
    pressure_pa: float
    density_kgm3: float
    speed_of_sound_mps: float


def standard_atmosphere(altitude_m: float) -> AtmosphereSample:
    altitude = max(float(altitude_m), 0.0)
    if altitude <= TROPOPAUSE_ALTITUDE_M:
        temperature = SEA_LEVEL_TEMPERATURE_K + TROPOSPHERE_LAPSE_K_PER_M * altitude
        pressure = SEA_LEVEL_PRESSURE_PA * (temperature / SEA_LEVEL_TEMPERATURE_K) ** (
            -GRAVITY_MPS2 / (TROPOSPHERE_LAPSE_K_PER_M * GAS_CONSTANT_AIR)
        )
    else:
        pressure_tropopause = SEA_LEVEL_PRESSURE_PA * (
            TROPOPAUSE_TEMPERATURE_K / SEA_LEVEL_TEMPERATURE_K
        ) ** (-GRAVITY_MPS2 / (TROPOSPHERE_LAPSE_K_PER_M * GAS_CONSTANT_AIR))
        temperature = TROPOPAUSE_TEMPERATURE_K
        pressure = pressure_tropopause * np.exp(
            -GRAVITY_MPS2
            * (altitude - TROPOPAUSE_ALTITUDE_M)
            / (GAS_CONSTANT_AIR * temperature)
        )

    density = pressure / (GAS_CONSTANT_AIR * temperature)
    speed_of_sound = float(np.sqrt(GAMMA_AIR * GAS_CONSTANT_AIR * temperature))
    return AtmosphereSample(
        altitude_m=altitude,
        temperature_k=float(temperature),
        pressure_pa=float(pressure),
        density_kgm3=float(density),
        speed_of_sound_mps=speed_of_sound,
    )


def mach_number(velocity_mps: float, altitude_m: float) -> float:
    atmosphere = standard_atmosphere(altitude_m)
    return float(velocity_mps / atmosphere.speed_of_sound_mps)


def dynamic_pressure_pa(velocity_mps: float, density_kgm3: float) -> float:
    return float(0.5 * density_kgm3 * velocity_mps**2)
