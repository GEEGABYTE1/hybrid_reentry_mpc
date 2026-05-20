"""Deterministic toy dynamics for attitude-control artifact generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from reentry_mpc.config import ExperimentConfig


@dataclass(frozen=True)
class ControllerSpec:
    """Controller parameters used by the smoke experiment."""

    name: str
    kp: float
    kd: float
    learned_bias_gain: float = 0.0


def run_attitude_simulation(
    config: ExperimentConfig, controller: ControllerSpec
) -> pd.DataFrame:
    """Run a deterministic single-axis reentry attitude-control simulation."""

    rng = np.random.default_rng(config.seed)
    angle = config.initial_angle_rad
    rate = config.initial_rate_rad_s
    rows: list[dict[str, float | str | int]] = []

    for step in range(config.horizon_steps):
        time_s = step * config.dt
        disturbance = _disturbance(step, config, rng)
        tracking_error = angle - config.target_angle_rad
        learned_bias = controller.learned_bias_gain * disturbance
        raw_control = (
            -controller.kp * tracking_error - controller.kd * rate - learned_bias
        )
        control = float(
            np.clip(raw_control, -config.control_limit, config.control_limit)
        )

        angular_accel = (control + disturbance - config.damping * rate) / config.inertia
        rate += angular_accel * config.dt
        angle += rate * config.dt

        rows.append(
            {
                "controller": controller.name,
                "step": step,
                "time_s": time_s,
                "angle_rad": angle,
                "rate_rad_s": rate,
                "target_angle_rad": config.target_angle_rad,
                "control": control,
                "disturbance": disturbance,
                "tracking_error_rad": angle - config.target_angle_rad,
            }
        )

    return pd.DataFrame(rows)


def _disturbance(
    step: int, config: ExperimentConfig, rng: np.random.Generator
) -> float:
    """Create repeatable aerodynamic torque variation."""

    periodic = config.disturbance_scale * np.sin(0.17 * step)
    noise = rng.normal(loc=0.0, scale=config.disturbance_scale * 0.08)
    return float(periodic + noise)
