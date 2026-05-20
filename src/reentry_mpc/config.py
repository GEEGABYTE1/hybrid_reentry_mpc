"""Configuration loading for reproducible experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    """Typed subset of the smoke experiment configuration."""

    seed: int
    horizon_steps: int
    dt: float
    initial_angle_rad: float
    initial_rate_rad_s: float
    target_angle_rad: float
    control_limit: float
    damping: float
    inertia: float
    baseline_kp: float
    baseline_kd: float
    learned_bias_gain: float
    disturbance_scale: float


def load_config(path: str | Path) -> ExperimentConfig:
    """Load an experiment YAML file into an immutable dataclass."""

    with Path(path).open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)

    return ExperimentConfig(
        seed=int(raw["seed"]),
        horizon_steps=int(raw["horizon_steps"]),
        dt=float(raw["dt"]),
        initial_angle_rad=float(raw["initial_state"]["angle_rad"]),
        initial_rate_rad_s=float(raw["initial_state"]["rate_rad_s"]),
        target_angle_rad=float(raw["target_angle_rad"]),
        control_limit=float(raw["control_limit"]),
        damping=float(raw["damping"]),
        inertia=float(raw["inertia"]),
        baseline_kp=float(raw["controllers"]["baseline_pd"]["kp"]),
        baseline_kd=float(raw["controllers"]["baseline_pd"]["kd"]),
        learned_bias_gain=float(
            raw["controllers"]["learning_augmented_pd"]["bias_gain"]
        ),
        disturbance_scale=float(raw["disturbance_scale"]),
    )
