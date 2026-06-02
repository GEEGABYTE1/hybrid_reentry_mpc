from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from reentry_mpc.artifacts import plt
from reentry_mpc.atmosphere import (
    dynamic_pressure_pa,
    mach_number,
    standard_atmosphere,
)


@dataclass(frozen=True)
class ReferenceProfileConfig:
    start_altitude_m: float
    end_altitude_m: float
    start_velocity_mps: float
    end_velocity_mps: float
    alpha_start_rad: float
    alpha_peak_rad: float
    alpha_end_rad: float
    theta_start_rad: float
    theta_end_rad: float


@dataclass(frozen=True)
class CorridorConstraints:
    alpha_margin_rad: float
    alpha_min_abs_rad: float
    alpha_max_abs_rad: float
    q_min_radps: float
    q_max_radps: float
    flap_min_rad: float
    flap_max_rad: float
    flap_rate_min_radps: float
    flap_rate_max_radps: float


@dataclass(frozen=True)
class DiagnosticsConfig:
    dynamic_pressure_limit_pa: float | None
    heating_proxy_limit: float | None


@dataclass(frozen=True)
class Phase2Config:
    seed: int
    duration_s: float
    dt: float
    reference_profile: ReferenceProfileConfig
    constraints: CorridorConstraints
    diagnostics: DiagnosticsConfig


def load_phase2_config(path: str | Path) -> Phase2Config:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)

    diagnostics = raw.get("diagnostics", {})
    return Phase2Config(
        seed=int(raw["seed"]),
        duration_s=float(raw["duration_s"]),
        dt=float(raw["dt"]),
        reference_profile=ReferenceProfileConfig(
            start_altitude_m=float(raw["reference_profile"]["start_altitude_m"]),
            end_altitude_m=float(raw["reference_profile"]["end_altitude_m"]),
            start_velocity_mps=float(raw["reference_profile"]["start_velocity_mps"]),
            end_velocity_mps=float(raw["reference_profile"]["end_velocity_mps"]),
            alpha_start_rad=float(raw["reference_profile"]["alpha_start_rad"]),
            alpha_peak_rad=float(raw["reference_profile"]["alpha_peak_rad"]),
            alpha_end_rad=float(raw["reference_profile"]["alpha_end_rad"]),
            theta_start_rad=float(raw["reference_profile"]["theta_start_rad"]),
            theta_end_rad=float(raw["reference_profile"]["theta_end_rad"]),
        ),
        constraints=CorridorConstraints(
            alpha_margin_rad=float(raw["constraints"]["alpha_margin_rad"]),
            alpha_min_abs_rad=float(raw["constraints"]["alpha_min_abs_rad"]),
            alpha_max_abs_rad=float(raw["constraints"]["alpha_max_abs_rad"]),
            q_min_radps=float(raw["constraints"]["q_min_radps"]),
            q_max_radps=float(raw["constraints"]["q_max_radps"]),
            flap_min_rad=float(raw["constraints"]["flap_min_rad"]),
            flap_max_rad=float(raw["constraints"]["flap_max_rad"]),
            flap_rate_min_radps=float(raw["constraints"]["flap_rate_min_radps"]),
            flap_rate_max_radps=float(raw["constraints"]["flap_rate_max_radps"]),
        ),
        diagnostics=DiagnosticsConfig(
            dynamic_pressure_limit_pa=_optional_float(
                diagnostics.get("dynamic_pressure_limit_pa")
            ),
            heating_proxy_limit=_optional_float(diagnostics.get("heating_proxy_limit")),
        ),
    )


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def smoothstep(progress: np.ndarray) -> np.ndarray:
    clipped = np.clip(progress, 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def build_reference_profile(config: Phase2Config) -> pd.DataFrame:
    time_s = np.arange(0.0, config.duration_s + 0.5 * config.dt, config.dt)
    tau = time_s / config.duration_s
    sigma = smoothstep(tau)
    ref = config.reference_profile

    altitude_m = (
        ref.start_altitude_m + (ref.end_altitude_m - ref.start_altitude_m) * sigma
    )
    velocity_mps = ref.start_velocity_mps + (
        ref.end_velocity_mps - ref.start_velocity_mps
    ) * smoothstep(tau**0.92)

    alpha_ref_rad = _alpha_reference(tau, ref)
    theta_ref_rad = ref.theta_start_rad + (
        ref.theta_end_rad - ref.theta_start_rad
    ) * smoothstep(tau)
    q_ref_radps = np.gradient(theta_ref_rad, config.dt)

    rows: list[dict[str, float | bool]] = []
    for idx, time in enumerate(time_s):
        atmosphere = standard_atmosphere(float(altitude_m[idx]))
        mach = mach_number(float(velocity_mps[idx]), float(altitude_m[idx]))
        qbar = dynamic_pressure_pa(float(velocity_mps[idx]), atmosphere.density_kgm3)
        heating_proxy = np.sqrt(atmosphere.density_kgm3) * float(velocity_mps[idx]) ** 3
        alpha_min, alpha_max = _alpha_corridor(alpha_ref_rad[idx], config.constraints)
        rows.append(
            {
                "time_s": float(time),
                "altitude_m": float(altitude_m[idx]),
                "velocity_mps": float(velocity_mps[idx]),
                "mach": mach,
                "density_kgm3": atmosphere.density_kgm3,
                "dynamic_pressure_pa": qbar,
                "heating_proxy": float(heating_proxy),
                "alpha_ref_rad": float(alpha_ref_rad[idx]),
                "theta_ref_rad": float(theta_ref_rad[idx]),
                "q_ref_radps": float(q_ref_radps[idx]),
                "alpha_min_rad": alpha_min,
                "alpha_max_rad": alpha_max,
                "q_min_radps": config.constraints.q_min_radps,
                "q_max_radps": config.constraints.q_max_radps,
                "flap_min_rad": config.constraints.flap_min_rad,
                "flap_max_rad": config.constraints.flap_max_rad,
                "flap_rate_min_radps": config.constraints.flap_rate_min_radps,
                "flap_rate_max_radps": config.constraints.flap_rate_max_radps,
                "dynamic_pressure_limit_pa": _nan_if_none(
                    config.diagnostics.dynamic_pressure_limit_pa
                ),
                "heating_proxy_limit": _nan_if_none(
                    config.diagnostics.heating_proxy_limit
                ),
                "dynamic_pressure_limit_exceeded": _exceeds(
                    qbar, config.diagnostics.dynamic_pressure_limit_pa
                ),
                "heating_proxy_limit_exceeded": _exceeds(
                    heating_proxy, config.diagnostics.heating_proxy_limit
                ),
            }
        )
    return pd.DataFrame(rows)


def _alpha_reference(tau: np.ndarray, ref: ReferenceProfileConfig) -> np.ndarray:
    climb = smoothstep(np.clip(tau / 0.45, 0.0, 1.0))
    descend = smoothstep(np.clip((tau - 0.45) / 0.55, 0.0, 1.0))
    alpha = ref.alpha_start_rad + (ref.alpha_peak_rad - ref.alpha_start_rad) * climb
    return alpha + (ref.alpha_end_rad - ref.alpha_peak_rad) * descend


def _alpha_corridor(
    alpha_ref_rad: float, constraints: CorridorConstraints
) -> tuple[float, float]:
    alpha_min = max(
        constraints.alpha_min_abs_rad,
        float(alpha_ref_rad) - constraints.alpha_margin_rad,
    )
    alpha_max = min(
        constraints.alpha_max_abs_rad,
        float(alpha_ref_rad) + constraints.alpha_margin_rad,
    )
    return float(alpha_min), float(alpha_max)


def _nan_if_none(value: float | None) -> float:
    if value is None:
        return float("nan")
    return float(value)


def _exceeds(value: float, limit: float | None) -> bool:
    return bool(limit is not None and value > limit)


def build_corridor_config(
    profile: pd.DataFrame, config: Phase2Config
) -> dict[str, Any]:
    return {
        "seed": config.seed,
        "duration_s": config.duration_s,
        "dt": config.dt,
        "reference_profile": asdict(config.reference_profile),
        "constraints": asdict(config.constraints),
        "diagnostics": {
            **asdict(config.diagnostics),
            "max_dynamic_pressure_pa": float(profile["dynamic_pressure_pa"].max()),
            "max_heating_proxy": float(profile["heating_proxy"].max()),
            "dynamic_pressure_limit_exceeded": bool(
                profile["dynamic_pressure_limit_exceeded"].any()
            ),
            "heating_proxy_limit_exceeded": bool(
                profile["heating_proxy_limit_exceeded"].any()
            ),
        },
        "columns_in_reference_profile_csv": list(profile.columns),
    }


def write_phase2_figures(profile: pd.DataFrame, output_dir: Path) -> dict[str, Path]:
    mach_alt_path = output_dir / "mach_vs_altitude.png"
    qbar_path = output_dir / "dynamic_pressure_vs_time.png"
    alpha_path = output_dir / "alpha_reference_corridor.png"
    flap_path = output_dir / "flap_authority_vs_time.png"

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.plot(profile["mach"], profile["altitude_m"] / 1000.0)
    ax.set_xlabel("Mach")
    ax.set_ylabel("Altitude (km)")
    ax.invert_xaxis()
    fig.tight_layout()
    fig.savefig(mach_alt_path, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.plot(profile["time_s"], profile["dynamic_pressure_pa"] / 1000.0)
    if profile["dynamic_pressure_limit_pa"].notna().any():
        ax.axhline(
            profile["dynamic_pressure_limit_pa"].dropna().iloc[0] / 1000.0,
            color="black",
            linestyle="--",
            linewidth=1.0,
        )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Dynamic pressure (kPa)")
    fig.tight_layout()
    fig.savefig(qbar_path, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.fill_between(
        profile["time_s"],
        profile["alpha_min_rad"],
        profile["alpha_max_rad"],
        alpha=0.25,
        label="corridor",
    )
    ax.plot(profile["time_s"], profile["alpha_ref_rad"], label="alpha_ref")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Alpha (rad)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(alpha_path, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.fill_between(
        profile["time_s"],
        profile["flap_min_rad"],
        profile["flap_max_rad"],
        alpha=0.22,
        label="flap position",
    )
    ax.plot(
        profile["time_s"],
        profile["flap_rate_max_radps"],
        linestyle="--",
        label="rate max",
    )
    ax.plot(
        profile["time_s"],
        profile["flap_rate_min_radps"],
        linestyle="--",
        label="rate min",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Flap / flap rate limits")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(flap_path, dpi=160)
    plt.close(fig)

    return {
        "mach_vs_altitude": mach_alt_path,
        "dynamic_pressure_vs_time": qbar_path,
        "alpha_reference_corridor": alpha_path,
        "flap_authority_vs_time": flap_path,
    }


def run_phase2_reference(
    config_path: str | Path = "configs/phase2_reference.yaml",
    output_dir: str | Path = "outputs/phase2_reference",
) -> dict[str, Path | pd.DataFrame | dict[str, Any]]:
    config = load_phase2_config(config_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    profile = build_reference_profile(config)
    corridor_config = build_corridor_config(profile, config)

    reference_path = output_path / "reference_profile.csv"
    corridor_path = output_path / "corridor_config.json"
    profile.to_csv(reference_path, index=False)
    with corridor_path.open("w", encoding="utf-8") as handle:
        json.dump(corridor_config, handle, indent=2, sort_keys=True)
        handle.write("\n")

    figure_paths = write_phase2_figures(profile, output_path)
    return {
        "profile": profile,
        "corridor_config": corridor_config,
        "reference_csv": reference_path,
        "corridor_json": corridor_path,
        **figure_paths,
    }
