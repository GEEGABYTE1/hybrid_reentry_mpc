from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from reentry_mpc.artifacts import plt
from reentry_mpc.longitudinal import (
    AeroParams,
    Phase1Config,
    VehicleParams,
    load_phase1_config,
    scheduled_pitching_moment,
)
from reentry_mpc.phase2 import build_reference_profile, load_phase2_config
from reentry_mpc.phase5 import load_phase5_config
from reentry_mpc.uncertainty import (
    UncertaintyScenario,
    perturb_aero,
    sample_scenario,
    uncertain_derivatives,
)

FEATURE_NAMES = [
    "alpha_rad",
    "q_radps",
    "theta_rad",
    "delta_flap_rad",
    "mach",
    "altitude_m",
    "velocity_mps",
    "density_kgm3",
    "dynamic_pressure_pa",
]

TARGET_NAMES = [
    "residual_alpha_dot",
    "residual_q_dot",
    "residual_theta_dot",
]


@dataclass(frozen=True)
class ResidualDatasetTier:
    name: str
    scenario_count: int
    samples_per_scenario: int


@dataclass(frozen=True)
class ResidualStateSampling:
    alpha_error_std_rad: float
    q_error_std_radps: float
    theta_error_std_rad: float
    flap_min_rad: float
    flap_max_rad: float


@dataclass(frozen=True)
class ResidualSplitConfig:
    train_fraction: float
    val_fraction: float
    test_fraction: float


@dataclass(frozen=True)
class Phase8Config:
    seed: int
    phase1_config: Path
    phase2_config: Path
    phase5_config: Path
    tiers: list[ResidualDatasetTier]
    state_sampling: ResidualStateSampling
    split: ResidualSplitConfig


def load_phase8_config(path: str | Path) -> Phase8Config:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)
    return Phase8Config(
        seed=int(raw["seed"]),
        phase1_config=Path(raw["phase1_config"]),
        phase2_config=Path(raw["phase2_config"]),
        phase5_config=Path(raw["phase5_config"]),
        tiers=[
            ResidualDatasetTier(
                name=str(tier["name"]),
                scenario_count=int(tier["scenario_count"]),
                samples_per_scenario=int(tier["samples_per_scenario"]),
            )
            for tier in raw["tiers"]
        ],
        state_sampling=ResidualStateSampling(
            alpha_error_std_rad=float(raw["state_sampling"]["alpha_error_std_rad"]),
            q_error_std_radps=float(raw["state_sampling"]["q_error_std_radps"]),
            theta_error_std_rad=float(raw["state_sampling"]["theta_error_std_rad"]),
            flap_min_rad=float(raw["state_sampling"]["flap_min_rad"]),
            flap_max_rad=float(raw["state_sampling"]["flap_max_rad"]),
        ),
        split=ResidualSplitConfig(
            train_fraction=float(raw["split"]["train_fraction"]),
            val_fraction=float(raw["split"]["val_fraction"]),
            test_fraction=float(raw["split"]["test_fraction"]),
        ),
    )


def run_phase8_residual_dataset(
    config_path: str | Path = "configs/phase8_residual_dataset.yaml",
    output_dir: str | Path = "outputs/phase8_residual_dataset",
) -> dict[str, Path | dict[str, Any]]:
    config = load_phase8_config(config_path)
    plant_config = load_phase1_config(config.phase1_config)
    phase2_config = load_phase2_config(config.phase2_config)
    phase5_config = load_phase5_config(config.phase5_config)
    reference_profile = build_reference_profile(phase2_config)
    dataset = generate_residual_samples(
        config=config,
        plant_config=plant_config,
        reference_profile=reference_profile,
        phase5_tiers={tier.name: tier for tier in phase5_config.tiers},
    )
    splits = split_dataset(dataset, config.split, config.seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    paths = {name: output_path / f"{name}.npz" for name in ["train", "val", "test"]}
    for name, split in splits.items():
        np.savez_compressed(
            paths[name],
            X=split["X"],
            y=split["y"],
            sample_id=split["sample_id"],
            scenario_id=split["scenario_id"],
            tier=split["tier"],
            mach=split["mach"],
            altitude_m=split["altitude_m"],
            delta_flap_rad=split["delta_flap_rad"],
            feature_names=np.array(FEATURE_NAMES),
            target_names=np.array(TARGET_NAMES),
        )

    metadata = build_metadata(config, dataset, splits)
    metadata_path = output_path / "dataset_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    figure_paths = write_residual_dataset_figures(dataset, output_path)
    return {
        "train_npz": paths["train"],
        "val_npz": paths["val"],
        "test_npz": paths["test"],
        "metadata_json": metadata_path,
        "metadata": metadata,
        **figure_paths,
    }


def generate_residual_samples(
    *,
    config: Phase8Config,
    plant_config: Phase1Config,
    reference_profile: pd.DataFrame,
    phase5_tiers: dict[str, Any],
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(config.seed)
    rows: list[dict[str, Any]] = []
    sample_id = 0
    for tier_idx, tier in enumerate(config.tiers):
        phase5_tier = phase5_tiers[tier.name]
        for scenario_id in range(tier.scenario_count):
            scenario_seed = config.seed + tier_idx * 10_000 + scenario_id
            scenario = sample_scenario(
                scenario_id=scenario_id,
                seed=scenario_seed,
                ranges=phase5_tier.uncertainty_ranges,
            )
            perturbed_aero = perturb_aero(plant_config.aero, scenario)
            indices = rng.integers(
                0, len(reference_profile), size=tier.samples_per_scenario
            )
            for ref_idx in indices:
                ref_row = reference_profile.iloc[int(ref_idx)]
                state = sample_state(ref_row, config.state_sampling, rng)
                delta = float(
                    rng.uniform(
                        config.state_sampling.flap_min_rad,
                        config.state_sampling.flap_max_rad,
                    )
                )
                nominal = nominal_derivatives_from_row(
                    state=state,
                    delta_flap_rad=delta,
                    row=ref_row,
                    vehicle=plant_config.vehicle,
                    aero=plant_config.aero,
                )
                truth = truth_derivatives_from_row(
                    state=state,
                    delta_flap_rad=delta,
                    row=ref_row,
                    vehicle=plant_config.vehicle,
                    aero=perturbed_aero,
                    scenario=scenario,
                )
                residual = truth - nominal
                rows.append(
                    {
                        "sample_id": sample_id,
                        "tier": tier.name,
                        "scenario_id": scenario_id,
                        "scenario_seed": scenario_seed,
                        "alpha_rad": float(state[0]),
                        "q_radps": float(state[1]),
                        "theta_rad": float(state[2]),
                        "delta_flap_rad": delta,
                        "mach": float(ref_row["mach"]),
                        "altitude_m": float(ref_row["altitude_m"]),
                        "velocity_mps": float(ref_row["velocity_mps"]),
                        "density_kgm3": float(ref_row["density_kgm3"]),
                        "dynamic_pressure_pa": float(ref_row["dynamic_pressure_pa"]),
                        "residual_alpha_dot": float(residual[0]),
                        "residual_q_dot": float(residual[1]),
                        "residual_theta_dot": float(residual[2]),
                    }
                )
                sample_id += 1
    frame = pd.DataFrame(rows)
    return {
        "frame": frame,
        "X": frame[FEATURE_NAMES].to_numpy(dtype=np.float32),
        "y": frame[TARGET_NAMES].to_numpy(dtype=np.float32),
        "sample_id": frame["sample_id"].to_numpy(dtype=np.int64),
        "scenario_id": frame["scenario_id"].to_numpy(dtype=np.int64),
        "tier": frame["tier"].to_numpy(dtype=str),
        "mach": frame["mach"].to_numpy(dtype=np.float32),
        "altitude_m": frame["altitude_m"].to_numpy(dtype=np.float32),
        "delta_flap_rad": frame["delta_flap_rad"].to_numpy(dtype=np.float32),
    }


def sample_state(
    row: pd.Series, sampling: ResidualStateSampling, rng: np.random.Generator
) -> np.ndarray:
    return np.array(
        [
            float(row["alpha_ref_rad"]) + rng.normal(0.0, sampling.alpha_error_std_rad),
            float(row["q_ref_radps"]) + rng.normal(0.0, sampling.q_error_std_radps),
            float(row["theta_ref_rad"]) + rng.normal(0.0, sampling.theta_error_std_rad),
        ],
        dtype=float,
    )


def nominal_derivatives_from_row(
    *,
    state: np.ndarray,
    delta_flap_rad: float,
    row: pd.Series,
    vehicle: VehicleParams,
    aero: AeroParams,
) -> np.ndarray:
    schedule = {
        "mach": float(row["mach"]),
        "altitude_m": float(row["altitude_m"]),
        "velocity_mps": float(row["velocity_mps"]),
        "dynamic_pressure_pa": float(row["dynamic_pressure_pa"]),
    }
    moment_nm, _cm, _effectiveness = scheduled_pitching_moment(
        state=state,
        delta_flap_rad=delta_flap_rad,
        schedule=schedule,
        vehicle=vehicle,
        aero=aero,
    )
    q_dot = moment_nm / vehicle.pitch_inertia_kgm2
    alpha_dot = state[1] - 0.22 * state[0]
    theta_dot = state[1]
    return np.array([alpha_dot, q_dot, theta_dot], dtype=float)


def truth_derivatives_from_row(
    *,
    state: np.ndarray,
    delta_flap_rad: float,
    row: pd.Series,
    vehicle: VehicleParams,
    aero: AeroParams,
    scenario: UncertaintyScenario,
) -> np.ndarray:
    return uncertain_derivatives(
        state=state,
        delta_flap_rad=delta_flap_rad,
        row=row,
        vehicle=vehicle,
        aero=aero,
        scenario=scenario,
    )


def split_dataset(
    dataset: dict[str, np.ndarray],
    split: ResidualSplitConfig,
    seed: int,
) -> dict[str, dict[str, np.ndarray]]:
    fractions = np.array(
        [split.train_fraction, split.val_fraction, split.test_fraction], dtype=float
    )
    if not np.isclose(fractions.sum(), 1.0):
        raise ValueError("Train/val/test split fractions must sum to 1.0.")
    sample_ids = dataset["sample_id"]
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(sample_ids))
    train_end = int(round(split.train_fraction * len(sample_ids)))
    val_end = train_end + int(round(split.val_fraction * len(sample_ids)))
    partitions = {
        "train": permutation[:train_end],
        "val": permutation[train_end:val_end],
        "test": permutation[val_end:],
    }
    return {
        name: {key: value[indices] for key, value in dataset.items() if key != "frame"}
        for name, indices in partitions.items()
    }


def build_metadata(
    config: Phase8Config,
    dataset: dict[str, np.ndarray],
    splits: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    targets = dataset["y"]
    return {
        "seed": config.seed,
        "feature_names": FEATURE_NAMES,
        "target_names": TARGET_NAMES,
        "total_samples": int(len(dataset["sample_id"])),
        "split_counts": {
            name: int(len(split["sample_id"])) for name, split in splits.items()
        },
        "tiers": [
            {
                "name": tier.name,
                "scenario_count": tier.scenario_count,
                "samples_per_scenario": tier.samples_per_scenario,
            }
            for tier in config.tiers
        ],
        "target_mean": dict(
            zip(TARGET_NAMES, targets.mean(axis=0).astype(float), strict=True)
        ),
        "target_std": dict(
            zip(TARGET_NAMES, targets.std(axis=0).astype(float), strict=True)
        ),
        "note": (
            "Residual targets are x_dot_true - x_dot_nominal. In the current "
            "reduced-order model, uncertainty primarily changes q_dot through "
            "aerodynamic moment and density scaling."
        ),
    }


def write_residual_dataset_figures(
    dataset: dict[str, np.ndarray], output_dir: Path
) -> dict[str, Path]:
    frame: pd.DataFrame = dataset["frame"]
    distribution_path = output_dir / "residual_distribution.png"
    mach_alt_path = output_dir / "residual_vs_mach_altitude.png"
    flap_path = output_dir / "residual_vs_flap.png"

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8))
    for ax, target in zip(axes, TARGET_NAMES, strict=True):
        ax.hist(frame[target], bins=40, color="#1f77b4", alpha=0.82)
        ax.set_title(target)
        ax.set_xlabel("Residual")
    axes[0].set_ylabel("Sample count")
    fig.tight_layout()
    fig.savefig(distribution_path, dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    sc0 = axes[0].scatter(
        frame["mach"],
        frame["residual_q_dot"],
        c=frame["altitude_m"] / 1000.0,
        s=10,
        alpha=0.6,
    )
    axes[0].set_xlabel("Mach")
    axes[0].set_ylabel("Residual q_dot (rad/s^2)")
    fig.colorbar(sc0, ax=axes[0], label="Altitude (km)")
    sc1 = axes[1].scatter(
        frame["altitude_m"] / 1000.0,
        frame["residual_q_dot"],
        c=frame["mach"],
        s=10,
        alpha=0.6,
    )
    axes[1].set_xlabel("Altitude (km)")
    axes[1].set_ylabel("Residual q_dot (rad/s^2)")
    fig.colorbar(sc1, ax=axes[1], label="Mach")
    fig.tight_layout()
    fig.savefig(mach_alt_path, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.scatter(
        frame["delta_flap_rad"],
        frame["residual_q_dot"],
        c=frame["dynamic_pressure_pa"] / 1000.0,
        s=10,
        alpha=0.58,
    )
    ax.set_xlabel("Flap deflection (rad)")
    ax.set_ylabel("Residual q_dot (rad/s^2)")
    ax.set_title("Residual Moment Error Changes With Flap and qbar")
    fig.tight_layout()
    fig.savefig(flap_path, dpi=160)
    plt.close(fig)

    return {
        "residual_distribution_figure": distribution_path,
        "residual_vs_mach_altitude_figure": mach_alt_path,
        "residual_vs_flap_figure": flap_path,
    }
