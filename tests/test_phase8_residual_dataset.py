import numpy as np

from reentry_mpc.longitudinal import load_phase1_config
from reentry_mpc.phase2 import build_reference_profile, load_phase2_config
from reentry_mpc.phase5 import load_phase5_config
from reentry_mpc.phase8 import (
    Phase8Config,
    ResidualDatasetTier,
    ResidualSplitConfig,
    ResidualStateSampling,
    generate_residual_samples,
    load_phase8_config,
    run_phase8_residual_dataset,
    split_dataset,
)


def test_phase8_dataset_shapes(tmp_path):
    artifacts = run_phase8_residual_dataset(output_dir=tmp_path)
    train = np.load(artifacts["train_npz"])
    assert train["X"].shape[1] == 9
    assert train["y"].shape[1] == 3
    assert train["X"].shape[0] == train["y"].shape[0]


def test_phase8_seed_is_deterministic():
    config = load_phase8_config("configs/phase8_residual_dataset.yaml")
    tiny = Phase8Config(
        seed=config.seed,
        phase1_config=config.phase1_config,
        phase2_config=config.phase2_config,
        phase5_config=config.phase5_config,
        tiers=[
            ResidualDatasetTier("moderate", scenario_count=2, samples_per_scenario=4)
        ],
        state_sampling=ResidualStateSampling(
            alpha_error_std_rad=0.01,
            q_error_std_radps=0.01,
            theta_error_std_rad=0.01,
            flap_min_rad=-0.1,
            flap_max_rad=0.1,
        ),
        split=config.split,
    )
    plant = load_phase1_config(tiny.phase1_config)
    reference = build_reference_profile(load_phase2_config(tiny.phase2_config))
    phase5 = load_phase5_config(tiny.phase5_config)
    tiers = {tier.name: tier for tier in phase5.tiers}

    first = generate_residual_samples(
        config=tiny, plant_config=plant, reference_profile=reference, phase5_tiers=tiers
    )
    second = generate_residual_samples(
        config=tiny, plant_config=plant, reference_profile=reference, phase5_tiers=tiers
    )

    np.testing.assert_allclose(first["X"], second["X"])
    np.testing.assert_allclose(first["y"], second["y"])


def test_phase8_split_sample_ids_are_disjoint():
    dataset = {
        "X": np.arange(90, dtype=np.float32).reshape(10, 9),
        "y": np.arange(30, dtype=np.float32).reshape(10, 3),
        "sample_id": np.arange(10, dtype=np.int64),
        "scenario_id": np.arange(10, dtype=np.int64),
        "tier": np.array(["moderate"] * 10),
        "mach": np.arange(10, dtype=np.float32),
        "altitude_m": np.arange(10, dtype=np.float32),
        "delta_flap_rad": np.arange(10, dtype=np.float32),
    }
    splits = split_dataset(
        dataset,
        ResidualSplitConfig(0.6, 0.2, 0.2),
        seed=123,
    )
    ids = {name: set(split["sample_id"].tolist()) for name, split in splits.items()}
    assert ids["train"].isdisjoint(ids["val"])
    assert ids["train"].isdisjoint(ids["test"])
    assert ids["val"].isdisjoint(ids["test"])
